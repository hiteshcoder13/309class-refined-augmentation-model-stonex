import cv2
import pickle
import numpy as np
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IMG_SIZE = 224
EMBED_DIM = 256
PROJ_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On the cloud deployment, reference images are not uploaded, so this will fail-safe
REFERENCE_DATA_DIR = Path("non-aug-complete-data/merged-rawdata-both")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SIMILAR_IMAGES_PER_FAMILY = 5


# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

VAL_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class StoneEmbedder(nn.Module):

    def __init__(self, num_classes, embed_dim=EMBED_DIM):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=IMG_SIZE,
        )

        bdim = self.backbone.num_features

        self.projector = nn.Sequential(
            nn.Linear(bdim, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(PROJ_DIM, embed_dim),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embedding=False):

        feat = self.backbone(x)

        emb = F.normalize(
            self.projector(feat),
            dim=-1
        )

        if return_embedding:
            return emb

        return emb, self.classifier(emb)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Load splits from the repository directory
    splits_path = Path("309refinedaugmodel/splits (5).pkl")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    family_names = splits["FAMILY_NAMES"]
    num_classes = splits["NUM_CLASSES"]

    model = StoneEmbedder(num_classes)

    # Load checkpoint from the repository directory
    ckpt_path = Path("309refinedaugmodel/best_stone_model.pt")
    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(ckpt["model"])

    model.to(DEVICE)
    model.eval()

    return model, family_names, splits


@st.cache_data(show_spinner=False)
def build_family_image_index(reference_data_dir):

    reference_data_dir = Path(reference_data_dir)

    if not reference_data_dir.exists():
        return {}

    family_index = {}

    for family_dir in sorted(reference_data_dir.iterdir()):

        if not family_dir.is_dir():
            continue

        image_paths = sorted(
            path for path in family_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )

        if image_paths:
            family_index[family_dir.name] = image_paths

    return family_index


def get_reference_names_from_splits(family, splits, limit=SIMILAR_IMAGES_PER_FAMILY):

    names = []
    seen = set()

    for image_path, family_idx in splits["train_records"] + splits["val_records"]:

        if splits["FAMILY_NAMES"][int(family_idx)] != family:
            continue

        image_name = Path(image_path).name

        if image_name in seen:
            continue

        names.append(image_name)
        seen.add(image_name)

        if len(names) == limit:
            break

    return names


def image_to_tensor(image_bgr):

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    tensor = VAL_TF(image=image_rgb)["image"]

    return tensor.unsqueeze(0).to(DEVICE)


def read_image_tensor(image_path):

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        return None

    return image_to_tensor(image_bgr)


@st.cache_data(show_spinner=False)
def embed_family_images(family, image_paths, _model, batch_size=16):

    embeddings = []
    names = []

    for start in range(0, len(image_paths), batch_size):

        batch_paths = image_paths[start:start + batch_size]
        batch_tensors = []
        batch_names = []

        for image_path in batch_paths:

            tensor = read_image_tensor(image_path)

            if tensor is None:
                continue

            batch_tensors.append(tensor)
            batch_names.append(Path(image_path).name)

        if not batch_tensors:
            continue

        batch = torch.cat(batch_tensors, dim=0)

        with torch.no_grad():

            batch_emb = _model(batch, return_embedding=True)

        embeddings.append(batch_emb.cpu().numpy())
        names.extend(batch_names)

    if not embeddings:
        return np.empty((0, EMBED_DIM), dtype=np.float32), []

    return np.concatenate(embeddings, axis=0), names


def find_similar_images(query_embedding, family, model, family_image_index, splits):

    image_paths = family_image_index.get(family, [])

    if image_paths:

        family_embeddings, image_names = embed_family_images(
            family,
            image_paths,
            model
        )

        if len(image_names) > 0:

            query = query_embedding.detach().cpu().numpy()[0]
            similarities = family_embeddings @ query
            top_idxs = np.argsort(similarities)[::-1][:SIMILAR_IMAGES_PER_FAMILY]

            return [
                {
                    "name": image_names[int(idx)],
                    "similarity": float(similarities[int(idx)]),
                    "source": "local"
                }
                for idx in top_idxs
            ]

    return [
        {
            "name": image_name,
            "similarity": None,
            "source": "splits"
        }
        for image_name in get_reference_names_from_splits(family, splits)
    ]


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_image(image_np, model, family_names, top_k=5):

    tensor = image_to_tensor(image_np)

    with torch.no_grad():

        emb, logits = model(tensor)

        probs = F.softmax(logits, dim=-1)[0]

        top_probs, top_idxs = torch.topk(
            probs,
            min(top_k, len(family_names))
        )

    results = []

    for idx, prob in zip(top_idxs, top_probs):

        results.append({
            "family": family_names[int(idx)],
            "confidence": float(prob)
        })

    return results, emb


# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(
    page_title="Stone Classifier (309 Refined)",
    page_icon="🪨",
    layout="wide"
)

st.title("🪨 Stone Family Classifier (309 Refined)")

st.write(
    "Upload one or multiple stone images and get top predictions using the 309 refined augmented model."
)

model, family_names, splits = load_model()
family_image_index = build_family_image_index(REFERENCE_DATA_DIR)

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        st.divider()

        col1, col2 = st.columns([1, 1])

        file_bytes = np.asarray(
            bytearray(file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        with col1:

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption=file.name,
                use_container_width=True
            )

        with st.spinner("Predicting..."):

            results, query_embedding = predict_image(
                image,
                model,
                family_names,
                top_k=5
            )

        with col2:

            st.subheader("Top Predictions")

            for rank, result in enumerate(results, start=1):

                st.write(
                    f"**{rank}. {result['family']}**"
                )

                st.progress(
                    min(result["confidence"], 1.0),
                    text=f"{result['confidence']:.2%}"
                )

                similar_images = find_similar_images(
                    query_embedding,
                    result["family"],
                    model,
                    family_image_index,
                    splits
                )

                if similar_images:

                    st.caption("Similar matched image names")

                    for match in similar_images:

                        if match["similarity"] is None:
                            st.write(f"- {match['name']}")
                        else:
                            st.write(
                                f"- {match['name']} "
                                f"(similarity {match['similarity']:.3f})"
                            )
