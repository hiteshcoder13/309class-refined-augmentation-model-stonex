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

IMG_SIZE    = 224
EMBED_DIM   = 256
PROJ_DIM    = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

VAL_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
        emb  = F.normalize(self.projector(feat), dim=-1)

        if return_embedding:
            return emb

        return emb, self.classifier(emb)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    splits_path = Path(
        "/home/Unthinkable/Downloads/stonex-preprocessing"
        "/309refinedaugmodel/splits (5).pkl"
    )
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    family_names = splits["FAMILY_NAMES"]
    num_classes  = splits["NUM_CLASSES"]

    model = StoneEmbedder(num_classes)

    ckpt_path = Path(
        "/home/Unthinkable/Downloads/stonex-preprocessing"
        "/309refinedaugmodel/best_stone_model.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    model.to(DEVICE)
    model.eval()

    return model, family_names


# --------------------------------------------------
# INFERENCE HELPERS
# --------------------------------------------------

def image_to_tensor(image_bgr):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor    = VAL_TF(image=image_rgb)["image"]
    return tensor.unsqueeze(0).to(DEVICE)


def predict_image(image_np, model, family_names, top_k=5):

    tensor = image_to_tensor(image_np)

    with torch.no_grad():
        _, logits = model(tensor)
        probs     = F.softmax(logits, dim=-1)[0]
        top_probs, top_idxs = torch.topk(probs, min(top_k, len(family_names)))

    return [
        {
            "family":     family_names[int(idx)],
            "confidence": float(prob),
        }
        for idx, prob in zip(top_idxs, top_probs)
    ]


# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(
    page_title="Stone Classifier (309 Refined)",
    page_icon="🪨",
    layout="wide",
)

st.title("🪨 Stone Family Classifier (309 Refined)")
st.write(
    "Upload one or multiple stone images and get top-5 predictions "
    "using the 309 refined augmented model."
)

model, family_names = load_model()

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:

    for file in uploaded_files:

        st.divider()

        col1, col2 = st.columns([1, 1])

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col1:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption=file.name,
                use_container_width=True,
            )

        with st.spinner("Predicting..."):
            results = predict_image(image, model, family_names, top_k=5)

        with col2:
            st.subheader("Top Predictions")

            for rank, result in enumerate(results, start=1):
                st.write(f"**{rank}. {result['family']}**")
                st.progress(
                    min(result["confidence"], 1.0),
                    text=f"{result['confidence']:.2%}",
                )
