# streamlit_app.py — Lung Cancer Disease Detection (Ensemble)
# Expected repo layout:
#   - streamlit_app.py
#   - cls_densenet201.pt
#   - yolo_cls_best.pt
#   - labels.json
#
# Inference settings: 448 letterboxed, no augmentations.

import io
import json
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import timm

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
PATH_DENSENET = ROOT / "cls_densenet201.pt"
PATH_YOLO = ROOT / "yolo_cls_best.pt"
PATH_LABELS = ROOT / "labels.json"

# ---------------- UI ----------------
st.set_page_config(page_title="Lung Cancer Disease Detection", layout="centered")
st.title("Lung Cancer Disease Detection")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
st.caption(f"Device: **{device.type.upper()}** • Repo root: `{ROOT.name}`")


# ---------------- Labels ----------------
@st.cache_data
def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["classes"]


if not PATH_LABELS.exists():
    st.error("labels.json not found in repo root.")
    st.stop()

CLASSES = load_labels(PATH_LABELS)
NUM_CLASSES = len(CLASSES)


# ---------------- Preprocess ----------------
def letterbox_448(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = 448
    w, h = img.size
    r = min(s / w, s / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - nw) // 2, (s - nh) // 2))
    return canvas


def to_tensor_normalized(pil_img: Image.Image) -> torch.Tensor:
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = np.asarray(pil_img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
    return (t - mean) / std


# ---------------- Model wrappers ----------------
class DenseNetWrapper(nn.Module):
    def __init__(self, weights_path: Path, num_classes: int):
        super().__init__()
        self.net = timm.create_model("densenet201", pretrained=False, num_classes=num_classes)
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(device)

    @torch.no_grad()
    def predict_proba(self, pil_img: Image.Image) -> np.ndarray:
        img = letterbox_448(pil_img)
        x = to_tensor_normalized(img).unsqueeze(0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = self.net(x)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        return probs


class YOLOClsWrapper:
    def __init__(self, weights_path: Path):
        # lazy import to avoid triggering cv2/ultralytics at app startup
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))

    @torch.no_grad()
    def predict_proba(self, pil_img: Image.Image) -> np.ndarray:
        img = letterbox_448(pil_img)
        res = self.model(img, imgsz=448, verbose=False)
        return np.asarray(res[0].probs.data, dtype=np.float32)


# Lazy-load models (so the app still runs if one weight is missing)
@st.cache_resource
def get_models(has_dn: bool, has_yolo: bool):
    dn = DenseNetWrapper(PATH_DENSENET, NUM_CLASSES) if has_dn else None
    yv = YOLOClsWrapper(PATH_YOLO) if has_yolo else None
    return dn, yv


HAS_DN = PATH_DENSENET.exists()
HAS_YOLO = PATH_YOLO.exists()
if not (HAS_DN or HAS_YOLO):
    st.error("No model weights found. Place cls_densenet201.pt and/or yolo_cls_best.pt in the repo root.")
    st.stop()

densenet_model, yolo_model = get_models(HAS_DN, HAS_YOLO)


def ensemble_predict(pil_img: Image.Image, w_dn=0.5, w_yolo=0.5):
    probs_list = []
    if densenet_model is not None and w_dn > 0:
        probs_list.append(w_dn * densenet_model.predict_proba(pil_img))
    if yolo_model is not None and w_yolo > 0:
        probs_list.append(w_yolo * yolo_model.predict_proba(pil_img))
    p = np.sum(probs_list, axis=0) if len(probs_list) > 1 else probs_list[0]
    idx = int(np.argmax(p))
    return idx, p


def show_topk(probs: np.ndarray, k: int = 5):
    k = min(k, len(probs))
    top_idx = np.argsort(probs)[::-1][:k]
    st.subheader("Top-k probabilities")
    for i in top_idx:
        st.write(f"{CLASSES[i]} — **{probs[i]*100:.2f}%**")


def fetch_image_from_url(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=10) as r:
        data = r.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------- Sidebar ----------------
st.sidebar.header("Ensemble Settings")
if HAS_DN and HAS_YOLO:
    w_dn = st.sidebar.slider("DenseNet201 weight", 0.0, 1.0, 0.5, 0.05)
    w_yolo = 1.0 - w_dn
    st.sidebar.write(f"YOLOv8-cls weight: **{w_yolo:.2f}**")
elif HAS_DN:
    w_dn, w_yolo = 1.0, 0.0
    st.sidebar.info("YOLO weights not found — using DenseNet201 only.")
else:
    w_dn, w_yolo = 0.0, 1.0
    st.sidebar.info("DenseNet weights not found — using YOLOv8-cls only.")
topk = st.sidebar.number_input(
    "Top-k",
    min_value=1,
    max_value=max(1, NUM_CLASSES),
    value=min(5, NUM_CLASSES),
    step=1,
)

# ---------------- Input tabs ----------------
tab1, tab2 = st.tabs(["📁 Upload Image", "🔗 Image URL"])
image = None

with tab1:
    up = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    )
    if up is not None:
        try:
            image = Image.open(up).convert("RGB")
            st.image(image, caption="Uploaded", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read image: {e}")

with tab2:
    url = st.text_input("Paste an image URL")
    if st.button("Load from URL", use_container_width=True):
        if url.strip():
            try:
                image = fetch_image_from_url(url.strip())
                st.image(image, caption="From URL", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to download image: {e}")

# ---------------- Predict ----------------
if image is not None:
    if st.button("Run Inference", type="primary", use_container_width=True):
        with st.spinner("Inferring..."):
            idx, probs = ensemble_predict(image, w_dn=w_dn, w_yolo=w_yolo)
        st.success(f"Prediction: **{CLASSES[idx]}** ({probs[idx]*100:.2f}%)")
        show_topk(probs, int(topk))

        with st.expander("Model breakdown"):
            if densenet_model is not None:
                p_dn = densenet_model.predict_proba(image)
                st.write("DenseNet201:", CLASSES[int(np.argmax(p_dn))], f"({p_dn.max()*100:.2f}%)")
            if yolo_model is not None:
                p_yv = yolo_model.predict_proba(image)
                st.write("YOLOv8-cls:", CLASSES[int(np.argmax(p_yv))], f"({p_yv.max()*100:.2f}%)")
else:
    st.info("Upload an image or provide a URL to begin.")
