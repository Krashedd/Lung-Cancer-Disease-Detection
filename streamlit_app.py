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
from ultralytics import YOLO

# ---------- Paths ----------
ROOT = Path(r"C:\Programming\Deploy James")
WEIGHTS_DIR = ROOT / "Weights"
DENSENET_WEIGHTS = WEIGHTS_DIR / "cls_densenet201.pt"
YOLO_WEIGHTS     = WEIGHTS_DIR / "yolo_cls_best.pt"
LABELS_PATH      = WEIGHTS_DIR / "labels.json"

# ---------- Device ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- UI ----------
st.set_page_config(page_title="Lung X-ray Ensemble — DenseNet201 + YOLOv8-cls", layout="centered")
st.title("Lung X-ray Ensemble — DenseNet201 + YOLOv8-cls")
st.caption(f"Device: **{device.type.upper()}** | Weights folder: `{WEIGHTS_DIR}`")

# ---------- Utilities ----------
class LetterboxSquare:
    def __init__(self, size=448, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill
    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        s = self.size
        scale = min(s / w, s / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        pad_w, pad_h = s - nw, s - nh
        pad_left = pad_w // 2
        pad_top  = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        # create padded canvas
        canvas = Image.new("RGB", (s, s), self.fill)
        canvas.paste(img, (pad_left, pad_top))
        return canvas

letterbox_448 = LetterboxSquare(448)

def to_tensor_normalized(pil_img: Image.Image) -> torch.Tensor:
    # ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = np.asarray(pil_img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
    return (t - mean) / std

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["classes"]

CLASSES = load_labels()
NUM_CLASSES = len(CLASSES)

# ---------- Model wrappers ----------
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
        self.model = YOLO(str(weights_path))
        # warmup may reduce initial latency on GPU

    @torch.no_grad()
    def predict_proba(self, pil_img: Image.Image) -> np.ndarray:
        img = letterbox_448(pil_img)  # keep exact same preprocessing
        res = self.model(img, imgsz=448, verbose=False)
        # Ultralytics returns probs in .probs.data
        return np.asarray(res[0].probs.data, dtype=np.float32)

@st.cache_resource
def load_ensemble():
    dn = DenseNetWrapper(DENSENET_WEIGHTS, NUM_CLASSES)
    yv = YOLOClsWrapper(YOLO_WEIGHTS)
    return dn, yv

densenet_model, yolo_model = load_ensemble()

def ensemble_predict(pil_img: Image.Image, w_dn=0.5, w_yolo=0.5):
    p1 = densenet_model.predict_proba(pil_img)
    p2 = yolo_model.predict_proba(pil_img)
    p  = w_dn * p1 + w_yolo * p2
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

# ---------- Sidebar controls ----------
st.sidebar.header("Ensemble Settings")
w_dn = st.sidebar.slider("DenseNet201 weight", 0.0, 1.0, 0.5, 0.05)
w_yolo = 1.0 - w_dn
st.sidebar.write(f"YOLOv8-cls weight: **{w_yolo:.2f}**")
topk = st.sidebar.number_input("Top-k", min_value=1, max_value=max(1, NUM_CLASSES), value=min(5, NUM_CLASSES), step=1)

# ---------- Input tabs ----------
tab1, tab2 = st.tabs(["📁 Upload Image", "🔗 Image URL"])

image: Image.Image | None = None

with tab1:
    up = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
    if up is not None:
        try:
            image = Image.open(up).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read image: {e}")

with tab2:
    url = st.text_input("Paste an image URL")
    if st.button("Load from URL", use_container_width=True):
        if url.strip():
            try:
                image = fetch_image_from_url(url.strip())
                st.image(image, caption="Image from URL", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to download image: {e}")

# ---------- Predict ----------
if image is not None:
    if st.button("Run Ensemble Inference", type="primary", use_container_width=True):
        with st.spinner("Inferring..."):
            idx, probs = ensemble_predict(image, w_dn=w_dn, w_yolo=w_yolo)
        st.success(f"Predicted: **{CLASSES[idx]}**")
        show_topk(probs, k=int(topk))

        # raw probs table
        with st.expander("View raw probabilities"):
            for i, cname in enumerate(CLASSES):
                st.write(f"{cname}: {probs[i]:.6f}")

        # individual model outputs (debug)
        with st.expander("Model breakdown (DenseNet201 vs YOLOv8-cls)"):
            p_dn = densenet_model.predict_proba(image)
            p_yv = yolo_model.predict_proba(image)
            st.write("DenseNet201 top-1:", CLASSES[int(np.argmax(p_dn))], f"({p_dn.max()*100:.2f}%)")
            st.write("YOLOv8-cls top-1:", CLASSES[int(np.argmax(p_yv))], f"({p_yv.max()*100:.2f}%)")

else:
    st.info("Upload an image or provide a URL to begin.")
