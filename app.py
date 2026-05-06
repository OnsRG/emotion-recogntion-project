import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.cm as cm
from torchvision import transforms, models
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear',
               'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Anger':    '#FF4B4B', 'Contempt': '#FF8C00',
    'Disgust':  '#9ACD32', 'Fear':     '#9B59B6',
    'Happy':    '#FFD700', 'Neutral':  '#95A5A6',
    'Sad':      '#3498DB', 'Surprise': '#FF69B4',
}

EMOTION_EMOJIS = {
    'Anger': '😠', 'Contempt': '😒', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Neutral':  '😐', 'Sad':     '😢', 'Surprise': '😲',
}

BEST = {
    "EmotionCNN":    {"weights": "outputs/weights/emotion_classifier_run_0.pt", "img_size": 96},
    "EmotionResNet": {"weights": "outputs/weights/emotion_classifier_run_4.pt", "img_size": 224},
    "YOLO":          {"weights": "runs/detect/outputs/yolo/emotion_yolo_run_0/weights/best.pt", "img_size": 96},
}

# ── Model definitions ──────────────────────────────────────────────────────────
class EmotionCNN(nn.Module):
    def __init__(self, num_emotions=8, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            self._conv_block(3,   32),
            self._conv_block(32,  64),
            self._conv_block(64, 128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.emotion_head(self.backbone(x).flatten(1))


class EmotionResNet(nn.Module):
    def __init__(self, num_emotions=8, dropout=0.3):
        super().__init__()
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, num_emotions)
        )

    def forward(self, x):
        return self.model(x)


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_gradcam_layer(model, model_type):
    if model_type == "EmotionCNN":
        return model.backbone[2]
    elif model_type == "EmotionResNet":
        return model.model.layer4[-1]
    return None


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha=0.38) -> Image.Image:
    img_size    = (400, 400)
    img_arr     = np.array(image.resize(img_size)).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img_size)) / 255.0
    heatmap     = cm.inferno(cam_resized)[:, :, :3]
    blended     = (1 - alpha) * img_arr + alpha * heatmap
    blended     = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pytorch_model(model_type, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EmotionCNN() if model_type == "EmotionCNN" else EmotionResNet()
    loaded = False
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        loaded = True
    model.to(device).eval()
    return model, device, loaded


@st.cache_resource
def load_yolo_model(weights_path):
    try:
        from ultralytics import YOLO
        if os.path.exists(weights_path):
            return YOLO(weights_path), True
        return None, False
    except ImportError:
        return None, False


# ── Inference ──────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image, img_size: int) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(image).unsqueeze(0)


def predict_pytorch(model, tensor, device):
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(device)), dim=1).cpu().numpy()[0]
    return CLASS_NAMES[int(np.argmax(probs))], probs


def predict_yolo(yolo_model, image: Image.Image):
    results = yolo_model(image, verbose=False)
    r = results[0]
    if r.probs is None:
        if r.boxes is None or len(r.boxes) == 0:
            probs = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
        else:
            boxes    = r.boxes
            best_idx = int(boxes.conf.argmax())
            class_id = int(boxes.cls[best_idx].item())
            conf     = float(boxes.conf[best_idx].item())
            probs    = np.zeros(len(CLASS_NAMES))
            if class_id < len(CLASS_NAMES):
                probs[class_id] = conf
            else:
                probs = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
    else:
        probs = r.probs.data.cpu().numpy()
    return CLASS_NAMES[int(np.argmax(probs))], probs


# ── UI components ──────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def confidence_chart(probs, highlight_color):
    winner = CLASS_NAMES[np.argmax(probs)]
    colors = [
        EMOTION_COLORS[CLASS_NAMES[i]] if CLASS_NAMES[i] == winner
        else "rgba(128, 128, 128, 0.3)"
        for i in range(len(CLASS_NAMES))
    ]
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=[round(float(p) * 100, 2) for p in probs],
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
        xaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color="white"), # Defaulting to white for Plotly
        height=260,
        margin=dict(t=10, b=10, l=0, r=0),
        showlegend=False,
    )
    return fig


def top3_html(probs, color):
    sorted_idx = np.argsort(probs)[::-1][:3]
    html = ""
    for rank, idx in enumerate(sorted_idx):
        name  = CLASS_NAMES[idx]
        pct   = probs[idx] * 100
        c     = EMOTION_COLORS[name]
        bold  = "bold" if rank == 0 else "normal"
        size  = "15px" if rank == 0 else "13px"
        # Using CSS var for text color and high contrast borders
        html += f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:10px 14px;margin-bottom:8px;border-radius:12px;
                    background:var(--secondary-background-color); border:2px solid {c if rank == 0 else 'rgba(128,128,128,0.4)'};">
            <span style="font-size:{size};font-weight:{bold};color:var(--text-color);">{EMOTION_EMOJIS[name]} {name}</span>
            <span style="font-size:{size};font-weight:bold;color:{c};">{pct:.1f}%</span>
        </div>"""
    return html


def heatmap_legend_html():
    return """
    <div style="display:flex;align-items:center;gap:8px;margin-top:12px;padding:5px;">
        <div style="width:120px;height:10px;border-radius:4px;
                    background:linear-gradient(to right,#000004,#56106e,#bb3754,#f98c09,#fcffa4); border: 1px solid #888;"></div>
        <span style="font-size:11px; font-weight:700; color:var(--text-color);">LOW</span>
        <span style="font-size:11px; font-weight:700; color:var(--text-color); margin-left:auto;">HIGH</span>
    </div>"""


# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    
    /* IMPROVED CONTRAST FOR LABELS */
    .label {
        font-size: 13px; font-weight: 900; color: var(--text-color);
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;
        opacity: 0.9;
    }
    
    .stImage img {
        border: 2px solid rgba(128,128,128,0.3);
        border-radius: 15px;
    }
    
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed rgba(128,128,128,0.6) !important;
        border-radius: 20px !important;
        background: var(--secondary-background-color) !important;
    }

    /* Target sidebar sub-header specifically for contrast */
    .sidebar-sub {
        color: var(--text-color);
        font-size: 14px;
        font-weight: 500;
        opacity: 1.0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Emotion Classifier")
    # Wrap in a class to apply high contrast CSS
    st.markdown("<div class='sidebar-sub'>Read faces. Understand emotions.</div>", 
                unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="label">Model Configuration</div>', unsafe_allow_html=True)
    model_type = st.selectbox("", list(BEST.keys()), label_visibility="collapsed")
    cfg = BEST[model_type]

    weights_exist = os.path.exists(cfg["weights"])
    st.markdown(
        f"<span style='font-size:13px; font-weight:800; color:{'#4CAF50' if weights_exist else '#FF4B4B'};'>"
        f"{'✅ WEIGHTS READY' if weights_exist else '⚠️ WEIGHTS MISSING'}</span>",
        unsafe_allow_html=True
    )

    st.divider()
    show_gradcam = st.toggle(
        "🔥 Show AI Attention",
        value=True,
        disabled=(model_type == "YOLO"),
    )

# ── Auto-load ──────────────────────────────────────────────────────────────────
if st.session_state.get("loaded_model") != model_type:
    with st.spinner(f"Loading {model_type}..."):
        if model_type == "YOLO":
            yolo, loaded = load_yolo_model(cfg["weights"])
            st.session_state.update({"yolo": yolo, "loaded": loaded})
        else:
            model, device, loaded = load_pytorch_model(model_type, cfg["weights"])
            st.session_state.update({"model": model, "device": device, "loaded": loaded})
        st.session_state["loaded_model"] = model_type

# ── Main UI ────────────────────────────────────────────────────────────────────
_, center_col, _ = st.columns([1, 3, 1])
with center_col:
    st.markdown("""
        <div style="text-align:center;padding:20px 0;">
            <div style="font-size:52px;">🎭</div>
            <div style="font-size:32px;font-weight:900;margin-top:8px;color:var(--text-color);">Emotion Classifier</div>
            <div style="font-size:16px;opacity:0.8;margin-top:4px;margin-bottom:24px;color:var(--text-color);">
                Upload a facial image for instant sentiment analysis
            </div>
        </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "webp"],
    )

if not uploaded:
    st.stop()

# ── Inference ──────────────────────────────────────────────────────────────────
image = Image.open(uploaded).convert("RGB")
mtype = st.session_state.get("loaded_model")

if mtype == "YOLO":
    yolo = st.session_state.get("yolo")
    if not yolo:
        st.error("YOLO model not loaded.")
        st.stop()
    emotion, probs = predict_yolo(yolo, image)
    cam_image = None
else:
    tensor         = preprocess(image, cfg["img_size"])
    model          = st.session_state["model"]
    device         = st.session_state["device"]
    emotion, probs = predict_pytorch(model, tensor, device)
    cam_image = None
    if show_gradcam:
        target_layer = get_gradcam_layer(model, mtype)
        if target_layer:
            gradcam   = GradCAM(model, target_layer)
            cam       = gradcam.generate(tensor.to(device))
            cam_image = overlay_heatmap(image, cam)

color      = EMOTION_COLORS[emotion]
emoji      = EMOTION_EMOJIS[emotion]
confidence = float(probs[CLASS_NAMES.index(emotion)]) * 100

# ── Reactive top border ────────────────────────────────────────────────────────
st.markdown(f"""
    <div style="position:fixed;top:0;left:0;right:0;height:6px;
                background:{color};z-index:9999;"></div>
""", unsafe_allow_html=True)

# ── Content Layout ─────────────────────────────────────────────────────────────
if cam_image:
    img_col1, img_col2 = st.columns(2, gap="large")
    with img_col1:
        st.markdown('<div class="label">📷 Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with img_col2:
        st.markdown('<div class="label">🔥 AI Focus Map</div>', unsafe_allow_html=True)
        st.image(cam_image, use_container_width=True)
        st.markdown(heatmap_legend_html(), unsafe_allow_html=True)
else:
    col, _ = st.columns([1, 1])
    with col:
        st.markdown('<div class="label">📷 Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

st.divider()

# ── Prediction Results ──────────────────────────────────────────────────────────
pred_col, chart_col = st.columns([1, 2], gap="large")

with pred_col:
    st.markdown('<div class="label">🔍 Detected Emotion</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background:var(--secondary-background-color); border:4px solid {color};
                    border-radius:24px;padding:35px 20px;text-align:center;margin-bottom:20px;">
            <div style="font-size:80px;line-height:1;">{emoji}</div>
            <div style="font-size:38px;font-weight:900;color:{color};margin-top:12px;
                        letter-spacing:-1px;text-transform:uppercase;">{emotion}</div>
            <div style="font-size:16px;font-weight:800;color:var(--text-color);opacity:0.9;margin-top:6px;">
                {confidence:.1f}% CONFIDENCE
            </div>
        </div>
        <div class="label" style="margin-top:14px;">Next Best Matches</div>
        {top3_html(probs, color)}
    """, unsafe_allow_html=True)

with chart_col:
    st.markdown('<div class="label">📊 Probability Spectrum</div>', unsafe_allow_html=True)
    st.plotly_chart(confidence_chart(probs, color), use_container_width=True)