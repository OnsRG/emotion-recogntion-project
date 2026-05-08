import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.cm as cm
from torchvision import transforms
from PIL import Image

# Import the models from your project structure
from src.modeling.model import EmotionCNN
from src.modeling.resnet_model import EmotionResNet

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

# Synced with your partner's CFG settings
BEST = {
    "EmotionCNN":    {"weights": "outputs/weights/emotion_classifier_run_1.pt", "img_size": 112, "dropout": 0.3},
    "EmotionResNet": {"weights": "outputs/weights/emotion_classifier_run_5.pt", "img_size": 224, "dropout": 0.5},
    "YOLO":          {"weights": "outputs/yolo/emotion_yolo_run_0/weights/best.pt", "img_size": 96},
}

# ── Grad-CAM Logic ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
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
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_gradcam_layer(model, model_type):
    if model_type == "EmotionCNN":
        return model.backbone[3] # Targets the new 4th conv block
    elif model_type == "EmotionResNet":
        return model.model.layer4[-1]
    return None

def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha=0.4) -> Image.Image:
    img_size = (400, 400)
    img_arr = np.array(image.resize(img_size)).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img_size)) / 255.0
    heatmap = cm.inferno(cam_resized)[:, :, :3]
    blended = (1 - alpha) * img_arr + alpha * heatmap
    return Image.fromarray(np.clip(blended * 255, 0, 255).astype(np.uint8))

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pytorch_model(model_type, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = config["weights"]
    
    # Passing mandatory arguments from your partner's files
    if model_type == "EmotionCNN":
        model = EmotionCNN(num_emotions=8, dropout=config["dropout"])
    else:
        model = EmotionResNet(num_emotions=8, dropout=config["dropout"])
    
    loaded = False
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        # Extracting model_state from the training checkpoint
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
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

# ── Inference Helpers ──────────────────────────────────────────────────────────
def preprocess(image: Image.Image, img_size: int) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Standard ImageNet normalization
    ])
    return t(image).unsqueeze(0)

# ── UI Layout ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion AI", page_icon="🎭", layout="wide")

with st.sidebar:
    st.title("🎭 Settings")
    model_type = st.selectbox("Choose Model", list(BEST.keys()))
    cfg = BEST[model_type]
    
    weights_ready = os.path.exists(cfg["weights"])
    st.info(f"Weights: {'✅ Found' if weights_ready else '❌ Missing'}")
    
    show_gradcam = st.toggle("Show Attention Map", value=True, disabled=(model_type == "YOLO"))

# Load Model
if model_type == "YOLO":
    model, loaded = load_yolo_model(cfg["weights"])
    device = "cpu"
else:
    model, device, loaded = load_pytorch_model(model_type, cfg)

if not loaded:
    st.error(f"Could not load {model_type} weights at {cfg['weights']}")
    st.stop()

# Main App Logic
st.markdown("<h1 style='text-align: center;'>Facial Emotion Recognition</h1>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a face...", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    # Prediction
    if model_type == "YOLO":
        results = model(image, verbose=False)[0]
        probs = results.probs.data.cpu().numpy() if results.probs is not None else np.zeros(8)
    else:
        tensor = preprocess(image, cfg["img_size"]).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    emotion_idx = np.argmax(probs)
    label = CLASS_NAMES[emotion_idx]
    conf = probs[emotion_idx]
    
    # Layout Columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Fixed parameter for 2026: width='stretch' instead of use_container_width=True
        st.image(image, caption="Original Image", width='stretch')
        
    with col2:
        if model_type != "YOLO" and show_gradcam:
            target = get_gradcam_layer(model, model_type)
            gc = GradCAM(model, target)
            cam = gc.generate(preprocess(image, cfg["img_size"]).to(device))
            # Fixed parameter for 2026: width='stretch'
            st.image(overlay_heatmap(image, cam), caption="AI Focus (Grad-CAM)", width='stretch')
        else:
            st.warning("Heatmap not available for YOLO or disabled.")

    # Results Display
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Detected Emotion", f"{EMOTION_EMOJIS[label]} {label}", f"{conf*100:.1f}%")
        
    with res_col2:
        fig = go.Figure(go.Bar(x=CLASS_NAMES, y=probs, marker_color=[EMOTION_COLORS[c] for c in CLASS_NAMES]))
        fig.update_layout(title="Probability Distribution", height=300, margin=dict(t=30, b=0, l=0, r=0))
        # Fixed parameter for 2026: width='stretch'
        st.plotly_chart(fig, width='stretch')