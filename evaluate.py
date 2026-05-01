import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.modeling.model import EmotionCNN
from src.data_loaders.data_loaders import get_data_loaders
def evaluate(cfg, class_names):
    # Load the saved model
    model = EmotionCNN(num_emotions=cfg["num_emotions"], dropout=cfg["dropout"]).to(cfg["device"])
    # This makes sure it loads correctly on CPU too
    checkpoint = torch.load(
        f"{cfg['save_path']}/weights/{cfg['experiment']}.pt",
        map_location=cfg["device"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # load the test data
    _, _, test_loader = get_data_loaders(cfg)
    # getting predictions and labels
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # calculate metrics
    print(classification_report(all_labels, all_preds, target_names=class_names))
    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Real")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg['save_path']}/plots/confusion_matrix.png")
    plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"
run = 0
experiment = f"emotion_classifier_run_{run}"
CFG = {
    # --- paths ---
    "dataset_path" : "data/YOLO_format",
    "save_dir"     : "checkpoints",

    # --- data ---
    "img_size"     : 96,
    "num_workers"  : 0,
    "device"       : device,

    # --- model ---
    "num_emotions" : 8,
    "dropout"      : 0.3,
    "save_path"    : "outputs",
    "experiment"   : experiment,

    # --- optimiser ---
    "lr"           : 1e-3,
    "weight_decay" : 1e-4,
    "batch_size"   : 42,

    # --- scheduler (ReduceLROnPlateau) ---
    "lr_patience"  : 3,
    "lr_factor"    : 0.5,

    # --- training loop ---
    "epochs"             : 20,
    "early_stop_patience": 7,
}


class_names = ['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sad','Surprise']
evaluate(CFG, class_names)