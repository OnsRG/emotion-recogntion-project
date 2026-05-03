import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
#from src.modeling.model import EmotionCNN
from src.modeling.resnet_model import EmotionResNet
from src.data_loaders.data_loaders import get_data_loaders


def evaluate(cfg, class_names):
    # Load the saved model
    #model = EmotionCNN(num_emotions=cfg["num_emotions"], dropout=cfg["dropout"]).to(cfg["device"])
    model = EmotionResNet(num_emotions=cfg["num_emotions"], dropout=cfg["dropout"]).to(cfg["device"])
    checkpoint = torch.load(
        f"{cfg['save_path']}/weights/{cfg['experiment']}.pt",
        map_location=cfg["device"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load test data
    _, _, test_loader = get_data_loaders(cfg)

    # Get predictions
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print metrics
    print("=" * 50)
    print(f"EVALUATION: {cfg['experiment']}")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save metrics as JSON ✅
    os.makedirs(f"{cfg['save_path']}/results", exist_ok=True)
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   output_dict=True)
    results_path = f"{cfg['save_path']}/results/{cfg['experiment']}_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Results saved to {results_path} ✅")

    # Confusion matrix ✅
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title(f"Confusion Matrix - {cfg['experiment']}")
    plt.ylabel("Real")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg['save_path']}/plots/{cfg['experiment']}_confusion_matrix.png")
    plt.show()
    print(f"Confusion matrix saved ✅")


# ── Run ────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
run = 3

experiment = f"emotion_classifier_run_{run}"

CFG = {
    "dataset_path"       : "data/YOLO_format",
    "save_dir"           : "checkpoints",
    "img_size"           : 224,  # ResNet native size
    "num_workers"        : 0,
    "device"             : device,
    "num_emotions"       : 8,
    "dropout"            : 0.3,
    "save_path"          : "outputs",
    "experiment"         : experiment,
    "lr"                 : 1e-3,
    "weight_decay"       : 1e-4,
    "batch_size"         : 42,
    "lr_patience"        : 3,
    "lr_factor"          : 0.5,
    "epochs"             : 30,
    "early_stop_patience": 7,
}

class_names = ['Anger', 'Contempt', 'Disgust', 'Fear',
               'Happy', 'Neutral', 'Sad', 'Surprise']

evaluate(CFG, class_names)