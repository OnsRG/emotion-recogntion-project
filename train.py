import sys
#sys.path.append(r"C:\Users\DELL\pythonprojects\emotion-recogntion-project")

import torch
from src.data_loaders.data_loaders import get_data_loaders
#from src.modeling.model import EmotionCNN
from src.modeling.resnet_model import EmotionResNet
from src.modeling.train_util import train, count_total_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"

run = 5
experiment = f"emotion_classifier_run_{run}"

CFG = {
    # --- paths ---
    "dataset_path" : "data/YOLO_format",
    "save_dir"     : "checkpoints",

    # --- data ---
    "img_size"     : 224,  # ResNet native size
    "num_workers"  : 0,
    "device"       : device,

    # --- model ---
    "num_emotions" : 8,
    "dropout"      : 0.5,
    "save_path"    : "outputs",
    "experiment"   : experiment,

    # --- optimiser ---
    "lr"           : 3e-4,
    "weight_decay" : 1e-3,
    "batch_size"   : 32,

    # --- scheduler (ReduceLROnPlateau) ---
    "lr_patience"  : 5,
    "lr_factor"    : 0.3,

    # --- training loop ---
    "epochs"             : 50,
    "early_stop_patience": 15,
}

#load the data 
train_loader, valid_loader, test_loader = get_data_loaders(CFG)

#model = EmotionCNN(num_emotions=CFG["num_emotions"], dropout=CFG["dropout"]).to(CFG['device'])
model = EmotionResNet(num_emotions=CFG["num_emotions"], dropout=CFG["dropout"]).to(CFG['device'])
count_total_parameters(model)

model, history = train(cfg=CFG, model=model, train_loader=train_loader, val_loader=valid_loader, device=CFG['device'])
