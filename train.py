import sys
#sys.path.append(r"C:\Users\DELL\pythonprojects\emotion-recogntion-project")

import torch
from src.data_loaders.data_loaders import get_data_loaders
from src.modeling.model import EmotionCNN
from src.modeling.train_util import train, plot_training_curves, count_total_parameters

import torch
### Training model
from src.modeling.train_util import train

### Setting Model
from src.modeling.model import EmotionCNN
from src.modeling.train_util import count_total_parameters

device = "cuda" if torch.cuda.is_available() else "cpu"

run = 0
experiment = f"emotion_classifier_run_{run}"

CFG = {
    # --- paths ---
    "dataset_path" : r"C:\Users\DELL\pythonprojects\emotion-recogntion-project\data\YOLO_format",
    "save_dir"     : "checkpoints",

    # --- data ---
    "img_size"     : 96,
    "num_workers"  : 0,
    "device"       : device,

    # --- model ---
    "num_emotions" : 8,
    "dropout"      : 0.3,
    "save_path"    : r"C:\Users\DELL\pythonprojects\emotion-recogntion-project\outputs",
    "experiment"   : experiment,

    # --- optimiser ---
    "lr"           : 1e-3,
    "weight_decay" : 1e-4,
    "batch_size"   : 32,

    # --- scheduler (ReduceLROnPlateau) ---
    "lr_patience"  : 3,
    "lr_factor"    : 0.5,

    # --- training loop ---
    "epochs"             : 1,
    "early_stop_patience": 7,
}

#load the data 
train_loader, valid_loader, test_loader = get_data_loaders(CFG)

model = EmotionCNN(num_emotions=CFG["num_emotions"], dropout=CFG["dropout"]).to(CFG['device'])
count_total_parameters(model)
for batch_size in (range(2,65,4)):
    CFG["batch_size"] = batch_size 
    model, history = train(cfg=CFG, model=model, train_loader=train_loader, val_loader=valid_loader, device=CFG['device'])


