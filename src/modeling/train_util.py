import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


from src.utils.helpers import save_json

def run_epoch(model, loader, cfg, device, optimizer=None):
    """
    Runs one full pass over `loader`.
    If optimizer is provided  -> training mode (backprop).
    If optimizer is None      -> eval mode (no grad).
    Returns dict with avg losses and accuracies.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_fn  = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    n_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader):
        #for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # accumulate losses
            bs = imgs.size(0)       # how many images in this batch (32)
            total_loss += loss.item() * bs   # add loss to total
            correct += (logits.argmax(1) == labels).sum().item()  # count correct
            n_samples  += bs        # count total images seen

            

    return {
        "loss"      : total_loss / n_samples,
        "acc"   : correct / n_samples if n_samples > 0 else 0.0,
    }


def train(cfg, model, train_loader, val_loader, device):
    #print(f"Using device: {device}\n")
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # if no save_path given, fall back to save_dir in cfg

    optimizer = torch.optim.Adam(model.parameters(),lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=cfg["lr_patience"], factor=cfg["lr_factor"])


    best_val_loss    = float("inf")   # best loss so far (starts very high)
    no_improve_count = 0              # how many epochs without improvement
    history = [] # stores all metrics
    from time import time 
    for epoch in range(1, cfg["epochs"] + 1):
        start = time()
        #print("epoch: ", epoch)
        train_m = run_epoch(model, train_loader, cfg, device, optimizer)
        val_m   = run_epoch(model, val_loader,   cfg, device)

        scheduler.step(val_m["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({"epoch":epoch,"train_loss":train_m['loss'],"val_loss":val_m['loss'],"train_acc":train_m['acc'],"val_acc":val_m['acc']})
        

        if val_m["loss"] < best_val_loss:
            best_val_loss    = val_m["loss"]
            no_improve_count = 0

            os.makedirs(f"{cfg['save_path']}/weights", exist_ok=True)
            ckpt_path = f"{cfg['save_path']}/weights/{cfg['experiment']}.pt"
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : best_val_loss
            }, ckpt_path)

            #print(f"  -> saved best model to {ckpt_path}  (val_loss={best_val_loss:.4f})")

   
        else:
            no_improve_count += 1
            if no_improve_count >= cfg["early_stop_patience"]:
                print(f"\nEarly stopping – no improvement for {cfg['early_stop_patience']} epochs.")
                break
        duration = time()- start
        print(
                f"Epoch {epoch:03d}/{cfg['epochs']}  "
                f"| train  loss={train_m['loss']:.4f}  acc={train_m['acc']:.3f}  "
                f"| val    loss={val_m['loss']:.4f}  acc={val_m['acc']:.3f}  "
                f"| lr={current_lr:.2e} | time ={duration :.2f}s | batch_size={cfg['batch_size']} |"
            
        )
    history_path = f"{cfg['save_path']}/history/{cfg['experiment']}.json"
    config_path = f"{cfg['save_path']}/configs/{cfg['experiment']}.json"
    plot_path = f"{cfg['save_path']}/plots/{cfg['experiment']}.png"
    save_json(data=history,json_path=history_path)
    save_json(data=cfg,json_path=config_path)
    plot_training_curves(history,plot_path)

    
    return model, history


def count_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print("=" * 50)
    print(f"Total Parameters in Model: {total:,}".center(50))
    print("=" * 50)


def plot_training_curves(history, save_path=None):
    epochs     = [el["epoch"]     for el in history]
    train_loss = [el["train_loss"] for el in history]
    val_loss   = [el["val_loss"]   for el in history]
    train_acc  = [el["train_acc"]  for el in history]
    val_acc    = [el["val_acc"]    for el in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # loss plot
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss,   label="val", linestyle="--")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True)

    # accuracy plot
    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc,   label="val", linestyle="--")
    axes[1].set_title("acc")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    plt.show()