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

    legib_loss_fn  = nn.CrossEntropyLoss()
    #number_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # skip non-legible rows

    total_loss = 0.0
    num_correct = 0
    n_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # accumulate losses
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            legib_loss += l1.item()   * bs
            num_loss   += l2.item()   * bs
            n_samples  += bs

            # legibility accuracy
            legib_correct += (legib_logits.argmax(1) == legible).sum().item()

            # number accuracy – only count samples that ARE legible
            mask = legible == 1
            if mask.any():
                preds = num_logits[mask].argmax(1)
                num_correct += (preds == number[mask]).sum().item()
                num_legible += mask.sum().item()

    return {
        "loss"      : total_loss / n_samples,
        "legib_loss": legib_loss / n_samples,
        "num_loss"  : num_loss   / n_samples,
        "legib_acc" : legib_correct / n_samples,
        "num_acc"   : num_correct / num_legible if num_legible > 0 else 0.0,
    }


def train(cfg, model, train_loader, val_loader, device):
    print(f"Using device: {device}\n")
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # if no save_path given, fall back to save_dir in cfg

    optimizer = torch.optim.Adam(model.parameters(),lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print("optimizer: ", optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=cfg["lr_patience"], factor=cfg["lr_factor"])
    print("scheduler: ", scheduler)

    best_val_loss    = float("inf")
    no_improve_count = 0
    history = {'train':[],'valid':[]}
    for epoch in range(1, cfg["epochs"] + 1):
        print("epoch: ", epoch)
        train_m = run_epoch(model, train_loader, cfg, device, optimizer)
        val_m   = run_epoch(model, val_loader,   cfg, device)

        scheduler.step(val_m["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history['train'].append(train_m)
        history['valid'].append(val_m)

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']}  "
            f"| train  loss={train_m['loss']:.4f}  "
            f"legib_acc={train_m['legib_acc']:.3f}  "
            f"num_acc={train_m['num_acc']:.3f}  "
            f"| val  loss={val_m['loss']:.4f}  "
            f"legib_acc={val_m['legib_acc']:.3f}  "
            f"num_acc={val_m['num_acc']:.3f}  "
            f"| lr={current_lr:.2e}"
        )

        if val_m["loss"] < best_val_loss:
            best_val_loss    = val_m["loss"]
            no_improve_count = 0

            # save weights
            ckpt_path = f"{cfg['save_path']}/weights/{cfg['experiment']}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": best_val_loss}, ckpt_path)


            print(f"  -> saved best model + cfg to {ckpt_path}  (val_loss={best_val_loss:.4f})")

        else:
            no_improve_count += 1
            if no_improve_count >= cfg["early_stop_patience"]:
                print(f"\nEarly stopping – no improvement for {cfg['early_stop_patience']} epochs.")
                break

    cfg_path = f"{cfg['save_path']}/configs/{cfg['experiment']}.json"
    save_json(cfg, cfg_path)

    history_path = f"{cfg['save_path']}/history/{cfg['experiment']}.json" 
    save_json(history, history_path)


    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    return model,  history

def count_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print("="*50)
    print(f"📊 Total Parameters in Model: {total:,}".center(50))
    print("="*50)


def plot_training_curves(train_history, val_history):
    """
    train_history / val_history: lists of the dicts returned by run_epoch
    e.g. train_history = [{"loss": 1.2, "legib_acc": 0.6, ...}, ...]
    """
    epochs = range(1, len(train_history) + 1)
    metrics = ["loss", "legib_loss", "num_loss", "legib_acc", "num_acc"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

    for ax, key in zip(axes, metrics):
        train_vals = [m[key] for m in train_history]
        val_vals   = [m[key] for m in val_history]

        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals,   label="val", linestyle="--")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()