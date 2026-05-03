import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── 1. Build samples list from YOLO folder structure ──────────────────────────
def load_dataset(dataset_path):
    train_data, valid_data, test_data = [], [], []

    for split in ["train", "valid", "test"]:
        images = os.listdir(f"{dataset_path}/{split}/images")
        for img in images:
            label_path = f"{dataset_path}/{split}/labels/{img[:-3]}txt"
            with open(label_path, "r") as f:
                line = f.readline()
                if line:
                    class_id = int(line.split()[0])

            sample = {
                "image_path": f"{dataset_path}/{split}/images/{img}",
                "class": class_id
            }

            if split == "train":
                train_data.append(sample)
            elif split == "valid":
                valid_data.append(sample)
            elif split == "test":
                test_data.append(sample)

    return train_data, valid_data, test_data


# ── 2. Dataset class ───────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, samples, img_size, transform=None):
        self.samples = samples

        # Use passed-in transform if provided, otherwise fall back to plain resize
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # cv2 reads BGR → convert to RGB → convert to PIL for torchvision transforms
        img_bgr = cv2.imread(item["image_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(img_rgb)      # ← convert to PIL Image

        img_tensor = self.transform(img)

        label = torch.tensor(item["class"], dtype=torch.long)

        return img_tensor, label


# ── 3. Data loaders ────────────────────────────────────────────────────────────
# def get_data_loaders(cfg):
#     train_data, valid_data, test_data = load_dataset(cfg["dataset_path"])

#     train_ds = EmotionDataset(train_data, cfg["img_size"])
#     valid_ds = EmotionDataset(valid_data, cfg["img_size"])
#     test_ds  = EmotionDataset(test_data,  cfg["img_size"])

#     train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
#     valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
#     test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

#     print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")

#     return train_loader, valid_loader, test_loader

def get_data_loaders(cfg):
    train_data, valid_data, test_data = load_dataset(cfg["dataset_path"])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = EmotionDataset(train_data, cfg["img_size"], transform=train_transform)
    valid_ds = EmotionDataset(valid_data, cfg["img_size"], transform=val_transform)
    test_ds  = EmotionDataset(test_data,  cfg["img_size"], transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")
    return train_loader, valid_loader, test_loader