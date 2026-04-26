import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
    def __init__(self, samples, img_size):
        self.samples = samples                    # list of {"image_path", "class"}
        self.transform = transforms.Compose([
            transforms.ToTensor(),                # HWC uint8 → CHW float [0,1]
            transforms.Resize((img_size, img_size)),
            transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean/std
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load image (cv2 reads BGR → convert to RGB)
        img = cv2.cvtColor(cv2.imread(item["image_path"]), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)

        # Emotion label
        label = torch.tensor(item["class"], dtype=torch.long)

        return img_tensor, label


# ── 3. Data loaders ────────────────────────────────────────────────────────────
def get_data_loaders(cfg):
    train_data, valid_data, test_data = load_dataset(cfg["dataset_path"])

    train_ds = EmotionDataset(train_data[:500], cfg["img_size"])
    valid_ds = EmotionDataset(valid_data[:100], cfg["img_size"])
    test_ds  = EmotionDataset(test_data,  cfg["img_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")

    return train_loader, valid_loader, test_loader