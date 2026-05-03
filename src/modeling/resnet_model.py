import torch
import torch.nn as nn
from torchvision import models


class EmotionResNet(nn.Module):
    def __init__(self, num_emotions, dropout):
        super().__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V1")

        # Freeze only early layers (layer1, layer2), fine-tune the rest
        freeze_layers = ["layer1", "layer2", "conv1", "bn1"]
        for name, param in self.model.named_parameters():
            if any(name.startswith(l) for l in freeze_layers):
                param.requires_grad = False

        # Better head with BatchNorm
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_emotions)
        )

    def forward(self, x):
        return self.model(x)