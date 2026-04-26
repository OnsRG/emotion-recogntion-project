import os
import numpy as np
import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_emotions, dropout):
        super().__init__()

        # Shared conv backbone
        self.backbone = nn.Sequential(
            self._conv_block(3,   32),    # -> (B, 32,  H/2,  W/2)
            self._conv_block(32,  64),    # -> (B, 64,  H/4,  W/4)
            self._conv_block(64, 128),    # -> (B, 128, H/8,  W/8)
            nn.AdaptiveAvgPool2d((4, 4))  # -> (B, 128, 4,    4)
        )

        flat_dim = 128 * 4 * 4  # 2048

        # Head 2 – emotion (multi-class)
        self.emotion_head = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        feats = self.backbone(x).flatten(1)
        #return self.legibility_head(feats), self.number_head(feats)
        return self.emotion_head(feats)
