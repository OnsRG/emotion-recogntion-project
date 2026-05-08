import os
import numpy as np
import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_emotions, dropout):
        super().__init__()

        self.backbone = nn.Sequential(
            self._conv_block(3,   32),   # 112 → 56
            self._conv_block(32,  64),   # 56  → 28
            self._conv_block(64,  128),  # 28  → 14
            self._conv_block(128, 256),  # 14  → 7  ← NEW
            nn.AdaptiveAvgPool2d((3, 3)) # 7   → 3×3 (less aggressive)
        )

        flat_dim = 256 * 3 * 3  # 2304

        self.emotion_head = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),      nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(128, num_emotions)
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
