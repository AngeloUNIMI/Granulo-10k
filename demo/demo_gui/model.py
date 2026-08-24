from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class StrandResNet18(nn.Module):
    """Single RGB image -> normalized [height, width, thickness]."""

    def __init__(self, pretrained: bool = False, hidden: int = 256):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.regressor(self.backbone(x))
