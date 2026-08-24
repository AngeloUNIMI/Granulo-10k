from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    """PointNet++ single-scale grouping classifier (standard ModelNet layout)."""

    def __init__(self, num_class: int, normal_channel: bool = True):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz: torch.Tensor):
        batch_size, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(batch_size, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, -1), l3_points


class get_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.nll_loss(pred, target)
