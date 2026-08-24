from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_cls_ssg


DINO_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
FEATURE_DIM = 768


class PointNetPPWithProjection(nn.Module):
    """PointNet++ encoder + projection + explicit XYZ bounding-box dimensions."""

    def __init__(
        self,
        proj_dim: int = FEATURE_DIM,
        checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = pointnet2_cls_ssg.get_model(num_class=40, normal_channel=False)

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.backbone.load_state_dict(state)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.fuse_dims = nn.Sequential(
            nn.Linear(proj_dim + 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, pc: torch.Tensor, pc_dims: torch.Tensor) -> torch.Tensor:
        # pc: [B, 3, N]
        #
        # PointNet++ performs radius queries and integer indexing based on
        # pairwise distances. Those operations are numerically unsafe in FP16
        # autocast and can produce invalid sentinel indices on CUDA. Keep the
        # PointNet++ branch in FP32 even when the rest of training uses AMP.
        #
        # We also bypass the pretrained ModelNet classification head because
        # only the 1024-D SA3 feature is needed. This avoids running its
        # BatchNorm1d layers, which are unnecessary here and fail for batch=1.
        with torch.autocast(device_type=pc.device.type, enabled=False):
            xyz = pc.float()

            if self.backbone.normal_channel:
                norm = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
            else:
                norm = None

            l1_xyz, l1_points = self.backbone.sa1(xyz, norm)
            l2_xyz, l2_points = self.backbone.sa2(l1_xyz, l1_points)
            _, l3_points = self.backbone.sa3(l2_xyz, l2_points)

            feat = l3_points.squeeze(-1)
            z = self.projector(feat)

            dims = pc_dims.float()
            if dims.dim() == 1:
                dims = dims.unsqueeze(0)

            out = self.fuse_dims(torch.cat([z, dims], dim=1))

        return out


class MMoEDecoder(nn.Module):
    """MMoE decoder copied structurally from the supplied paper implementation."""

    def __init__(
        self,
        in_features: int = FEATURE_DIM,
        num_tasks: int = 3,
        num_experts: int = 64,
        expert_hidden: int = 256,
        tower_hidden: int = 64,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.gate_temperature = gate_temperature

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, expert_hidden),
                    nn.ReLU(),
                    nn.LayerNorm(expert_hidden),
                    nn.Dropout(0.1),
                    nn.Linear(expert_hidden, expert_hidden),
                    nn.ReLU(),
                    nn.LayerNorm(expert_hidden),
                )
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(in_features, num_experts) for _ in range(num_tasks)]
        )
        self.task_towers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(expert_hidden, tower_hidden), nn.ReLU())
                for _ in range(num_tasks)
            ]
        )
        self.heads = nn.ModuleList([nn.Linear(tower_hidden, 1) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor, return_gates: bool = False):
        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        outputs = []
        tower_features = []
        gate_weights_all = []

        for task_idx in range(self.num_tasks):
            logits = self.gates[task_idx](x) / self.gate_temperature
            weights = F.softmax(logits, dim=1)
            mixed = torch.sum(expert_stack * weights.unsqueeze(-1), dim=1)
            tower = self.task_towers[task_idx](mixed)
            tower_features.append(tower)
            outputs.append(self.heads[task_idx](tower))
            if return_gates:
                gate_weights_all.append(weights)

        outputs = torch.cat(outputs, dim=1)
        if return_gates:
            return outputs, tower_features, gate_weights_all
        return outputs, tower_features


class UncertaintyMultiTaskLoss(nn.Module):
    """
    Kendall-style uncertainty regression loss used by the supplied code.

    By default this intentionally reproduces the supplied implementation:
    all three regression targets contribute for every acquisition and the
    visibility vector is ignored. Set use_visibility=True for the masked variant.
    """

    def __init__(self, num_tasks: int = 3, use_visibility: bool = False):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.use_visibility = use_visibility

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = preds.new_tensor(0.0)
        for task_idx in range(preds.shape[1]):
            squared = (preds[:, task_idx] - targets[:, task_idx]) ** 2
            if self.use_visibility and visibility is not None:
                weights = visibility[:, task_idx]
                denom = weights.sum().clamp_min(1.0)
                mse = (squared * weights).sum() / denom
            else:
                mse = squared.mean()
            precision = torch.exp(-self.log_vars[task_idx])
            total = total + 0.5 * precision * mse + 0.5 * self.log_vars[task_idx]
        return total


class GranuloPaperModel(nn.Module):
    """
    Two DINOv3 ViT-B/16 views + PointNet++ -> elementwise max fusion -> 64-expert MMoE.

    This is the frozen DINOv3 ViT-B/16 backbone used by Experiment 1.

    The baseline code instantiates two frozen image encoders with identical
    pretrained weights. Here a single frozen encoder is shared between A and B;
    because the image encoders are frozen and architecturally identical, this
    yields the same feature function while using substantially less GPU memory.
    """

    def __init__(
        self,
        pointnet_checkpoint: str | Path | None,
        num_experts: int = 64,
        pretrained_dino: bool = True,
        freeze_dino: bool = True,
    ):
        super().__init__()
        self.image_encoder = timm.create_model(
            DINO_MODEL_NAME,
            pretrained=pretrained_dino,
            num_classes=0,
        )
        if freeze_dino:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

        self.point_encoder = PointNetPPWithProjection(
            proj_dim=FEATURE_DIM,
            checkpoint_path=pointnet_checkpoint,
            freeze_backbone=False,
        )
        self.decoder = MMoEDecoder(
            in_features=FEATURE_DIM,
            num_tasks=3,
            num_experts=num_experts,
            expert_hidden=256,
            tower_hidden=64,
        )

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        point_cloud: torch.Tensor,
        pc_dims: torch.Tensor,
        return_gates: bool = False,
    ):
        # DINOv3 is frozen for the controlled backbone-swap experiment. Avoid keeping activations.
        with torch.no_grad():
            features_a = self.image_encoder(image_a)
            features_b = self.image_encoder(image_b)

        # Dataset provides [B, N, 3]; PointNet++ expects [B, 3, N].
        pc = point_cloud.transpose(1, 2).contiguous()
        features_pc = self.point_encoder(pc, pc_dims)

        fused, _ = torch.max(
            torch.stack([features_a, features_b, features_pc], dim=0),
            dim=0,
        )

        return self.decoder(fused, return_gates=return_gates)