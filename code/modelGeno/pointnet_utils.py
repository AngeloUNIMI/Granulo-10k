import sys
import torch
import torch.nn as nn

sys.path.append("Pointnet_Pointnet2_pytorch/models")
import pointnet2_cls_ssg


class PointNetPPWithProjection(nn.Module):
    """
    PointNet++ encoder with projection head and explicit
    geometric-dimension conditioning (W, H, T).
    """

    def __init__(
        self,
        proj_dim=512,
        ckpt_path=(
            "Pointnet_Pointnet2_pytorch/log/classification/"
            "pointnet2_ssg_wo_normals/"
            "checkpoints/best_model.pth"
        ),
        num_classes=40,
        use_normals=False,
        freeze_backbone=False,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. PointNet++ backbone
        # --------------------------------------------------
        self.backbone = pointnet2_cls_ssg.get_model(
            num_class=num_classes,
            normal_channel=use_normals
        )

        # Load pretrained weights
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False
        )
        self.backbone.load_state_dict(ckpt["model_state_dict"])

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --------------------------------------------------
        # 2. Projection head (PointNet++ -> latent)
        # --------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # --------------------------------------------------
        # 3. Dimension fusion head (W, H, T)
        # --------------------------------------------------
        self.fuse_dims = nn.Sequential(
            nn.Linear(proj_dim + 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, pc, pc_dims):
        """
        Args:
            pc:       Tensor [B, 3, N]   point cloud (centered, NOT scaled)
            pc_dims:  Tensor [B, 3]      [width, height, thickness]

        Returns:
            z:        Tensor [B, proj_dim] fused latent representation
        """

        # --------------------------------------------------
        # 1. PointNet++ feature extraction
        # --------------------------------------------------
        _, l3_points = self.backbone(pc)          # l3_points: [B, 1024, 1]
        feat = l3_points.squeeze(-1)              # [B, 1024]

        # --------------------------------------------------
        # 2. Projection
        # --------------------------------------------------
        z = self.projector(feat)                  # [B, proj_dim]

        # --------------------------------------------------
        # 3. Fuse explicit dimensions
        # --------------------------------------------------
        if pc_dims.dim() == 1:
            pc_dims = pc_dims.unsqueeze(0)

        z = torch.cat([z, pc_dims], dim=1)        # [B, proj_dim + 3]
        z = self.fuse_dims(z)                     # [B, proj_dim]

        return z