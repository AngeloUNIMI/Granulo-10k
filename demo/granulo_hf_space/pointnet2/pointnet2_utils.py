from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance between src [B,N,C] and dst [B,M,C]."""
    return (
        torch.sum(src ** 2, dim=-1, keepdim=True)
        + torch.sum(dst ** 2, dim=-1).unsqueeze(1)
        - 2.0 * torch.matmul(src, dst.transpose(1, 2))
    )


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points [B,N,C] with idx [B,...]."""
    device = points.device
    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    batch_indices = batch_indices.view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling on xyz [B,N,3]."""
    device = xyz.device
    batch_size, n, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, n), 1e10, device=device)
    farthest = torch.randint(0, n, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """Find up to nsample points within radius for each centroid."""
    device = xyz.device
    batch_size, n, _ = xyz.shape
    _, s, _ = new_xyz.shape

    group_idx = torch.arange(n, dtype=torch.long, device=device).view(1, 1, n)
    group_idx = group_idx.repeat(batch_size, s, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = n
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # If a neighborhood has fewer than nsample valid points, repeat the first
    # valid index. For degenerate neighborhoods this still avoids the sentinel
    # N index from escaping into index_points().
    group_first = group_idx[:, :, 0].view(batch_size, s, 1).repeat(1, 1, nsample)
    no_valid = group_first == n
    if torch.any(no_valid):
        nearest = torch.argmin(sqrdists, dim=-1, keepdim=True).repeat(1, 1, nsample)
        group_first = torch.where(no_valid, nearest, group_first)

    mask = group_idx == n
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
    returnfps: bool = False,
):
    """FPS + local grouping. xyz [B,N,3], points [B,N,D] or None."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(new_xyz.shape[0], npoint, 1, 3)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor,
    points: torch.Tensor | None,
):
    """Group the complete point set into one region."""
    device = xyz.device
    batch_size, n, c = xyz.shape
    new_xyz = torch.zeros(batch_size, 1, c, device=device)
    grouped_xyz = xyz.view(batch_size, 1, n, c)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(batch_size, 1, n, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int | None,
        radius: float | None,
        nsample: int | None,
        in_channel: int,
        mlp: list[int],
        group_all: bool,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input xyz [B,C,N], points [B,D,N]; output new_xyz [B,C,S], new_points [B,D',S].
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            assert self.npoint is not None and self.radius is not None and self.nsample is not None
            new_xyz, new_points = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points,
            )

        # [B,S,K,C+D] -> [B,C+D,K,S]
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
