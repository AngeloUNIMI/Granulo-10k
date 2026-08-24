from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ACQUISITION_RE = re.compile(r"^(\d{4})_(\d{4})_([AB])$", re.IGNORECASE)


@dataclass(frozen=True)
class StrandSample:
    strand_id: int
    acquisition_id: str
    image_a: Path
    image_b: Path
    point_cloud: Path
    orientation: str
    target_mm: tuple[float, float, float]


def load_measurements(path: Path) -> dict[int, tuple[float, float, float]]:
    """Read measurements.txt: strand_id height width thickness."""
    measurements: dict[int, tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                strand_id = int(parts[0])
                height = float(parts[1])
                width = float(parts[2])
                thickness = float(parts[3])
            except ValueError:
                continue
            measurements[strand_id] = (height, width, thickness)
    if not measurements:
        raise ValueError(f"No measurements found in {path}")
    return measurements


def load_thickness_acquisitions(path: Path) -> set[str]:
    """Read identifiers such as 0001_0010 from strands_ok_for_thickness.txt."""
    ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            token = line.strip().split()[0] if line.strip() else ""
            if re.fullmatch(r"\d{4}_\d{4}", token):
                ids.add(token)
    return ids


def _point_cloud_has_xyz_data(path: Path) -> bool:
    """
    Return True only if the file contains at least one finite XYZ row.

    Some Granulo-10k .xyz files are non-zero in size but contain only
    whitespace/comments, so checking st_size alone is not sufficient.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                parts = stripped.replace(",", " ").split()

                if len(parts) < 3:
                    continue

                try:
                    xyz = np.asarray(
                        [float(parts[0]), float(parts[1]), float(parts[2])],
                        dtype=np.float32,
                    )
                except ValueError:
                    continue

                if np.all(np.isfinite(xyz)):
                    return True
    except OSError:
        return False

    return False


def _choose_point_cloud(
    pc_dir: Path,
    acquisition_id: str,
    orientation: str,
) -> Path | None:
    if orientation == "sideways":
        preferred = pc_dir / f"{acquisition_id}_PC_thickness.xyz"
    else:
        preferred = pc_dir / f"{acquisition_id}_PC_lungh_largh.xyz"

    if _point_cloud_has_xyz_data(preferred):
        return preferred

    # If the preferred PC is empty/invalid, try another PC for the same
    # acquisition before marking the sample as incomplete.
    for candidate in sorted(pc_dir.glob(f"{acquisition_id}_PC_*.xyz")):
        if candidate == preferred:
            continue

        if _point_cloud_has_xyz_data(candidate):
            return candidate

    return None


def discover_samples(dataset_root: Path, strict: bool = True) -> list[StrandSample]:
    """
    Build one multimodal sample per acquisition:
      A image + B image + point cloud -> [height, width, thickness].
    """
    dataset_root = dataset_root.resolve()
    images_root = dataset_root / "Images" / "Strands_compliant"
    pcs_root = dataset_root / "PCs" / "Strands_compliant"
    measurements_path = images_root / "measurements.txt"
    thickness_path = images_root / "strands_ok_for_thickness.txt"

    for required in (images_root, pcs_root, measurements_path, thickness_path):
        if not required.exists():
            raise FileNotFoundError(f"Required dataset path not found: {required}")

    measurements = load_measurements(measurements_path)
    thickness_ids = load_thickness_acquisitions(thickness_path)

    samples: list[StrandSample] = []
    missing: list[str] = []

    for image_a in sorted(images_root.glob("*/*_A.png")):
        match = ACQUISITION_RE.match(image_a.stem)
        if not match:
            continue

        strand_id = int(match.group(1))
        acquisition_id = f"{match.group(1)}_{match.group(2)}"
        image_b = image_a.with_name(f"{acquisition_id}_B.png")
        pc_dir = pcs_root / image_a.parent.name
        orientation = "sideways" if acquisition_id in thickness_ids else "frontal"
        point_cloud = _choose_point_cloud(pc_dir, acquisition_id, orientation)

        if strand_id not in measurements:
            missing.append(f"{acquisition_id}: measurement missing")
            continue
        if not image_b.exists():
            missing.append(f"{acquisition_id}: B image missing")
            continue
        if point_cloud is None:
            missing.append(f"{acquisition_id}: point cloud missing/empty")
            continue

        samples.append(
            StrandSample(
                strand_id=strand_id,
                acquisition_id=acquisition_id,
                image_a=image_a,
                image_b=image_b,
                point_cloud=point_cloud,
                orientation=orientation,
                target_mm=measurements[strand_id],
            )
        )

    if strict and missing:
        preview = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise RuntimeError(
            f"Dataset discovery found {len(missing)} incomplete acquisitions:\n"
            f"{preview}{more}"
        )

    if not samples:
        raise RuntimeError(f"No complete multimodal samples found under {dataset_root}")

    return samples


def paper_five_fold_split(
    samples: Sequence[StrandSample],
    seed: int = 42,
    fold: int = 0,
    num_folds: int = 5,
) -> tuple[list[StrandSample], list[StrandSample], list[StrandSample]]:
    """
    Reproduce the supplied code's panel/strand-wise nested KFold structure.

    For 5 folds this gives approximately 60/20/20% strands for train/val/test.
    """
    try:
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the paper-style KFold split. "
            "Install it with: pip install scikit-learn"
        ) from exc

    strand_ids = np.array(sorted({s.strand_id for s in samples}), dtype=np.int64)
    if not 0 <= fold < num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")

    outer = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(strand_ids))
    trainval_idx, test_idx = outer_splits[fold]
    trainval_ids = strand_ids[trainval_idx]
    test_ids = set(strand_ids[test_idx].tolist())

    inner = KFold(n_splits=num_folds - 1, shuffle=True, random_state=seed + fold)
    inner_train_idx, val_idx = next(inner.split(trainval_ids))
    train_ids = set(trainval_ids[inner_train_idx].tolist())
    val_ids = set(trainval_ids[val_idx].tolist())

    train = [s for s in samples if s.strand_id in train_ids]
    val = [s for s in samples if s.strand_id in val_ids]
    test = [s for s in samples if s.strand_id in test_ids]
    return train, val, test


def compute_label_stats(samples: Sequence[StrandSample]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std over unique strands, matching the source label normalization."""
    by_strand: dict[int, tuple[float, float, float]] = {}
    for sample in samples:
        by_strand[sample.strand_id] = sample.target_mm
    values = np.asarray(list(by_strand.values()), dtype=np.float64)
    mean = values.mean(axis=0)
    # statistics.stdev in the supplied code uses sample std (ddof=1).
    std = values.std(axis=0, ddof=1)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_scalar_image_stats(
    samples: Sequence[StrandSample],
    image_size: int = 518,
    cache_path: Path | None = None,
) -> tuple[float, float]:
    """
    Compute the dataset scalar image mean/std used by the supplied training code.

    Both A and B views are included. The result is cached when cache_path is given.
    """
    if cache_path is not None and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return float(data["mean"]), float(data["std"])

    resize = transforms.Resize((image_size, image_size))
    means: list[float] = []
    stds: list[float] = []

    total = len(samples) * 2
    done = 0
    for sample in samples:
        for path in (sample.image_a, sample.image_b):
            with Image.open(path) as im:
                im = resize(im.convert("RGB"))
                arr = np.asarray(im, dtype=np.float32) / 255.0
            means.append(float(arr.mean()))
            stds.append(float(arr.std(ddof=1)))
            done += 1
            if done % 250 == 0 or done == total:
                print(f"Image statistics: {done}/{total}")

    mean = float(np.mean(means))
    std = float(np.mean(stds))
    if std <= 0:
        raise ValueError("Computed image std is zero.")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"mean": mean, "std": std}, indent=2),
            encoding="utf-8",
        )
    return mean, std


class GranuloMultimodalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[StrandSample],
        label_mean: Sequence[float],
        label_std: Sequence[float],
        image_mean: float,
        image_std: float,
        image_size: int = 518,
        sample_points: int = 2048,
        augment_point_cloud: bool = False,
    ):
        self.samples = list(samples)
        self.label_mean = torch.tensor(label_mean, dtype=torch.float32)
        self.label_std = torch.tensor(label_std, dtype=torch.float32)
        self.sample_points = int(sample_points)
        self.augment_point_cloud = bool(augment_point_cloud)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[image_mean, image_mean, image_mean],
                    std=[image_std, image_std, image_std],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_xyz(path: Path) -> np.ndarray:
        rows = []

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()

                    if not stripped or stripped.startswith("#"):
                        continue

                    parts = stripped.replace(",", " ").split()

                    if len(parts) < 3:
                        continue

                    try:
                        xyz = [
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                        ]
                    except ValueError:
                        continue

                    if np.all(np.isfinite(xyz)):
                        rows.append(xyz)
        except OSError as exc:
            raise RuntimeError(f"Could not read point cloud: {path}") from exc

        if not rows:
            raise ValueError(f"Empty or invalid point cloud: {path}")

        return np.asarray(
            rows,
            dtype=np.float32,
        )

    def _resample(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        replace = n < self.sample_points
        idx = np.random.choice(n, self.sample_points, replace=replace)
        return points[idx].copy()

    @staticmethod
    def _augment_pc(points: np.ndarray) -> np.ndarray:
        # Supplied code: arbitrary Z rotation + XY Gaussian jitter.
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        points = points @ rotation.T
        jitter = np.clip(0.005 * np.random.randn(points.shape[0], 2), -0.02, 0.02)
        points[:, :2] += jitter.astype(np.float32)
        return points.astype(np.float32, copy=False)

    @staticmethod
    def _center_and_dimensions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Bounding-box dimensions are translation invariant; compute before/after centering is equivalent.
        dims = points.max(axis=0) - points.min(axis=0)
        centered = points - points.mean(axis=0, keepdims=True)
        return centered.astype(np.float32), dims.astype(np.float32)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_a) as im:
            image_a = self.image_transform(im.convert("RGB"))
        with Image.open(sample.image_b) as im:
            image_b = self.image_transform(im.convert("RGB"))

        points = self._resample(self._load_xyz(sample.point_cloud))
        if self.augment_point_cloud:
            points = self._augment_pc(points)
        points, pc_dims = self._center_and_dimensions(points)

        target_mm = torch.tensor(sample.target_mm, dtype=torch.float32)
        target_norm = (target_mm - self.label_mean) / self.label_std

        # Kept for diagnostics and optional masked loss experiments.
        if sample.orientation == "sideways":
            visibility = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        else:
            visibility = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

        return {
            "image_a": image_a,
            "image_b": image_b,
            "point_cloud": torch.from_numpy(points),  # [N, 3]
            "pc_dims": torch.from_numpy(pc_dims),
            "target_mm": target_mm,
            "target_norm": target_norm,
            "visibility": visibility,
            "strand_id": sample.strand_id,
            "acquisition_id": sample.acquisition_id,
            "orientation": sample.orientation,
            "point_cloud_path": str(sample.point_cloud),
        }