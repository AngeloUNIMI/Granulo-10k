from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _ensure_pointnet2_on_path():
    """
    Make the project's PointNet++ package importable by both the original
    DINOv2 model and the Experiment-1 DINOv3 model.
    """
    here = Path(__file__).resolve().parent
    project_root = here.parent

    candidates = [
        project_root / "code" / "multimodal_paper",
        project_root / "code" / "multimodal_paper_dinov3",
        Path.cwd() / "code" / "multimodal_paper",
        Path.cwd() / "code" / "multimodal_paper_dinov3",
    ]

    for candidate in candidates:
        if (candidate / "pointnet2").is_dir():
            candidate_str = str(candidate.resolve())

            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

            return candidate.resolve()

    checked = "\n".join(
        f"  - {candidate / 'pointnet2'}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not find the PointNet++ implementation required by the model.\n\n"
        "Checked:\n"
        f"{checked}"
    )


def _load_model_class_for_checkpoint(
    checkpoint: dict,
):
    """
    Select the architecture implementation from checkpoint metadata.

    Supported GUI checkpoints:
      - original DINOv2 ViT-B/14 paper-model checkpoints;
      - Experiment 1 DINOv3 ViT-B/16 global-feature checkpoints.

    The DINOv3 implementation is kept in demo_gui/model_dinov3.py so using
    the new checkpoint does not require replacing the original training code.
    """
    here = Path(__file__).resolve().parent
    project_root = here.parent

    config = checkpoint.get("config", {}) or {}
    architecture = str(
        checkpoint.get("architecture", "")
    )
    dino_model = str(
        config.get("dino_model", "")
    )
    experiment = str(
        config.get("experiment", "")
    )

    identity = " ".join(
        [
            architecture,
            dino_model,
            experiment,
        ]
    ).lower()

    is_dinov3 = (
        "dinov3" in identity
        or "patch16_dinov3" in identity
    )

    _ensure_pointnet2_on_path()

    if is_dinov3:
        model_path = here / "model_dinov3.py"
        module_name = "granulo_demo_dinov3_model"

        if not model_path.exists():
            raise FileNotFoundError(
                "This is a DINOv3 checkpoint, but the GUI is missing:\n"
                f"{model_path}\n\n"
                "Place model_dinov3.py in the same demo_gui directory "
                "as inference.py."
            )

        backbone_label = "DINOv3 ViT-B/16"

    else:
        candidates = [
            project_root / "code" / "multimodal_paper" / "model.py",
            Path.cwd() / "code" / "multimodal_paper" / "model.py",
        ]

        model_path = next(
            (
                candidate.resolve()
                for candidate in candidates
                if candidate.exists()
            ),
            None,
        )

        if model_path is None:
            checked = "\n".join(
                f"  - {candidate}"
                for candidate in candidates
            )

            raise FileNotFoundError(
                "Could not find the original DINOv2 multimodal model.\n\n"
                "Checked:\n"
                f"{checked}"
            )

        module_name = "granulo_demo_dinov2_model"
        backbone_label = "DINOv2 ViT-B/14"

    module = sys.modules.get(
        module_name
    )

    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            model_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load model module from: {model_path}"
            )

        module = importlib.util.module_from_spec(
            spec
        )
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return (
        module.GranuloPaperModel,
        backbone_label,
        is_dinov3,
    )


def _load_xyz(path: Path) -> np.ndarray:
    """Load finite XYZ rows, tolerating comments/blank lines/extra columns."""
    rows = []

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

    if not rows:
        raise ValueError(
            f"Point cloud contains no valid XYZ points:\n{path}"
        )

    return np.asarray(rows, dtype=np.float32)


def _resample_points(
    points: np.ndarray,
    sample_points: int,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic version of the training-time random point resampling."""
    rng = np.random.default_rng(seed)

    n = len(points)
    replace = n < sample_points

    indices = rng.choice(
        n,
        size=sample_points,
        replace=replace,
    )

    return points[indices].copy()


def _center_and_dimensions(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dims = points.max(axis=0) - points.min(axis=0)
    centered = points - points.mean(axis=0, keepdims=True)

    return (
        centered.astype(np.float32, copy=False),
        dims.astype(np.float32, copy=False),
    )


def resolve_view_pair(
    selected_image: str | Path,
) -> tuple[Path, Path]:
    """
    Resolve ..._A.png and ..._B.png for the selected acquisition.
    Either A or B may be selected by the user.
    """
    selected_image = Path(selected_image)

    stem = selected_image.stem

    if not (stem.endswith("_A") or stem.endswith("_B")):
        raise ValueError(
            "Expected an acquisition image named like "
            "'0001_0013_A.png' or '0001_0013_B.png'."
        )

    acquisition_prefix = stem[:-2]
    suffix = selected_image.suffix

    image_a = selected_image.with_name(
        f"{acquisition_prefix}_A{suffix}"
    )
    image_b = selected_image.with_name(
        f"{acquisition_prefix}_B{suffix}"
    )

    if not image_a.exists():
        raise FileNotFoundError(
            f"Paired View A was not found:\n{image_a}"
        )

    if not image_b.exists():
        raise FileNotFoundError(
            f"Paired View B was not found:\n{image_b}"
        )

    return image_a, image_b


class StrandPredictor:
    """
    Inference wrapper for Granulo-10k multimodal checkpoints.

    The checkpoint metadata is used to select either the original
    DINOv2 ViT-B/14 architecture or Experiment-1 DINOv3 ViT-B/16.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{self.checkpoint_path}"
            )

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "This checkpoint does not contain 'model_state_dict'. "
                "Please select a checkpoint produced by "
                "code/multimodal_paper/train.py."
            )

        self.architecture = checkpoint.get(
            "architecture",
            "Granulo multimodal paper model",
        )

        config = checkpoint.get("config", {})
        normalization = checkpoint.get("normalization", {})

        self.config = dict(config)
        self.fold = config.get("fold", None)
        self.best_val_metrics = checkpoint.get(
            "best_val_metrics",
            None,
        )

        self.image_size = int(
            config.get("image_size", 518)
        )
        self.sample_points = int(
            config.get("sample_points", 2048)
        )
        self.num_experts = int(
            config.get("num_experts", 64)
        )

        try:
            image_mean = float(
                normalization["image_mean_scalar"]
            )
            image_std = float(
                normalization["image_std_scalar"]
            )
            label_mean = normalization["label_mean"]
            label_std = normalization["label_std"]
        except KeyError as exc:
            raise ValueError(
                "The checkpoint is missing the normalization metadata "
                "required by the multimodal predictor."
            ) from exc

        self.label_mean = torch.tensor(
            label_mean,
            dtype=torch.float32,
            device=self.device,
        )

        self.label_std = torch.tensor(
            label_std,
            dtype=torch.float32,
            device=self.device,
        )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        image_mean,
                        image_mean,
                        image_mean,
                    ],
                    std=[
                        image_std,
                        image_std,
                        image_std,
                    ],
                ),
            ]
        )

        (
            GranuloPaperModel,
            self.backbone_label,
            self.is_dinov3,
        ) = _load_model_class_for_checkpoint(
            checkpoint
        )

        self.dino_model_name = str(
            config.get(
                "dino_model",
                (
                    "vit_base_patch16_dinov3.lvd1689m"
                    if self.is_dinov3
                    else "vit_base_patch14_dinov2.lvd142m"
                ),
            )
        )

        # Full image-encoder weights are already stored in the training
        # checkpoint, so no internet download is required here.
        try:
            self.model = GranuloPaperModel(
                pointnet_checkpoint=None,
                num_experts=self.num_experts,
                pretrained_dino=False,
                freeze_dino=True,
            )
        except Exception as exc:
            if self.is_dinov3:
                raise RuntimeError(
                    "Could not construct the DINOv3 ViT-B/16 model. "
                    "Make sure the GUI environment has a recent timm version "
                    "(recommended: timm>=1.0.20).\n\n"
                    f"Original error: {exc}"
                ) from exc
            raise

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.best_epoch = checkpoint.get(
            "best_epoch",
            None,
        )
        self.test_metrics = checkpoint.get(
            "test_metrics",
            None,
        )

    def _prepare_image(
        self,
        path: Path,
    ) -> torch.Tensor:
        with Image.open(path) as im:
            tensor = self.image_transform(
                im.convert("RGB")
            )

        return tensor.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

    def _prepare_point_cloud(
        self,
        path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points = _load_xyz(path)

        points = _resample_points(
            points,
            self.sample_points,
            seed=0,
        )

        points, dims = _center_and_dimensions(
            points
        )

        point_cloud = torch.from_numpy(
            points
        ).unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        pc_dims = torch.from_numpy(
            dims
        ).unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        return point_cloud, pc_dims

    @torch.inference_mode()
    def predict(
        self,
        selected_image: str | Path,
        point_cloud_path: str | Path | None,
        orientation: str,
    ) -> dict:
        """
        Run the full multimodal model when a valid point cloud is available.

        If point_cloud_path is None (or no longer exists), use an image-only
        fallback: max-fuse the the two image-encoder embeddings and feed that feature to
        the same trained MMoE decoder.

        The checkpoint is not modified. Image-only inference is therefore a
        fallback mode and may have different accuracy because training used
        the point-cloud branch whenever a valid PC was available.
        """
        orientation = orientation.lower().strip()

        if orientation not in {
            "frontal",
            "sideways",
        }:
            raise ValueError(
                "orientation must be 'frontal' or 'sideways'"
            )

        image_a_path, image_b_path = (
            resolve_view_pair(selected_image)
        )

        image_a = self._prepare_image(
            image_a_path
        )
        image_b = self._prepare_image(
            image_b_path
        )

        pc_path = None

        if point_cloud_path is not None:
            candidate = Path(point_cloud_path)

            if candidate.exists():
                pc_path = candidate

        amp_enabled = (
            self.device.type == "cuda"
        )

        if pc_path is not None:
            point_cloud, pc_dims = (
                self._prepare_point_cloud(
                    pc_path
                )
            )

            # PointNet++ starts farthest-point sampling from a random point.
            # Fix the seed so repeated GUI predictions are stable.
            torch.manual_seed(0)

            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(0)

            with torch.autocast(
                device_type=self.device.type,
                enabled=amp_enabled,
            ):
                pred_norm, _, gate_weights = self.model(
                    image_a,
                    image_b,
                    point_cloud,
                    pc_dims,
                    return_gates=True,
                )

            input_mode = "multimodal"

        else:
            # Image-only fallback. This intentionally bypasses PointNet++ and
            # max-fuses only the the two frozen image-encoder feature vectors.
            with torch.autocast(
                device_type=self.device.type,
                enabled=amp_enabled,
            ):
                with torch.no_grad():
                    features_a = self.model.image_encoder(
                        image_a
                    )
                    features_b = self.model.image_encoder(
                        image_b
                    )

                fused, _ = torch.max(
                    torch.stack(
                        [features_a, features_b],
                        dim=0,
                    ),
                    dim=0,
                )

                pred_norm, _, gate_weights = self.model.decoder(
                    fused,
                    return_gates=True,
                )

            input_mode = "image_only"

        pred_mm = (
            pred_norm.float()[0]
            * self.label_std
            + self.label_mean
        )

        height_mm, width_mm, thickness_mm = [
            float(v)
            for v in pred_mm.detach().cpu()
        ]

        task_names = (
            "height",
            "width",
            "thickness",
        )

        gate_top5 = {}

        for task_name, weights in zip(
            task_names,
            gate_weights,
        ):
            task_weights = (
                weights[0]
                .float()
                .detach()
                .cpu()
            )

            k = min(
                5,
                int(task_weights.numel()),
            )

            values, indices = torch.topk(
                task_weights,
                k=k,
            )

            gate_top5[task_name] = [
                {
                    "expert": int(index) + 1,
                    "weight": float(value),
                }
                for value, index
                in zip(values, indices)
            ]

        return {
            "orientation": orientation,
            "height_mm": height_mm,
            "width_mm": width_mm,
            "thickness_mm": thickness_mm,
            "image_a_path": str(image_a_path),
            "image_b_path": str(image_b_path),
            "point_cloud_path": (
                str(pc_path)
                if pc_path is not None
                else None
            ),
            "input_mode": input_mode,
            "gate_top5": gate_top5,
        }
