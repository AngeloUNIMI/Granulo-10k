from __future__ import annotations

import argparse
import base64
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Hugging Face ZeroGPU exposes the `spaces` module inside the Space runtime.
# Keep a no-op fallback so this same file still runs locally.
try:
    import spaces  # type: ignore
except ImportError:  # local development
    spaces = None

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from PIL import Image, ImageFilter

from inference import StrandPredictor, resolve_view_pair
from overlay import annotate_measurements


APP_TITLE = "Granulo-10k - Multimodal Strand Measurement Demo"
DATASET_URL = "https://github.com/AngeloUNIMI/Granulo-10k"
IEBIL_URL = "https://iebil.di.unimi.it/"
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_REF_H = 115.0
DEFAULT_REF_W = 20.0
DEFAULT_REF_T = 0.7
DEFAULT_EMA_SPAN = 10
MAX_POINT_DISPLAY = 7000
ZERO_GPU_DURATION = int(os.environ.get("GRANULO_ZERO_GPU_DURATION", "20"))

# The Gradio queue already serializes the inference event, but the lock also
# protects the single shared model if run_prediction() is ever called outside
# the Gradio event queue.
PREDICT_LOCK = threading.Lock()


def zero_gpu(duration: int = 20):
    """Return the HF ZeroGPU decorator, or a no-op decorator locally."""
    if spaces is None:
        def identity(fn):
            return fn
        return identity
    return spaces.GPU(duration=duration)


@dataclass(frozen=True)
class AcquisitionRecord:
    strand_group: str
    acquisition: str
    image_a: Path


class DatasetIndex:
    def __init__(self, data_root: str | Path | None):
        self.images_root = find_images_root(data_root)
        self.records: list[AcquisitionRecord] = []
        self.by_group: dict[str, list[AcquisitionRecord]] = {}
        self.by_path: dict[str, AcquisitionRecord] = {}
        self._scan()

    def _scan(self):
        if self.images_root is None or not self.images_root.exists():
            return

        for candidate in sorted(self.images_root.rglob("*_A.*")):
            if not candidate.is_file() or candidate.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                continue
            try:
                image_a, image_b = resolve_view_pair(candidate)
            except Exception:
                continue
            if not image_a.exists() or not image_b.exists():
                continue

            try:
                parent_rel = image_a.parent.relative_to(self.images_root)
                group = str(parent_rel) if str(parent_rel) != "." else "root"
            except ValueError:
                group = image_a.parent.name

            record = AcquisitionRecord(
                strand_group=group,
                acquisition=image_a.stem[:-2],
                image_a=image_a,
            )
            self.records.append(record)
            self.by_group.setdefault(group, []).append(record)
            self.by_path[str(image_a.resolve())] = record

        for records in self.by_group.values():
            records.sort(key=lambda item: item.acquisition)

    @property
    def groups(self) -> list[str]:
        return sorted(self.by_group)

    def records_for_group(self, group: str | None) -> list[AcquisitionRecord]:
        if group is None:
            return []
        return self.by_group.get(str(group), [])

    def record_for(self, group: str | None, acquisition: str | None) -> AcquisitionRecord | None:
        if group is None or acquisition is None:
            return None
        for record in self.records_for_group(group):
            if record.acquisition == acquisition:
                return record
        return None

    def record_for_path(self, path: str | Path | None) -> AcquisitionRecord | None:
        if not path:
            return None
        try:
            key = str(Path(path).resolve())
        except Exception:
            return None
        return self.by_path.get(key)

    def adjacent(self, path: str | Path | None, offset: int) -> AcquisitionRecord | None:
        record = self.record_for_path(path)
        if record is None:
            return None
        records = self.records_for_group(record.strand_group)
        try:
            index = records.index(record)
        except ValueError:
            return None
        new_index = max(0, min(len(records) - 1, index + int(offset)))
        return records[new_index]

    def random_record(self) -> AcquisitionRecord | None:
        return random.choice(self.records) if self.records else None


def find_images_root(data_root: str | Path | None) -> Path | None:
    candidates: list[Path] = []

    if data_root:
        candidates.append(Path(data_root).expanduser())

    env_root = os.environ.get("GRANULO_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    here = Path(__file__).resolve().parent
    project_root = here.parent
    candidates.extend(
        [
            project_root / "data" / "Granulo-10k",
            Path.cwd() / "data" / "Granulo-10k",
            Path.cwd() / "Granulo-10k",
        ]
    )

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.extend(
            [
                candidate,
                candidate / "Images" / "Strands_compliant",
                candidate / "Strands_compliant",
            ]
        )
        if candidate.name == "Images":
            expanded.append(candidate / "Strands_compliant")

    seen = set()
    for candidate in expanded:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if resolved.is_dir() and resolved.name == "Strands_compliant":
            return resolved

    return None


def load_measurements(measurements_file: Path) -> dict[int, dict[str, float]]:
    measurements: dict[int, dict[str, float]] = {}
    with measurements_file.open("r", encoding="utf-8-sig", errors="ignore") as f:
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
            measurements[strand_id] = {
                "height_mm": height,
                "width_mm": width,
                "thickness_mm": thickness,
            }
    return measurements


_MEASUREMENTS_CACHE: dict[str, dict[int, dict[str, float]]] = {}
_THICKNESS_CACHE: dict[str, set[str]] = {}


def candidate_measurements_files(image_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for parent in (image_path.parent, *image_path.parents):
        if parent.name == "Strands_compliant":
            candidate = parent / "measurements.txt"
            if candidate.exists():
                candidates.append(candidate)
            break

    current = image_path.parent
    for _ in range(10):
        candidate = current / "measurements.txt"
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)
        if current.parent == current:
            break
        current = current.parent
    return candidates


def ground_truth_for_image(image_path: Path) -> dict[str, float] | None:
    match = re.match(r"(\d{4})_\d{4}_[AB]$", image_path.stem, re.IGNORECASE)
    if not match:
        return None
    strand_id = int(match.group(1))

    for measurements_file in candidate_measurements_files(image_path):
        key = str(measurements_file.resolve())
        try:
            if key not in _MEASUREMENTS_CACHE:
                _MEASUREMENTS_CACHE[key] = load_measurements(measurements_file)
            ground_truth = _MEASUREMENTS_CACHE[key].get(strand_id)
        except Exception:
            continue
        if ground_truth is not None:
            return dict(ground_truth)
    return None


def find_thickness_file(image_path: Path) -> Path | None:
    current = image_path.parent
    for _ in range(8):
        candidate = current / "strands_ok_for_thickness.txt"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def thickness_set(image_path: Path) -> set[str]:
    thickness_file = find_thickness_file(image_path)
    if thickness_file is None:
        return set()
    key = str(thickness_file.resolve())
    if key in _THICKNESS_CACHE:
        return _THICKNESS_CACHE[key]

    acquisitions: set[str] = set()
    try:
        with thickness_file.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                acquisition = tokens[0].strip()
                if re.fullmatch(r"\d{4}_\d{4}", acquisition):
                    acquisitions.add(acquisition)
    except OSError:
        acquisitions = set()
    _THICKNESS_CACHE[key] = acquisitions
    return acquisitions


def orientation_for_image(image_path: Path) -> str:
    match = re.match(r"(\d{4}_\d{4})_[AB]$", image_path.stem, re.IGNORECASE)
    if not match:
        return "frontal"
    return "sideways" if match.group(1) in thickness_set(image_path) else "frontal"


def mask_path_for_image(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    try:
        images_index = parts.index("Images")
    except ValueError:
        return None
    parts[images_index] = "Masks"
    candidate = Path(*parts)
    return candidate if candidate.exists() else None


def load_mask(image_path: Path) -> Image.Image | None:
    candidate = mask_path_for_image(image_path)
    if candidate is None:
        return None
    try:
        with Image.open(candidate) as mask:
            return mask.convert("L").copy()
    except Exception:
        return None


def apply_mask_outline(image: Image.Image, mask: Image.Image | None) -> Image.Image:
    if mask is None:
        return image.convert("RGB")

    mask_l = mask.convert("L")
    if mask_l.size != image.size:
        mask_l = mask_l.resize(image.size, Image.Resampling.NEAREST)

    mask_array = np.asarray(mask_l)
    binary = mask_array > 127
    if binary.mean() > 0.5:
        binary = ~binary

    binary_image = Image.fromarray(binary.astype(np.uint8) * 255)
    eroded = np.asarray(binary_image.filter(ImageFilter.MinFilter(3))) > 0
    contour = binary & ~eroded

    rgba = np.asarray(image.convert("RGBA")).copy()
    rgba[contour] = np.array([255, 215, 0, 255], dtype=np.uint8)
    return Image.fromarray(rgba).convert("RGB")


def point_cloud_has_xyz_data(path: Path) -> bool:
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
                    xyz = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
                if np.all(np.isfinite(xyz)):
                    return True
    except OSError:
        return False
    return False


def pc_dir_for_image(image_path: Path) -> Path | None:
    parts = list(image_path.parent.parts)
    try:
        images_index = parts.index("Images")
    except ValueError:
        return None
    parts[images_index] = "PCs"
    return Path(*parts)


def point_cloud_candidates(image_path: Path) -> list[Path]:
    match = re.match(r"(\d{4}_\d{4})_[AB]$", image_path.stem, re.IGNORECASE)
    if not match:
        return []
    pc_dir = pc_dir_for_image(image_path)
    if pc_dir is None or not pc_dir.exists():
        return []
    prefix = match.group(1)
    return sorted(
        path for path in pc_dir.glob(f"{prefix}_PC_*.xyz") if point_cloud_has_xyz_data(path)
    )


def select_point_cloud(image_path: Path, orientation: str) -> Path | None:
    candidates = point_cloud_candidates(image_path)
    if not candidates:
        return None

    if orientation == "sideways":
        preferred = [
            path
            for path in candidates
            if "thickness" in path.stem.lower() or "spess" in path.stem.lower()
        ]
    else:
        preferred = [
            path
            for path in candidates
            if any(
                token in path.stem.lower()
                for token in ("lungh", "largh", "width", "height")
            )
        ]
        if not preferred:
            preferred = [
                path
                for path in candidates
                if "thickness" not in path.stem.lower() and "spess" not in path.stem.lower()
            ]

    return preferred[0] if preferred else candidates[0]


def load_xyz(path: Path) -> np.ndarray:
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
                xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError:
                continue
            if np.all(np.isfinite(xyz)):
                rows.append(xyz)
    if not rows:
        raise ValueError(f"Point cloud has no valid XYZ points: {path}")
    return np.asarray(rows, dtype=np.float32)


def point_cloud_figure(path: Path | None) -> go.Figure:
    fig = go.Figure()
    if path is None:
        fig.update_layout(
            title="Point cloud unavailable - image-only fallback",
            height=560,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    try:
        points = load_xyz(path)
        if len(points) > MAX_POINT_DISPLAY:
            indices = np.linspace(0, len(points) - 1, MAX_POINT_DISPLAY, dtype=int)
            points = points[indices]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2, opacity=0.72),
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                name="XYZ",
            )
        )
        fig.update_layout(
            title=path.name,
            scene=dict(
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                # Preserve the real XYZ proportions.  The plot container can
                # still span the full page width, but the geometry itself must
                # never be stretched independently along X/Y/Z.
                aspectmode="data",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            autosize=True,
            height=560,
            margin=dict(l=0, r=0, t=50, b=0),
        )
    except Exception as exc:
        fig.update_layout(
            title=f"Could not render point cloud: {exc}",
            height=560,
            margin=dict(l=0, r=0, t=50, b=0),
        )
    return fig


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def render_views(
    image_a_path: Path,
    prediction: dict[str, Any] | None,
    show_mask_outline: bool,
) -> tuple[Image.Image, Image.Image]:
    image_a_path, image_b_path = resolve_view_pair(image_a_path)
    image_a = load_rgb(image_a_path)
    image_b = load_rgb(image_b_path)
    mask_a = load_mask(image_a_path)
    mask_b = load_mask(image_b_path)

    display_a = (
        annotate_measurements(image_a, prediction, mask_a)
        if prediction is not None
        else image_a.copy()
    )
    display_b = image_b.copy()

    if show_mask_outline:
        display_a = apply_mask_outline(display_a, mask_a)
        display_b = apply_mask_outline(display_b, mask_b)

    return display_a, display_b


def measurement_markdown(
    prediction: dict[str, Any] | None,
    ground_truth: dict[str, float] | None,
    inference_ms: float | None = None,
) -> str:
    def fmt(value: float | None, digits: int) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    rows = []
    specs = [
        ("Height", "height_mm", 2),
        ("Width", "width_mm", 2),
        ("Thickness", "thickness_mm", 3),
    ]
    for label, key, digits in specs:
        pred = float(prediction[key]) if prediction is not None else None
        gt = float(ground_truth[key]) if ground_truth is not None else None
        error = pred - gt if pred is not None and gt is not None else None
        rows.append(
            f"| {label} | {fmt(pred, digits)} | {fmt(gt, digits)} | "
            f"{('—' if error is None else f'{error:+.{digits}f}')} |"
        )

    timing = "—" if inference_ms is None else f"{inference_ms:.0f} ms"
    mode = "—"
    if prediction is not None:
        mode = "Multimodal" if prediction.get("input_mode") == "multimodal" else "Image-only fallback"

    return (
        "### Measurements [mm]\n\n"
        "| Quantity | Prediction | Ground truth | Error |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + f"\n\n**Input mode:** {mode}  ·  **Inference:** {timing}"
    )


def acquisition_info_markdown(
    image_a_path: Path,
    orientation: str,
    pc_path: Path | None,
    ground_truth: dict[str, float] | None,
) -> str:
    image_a, image_b = resolve_view_pair(image_a_path)
    pc_text = pc_path.name if pc_path is not None else "not available (image-only fallback)"
    gt_text = "available" if ground_truth is not None else "not found"
    return (
        f"**Acquisition:** `{image_a.stem[:-2]}`  ·  "
        f"**Orientation:** {orientation.title()}  ·  **Ground truth:** {gt_text}  \n"
        f"**View A:** `{image_a.name}`  ·  **View B:** `{image_b.name}`  ·  "
        f"**Point cloud:** `{pc_text}`"
    )


def ema_same_length(values: np.ndarray, span: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values.copy()
    span = max(1, int(span))
    if span <= 1:
        return values.copy()
    alpha = 2.0 / (float(span) + 1.0)
    result = np.full_like(values, np.nan, dtype=np.float64)
    previous = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        previous = float(value) if previous is None else alpha * float(value) + (1.0 - alpha) * previous
        result[index] = previous
    return result


def history_figure(
    history: list[dict[str, Any]] | None,
    smoothing_enabled: bool,
    ema_span: int,
    ref_h: float | None,
    ref_w: float | None,
    ref_t: float | None,
) -> go.Figure:
    history = history or []
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Height [mm]", "Width [mm]", "Thickness [mm]"),
    )

    if not history:
        fig.update_layout(
            height=700,
            title="Prediction history - no predictions in this browser session yet",
            margin=dict(l=45, r=20, t=80, b=45),
        )
        return fig

    x = np.arange(1, len(history) + 1)
    labels = [item["acquisition"] for item in history]
    specs = [
        (1, "height_mm", "Height", ref_h),
        (2, "width_mm", "Width", ref_w),
        (3, "thickness_mm", "Thickness", ref_t),
    ]
    span = max(1, min(100, int(ema_span or DEFAULT_EMA_SPAN)))

    for row, key, task_name, reference in specs:
        predicted = np.asarray([item["prediction"][key] for item in history], dtype=np.float64)
        gt = np.asarray(
            [
                item["ground_truth"][key]
                if item.get("ground_truth") is not None and key in item["ground_truth"]
                else np.nan
                for item in history
            ],
            dtype=np.float64,
        )
        if smoothing_enabled:
            predicted = ema_same_length(predicted, span)
            gt = ema_same_length(gt, span)
            suffix = f" (EMA {span})"
        else:
            suffix = ""

        fig.add_trace(
            go.Scatter(
                x=x,
                y=predicted,
                mode="lines+markers",
                name=f"{task_name} prediction{suffix}",
                legendgroup=f"{task_name}-prediction",
            ),
            row=row,
            col=1,
        )
        if np.any(np.isfinite(gt)):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gt,
                    mode="lines+markers",
                    line=dict(dash="dash"),
                    name=f"{task_name} ground truth{suffix}",
                    legendgroup=f"{task_name}-gt",
                ),
                row=row,
                col=1,
            )
        if reference is not None and np.isfinite(float(reference)):
            fig.add_hline(
                y=float(reference),
                line_dash="dot",
                annotation_text=f"ref {float(reference):g}",
                row=row,
                col=1,
            )

    tick_stride = max(1, len(history) // 12)
    tick_values = x[::tick_stride]
    tick_labels = labels[::tick_stride]
    fig.update_xaxes(
        title_text="Prediction sequence",
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        tickangle=-30,
        row=3,
        col=1,
    )
    fig.update_layout(
        height=720,
        title=f"Prediction history - {len(history)} predictions in this browser session",
        hovermode="x unified",
        margin=dict(l=45, r=20, t=80, b=90),
    )
    return fig


def gate_figure(prediction: dict[str, Any] | None) -> go.Figure:
    fig = go.Figure()
    if not prediction or not prediction.get("gate_top5"):
        fig.update_layout(
            height=380,
            title="Expert gates - run a prediction to inspect the top experts",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    for task_name, entries in prediction["gate_top5"].items():
        fig.add_trace(
            go.Bar(
                x=[f"E{entry['expert']}" for entry in entries],
                y=[entry["weight"] for entry in entries],
                name=task_name.title(),
            )
        )
    fig.update_layout(
        barmode="group",
        height=390,
        title="Top-5 MMoE expert weights",
        xaxis_title="Expert",
        yaxis_title="Gate weight",
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def append_history(
    history: list[dict[str, Any]] | None,
    image_path: Path,
    orientation: str,
    checkpoint_name: str,
    prediction: dict[str, Any],
    ground_truth: dict[str, float] | None,
) -> list[dict[str, Any]]:
    updated = list(history or [])
    match = re.match(r"(\d{4})_\d{4}_[AB]$", image_path.stem, re.IGNORECASE)
    updated.append(
        {
            "sequence": len(updated) + 1,
            "acquisition": image_path.stem[:-2],
            "strand_id": int(match.group(1)) if match else None,
            "orientation": orientation,
            "model": checkpoint_name,
            "prediction": {
                "height_mm": float(prediction["height_mm"]),
                "width_mm": float(prediction["width_mm"]),
                "thickness_mm": float(prediction["thickness_mm"]),
            },
            "ground_truth": dict(ground_truth) if ground_truth is not None else None,
        }
    )
    return updated


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_header_html() -> str:
    assets = Path(__file__).resolve().parent / "assets"
    unimi = image_data_uri(assets / "minerva2011.png")
    icip = image_data_uri(assets / "icip_logo.png")
    iebil = image_data_uri(assets / "iebil_logo.png")
    qr = image_data_uri(assets / "qrcode.png")
    return f"""
    <div class="granulo-header">
      <div class="header-side header-left">
        {f'<span class="brand-logo-box"><img src="{unimi}" class="brand-logo unimi-logo" alt="Universita degli Studi di Milano" /></span>' if unimi else ''}
        {f'<a href="{IEBIL_URL}" target="_blank" rel="noopener noreferrer" class="brand-logo-box iebil-link" title="IEBIL Lab"><img src="{iebil}" class="brand-logo iebil-logo" alt="IEBIL Lab" /></a>' if iebil else ''}
      </div>
      <div class="header-center">
        <div class="header-title">Granulo-10k</div>
        <div class="header-subtitle">Multimodal Strand Measurement Demo</div>
        <div class="header-authors">Pasquale Coscia &middot; Angelo Genovese &middot; Vincenzo Piuri &middot; Fabio Scotti</div>
        <div class="header-affiliation">Department of Computer Science, Universit&agrave; degli Studi di Milano, Italy</div>
        {f'<img src="{icip}" class="icip-logo" alt="ICIP 2026 Tampere" />' if icip else ''}
      </div>
      <div class="header-side header-right">
        {f'<a href="{DATASET_URL}" target="_blank" rel="noopener noreferrer" title="Granulo-10k dataset and code"><img src="{qr}" class="qr-logo" alt="Dataset and code QR code" /></a>' if qr else ''}
      </div>
    </div>
    """


CSS = """
html, body {
  height: 100% !important;
  margin: 0 !important;
  overflow: hidden !important;
}

gradio-app {
  display: block !important;
  height: 100vh !important;
  height: 100dvh !important;
  overflow: hidden !important;
}

.gradio-container {
  max-width: 1500px !important;

  /* Keep scrolling inside the Gradio app when embedded on huggingface.co */
  height: 100vh !important;
  height: 100dvh !important;
  max-height: 100dvh !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  scrollbar-gutter: stable;
  -webkit-overflow-scrolling: touch;
}
.granulo-header {
  display: grid; grid-template-columns: 1fr 2fr 1fr; align-items: center;
  background: #f5efe3; border: 1px solid #e6ddce; border-radius: 14px;
  padding: 12px 22px; margin-bottom: 14px;
}
.header-side { display: flex; align-items: center; min-height: 100px; }
.header-left { justify-content: flex-start; gap: 14px; }
.header-right { justify-content: flex-end; }
.header-center { text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.brand-logo-box {
  width: 82px !important; height: 82px !important; flex: 0 0 82px !important;
  display: inline-flex !important; align-items: center !important; justify-content: center !important;
  overflow: visible !important; text-decoration: none !important;
}
.brand-logo-box > img.brand-logo {
  width: 82px !important; height: 82px !important;
  min-width: 82px !important; min-height: 82px !important;
  max-width: 82px !important; max-height: 82px !important;
  object-fit: contain !important; display: block !important;
}
.iebil-logo { transition: transform 0.15s ease, filter 0.15s ease; }
.iebil-link:hover .iebil-logo { transform: scale(1.06); filter: brightness(1.04); }
.icip-logo { max-height: 38px; max-width: 210px; object-fit: contain; margin-top: 8px; }
.qr-logo { height: 78px; width: 78px; object-fit: contain; }
.header-title { font-size: 30px; font-weight: 700; color: #111827; line-height: 1.05; }
.header-subtitle { font-size: 15px; color: #0b4f7a; font-weight: 600; margin-top: 4px; }
.header-authors { font-size: 13px; color: #1f2937; font-weight: 600; margin-top: 7px; line-height: 1.25; }
.header-affiliation { font-size: 12px; color: #4b5563; font-style: italic; margin-top: 2px; line-height: 1.25; }
@media (max-width: 900px) {
  .granulo-header { grid-template-columns: 90px 1fr 90px; padding: 10px 12px; }
  .header-left { gap: 6px; flex-direction: column; }
  .brand-logo-box { width: 52px !important; height: 52px !important; flex-basis: 52px !important; }
  .brand-logo-box > img.brand-logo {
    width: 52px !important; height: 52px !important;
    min-width: 52px !important; min-height: 52px !important;
    max-width: 52px !important; max-height: 52px !important;
  }
  .qr-logo { height: 62px; width: 62px; }
  .header-title { font-size: 25px; }
  .header-subtitle { font-size: 13px; }
  .header-authors { font-size: 11px; }
  .header-affiliation { font-size: 10px; }
  .icip-logo { max-height: 32px; max-width: 175px; }
}
.status-card { border-left: 4px solid #0b4f7a; padding-left: 10px; }
#run-prediction { min-height: 52px; font-weight: 700; font-size: 16px; }
#point-cloud-plot { width: 100% !important; max-width: none !important; }
#point-cloud-plot > div,
#point-cloud-plot .wrap,
#point-cloud-plot .plot-container,
#point-cloud-plot .js-plotly-plot,
#point-cloud-plot .plotly,
#point-cloud-plot .plotly-graph-div {
  width: 100% !important;
  max-width: 100% !important;
  flex: 1 1 100% !important;
}
#point-cloud-plot .svg-container { width: 100% !important; max-width: 100% !important; }
"""


class GranuloWebApp:
    def __init__(
        self,
        checkpoint: str | Path | None,
        data_root: str | Path | None,
        device: str | None = None,
    ):
        self.dataset = DatasetIndex(data_root)
        self.predictor: StrandPredictor | None = None
        self.model_error: str | None = None
        self.checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None

        if self.checkpoint_path is not None:
            try:
                self.predictor = StrandPredictor(self.checkpoint_path, device=device)
            except Exception as exc:
                self.model_error = str(exc)

    @property
    def model_status(self) -> str:
        if self.predictor is not None:
            best_epoch = (
                f", best epoch {self.predictor.best_epoch}"
                if self.predictor.best_epoch is not None
                else ""
            )
            return (
                f"✅ **Model:** `{self.predictor.checkpoint_path.name}` · "
                f"{self.predictor.backbone_label} · device `{self.predictor.device}`{best_epoch}"
            )
        if self.model_error:
            return f"❌ **Model failed to load:** {self.model_error}"
        return "⚠️ **Model not loaded.** Set `GRANULO_MODEL_PATH` or place a checkpoint in `models/`."

    @property
    def data_status(self) -> str:
        if self.dataset.images_root is None:
            return "❌ **Dataset:** `Strands_compliant` not found. Set `GRANULO_DATA_ROOT` or place demo data under `data/`."
        return (
            f"✅ **Dataset:** `{self.dataset.images_root}` · "
            f"{len(self.dataset.records)} paired acquisitions · {len(self.dataset.groups)} strand folders"
        )

    def create(self) -> gr.Blocks:
        groups = self.dataset.groups
        initial_record = self.dataset.records[0] if self.dataset.records else None
        initial_group = initial_record.strand_group if initial_record else None
        initial_acquisitions = [r.acquisition for r in self.dataset.records_for_group(initial_group)]
        initial_acquisition = initial_record.acquisition if initial_record else None

        if initial_record:
            initial_orientation = orientation_for_image(initial_record.image_a)
            initial_pc = select_point_cloud(initial_record.image_a, initial_orientation)
            initial_gt = ground_truth_for_image(initial_record.image_a)
            initial_a, initial_b = render_views(initial_record.image_a, None, False)
            initial_pc_fig = point_cloud_figure(initial_pc)
            initial_info = acquisition_info_markdown(
                initial_record.image_a, initial_orientation, initial_pc, initial_gt
            )
            initial_path = str(initial_record.image_a)
        else:
            initial_orientation = "frontal"
            initial_gt = None
            initial_a = initial_b = None
            initial_pc_fig = point_cloud_figure(None)
            initial_info = "No Granulo-10k acquisitions were found."
            initial_path = None

        blocks_kwargs = {"css": CSS} if int(gr.__version__.split(".")[0]) < 6 else {}
        with gr.Blocks(title=APP_TITLE, **blocks_kwargs) as demo:
            gr.HTML(build_header_html())

            with gr.Row():
                gr.Markdown(self.model_status, elem_classes="status-card")
                gr.Markdown(self.data_status, elem_classes="status-card")

            current_path = gr.State(initial_path)
            last_prediction = gr.State(None)
            history = gr.State([])

            with gr.Row():
                with gr.Column(scale=4):
                    strand_group = gr.Dropdown(
                        choices=groups,
                        value=initial_group,
                        label="Strand folder",
                        interactive=bool(groups),
                    )
                    acquisition = gr.Dropdown(
                        choices=initial_acquisitions,
                        value=initial_acquisition,
                        label="Acquisition",
                        interactive=bool(initial_acquisitions),
                    )
                with gr.Column(scale=3):
                    orientation = gr.Radio(
                        choices=[("Frontal", "frontal"), ("Sideways", "sideways")],
                        value=initial_orientation,
                        label="Acquisition view",
                    )
                    show_mask = gr.Checkbox(value=False, label="Show segmentation outline")
                with gr.Column(scale=3):
                    with gr.Row():
                        previous_button = gr.Button("◀ Previous")
                        random_button = gr.Button("Random")
                        next_button = gr.Button("Next ▶")
                    run_button = gr.Button("Run prediction", variant="primary", elem_id="run-prediction")

            acquisition_info = gr.Markdown(initial_info)

            with gr.Row(equal_height=True):
                view_a = gr.Image(
                    value=initial_a,
                    label="View A - prediction overlay",
                    type="pil",
                    interactive=False,
                    height=500,
                )
                view_b = gr.Image(
                    value=initial_b,
                    label="View B - image encoder input",
                    type="pil",
                    interactive=False,
                    height=500,
                )

            measurements = gr.Markdown(
                measurement_markdown(None, initial_gt),
                elem_classes="status-card",
            )
            event_status = gr.Markdown("Ready.")

            with gr.Tabs():
                with gr.Tab("Point cloud"):
                    pc_plot = gr.Plot(value=initial_pc_fig, label="Point cloud", elem_id="point-cloud-plot")
                with gr.Tab("Prediction history"):
                    with gr.Row():
                        smooth = gr.Checkbox(value=False, label="EMA smoothing")
                        ema_span = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=DEFAULT_EMA_SPAN,
                            label="EMA span",
                        )
                    with gr.Row():
                        ref_h = gr.Number(value=DEFAULT_REF_H, label="Reference H [mm]")
                        ref_w = gr.Number(value=DEFAULT_REF_W, label="Reference W [mm]")
                        ref_t = gr.Number(value=DEFAULT_REF_T, label="Reference T [mm]")
                    history_plot = gr.Plot(
                        value=history_figure([], False, DEFAULT_EMA_SPAN, DEFAULT_REF_H, DEFAULT_REF_W, DEFAULT_REF_T),
                        label="Prediction history",
                    )
                    clear_history = gr.Button("Clear session history")
                with gr.Tab("Expert gates"):
                    gates_plot = gr.Plot(value=gate_figure(None), label="MMoE expert gates")

            def load_record(record: AcquisitionRecord | None, show_outline: bool):
                if record is None:
                    raise gr.Error("Acquisition not found.")
                path = record.image_a
                orient = orientation_for_image(path)
                pc_path = select_point_cloud(path, orient)
                gt = ground_truth_for_image(path)
                display_a, display_b = render_views(path, None, bool(show_outline))
                info = acquisition_info_markdown(path, orient, pc_path, gt)
                return (
                    record.strand_group,
                    gr.Dropdown(
                        choices=[r.acquisition for r in self.dataset.records_for_group(record.strand_group)],
                        value=record.acquisition,
                    ),
                    str(path),
                    orient,
                    display_a,
                    display_b,
                    point_cloud_figure(pc_path),
                    measurement_markdown(None, gt),
                    info,
                    None,
                    gate_figure(None),
                    f"Loaded `{record.acquisition}`.",
                )

            navigation_outputs = [
                strand_group,
                acquisition,
                current_path,
                orientation,
                view_a,
                view_b,
                pc_plot,
                measurements,
                acquisition_info,
                last_prediction,
                gates_plot,
                event_status,
            ]

            def on_group_change(group: str, show_outline: bool):
                records = self.dataset.records_for_group(group)
                return load_record(records[0] if records else None, show_outline)

            strand_group.input(
                on_group_change,
                inputs=[strand_group, show_mask],
                outputs=navigation_outputs,
                queue=False,
            )

            def on_acquisition_change(group: str, acquisition_name: str, show_outline: bool):
                return load_record(self.dataset.record_for(group, acquisition_name), show_outline)

            acquisition.input(
                on_acquisition_change,
                inputs=[strand_group, acquisition, show_mask],
                outputs=navigation_outputs,
                queue=False,
            )

            def navigate(path: str | None, offset: int, show_outline: bool):
                return load_record(self.dataset.adjacent(path, offset), show_outline)

            previous_button.click(
                lambda path, outline: navigate(path, -1, outline),
                inputs=[current_path, show_mask],
                outputs=navigation_outputs,
                queue=False,
            )
            next_button.click(
                lambda path, outline: navigate(path, 1, outline),
                inputs=[current_path, show_mask],
                outputs=navigation_outputs,
                queue=False,
            )
            random_button.click(
                lambda outline: load_record(self.dataset.random_record(), outline),
                inputs=[show_mask],
                outputs=navigation_outputs,
                queue=False,
            )

            def on_orientation_change(path: str | None, orient: str, outline: bool):
                if not path:
                    return point_cloud_figure(None), measurement_markdown(None, None), None, gate_figure(None), "No acquisition selected."
                image_path = Path(path)
                pc_path = select_point_cloud(image_path, orient)
                gt = ground_truth_for_image(image_path)
                display_a, display_b = render_views(image_path, None, outline)
                info = acquisition_info_markdown(image_path, orient, pc_path, gt)
                return (
                    point_cloud_figure(pc_path),
                    measurement_markdown(None, gt),
                    None,
                    gate_figure(None),
                    display_a,
                    display_b,
                    info,
                    "Orientation changed; previous prediction cleared.",
                )

            orientation.input(
                on_orientation_change,
                inputs=[current_path, orientation, show_mask],
                outputs=[pc_plot, measurements, last_prediction, gates_plot, view_a, view_b, acquisition_info, event_status],
                queue=False,
            )

            def on_mask_change(path: str | None, prediction: dict[str, Any] | None, outline: bool):
                if not path:
                    return None, None
                return render_views(Path(path), prediction, outline)

            show_mask.input(
                on_mask_change,
                inputs=[current_path, last_prediction, show_mask],
                outputs=[view_a, view_b],
                queue=False,
            )

            @zero_gpu(duration=ZERO_GPU_DURATION)
            def run_prediction(
                path: str | None,
                orient: str,
                outline: bool,
                history_value: list[dict[str, Any]] | None,
                smoothing: bool,
                span: int,
                reference_h: float | None,
                reference_w: float | None,
                reference_t: float | None,
            ):
                if self.predictor is None:
                    raise gr.Error(
                        self.model_error
                        or "No model is loaded. Configure GRANULO_MODEL_PATH or the models/ directory."
                    )
                if not path:
                    raise gr.Error("No acquisition is selected.")

                image_path = Path(path)
                pc_path = select_point_cloud(image_path, orient)
                gt = ground_truth_for_image(image_path)

                if self.predictor.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with PREDICT_LOCK:
                    prediction = self.predictor.predict(
                        selected_image=image_path,
                        point_cloud_path=pc_path,
                        orientation=orient,
                    )
                if self.predictor.device.type == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - start) * 1000.0

                updated_history = append_history(
                    history_value,
                    image_path,
                    orient,
                    self.predictor.checkpoint_path.name,
                    prediction,
                    gt,
                )
                display_a, display_b = render_views(image_path, prediction, outline)
                status = (
                    "Multimodal prediction complete."
                    if prediction.get("input_mode") == "multimodal"
                    else "Image-only fallback prediction complete (point cloud unavailable)."
                )
                return (
                    display_a,
                    display_b,
                    measurement_markdown(prediction, gt, inference_ms),
                    prediction,
                    updated_history,
                    history_figure(updated_history, smoothing, span, reference_h, reference_w, reference_t),
                    gate_figure(prediction),
                    point_cloud_figure(pc_path),
                    status,
                )

            prediction_event = run_button.click(
                run_prediction,
                inputs=[
                    current_path,
                    orientation,
                    show_mask,
                    history,
                    smooth,
                    ema_span,
                    ref_h,
                    ref_w,
                    ref_t,
                ],
                outputs=[
                    view_a,
                    view_b,
                    measurements,
                    last_prediction,
                    history,
                    history_plot,
                    gates_plot,
                    pc_plot,
                    event_status,
                ],
                concurrency_limit=1,
                concurrency_id="granulo_gpu",
            )

            # Plotly 3-D plots can retain their default ~700 px canvas width after
            # other Gradio components are updated.  Force a browser-side reflow
            # after every inference so the point-cloud canvas follows the full
            # width of its tab without changing the XYZ aspect ratio.
            prediction_event.then(
                fn=None,
                js="""() => {
                    const resizePointCloud = () => {
                        const root = document.querySelector('#point-cloud-plot');
                        if (!root) return;
                        const plot = root.querySelector('.js-plotly-plot, .plotly-graph-div');
                        if (plot && window.Plotly && window.Plotly.Plots) {
                            window.Plotly.Plots.resize(plot);
                        } else {
                            window.dispatchEvent(new Event('resize'));
                        }
                    };
                    requestAnimationFrame(() => {
                        resizePointCloud();
                        setTimeout(resizePointCloud, 100);
                        setTimeout(resizePointCloud, 350);
                    });
                }""",
                queue=False,
            )

            def redraw_history(
                history_value,
                smoothing,
                span,
                reference_h,
                reference_w,
                reference_t,
            ):
                return history_figure(history_value, smoothing, span, reference_h, reference_w, reference_t)

            history_controls = [smooth, ema_span, ref_h, ref_w, ref_t]
            for component in history_controls:
                component.change(
                    redraw_history,
                    inputs=[history, smooth, ema_span, ref_h, ref_w, ref_t],
                    outputs=history_plot,
                    queue=False,
                )

            def clear_history_fn(smoothing, span, reference_h, reference_w, reference_t):
                return [], history_figure([], smoothing, span, reference_h, reference_w, reference_t), "Session prediction history cleared."

            clear_history.click(
                clear_history_fn,
                inputs=[smooth, ema_span, ref_h, ref_w, ref_t],
                outputs=[history, history_plot, event_status],
                queue=False,
            )

        demo.queue(default_concurrency_limit=1, max_size=32)
        return demo


def parse_args():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--model",
        default=os.environ.get("GRANULO_MODEL_PATH"),
        help="Path to the trained multimodal .pt checkpoint. Can also use GRANULO_MODEL_PATH.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("GRANULO_DATA_ROOT"),
        help="Granulo-10k root or Images/Strands_compliant path. Can also use GRANULO_DATA_ROOT.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port. Default: 7860")
    parser.add_argument("--share", action="store_true", help="Ask Gradio for a temporary public share link.")
    parser.add_argument(
        "--root-path",
        default=os.environ.get("GRANULO_ROOT_PATH"),
        help="Optional reverse-proxy subpath, e.g. /granulo. Can also use GRANULO_ROOT_PATH.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = GranuloWebApp(args.model, args.data_root, args.device)
    demo = app.create()
    launch_kwargs = {}
    if int(gr.__version__.split(".")[0]) >= 6:
        launch_kwargs["css"] = CSS
    favicon = Path(__file__).resolve().parent / "granulo_demo_icon.png"
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=favicon if favicon.exists() else None,
        root_path=args.root_path,
        **launch_kwargs,
    )


if __name__ == "__main__":
    main()
