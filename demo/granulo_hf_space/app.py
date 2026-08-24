from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from gradio_app import APP_TITLE, CSS, GranuloWebApp


HERE = Path(__file__).resolve().parent


def _first_existing_pt(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    checkpoints = sorted(directory.glob("*.pt")) + sorted(directory.glob("*.pth"))
    return checkpoints[0] if checkpoints else None


def resolve_model_path() -> Path | None:
    """Resolve a local checkpoint first, then optionally download from HF Hub."""
    explicit = os.environ.get("GRANULO_MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path

    bundled = _first_existing_pt(HERE / "models")
    if bundled is not None:
        return bundled

    repo_id = os.environ.get("GRANULO_MODEL_REPO")
    filename = os.environ.get("GRANULO_MODEL_FILE")
    if repo_id and filename:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=os.environ.get("GRANULO_MODEL_REPO_TYPE", "model"),
            revision=os.environ.get("GRANULO_MODEL_REVISION"),
            token=os.environ.get("HF_TOKEN") or None,
        )
        return Path(downloaded)

    return None


def resolve_data_root() -> Path | None:
    """Resolve bundled demo data first, then optionally an HF dataset snapshot."""
    explicit = os.environ.get("GRANULO_DATA_ROOT")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path

    candidates = [
        HERE / "data" / "Granulo-10k",
        HERE / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            # DatasetIndex will discover Images/Strands_compliant below here.
            return candidate

    repo_id = os.environ.get("GRANULO_DATASET_REPO")
    if repo_id:
        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=os.environ.get("GRANULO_DATASET_REVISION"),
            token=os.environ.get("HF_TOKEN") or None,
        )
        subdir = os.environ.get("GRANULO_DATASET_SUBDIR")
        return Path(downloaded) / subdir if subdir else Path(downloaded)

    return None


def resolve_device() -> str:
    explicit = os.environ.get("GRANULO_DEVICE")
    if explicit:
        return explicit
    # ZeroGPU exposes CUDA emulation during startup, so this resolves to cuda
    # there; local CPU-only development still remains usable.
    return "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = resolve_model_path()
DATA_ROOT = resolve_data_root()
DEVICE = resolve_device()

web_app = GranuloWebApp(
    checkpoint=MODEL_PATH,
    data_root=DATA_ROOT,
    device=DEVICE,
)

demo = web_app.create()


if __name__ == "__main__":
    launch_kwargs = {}
    if int(gr.__version__.split(".")[0]) >= 6:
        launch_kwargs["css"] = CSS

    favicon = HERE / "granulo_demo_icon.png"
    demo.launch(
        show_error=True,
        favicon_path=favicon if favicon.exists() else None,
        **launch_kwargs,
    )
