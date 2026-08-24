---
title: Granulo-10k Strand Measurement Demo
emoji: 🪵
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.22.0
python_version: "3.12.12"
app_file: app.py
pinned: false
header: mini
---

# Granulo-10k Multimodal Strand Measurement Demo

Interactive Gradio demo for multimodal strand measurement using two RGB views,
a 3-D point cloud, DINOv3 ViT-B/16, PointNet++, and an MMoE regression head.

The Space is prepared for **Hugging Face ZeroGPU**. Select **ZeroGPU** in the
Space hardware settings before testing inference.

## Required files

### Model checkpoint

The easiest first deployment is to upload the Experiment-1 checkpoint to:

```text
models/granulo_dinov3_pointnet_mmoe_fold0.pt
```

Any `.pt`/`.pth` file placed directly in `models/` will be auto-detected.

Alternatively, keep the checkpoint in a separate Hugging Face model repository
and configure these Space variables/secrets:

```text
GRANULO_MODEL_REPO=<namespace/model-repo>
GRANULO_MODEL_FILE=<checkpoint-file.pt>
```

For a private model repository, add `HF_TOKEN` as a **Space Secret**, not as a
plain variable.

### Demo dataset

Place the demo subset under:

```text
data/Granulo-10k/
├── Images/
│   └── Strands_compliant/
├── Masks/
│   └── Strands_compliant/
└── PCs/
    └── Strands_compliant/
```

Keep the existing Granulo-10k internal subfolders beneath `Strands_compliant`.
The app also looks for `measurements.txt` and `strands_ok_for_thickness.txt`
using the same layout as the local demo.

Alternatively, store the demo data in an HF Dataset repository and set:

```text
GRANULO_DATASET_REPO=<namespace/dataset-repo>
```

If the Granulo root is inside a subdirectory of the dataset repository, also set:

```text
GRANULO_DATASET_SUBDIR=<relative/path>
```

## ZeroGPU

The expensive prediction callback is decorated with `@spaces.GPU` and requests
20 seconds by default. The model itself is loaded once at Space startup and is
placed on CUDA at module level, as recommended for ZeroGPU.

To change the requested maximum duration, set:

```text
GRANULO_ZERO_GPU_DURATION=20
```

Do **not** add `spaces` to `requirements.txt`; the ZeroGPU runtime manages that
package.

## Optional variables

```text
GRANULO_MODEL_PATH
GRANULO_DATA_ROOT
GRANULO_DEVICE
GRANULO_ZERO_GPU_DURATION
GRANULO_MODEL_REPO
GRANULO_MODEL_FILE
GRANULO_MODEL_REPO_TYPE
GRANULO_MODEL_REVISION
GRANULO_DATASET_REPO
GRANULO_DATASET_REVISION
GRANULO_DATASET_SUBDIR
```
