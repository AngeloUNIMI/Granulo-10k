# Granulo-10k Experiment 1: DINOv3 ViT-B/16 backbone swap

This is a controlled variant of the current multimodal training code for the first DINOv3 experiment.

## What changes relative to the current DINOv2 baseline

Only the image-backbone experiment is intentionally changed:

- **DINOv2 ViT-B/14 -> DINOv3 ViT-B/16**
- model name: `vit_base_patch16_dinov3.lvd1689m`
- image input: **512 x 512** instead of 518 x 518
- image backbone remains **frozen**
- one shared frozen image encoder is used for both A/B views, as in the current clean baseline
- global 768-D image embeddings are used; there is **no patch-token/mask-guided pooling** in this experiment

Everything else is kept the same to make this a controlled comparison:

- two RGB views A/B
- PointNet++ SSG and the same bundled ModelNet40-pretrained checkpoint
- PointNet++ projection to 768-D
- explicit point-cloud bounding-box dimensions
- element-wise max fusion of A, B, and point-cloud embeddings
- 64-expert MMoE by default
- task-specific H/W/T gates, towers, and heads
- uncertainty-weighted multitask regression loss
- same strand-disjoint nested 5-fold split
- same optimizer/loss/training logic and point-cloud augmentation

DINOv3 ViT-B/16 also outputs a **768-D embedding**, so the PointNet adapter and MMoE dimensions are unchanged.

## Requirement

DINOv3 support in `timm` requires:

```bash
pip install "timm>=1.0.20"
```

The full requirements file has already been updated accordingly.

## Recommended controlled comparison

Use the same fold, seed, epochs, batch size, and other flags as the DINOv2 run.

Example on the H100:

```bash
python code/multimodal_paper_dinov3/train.py \
  --dataset-root data_full/Granulo-10k \
  --output models/granulo_dinov3_pointnet_mmoe_fold0.pt \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 8 \
  --fold 0 \
  --allow-incomplete
```

If your baseline uses a different batch size or worker count, keep those same values for the fairest timing comparison.

## Expected first-run behavior

On the first model run, `timm` will download the pretrained DINOv3 ViT-B/16 weights. The DINOv3 model is licensed under Meta's DINOv3 license.

## Checkpoint metadata

The saved checkpoint identifies the experiment explicitly:

```text
architecture = DINOv3-ViT-B/16 + PointNet++ + max fusion + MMoE
experiment   = exp1_dinov3_vitb16_frozen_global
dino_model   = vit_base_patch16_dinov3.lvd1689m
image_size   = 512
feature_dim  = 768
```

This makes it easy to distinguish it from the current DINOv2 checkpoints during evaluation or GUI integration.
