from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    GranuloMultimodalDataset,
    compute_label_stats,
    compute_scalar_image_stats,
    discover_samples,
    paper_five_fold_split,
)


TASKS = ("height", "width", "thickness")

# Experiment 1: frozen DINOv3 ViT-B/16 global embedding.
# 512 is divisible by the ViT-B/16 patch size and keeps the comparison close
# to the 518x518 DINOv2 baseline.
IMAGE_SIZE = 512
DINO_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset,
    batch_size,
    shuffle,
    workers,
    pin_memory,
    prefetch_factor,
):
    generator = torch.Generator()
    generator.manual_seed(42)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "worker_init_fn": worker_seed if workers > 0 else None,
        "generator": generator,
    }

    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


def configure_cuda_acceleration(device, args):
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = None
    gpu_name = None
    capability = None

    if device.type != "cuda":
        return amp_enabled, amp_dtype, gpu_name, capability

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark

    tf32_enabled = not args.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled

    if tf32_enabled:
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)

    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    if amp_enabled:
        if args.amp_dtype == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "--amp-dtype bf16 was requested, but BF16 is not supported."
                )
            amp_dtype = torch.bfloat16
        elif args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        else:
            amp_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

    return amp_enabled, amp_dtype, gpu_name, capability


def autocast_context(amp_enabled, amp_dtype):
    if not amp_enabled:
        return nullcontext()

    return torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=True,
    )


def denormalize(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return pred_norm * std + mean


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    label_mean,
    label_std,
    device,
    amp_enabled,
    amp_dtype,
):
    model.eval()
    criterion.eval()
    total_loss = 0.0
    total_n = 0
    abs_error = torch.zeros(3, device=device)

    for batch in loader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        pc = batch["point_cloud"].to(device, non_blocking=True)
        pc_dims = batch["pc_dims"].to(device, non_blocking=True)
        target_norm = batch["target_norm"].to(device, non_blocking=True)
        target_mm = batch["target_mm"].to(device, non_blocking=True)
        visibility = batch["visibility"].to(device, non_blocking=True)

        with autocast_context(amp_enabled, amp_dtype):
            pred_norm, _ = model(image_a, image_b, pc, pc_dims)
            loss = criterion(pred_norm, target_norm, visibility)

        pred_mm = denormalize(pred_norm.float(), label_mean, label_std)
        abs_error += (pred_mm - target_mm).abs().sum(dim=0)
        batch_n = target_mm.shape[0]
        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

    return {
        "loss": total_loss / max(total_n, 1),
        "mae_height_mm": float(abs_error[0].item() / max(total_n, 1)),
        "mae_width_mm": float(abs_error[1].item() / max(total_n, 1)),
        "mae_thickness_mm": float(abs_error[2].item() / max(total_n, 1)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 1: frozen DINOv3 ViT-B/16 + PointNet++ + MMoE"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Granulo-10k"),
        help="Folder containing Images/, Masks/, and PCs/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/granulo_dinov3_pointnet_mmoe.pt"),
    )
    parser.add_argument(
        "--pointnet-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "pointnet2_ssg_best_model.pth",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-points", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP) and train/evaluate in FP32.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16"),
        default="auto",
        help="AMP dtype. Auto prefers BF16 on H100/A100-class GPUs.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 acceleration for remaining FP32 CUDA operations.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="Disable cuDNN benchmark/autotuning.",
    )
    parser.add_argument(
        "--no-fused-adamw",
        action="store_true",
        help="Disable fused CUDA AdamW.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched by each DataLoader worker.",
    )
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--masked-uncertainty-loss",
        action="store_true",
        help=(
            "Use frontal/sideways visibility masks in the uncertainty loss. "
            "Default reproduces the supplied source, which passes the mask but does not use it."
        ),
    )
    parser.add_argument(
        "--no-pc-augmentation",
        action="store_true",
        help="Disable the supplied Z-rotation + XY-jitter point-cloud augmentation.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip incomplete acquisitions instead of failing dataset discovery.",
    )
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Discover samples and print split counts without loading/training the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    samples = discover_samples(args.dataset_root, strict=not args.allow_incomplete)
    train_samples, val_samples, test_samples = paper_five_fold_split(
        samples, seed=args.seed, fold=args.fold, num_folds=5
    )

    print(f"Complete multimodal acquisitions: {len(samples)}")
    print(f"Unique strands: {len(set(s.strand_id for s in samples))}")
    print(
        "Split (paper-style strand-wise 5-fold): "
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )
    print(
        "Split strands: "
        f"train={len(set(s.strand_id for s in train_samples))}, "
        f"val={len(set(s.strand_id for s in val_samples))}, "
        f"test={len(set(s.strand_id for s in test_samples))}"
    )
    print(
        "Orientations: "
        f"frontal={sum(s.orientation == 'frontal' for s in samples)}, "
        f"sideways={sum(s.orientation == 'sideways' for s in samples)}"
    )

    if args.check_data_only:
        return

    # Import the heavy model stack only after data validation, so --check-data-only
    # can run before TIMM is installed/downloaded.
    from model import GranuloPaperModel, UncertaintyMultiTaskLoss

    # Source implementation normalizes labels globally before the split.
    label_mean_np, label_std_np = compute_label_stats(samples)
    print(f"Label mean [H,W,T]: {label_mean_np.tolist()}")
    print(f"Label std  [H,W,T]: {label_std_np.tolist()}")

    stats_cache = args.output.with_suffix(".image_stats.json")
    # Use train+val only for image statistics to avoid test leakage while keeping
    # the supplied code's single-scalar normalization scheme.
    image_mean, image_std = compute_scalar_image_stats(
        train_samples + val_samples,
        image_size=IMAGE_SIZE,
        cache_path=stats_cache,
    )
    print(f"Image scalar mean/std: {image_mean:.6f} / {image_std:.6f}")

    train_ds = GranuloMultimodalDataset(
        train_samples,
        label_mean_np,
        label_std_np,
        image_mean,
        image_std,
        image_size=IMAGE_SIZE,
        sample_points=args.sample_points,
        augment_point_cloud=not args.no_pc_augmentation,
    )
    val_ds = GranuloMultimodalDataset(
        val_samples,
        label_mean_np,
        label_std_np,
        image_mean,
        image_std,
        image_size=IMAGE_SIZE,
        sample_points=args.sample_points,
        augment_point_cloud=False,
    )
    test_ds = GranuloMultimodalDataset(
        test_samples,
        label_mean_np,
        label_std_np,
        image_mean,
        image_std,
        image_size=IMAGE_SIZE,
        sample_points=args.sample_points,
        augment_point_cloud=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled, amp_dtype, gpu_name, capability = configure_cuda_acceleration(
        device,
        args,
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {gpu_name}")
        print(f"Compute capability: {capability[0]}.{capability[1]}")

    if amp_enabled:
        dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        print(f"AMP: enabled ({dtype_name})")
    else:
        print("AMP: disabled (FP32)")

    if device.type == "cuda":
        print(f"TF32: {'disabled' if args.no_tf32 else 'enabled'}")
        print(
            "cuDNN benchmark: "
            f"{'disabled' if args.no_cudnn_benchmark else 'enabled'}"
        )

    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
        args.prefetch_factor,
    )
    val_loader = make_loader(
        val_ds,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
        args.prefetch_factor,
    )
    test_loader = make_loader(
        test_ds,
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
        args.prefetch_factor,
    )


    model = GranuloPaperModel(
        pointnet_checkpoint=args.pointnet_checkpoint,
        num_experts=args.num_experts,
        pretrained_dino=True,
        freeze_dino=True,
    ).to(device)

    criterion = UncertaintyMultiTaskLoss(
        num_tasks=3,
        use_visibility=args.masked_uncertainty_loss,
    ).to(device)

    # Supplied foundation-model regression optimizer: AdamW; encoder branch at
    # 5% of base LR, decoder and uncertainty parameters at base LR.
    point_params = [p for p in model.point_encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    optimizer_groups = [
        {
            "params": point_params,
            "lr": args.base_lr * 0.05,
            "weight_decay": 0.05,
        },
        {
            "params": decoder_params,
            "lr": args.base_lr,
            "weight_decay": 0.0,
        },
        {
            "params": criterion.parameters(),
            "lr": args.base_lr,
            "weight_decay": 0.0,
        },
    ]

    use_fused_adamw = device.type == "cuda" and not args.no_fused_adamw

    try:
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            fused=use_fused_adamw,
        )
    except (TypeError, RuntimeError):
        use_fused_adamw = False
        optimizer = torch.optim.AdamW(optimizer_groups)

    print(f"AdamW: {'fused' if use_fused_adamw else 'standard'}")

    # BF16 does not require GradScaler. FP16 still benefits from it.
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
    )


    label_mean = torch.tensor(label_mean_np, device=device)
    label_std = torch.tensor(label_std_np, device=device)

    best_val = float("inf")
    best_state = None
    history = []
    since = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        # Keep the frozen DINO branch frozen but allow the source-like train/eval
        # module mode to follow the overall model.
        criterion.train()
        running_loss = 0.0
        running_n = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            image_a = batch["image_a"].to(device, non_blocking=True)
            image_b = batch["image_b"].to(device, non_blocking=True)
            pc = batch["point_cloud"].to(device, non_blocking=True)
            pc_dims = batch["pc_dims"].to(device, non_blocking=True)
            target_norm = batch["target_norm"].to(device, non_blocking=True)
            visibility = batch["visibility"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled, amp_dtype):
                pred_norm, _ = model(image_a, image_b, pc, pc_dims)
                loss = criterion(pred_norm, target_norm, visibility)

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            n = target_norm.shape[0]
            running_loss += float(loss.item()) * n
            running_n += n

            if batch_idx % 25 == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"loss={running_loss / max(running_n, 1):.5f}"
                )

        train_loss = running_loss / max(running_n, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            label_mean,
            label_std,
            device,
            amp_enabled,
            amp_dtype,
        )
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        epoch_seconds = time.perf_counter() - epoch_start
        train_samples_per_second = running_n / max(epoch_seconds, 1e-9)

        perf_suffix = (
            f" | {epoch_seconds:.1f} s, "
            f"{train_samples_per_second:.1f} train samples/s"
        )

        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            perf_suffix += f", peak VRAM={peak_gb:.2f} GB"

        print(
            f"Epoch {epoch:03d}: train={train_loss:.5f} "
            f"val={val_metrics['loss']:.5f} | "
            f"MAE H={val_metrics['mae_height_mm']:.3f} mm, "
            f"W={val_metrics['mae_width_mm']:.3f} mm, "
            f"T={val_metrics['mae_thickness_mm']:.4f} mm"
            f"{perf_suffix}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "criterion": {k: v.detach().cpu() for k, v in criterion.state_dict().items()},
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            print("  -> new best validation checkpoint")

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    model.load_state_dict(best_state["model"])
    criterion.load_state_dict(best_state["criterion"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        label_mean,
        label_std,
        device,
        amp_enabled,
        amp_dtype,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "architecture": "DINOv3-ViT-B/16 + PointNet++ + max fusion + MMoE",
        "model_state_dict": best_state["model"],
        "criterion_state_dict": best_state["criterion"],
        "config": {
            "dino_model": DINO_MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "feature_dim": 768,
            "sample_points": args.sample_points,
            "num_experts": args.num_experts,
            "expert_hidden": 256,
            "tower_hidden": 64,
            "shared_frozen_dino": True,
            "experiment": "exp1_dinov3_vitb16_frozen_global",
            "amp_enabled": amp_enabled,
            "amp_dtype": (
                "bfloat16"
                if amp_dtype == torch.bfloat16
                else (
                    "float16"
                    if amp_dtype == torch.float16
                    else "float32"
                )
            ),
            "tf32_enabled": device.type == "cuda" and not args.no_tf32,
            "cudnn_benchmark": device.type == "cuda" and not args.no_cudnn_benchmark,
            "fused_adamw": use_fused_adamw,
            "prefetch_factor": args.prefetch_factor,
            "masked_uncertainty_loss": args.masked_uncertainty_loss,
            "seed": args.seed,
            "fold": args.fold,
        },
        "normalization": {
            "image_mean_scalar": image_mean,
            "image_std_scalar": image_std,
            "label_mean": label_mean_np.tolist(),
            "label_std": label_std_np.tolist(),
        },
        "splits": {
            "train_strands": sorted({s.strand_id for s in train_samples}),
            "val_strands": sorted({s.strand_id for s in val_samples}),
            "test_strands": sorted({s.strand_id for s in test_samples}),
        },
        "best_epoch": best_state["epoch"],
        "best_val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }
    torch.save(checkpoint, args.output)

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.time() - since
    print("\nTraining complete")
    print(f"Best epoch: {best_state['epoch']}")
    print(f"Test metrics: {test_metrics}")
    print(f"Checkpoint: {args.output}")
    print(f"History: {history_path}")
    print(f"Elapsed: {elapsed / 60.0:.1f} min")


if __name__ == "__main__":
    main()
