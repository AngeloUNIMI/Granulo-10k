#!/bin/bash

echo "Launching experiments for selected models..."

# List of models to be tested
MODELS=(
    "resnet18"
    "resnet34"
    "resnet50"
    "resnet101"
    "resnet152"
    "resnext50_32x4d"
    "resnext101_32x8d"
    "resnext101_64x4d"
    "wide_resnet50_2"
    "wide_resnet101_2"
    "dino_vitb14"
    "clip_vitl14"
    "eva02_clip_l14"
    "convnextv2_base"
)


for model in "${MODELS[@]}"; do
    echo "========================================================"
    echo "Running all experiments for model: $model"
    echo "========================================================"
    echo "  Model:           $model"
    python3 cnn_osb.py --models "$model" --decoder plain
done

echo "All experiments completed."
