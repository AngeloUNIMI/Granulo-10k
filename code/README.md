# Granulo-10k: Multiple-View Industrial Granulometry

Official PyTorch implementation associated with the paper:

> **Granulo-10k: A Large-Scale Benchmark Dataset for Multiple-View Industrial Granulometry**  
> Pasquale Coscia, Angelo Genovese, Vincenzo Piuri, Fabio Scotti

The code trains multi-view neural networks to estimate three geometric properties of OSB wood strands:

- height;
- width;
- thickness.

Each sample can combine two synchronized RGB views with a 3D point cloud. The repository includes convolutional and foundation-model image encoders, PointNet++ point-cloud encoding, and three decoder variants: plain, gated, and Multi-gate Mixture-of-Experts (MMoE).

## Repository structure

```text
.
├── cnn_osb.py                         # Main training and evaluation program
├── format.py
├── gated_exps.sh                      # Runs the gated decoder on several backbones
├── mmoe_exps.sh                       # Runs MMoE experiments
├── plain_exps.sh                      # Runs the plain decoder on several backbones
├── test_resnets_original_fusion.sh    # Runs the residual backbones
├── modelGeno/
│   ├── resnet_geno.py                 # Image encoders and decoder definitions
│   └── pointnet_utils.py              # PointNet++ wrapper and projection head
├── functions/
│   ├── trainPanelClassify.py           # Optional classification pre-training stage
│   ├── trainStrandRegr.py              # Multi-task regression training loop
│   └── test.py                         # Test procedure
├── util/
│   ├── imageFolderWithSizePC.py        # Two-view image and point-cloud dataset loader
│   ├── losses.py                       # Regression and uncertainty losses
│   ├── models.py                       # TIMM encoders, optimizers, input resolutions
│   ├── splitPanels.py                  # Dataset split generation and loading
│   ├── resultsLogger.py                # Experiment logging
│   └── calib_opencv_simple.yml         # Stereo calibration parameters
└── Pointnet_Pointnet2_pytorch/         # Bundled PointNet/PointNet++ implementation
```

## Main architecture

The model contains three encoders:

1. an image encoder for camera A;
2. an image encoder for camera B;
3. a PointNet++ encoder for the associated point cloud.

The PointNet++ representation is projected to the same feature dimensionality as the image embeddings. The point-cloud branch also receives three dimensions computed from the point cloud. The three modality embeddings are fused and passed to one of the supported decoders.

### Decoder options

- `plain`: fully connected regression decoder;
- `gated`: task-specific gated decoder;
- `mmoe`: Multi-gate Mixture-of-Experts decoder.

The MMoE decoder uses shared experts and separate task-specific gates and heads for height, width, and thickness. The default number of experts is 64.

## Supported image backbones

### Residual networks

- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `resnext50_32x4d`
- `resnext101_32x8d`
- `resnext101_64x4d`
- `wide_resnet50_2`
- `wide_resnet101_2`

### Foundation and modern image models

- `dino_vitb14`
- `clip_vitl14`
- `eva02_clip_l14`
- `convnextv2_base`

The corresponding input resolutions are selected automatically:

| Backbone family | Input resolution |
|---|---:|
| ResNet, ResNeXt, Wide ResNet | 320 x 240 |
| DINO ViT-B/14 | 518 x 518 |
| CLIP ViT-L/14 | 224 x 224 |
| EVA02 | 336 x 336 |
| ConvNeXtV2 Base | 384 x 384 |

## Requirements

A CUDA-capable GPU is strongly recommended.

The code requires Python 3 and the following principal packages:

```text
torch
torchvision
timm
numpy
scikit-learn
matplotlib
Pillow
PyYAML
```

A typical installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision timm numpy scikit-learn matplotlib Pillow PyYAML
```

Install the PyTorch build appropriate for the CUDA version available on the system.

## Dataset preparation

The current program uses a fixed dataset root in `cnn_osb.py`:

```python
baseDir = '../../../data/CNN_OSB/'
```

Before running the code, either place the data at that location relative to the repository or update `baseDir` locally.

The expected structure is:

```text
../../../data/CNN_OSB/
├── DB Wood (test)/
│   └── DB_strand_IPAN3D_TII_buoni_e_sottili_JPG/
│       ├── datastore_A/
│       │   └── <class-folder>/
│       │       └── <strand>_<acquisition>_A.jpg
│       ├── datastore_B/
│       │   └── <class-folder>/
│       │       └── <strand>_<acquisition>_B.jpg
│       ├── datastore_PC/
│       │   └── <class-folder>/
│       │       ├── <strand>_<acquisition>_PC_lungh_largh.xyz
│       │       └── <strand>_<acquisition>_PC_thickness.xyz
│       └── datastore_train_test/       # Generated splits and cached statistics
└── DB Wood (orig)/
    └── DB_strand_IPAN3D_TII/
        ├── Misure_buoni_e_sottili.csv
        └── Strand_per_spessore_buoni.csv
```

The image folders follow the directory convention required by `torchvision.datasets.ImageFolder`; images must therefore be stored inside one or more class subdirectories.

For each camera-A image, the loader derives the corresponding camera-B image and point-cloud filename from the first two underscore-separated fields of the filename.

Point clouds are read from `.xyz` text files containing three columns. The loader resamples each cloud to 2,048 points.

## PointNet++ checkpoint

The point-cloud encoder expects a pretrained PointNet++ classification checkpoint at:

```text
Pointnet_Pointnet2_pytorch/log/classification/
└── pointnet2_ssg_wo_normals/
    └── checkpoints/
        └── best_model.pth
```

The checkpoint must contain a `model_state_dict` entry compatible with the bundled `pointnet2_cls_ssg` implementation.

A compatible checkpoint can be obtained by training the bundled PointNet++ implementation on ModelNet40 or by placing the checkpoint used for the experiments in the path above.

## Running an experiment

Run commands from the repository root, where `cnn_osb.py` is located.

### Basic run

```bash
python3 cnn_osb.py
```

This uses the default configuration:

- backbone: `resnet18`;
- decoder: `plain`;
- training mode: `imagenet`;
- five iterations;
- 50 regression epochs;
- base learning rate: `0.0012`;
- 64 experts when MMoE is selected.

### DINO with MMoE

```bash
python3 cnn_osb.py \
  --models dino_vitb14 \
  --decoder mmoe \
  --num_experts 64 \
  --base_lr 0.0012 \
  --num_epochs_regr 50 \
  --num_iterations 5
```

### Multiple backbones in one command

```bash
python3 cnn_osb.py \
  --models resnet18 resnet34 dino_vitb14 \
  --decoder mmoe
```

The program processes the requested models sequentially.

### Plain decoder

```bash
python3 cnn_osb.py \
  --models dino_vitb14 \
  --decoder plain
```

### Gated decoder

```bash
python3 cnn_osb.py \
  --models dino_vitb14 \
  --decoder gated
```

### MMoE ablation on the number of experts

```bash
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --num_experts 1
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --num_experts 8
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --num_experts 32
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --num_experts 64
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --num_experts 128
```

### Learning-rate ablation

```bash
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --base_lr 0.0006
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --base_lr 0.0012
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --base_lr 0.0024
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --base_lr 0.0036
python3 cnn_osb.py --models dino_vitb14 --decoder mmoe --base_lr 0.0048
```

## Batch scripts

The repository includes shell scripts for launching groups of experiments:

```bash
bash plain_exps.sh
bash gated_exps.sh
bash mmoe_exps.sh
bash test_resnets_original_fusion.sh
```

Edit the `MODELS` array inside a script to enable or disable particular backbones.

## Command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--seed_adams` | `42` | Random seed used by the experiment |
| `--plotta` / `--no-plotta` | disabled | Enables or disables plotting |
| `--log` / `--no-log` | enabled | Enables or disables result logging |
| `--num_iterations` | `5` | Number of cross-validation iterations |
| `--batch_size` | `256` | Training and validation batch size |
| `--batch_size_norm` | `256` | Batch size used to compute normalization statistics |
| `--batch_size_test` | `256` | Test batch size |
| `--numWorkersP` | `8` | Number of DataLoader workers |
| `--class_switch` | `0` | Enables the optional classification stage when nonzero |
| `--num_epochs_class` | `10` | Number of classification epochs |
| `--num_epochs_regr` | `50` | Number of regression epochs |
| `--base_lr` | `0.0012` | Base learning rate |
| `--models` | `resnet18` | One or more backbone names |
| `--decoder` | `plain` | Decoder type: `plain`, `gated`, or `mmoe` |
| `--num_experts` | `64` | Number of experts used by MMoE |
| `--trainModes` | `imagenet` | One or more initialization modes, such as `imagenet` or `scratch` |

Example using smaller batches:

```bash
python3 cnn_osb.py \
  --models dino_vitb14 \
  --decoder mmoe \
  --batch_size 16 \
  --batch_size_norm 16 \
  --batch_size_test 16 \
  --numWorkersP 4
```

## Training procedure

For each selected backbone and training mode, the program:

1. loads strand measurements and thickness-visibility information;
2. constructs the two image encoders and the PointNet++ branch;
3. optionally performs the classification stage;
4. creates the selected regression decoder;
5. generates or loads dataset splits;
6. trains the multi-task regression model;
7. evaluates height, width, and thickness prediction;
8. stores per-iteration and aggregate results.

The regression loss is based on learnable task-dependent uncertainty and jointly balances the three target dimensions.

## Output files

Results are written under:

```text
results/
└── DB_strand_IPAN3D_TII_buoni_e_sottili_JPG/
    └── <base-learning-rate>/
        └── <training-mode>/
            └── <backbone>/
```

The directory may contain:

- timestamped text logs;
- `results_<iteration>.dat` files;
- `resultsFinal.dat` with aggregate results;
- `results.txt` generated by the result logger.

Dataset splits and normalization statistics are stored in the dataset's `datastore_train_test` directory.

## GPU memory

The default batch size of 256 may exceed the memory available on many GPUs, particularly for ViT-L, EVA, DINO, and ConvNeXt backbones. Reduce all three batch-size options when an out-of-memory error occurs:

```bash
python3 cnn_osb.py \
  --models eva02_clip_l14 \
  --decoder mmoe \
  --batch_size 8 \
  --batch_size_norm 8 \
  --batch_size_test 8
```

## Pretrained weights and internet access

When `--trainModes imagenet` is used, TorchVision or TIMM may download pretrained image-model weights automatically. The first run therefore requires internet access unless the weights are already cached.

PointNet++ weights are not downloaded automatically and must be placed at the path described above.

## Implementation notes

- The current release assumes that the dataset path and file names match the structure described above.
- Stereo augmentation uses `util/calib_opencv_simple.yml`.
- The point-cloud loader checks first for `_PC_lungh_largh.xyz` and then for `_PC_thickness.xyz`.
- The current training pipeline uses point-cloud input in the regression dataset loader.
- The program should be launched from the repository root because several paths are relative to the current working directory.
- `script_lambdas.sh` refers to experimental arguments that are not part of the current `cnn_osb.py` command-line interface; it is retained as an auxiliary historical script and is not required for the main experiments.

## Troubleshooting

### `FileNotFoundError` for the dataset

Check the `baseDir` value in `cnn_osb.py` and ensure the expected directories and CSV files exist.

### Missing PointNet++ checkpoint

Place `best_model.pth` in:

```text
Pointnet_Pointnet2_pytorch/log/classification/
pointnet2_ssg_wo_normals/checkpoints/
```

### CUDA out-of-memory error

Reduce `--batch_size`, `--batch_size_norm`, and `--batch_size_test`.

### DataLoader worker errors

Try:

```bash
python3 cnn_osb.py --numWorkersP 0
```

### TIMM model or pretrained-weight error

Update TIMM and TorchVision:

```bash
pip install --upgrade timm torchvision
```

## Citation

When using this code or the Granulo-10k dataset, please cite the associated paper:

```bibtex
@inproceedings{coscia2026granulo10k,
  title     = {Granulo-10k: A Large-Scale Benchmark Dataset for Multiple-View Industrial Granulometry},
  author    = {Coscia, Pasquale and Genovese, Angelo and Piuri, Vincenzo and Scotti, Fabio},
  year      = {2026}
}
```

Please update the BibTeX entry with the final conference name, pages, DOI, and publication details when available.

## Acknowledgments

This repository includes code derived from the PointNet/PointNet++ PyTorch implementation distributed in the `Pointnet_Pointnet2_pytorch` directory. Refer to its own README and license information for attribution and usage conditions.
