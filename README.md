# Granulo-10k

<p align="center">
  <img href="imgs/example.png" />
  <b>A large-scale benchmark dataset for multiple-view industrial granulometry of OSB wood strands.</b>
</p>

<p align="center">
  <a href="#dataset">Dataset</a> •
  <a href="#download">Download</a> •
  <a href="#tasks">Tasks</a> •
  <a href="#baselines">Baselines</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**Granulo-10k** is an open benchmark dataset for research on **Oriented Strand Board (OSB) strand analysis**, with a focus on multiple-view granulometry and three-dimensional geometric estimation.

The dataset contains high-resolution paired images of wood strands acquired with a calibrated two-camera setup, together with segmentation masks, granulometric ground truth, and 3D point clouds. It is designed to support reproducible research on automated strand segmentation and estimation of:

- **height** (`h`)
- **width** (`w`)
- **thickness** (`t`)

Granulo-10k addresses a key limitation in industrial wood vision research: most existing approaches estimate geometry from 2D projections and often do not provide individual strand-level thickness measurements. This dataset provides multiple views and point-cloud information to encourage research on full 3D strand characterization.

---

## Highlights

- **9,600 RGB images** at `1280 x 960` resolution
- **200 unique OSB wood strands**
- **24 acquisitions per strand**
- **2 synchronized camera views** per acquisition
- **Segmentation masks** for each paired acquisition
- **3D point clouds** associated with each paired acquisition
- **Ground-truth granulometric measurements**: height, width, and thickness
- **Compliant / non-compliant strand labels** based on manufacturer reference dimensions
- **Strand-disjoint evaluation protocol** to avoid train-test leakage

---

## Dataset

Granulo-10k contains images of thinly chopped wood pieces, known as **strands**, used in OSB panel production.

### Strand categories

The dataset includes 200 manually measured strands, divided into two classes:

| Category | Number of strands | Average size |
| --- | ---: | --- |
| Compliant | 100 | `h x w x t = 115 x 20 x 0.70 mm` |
| Non-compliant | 100 | `h x w x t = 91 x 9 x 0.65 mm` |

For each strand, the maximum height and width were measured using a caliper. Since thickness may vary across the strand surface, multiple thickness measurements were collected at different points and averaged.

### Acquisition protocol

Images were collected using a calibrated multiple-view acquisition system composed of:

- two synchronized **Sony SX90CR** color cameras
- a trigger mechanism connected to a photocell
- four LED bars arranged to provide approximately uniform illumination
- calibrated camera geometry for multiple-view reconstruction

The two cameras were placed at the same height and oriented at an angle of approximately `85 deg` with respect to the support, with a camera distance of `125 mm`. LED bars were placed at approximately `90 mm` from the cameras.

<p align="center">
  <img src="figures/fig1_acquisition_setup.png" alt="Outline of the multiple-view acquisition setup" width="520">
</p>

<p align="center">
  <em>Fig. 1. Outline of the multiple-view acquisition setup used to capture synchronized views of falling strands.</em>
</p>

Each strand was dropped from random positions above the cameras, while ensuring that it fell inside the intersection of the two fields of view. To increase acquisition variability, each strand was acquired 24 times:

- 8 frontal drops
- 8 sideways drops
- 8 intermediate-orientation drops

This results in:

```text
200 strands x 24 acquisitions x 2 cameras = 9,600 RGB images
```

### Data modalities

For each paired acquisition, Granulo-10k provides:

- RGB image from camera 1
- RGB image from camera 2
- segmentation mask for camera 1
- segmentation mask for camera 2
- associated 3D point cloud
- ground-truth height, width, and thickness measurements
- compliance category

<p align="center">
  <img src="figures/fig2_dataset_examples.png" alt="Examples of Granulo-10k strands with images, masks, and point clouds" width="900">
</p>

<p align="center">
  <em>Fig. 2. Examples of three strands captured with the two-camera setup. Each acquisition includes RGB images, segmentation masks, and a corresponding 3D point cloud.</em>
</p>

---

## Download

Dataset can be downloaded from Hugging Face [here](https://huggingface.co/datasets/AngeloUNIMI/Granulo-10k).

---

## Tasks

Granulo-10k supports several research tasks:

### 1. Strand segmentation

Estimate binary masks for OSB strands from single-view or paired RGB images.

### 2. Multiple-view granulometry

Estimate strand height, width, and thickness using one or more of the available modalities:

- image-only input
- point-cloud-only input
- fused image and point-cloud input

### 3. Compliant vs. non-compliant classification

Classify strands according to whether their geometry is compliant with manufacturer reference dimensions.

### 4. Multi-modal geometric learning

Develop methods that combine two RGB views and 3D point clouds for robust geometric reasoning.

---

## Baselines

The accompanying paper evaluates modern visual backbones and multi-modal fusion strategies for joint estimation of height, width, and thickness.

### Architecture summary

The proposed baseline uses:

- two frozen image encoders, one for each camera view
- a PointNet++ encoder for the point cloud
- an MLP adapter to align point-cloud features with image embeddings
- max-pooling feature fusion
- a Multi-gate Mixture-of-Experts (MMoE) decoder
- task-specific heads for height, width, and thickness regression
- a multi-task uncertainty-weighted loss

### Evaluation protocol

Experiments use:

- 5-fold cross-validation
- 50 training epochs per fold
- learning rate of `1.2e-3`
- strand-disjoint splits, so all acquisitions of the same strand are assigned to the same split
- MAE and MAPE as evaluation metrics

### Reported results

The following table summarizes representative results from the paper. Values are reported as mean ± standard deviation.

| Backbone | Point cloud | Height MAE [mm] | Height MAPE [%] | Width MAE [mm] | Width MAPE [%] | Thickness MAE [mm] | Thickness MAPE [%] |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean value baseline | - | 17.88 ± 1.45 | 21.58 ± 2.86 | 6.24 ± 0.50 | 56.22 ± 3.56 | 0.18 ± 0.03 | 29.29 ± 2.32 |
| DINO ViT-B/14 | No | 2.70 ± 0.34 | 2.99 ± 0.34 | 1.65 ± 0.21 | 12.13 ± 1.67 | 0.09 ± 0.01 | 13.70 ± 0.67 |
| ConvNeXtV2 | No | 2.95 ± 0.31 | 3.30 ± 0.37 | 1.64 ± 0.13 | 11.67 ± 1.16 | 0.09 ± 0.01 | 14.91 ± 1.52 |
| EVA02-CLIP ViT-L/14 | No | 2.82 ± 0.34 | 3.19 ± 0.35 | 1.73 ± 0.10 | 12.11 ± 0.70 | 0.09 ± 0.00 | 14.63 ± 1.33 |
| CLIP ViT-L/14 | No | 2.99 ± 0.24 | 3.34 ± 0.16 | 1.84 ± 0.17 | 13.38 ± 1.69 | 0.11 ± 0.01 | 17.77 ± 1.67 |
| DINO ViT-B/14 | Yes | **2.53 ± 0.08** | **2.77 ± 0.09** | **1.59 ± 0.02** | **11.94 ± 0.35** | 0.10 ± 0.00 | 14.58 ± 0.57 |
| ConvNeXtV2 | Yes | 2.93 ± 0.16 | 3.24 ± 0.16 | 1.79 ± 0.06 | 13.94 ± 0.38 | 0.10 ± 0.00 | 15.13 ± 0.13 |
| EVA02-CLIP ViT-L/14 | Yes | 3.01 ± 0.07 | 3.31 ± 0.08 | 1.78 ± 0.07 | 13.47 ± 0.93 | 0.11 ± 0.00 | 17.44 ± 0.79 |
| CLIP ViT-L/14 | Yes | 3.41 ± 0.14 | 3.80 ± 0.12 | 2.06 ± 0.17 | 15.93 ± 2.02 | 0.13 ± 0.01 | 19.96 ± 1.19 |

DINO ViT-B/14 with point-cloud input achieves the strongest performance for height and width estimation, while thickness estimation remains the most challenging target.

---

## 📖 Citation

If you use Granulo-10k in your research, please cite the associated paper:

```bibtex
@inproceedings{coscia2026granulo10k,
  title     = {Granulo-10k: A Large-Scale Benchmark Dataset for Multiple-View Industrial Granulometry},
  author    = {Coscia, Pasquale and Genovese, Angelo and Piuri, Vincenzo and Scotti, Fabio},
  booktitle = {Proceedings of the IEEE International Conference on Image Processing (ICIP)},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported in part by the EC under project **EdgeAI** (`101097300`). Project EdgeAI is supported by the Chips Joint Undertaking and its members, including top-up funding by Austria, Belgium, France, Greece, Italy, Latvia, the Netherlands, and Norway.

The authors thank **IMAL s.r.l.**, San Damaso, Modena, Italy, for cooperation in providing data and sample classification. The authors also acknowledge Prof. **Ruggero Donida Labati** for his contribution to the data collection process.


