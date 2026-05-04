# Granulo-10k
Dataset can be downloaded from Hugging Face [here](https://huggingface.co/datasets/AngeloUNIMI/Granulo-10k).

## Description
Granulo-10k: A Large-Scale Benchmark Dataset for Multiple-View Industrial Granulometry, ICIP 2026.

## Structure
Images
- strands_compliant
- strands_non_compliant

Masks
- strands_compliant
- strands_non_compliant

PCs
- strands_compliant
- strands_non_compliant

## Size
~20GB

## How to load:
```bash
pip install datasets
```
```python
from datasets import load_dataset
dataset = load_dataset("AngeloUNIMI/Granulo-10k")
```

