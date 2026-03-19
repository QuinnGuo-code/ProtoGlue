# ProtoGlue

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ProtoGlue** is a unified graph-based deep learning framework for spatial multi-omics integration and spatial domain identification. It jointly leverages multimodal molecular profiles and spatial neighborhood information to learn biologically meaningful and spatially structured latent representations, identify spatial domains, and support downstream visualization and quantitative evaluation.

## Highlights

- **Unified framework** for spatial multi-omics integration and spatial domain identification
- **Multi-scale spatial graph construction** for modeling local tissue organization
- **Modality-specific graph-based encoding** for learning complementary latent features
- **Cross-modal reconstruction correspondence** for improving inter-modality consistency
- **Uncertainty-weighted multi-objective optimization** for balanced joint training
- **Automatic cluster-number selection** for unsupervised domain identification

## Overview

The overall workflow of ProtoGlue is illustrated below.

![ProtoGlue overview](docs/_static/protoglue_overview.png)

## Installation

### Option 1: Install from GitHub

```bash
pip install git+https://github.com/QuinnGuo-code/ProtoGlue.git
```

### Option 2: Install from source

```bash
git clone https://github.com/QuinnGuo-code/ProtoGlue.git
cd ProtoGlue
pip install -e .
```

### Option 3: Using conda

```bash
conda env create -f environment.yml
conda activate protoglue
pip install -e .
```

> **GPU users:** The default `environment.yml` installs CPU-only PyTorch. For GPU support, edit `environment.yml` and replace `cpuonly` with `pytorch-cuda=11.8` (or `12.1`), or install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/).

### Dependencies

ProtoGlue requires Python >= 3.8 and PyTorch >= 1.12. GPU support (NVIDIA CUDA) is strongly recommended for training. See [requirements.txt](requirements.txt) for the full dependency list.

## Quick Start

```python
from pathlib import Path
import protoglue as pg

result = pg.run_pipeline(
    data_dir=Path("data/human_lymph_node"),
    output_dir=Path("results/human_lymph_node"),
    device="cpu",
)

print(result["metrics"])
# {'n_clusters': 6, 'ARI': 0.52, 'NMI': 0.61, 'spatial_edge_purity': 0.87, ...}
```

The output directory will be created automatically if it does not exist.

See [examples/minimal_example.py](examples/minimal_example.py) for a command-line example.

## Tutorials

We provide step-by-step tutorial notebooks for three benchmark datasets:

| Tutorial | Dataset | Modalities | Ground Truth |
|----------|---------|------------|:------------:|
| [Tutorial 1](tutorials/tutorial_human_lymph_node.ipynb) | Human Lymph Node | RNA + ADT | ✓ |
| [Tutorial 2](tutorials/tutorial_human_breast_cancer.ipynb) | Human Breast Cancer | RNA + ADT | ✗ |
| [Tutorial 3](tutorials/tutorial_mouse_brain.ipynb) | Mouse Brain | RNA + ATAC | ✗ |

## Data Preparation

The raw datasets are not distributed in this repository. Please follow the instructions in [data/README.md](data/README.md) to download and organize the input files.

Expected input format: two `.h5ad` files per dataset (one per modality), with matching barcodes and spatial coordinates.

## Output

ProtoGlue produces:

- predicted spatial domain labels
- low-dimensional fused embeddings
- clustering and evaluation metrics, including ARI, NMI, silhouette score, spatial purity, and Moran's I
- visualization outputs for downstream analysis

See [results/README.md](results/README.md) for details on output format.

## Documentation

Online documentation is available at [Read the Docs](https://protoglue.readthedocs.io/en/latest/).

Documentation source files are also available in the `docs/` directory and include:

- [Installation guide](docs/installation.rst)
- [Quick start](docs/quickstart.rst)
- [Tutorials](docs/tutorials.rst)
- [API reference](docs/api.rst)

Step-by-step tutorial notebooks are also provided in the `tutorials/` directory for the benchmark datasets.

## Architecture

```text
Input (PCA₁, PCA₂, spatial coords)
  │
  ├─ Multi-Scale Spatial GCN ──┐     ├─ Multi-Scale Spatial GCN ──┐
  ├─ Feature Graph GCN ────────┤     ├─ Feature Graph GCN ────────┤
  │                             │     │                             │
  └── Intra-modal Attention ───→ y₁   └── Intra-modal Attention ───→ y₂
                                 │                                   │
                                 └── Cross-Attention Gated Fusion ──→ z
                                                   │
                              ┌────────────────────┼────────────────────┐
                              │                    │                    │
                         Self-Recon           DEC Clustering      Cross-Recon
                        (z → x̂₁, x̂₂)      (z → soft labels)   (y₁ → x̂₂, y₂ → x̂₁)
```

## Configuration

All hyperparameters have sensible defaults and can be overridden via the `config` argument:

```python
result = pg.run_pipeline(
    data_dir="data/my_dataset",
    output_dir="results/my_output",
    config={
        "N_CLUSTERS": 8,
        "PRETRAIN_EPOCHS": 500,
        "FINETUNE_EPOCHS": 800,
        "LATENT_DIM": 128,
        "SPATIAL_SCALES": [3, 5, 10],
    },
)
```

See [protoglue/config.py](protoglue/config.py) for the full list of configurable parameters.

## Citation

If you use ProtoGlue in your research, please cite:

```bibtex
@misc{protoglue2026,
  title={ProtoGlue: a unified framework for spatial multi-omics integration and spatial domain identification},
  author={Guo, Xuan and Xian, Xuewei and Wu, Siyi and Wang, Yanqin and Zhang, Lanyue and Liu, Wei},
  year={2026},
  note={Manuscript under review}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions, please open an [issue](https://github.com/QuinnGuo-code/ProtoGlue/issues) on GitHub.
