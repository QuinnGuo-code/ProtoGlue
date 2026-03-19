"""
ProtoGlue: Graph-based deep learning for spatial multi-omics integration
and spatial domain identification.

ProtoGlue integrates spatial multi-omics data through:
- Multi-scale spatial graph construction with density-adaptive neighbors
- Asymmetric encode-decode architecture with cross-reconstruction correspondence
- Prototype-based deep embedded clustering with spatial refinement

Example
-------
>>> import protoglue as pg
>>> adata_rna, adata_om2 = pg.load_data("path/to/data")
>>> model = pg.ProtoGlue(in1=50, in2=50, n_scales=3)
>>> result = pg.run_pipeline("path/to/data", "path/to/output")
"""

__version__ = "0.1.0"
__author__ = "Xuan Guo"

from . import config
from .models import ProtoGlue
from .models.decoders import DECHead
from .data.preprocessing import load_and_preprocess
from .data.graph import build_graphs, graphs_to_torch
from .data.utils import fix_seed
from .training.trainer import run_stage
from .training.k_selection import scan_select_k, eval_labels
from .losses.total import total_loss
from .api import run_pipeline

__all__ = [
    "config",
    "ProtoGlue",
    "DECHead",
    "load_and_preprocess",
    "build_graphs",
    "graphs_to_torch",
    "fix_seed",
    "run_stage",
    "scan_select_k",
    "eval_labels",
    "total_loss",
    "run_pipeline",
]
