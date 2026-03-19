"""Training utilities for ProtoGlue."""

from .trainer import run_stage, temp_schedule
from .k_selection import scan_select_k, eval_labels, cluster_gmm_full
from .callbacks import detect_collapse_from_q, reset_dead_prototypes

__all__ = [
    "run_stage",
    "temp_schedule",
    "scan_select_k",
    "eval_labels",
    "cluster_gmm_full",
    "detect_collapse_from_q",
    "reset_dead_prototypes",
]
