"""Loss functions for ProtoGlue training."""

from .vicreg import stable_vicreg_loss
from .smooth import laplacian_smooth_loss, calibrate_lam_smooth
from .clustering import dec_kl, balance_kl, entropy_loss, smooth_target
from .total import total_loss, AdaptiveLossWeights

__all__ = [
    "stable_vicreg_loss",
    "laplacian_smooth_loss",
    "calibrate_lam_smooth",
    "dec_kl",
    "balance_kl",
    "entropy_loss",
    "smooth_target",
    "total_loss",
    "AdaptiveLossWeights",
]
