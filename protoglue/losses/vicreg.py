"""VICReg (Variance-Invariance-Covariance Regularization) loss."""

import torch
import torch.nn.functional as F

from protoglue.config import VICREG_SIM_COEFF, VICREG_VAR_COEFF, VICREG_COV_COEFF


def stable_vicreg_loss(z1, z2, sim_coeff=None, var_coeff=None, cov_coeff=None, eps=1e-8):
    """Stable VICReg loss for cross-modal embedding alignment.

    Parameters
    ----------
    z1, z2 : Tensor, shape (n, d)
        Embeddings from two modalities.
    sim_coeff, var_coeff, cov_coeff : float, optional
        Override default coefficients from config.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        Scalar VICReg loss.
    """
    _sim = float(sim_coeff or VICREG_SIM_COEFF)
    _var = float(var_coeff or VICREG_VAR_COEFF)
    _cov = float(cov_coeff or VICREG_COV_COEFF)

    z1 = z1 - z1.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)

    sim_loss = F.mse_loss(z1, z2)

    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.softplus(1.0 - std1)) + torch.mean(F.softplus(1.0 - std2))

    n = z1.shape[0]
    cov1 = (z1.T @ z1) / max(n - 1, 1)
    cov2 = (z2.T @ z2) / max(n - 1, 1)
    cov1 -= torch.diag(torch.diag(cov1))
    cov2 -= torch.diag(torch.diag(cov2))
    cov_loss = (cov1 ** 2).mean() + (cov2 ** 2).mean()

    return _sim * sim_loss + _var * var_loss + _cov * cov_loss
