"""Combined loss function for ProtoGlue training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from protoglue.config import (
    CORRESPONDENCE_TYPE, LAM_RECON, LAM_CORR, LAM_SMOOTH, LAM_VICREG,
    LAM_CLUSTER, LAM_BALANCE, LAM_ALPHA_ENT, LAM_ALPHA_VAR, LAM_ALPHA_SMOOTH,
    BASE_LOSS_WEIGHTS, USE_ADAPTIVE_LOSS, ADAPTIVE_LOSS_TERMS,
)
from .smooth import laplacian_smooth_loss
from .vicreg import stable_vicreg_loss
from .clustering import dec_kl, balance_kl, entropy_loss


class AdaptiveLossWeights(nn.Module):
    """Learns per-term loss weights via log-variance (uncertainty weighting).

    Parameters
    ----------
    terms : list of str
        Names of loss terms to weight.
    init_logvar : float
        Initial log-variance value.
    clamp : float
        Clamping range for log-variance.
    """

    def __init__(self, terms, init_logvar=0.0, clamp=8.0):
        super().__init__()
        self.terms = list(terms)
        self.clamp = float(clamp)
        self.log_vars = nn.Parameter(
            torch.full((len(self.terms),), float(init_logvar))
        )

    def forward(self, losses: dict):
        total = 0.0
        weights = {}
        logvars = {}
        for i, t in enumerate(self.terms):
            if t not in losses or losses[t] is None:
                continue
            s = torch.clamp(self.log_vars[i], -self.clamp, self.clamp)
            w = torch.exp(-s)
            total = total + w * losses[t] + s
            weights[t] = w
            logvars[t] = s
        return total, weights, logvars


def total_loss(out, x1, x2, A_sp_smooth, p_target=None, warm=1.0, loss_weighter=None):
    """Compute total ProtoGlue loss with all components.

    Components: reconstruction, correspondence, smoothness, VICReg,
    clustering (DEC KL), balance, and modality weight regularization.

    Parameters
    ----------
    out : dict
        Model forward output dictionary.
    x1, x2 : Tensor
        Original input features.
    A_sp_smooth : sparse Tensor
        Spatial adjacency for smoothness computation.
    p_target : Tensor, optional
        DEC target distribution (None during pretraining).
    warm : float
        Warmup coefficient for clustering losses (0-1).
    loss_weighter : AdaptiveLossWeights, optional
        Adaptive loss weight module.

    Returns
    -------
    loss : Tensor
        Scalar total loss.
    parts : dict
        Dictionary of individual loss values for logging.
    """
    d1, d2 = x1.shape[1], x2.shape[1]

    # Self-reconstruction loss (dimension-normalized)
    l_recon = (
        (d2 / (d1 + d2)) * F.mse_loss(out["x1_hat"], x1)
        + (d1 / (d1 + d2)) * F.mse_loss(out["x2_hat"], x2)
    )

    # Correspondence loss (cross-reconstruction or cycle)
    if CORRESPONDENCE_TYPE == "cross_recon":
        l_corr = 0.5 * F.mse_loss(out["y1_hat"], x2) + 0.5 * F.mse_loss(out["y2_hat"], x1)
    else:
        l_corr = 0.5 * F.mse_loss(out["y1"], out["y1_hat"]) + 0.5 * F.mse_loss(out["y2"], out["y2_hat"])

    # Smoothness loss
    l_smooth = laplacian_smooth_loss(out["z"], A_sp_smooth)

    # VICReg loss
    l_v = (
        stable_vicreg_loss(out["y1"], out["y2"])
        if LAM_VICREG > 0
        else torch.tensor(0.0, device=x1.device)
    )

    # Clustering loss
    l_c = (
        dec_kl(out["q"], p_target)
        if (p_target is not None and out["q"] is not None)
        else torch.tensor(0.0, device=x1.device)
    )
    l_b = (
        balance_kl(out["q"])
        if (out["q"] is not None and LAM_BALANCE > 0)
        else torch.tensor(0.0, device=x1.device)
    )

    # Modality weight (alpha) regularization
    l_alpha_ent = (
        entropy_loss(out["alpha_modality"])
        if (LAM_ALPHA_ENT > 0 and out.get("alpha_modality") is not None)
        else torch.tensor(0.0, device=x1.device)
    )
    if LAM_ALPHA_VAR > 0 and out.get("alpha_modality") is not None:
        alpha_rna = out["alpha_modality"][:, 0]
        l_alpha_var = -torch.var(alpha_rna)
        l_alpha_smooth = laplacian_smooth_loss(alpha_rna.unsqueeze(1), A_sp_smooth)
    else:
        l_alpha_var = l_alpha_smooth = torch.tensor(0.0, device=x1.device)

    # Collect all loss components
    losses = {
        "recon": l_recon,
        "corr": l_corr,
        "smooth": l_smooth,
        "alpha_ent": l_alpha_ent,
        "alpha_var": l_alpha_var,
        "alpha_smooth": l_alpha_smooth,
        "vicreg": l_v,
        "cluster": (
            l_c * float(warm)
            if (warm and out["q"] is not None and p_target is not None and LAM_CLUSTER > 0)
            else torch.tensor(0.0, device=x1.device)
        ),
        "balance": (
            l_b * float(warm)
            if (warm and out["q"] is not None and LAM_BALANCE > 0)
            else torch.tensor(0.0, device=x1.device)
        ),
    }

    # Compute total loss
    w_dict = {}
    lv_dict = {}
    if loss_weighter is not None:
        losses_scaled = {
            k: losses[k] * float(BASE_LOSS_WEIGHTS.get(k, 1.0)) for k in losses
        }
        loss, w_dict_t, lv_dict_t = loss_weighter(losses_scaled)
        w_dict = {f"w_{k}": float(v.detach().cpu()) for k, v in w_dict_t.items()}
        lv_dict = {f"logvar_{k}": float(v.detach().cpu()) for k, v in lv_dict_t.items()}
    else:
        loss = (
            LAM_RECON * l_recon + LAM_CORR * l_corr + LAM_SMOOTH * l_smooth
            + LAM_VICREG * l_v + LAM_ALPHA_ENT * l_alpha_ent
            + LAM_ALPHA_VAR * l_alpha_var + LAM_ALPHA_SMOOTH * l_alpha_smooth
            + LAM_CLUSTER * float(warm) * l_c + LAM_BALANCE * float(warm) * l_b
        )

    # Log individual parts
    parts = {
        "loss": float(loss.detach().cpu()),
        "recon": float(l_recon.detach().cpu()),
        "corr": float(l_corr.detach().cpu()),
        "smooth": float(l_smooth.detach().cpu()),
        "vicreg": float(l_v.detach().cpu()),
        "cluster": float(l_c.detach().cpu()),
        "balance": float(l_b.detach().cpu()),
        "alpha_ent": float(l_alpha_ent.detach().cpu()),
        "alpha_var": float(l_alpha_var.detach().cpu()),
        "alpha_smooth": float(l_alpha_smooth.detach().cpu()),
        "warm": float(warm),
    }
    parts.update(w_dict)
    parts.update(lv_dict)
    return loss, parts
