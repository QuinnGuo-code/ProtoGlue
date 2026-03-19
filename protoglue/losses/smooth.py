"""Spatial smoothness losses for ProtoGlue."""

import torch
import torch.nn.functional as F


def laplacian_smooth_loss(z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Laplacian smoothness loss: penalizes embedding differences between neighbors.

    Parameters
    ----------
    z : Tensor, shape (n, d)
        Latent embeddings.
    A : sparse Tensor
        Adjacency matrix.

    Returns
    -------
    Tensor
        Scalar smoothness loss.
    """
    A = A.coalesce()
    i, j = A.indices()[0], A.indices()[1]
    return torch.mean(torch.sum((z[i] - z[j]) ** 2, dim=1))


def calibrate_lam_smooth(model, X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t, target_ratio=0.05):
    """Estimate initial LAM_SMOOTH so smooth_loss ~ target_ratio * recon_loss.

    Parameters
    ----------
    model : ProtoGlue
        Model in eval mode.
    X1_t, X2_t : Tensor
        Input features.
    A_sp_t : list of sparse Tensor
        Spatial adjacency matrices.
    A_f1_t, A_f2_t : sparse Tensor
        Feature adjacency matrices.
    target_ratio : float
        Desired ratio of smooth to recon loss.

    Returns
    -------
    float
        Calibrated smoothness weight.
    """
    model.eval()
    with torch.no_grad():
        out = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t)
        l_recon = (
            F.mse_loss(out["x1_hat"], X1_t) + F.mse_loss(out["x2_hat"], X2_t)
        ).item()
        l_smooth = laplacian_smooth_loss(out["z"], A_sp_t[0]).item()
    if l_smooth < 1e-10:
        return 0.0
    return round(min(max(target_ratio * l_recon / l_smooth, 0.0), 1.0), 6)
