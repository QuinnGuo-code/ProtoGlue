"""Clustering-related losses for DEC training."""

import torch
import torch.nn.functional as F


def dec_kl(q, p):
    """KL divergence loss for DEC: KL(P || Q).

    Parameters
    ----------
    q : Tensor, shape (n, K)
        Soft cluster assignments.
    p : Tensor, shape (n, K)
        Target distribution.

    Returns
    -------
    Tensor
        Scalar KL divergence.
    """
    q = torch.clamp(q, 1e-8, 1.0)
    p = torch.clamp(p, 1e-8, 1.0)
    return torch.mean(torch.sum(p * torch.log(p / q), dim=1))


def balance_kl(q):
    """Balance KL: encourages uniform marginal cluster assignment.

    Parameters
    ----------
    q : Tensor, shape (n, K)
        Soft cluster assignments.

    Returns
    -------
    Tensor
        Scalar balance loss.
    """
    q = torch.clamp(q, 1e-8, 1.0)
    m = torch.mean(q, dim=0)
    u = torch.full_like(m, 1.0 / m.numel())
    return torch.sum(m * torch.log(m / u))


def entropy_loss(p):
    """Entropy loss: encourages sharp (low-entropy) assignment distributions.

    Parameters
    ----------
    p : Tensor, shape (n, K)
        Probability distribution.

    Returns
    -------
    Tensor
        Mean negative entropy.
    """
    p = torch.clamp(p, 1e-8, 1.0)
    return torch.mean(-torch.sum(p * torch.log(p), dim=1))


def smooth_target(p, A_sp, alpha):
    """Apply spatial smoothing to the DEC target distribution.

    Parameters
    ----------
    p : Tensor, shape (n, K)
        Target distribution.
    A_sp : sparse Tensor
        Spatial adjacency matrix.
    alpha : float
        Smoothing strength (0 = no smoothing).

    Returns
    -------
    Tensor
        Spatially smoothed target distribution.
    """
    if alpha <= 0:
        return p
    p2 = (1.0 - alpha) * p + alpha * torch.sparse.mm(A_sp, p)
    return p2 / (torch.sum(p2, dim=1, keepdim=True) + 1e-12)
