"""Training callbacks: cluster collapse detection and prototype reset."""

import torch
import numpy as np


def detect_collapse_from_q(q: torch.Tensor, K: int, min_frac: float = 0.01):
    """Detect cluster collapse (clusters that are too small).

    Parameters
    ----------
    q : Tensor, shape (n, K)
        Soft cluster assignments.
    K : int
        Number of clusters.
    min_frac : float
        Minimum fraction threshold below which a cluster is considered collapsed.

    Returns
    -------
    bad_idx : ndarray
        Indices of collapsed clusters.
    frac : ndarray
        Fraction of spots per cluster.
    counts : ndarray
        Absolute counts per cluster.
    """
    lab = q.argmax(dim=1).detach().cpu().numpy().astype(int)
    n = len(lab)
    counts = np.bincount(lab, minlength=K)
    frac = counts / max(1, n)
    bad = np.where(frac < float(min_frac))[0]
    return bad, frac, counts


def reset_dead_prototypes(dec_head, bad_idx, z: torch.Tensor, seed: int = 0):
    """Reset collapsed cluster prototypes to random embedding points.

    Parameters
    ----------
    dec_head : DECHead
        Clustering head with ``mu`` parameter.
    bad_idx : ndarray
        Indices of collapsed clusters to reset.
    z : Tensor, shape (n, d)
        Current latent embeddings (detached).
    seed : int
        Random seed for selecting replacement points.
    """
    if bad_idx is None or len(bad_idx) == 0:
        return
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        n = z.shape[0]
        pick = (
            rng.choice(n, size=len(bad_idx), replace=False)
            if n >= len(bad_idx)
            else rng.choice(n, size=len(bad_idx), replace=True)
        )
        dec_head.mu[torch.tensor(bad_idx, device=z.device, dtype=torch.long)] = z[pick]
