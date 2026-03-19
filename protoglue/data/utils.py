"""Utility functions for data handling."""

import random
import numpy as np
import torch
import scipy.sparse as sp


def fix_seed(seed: int = 2022) -> None:
    """Fix random seeds for reproducibility across numpy, torch, and CUDA.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dense(X):
    """Convert sparse matrix to dense array if needed."""
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def clr_normalize_per_cell(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Centered log-ratio normalization per cell (for ADT/protein data).

    Parameters
    ----------
    X : ndarray, shape (n_cells, n_features)
        Raw count matrix.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    ndarray
        CLR-normalized matrix.
    """
    X = np.log1p(np.maximum(X, 0.0) + eps)
    return X - X.mean(axis=1, keepdims=True)


def spatial_coords_from_adata(adata) -> np.ndarray:
    """Extract 2D spatial coordinates from an AnnData object.

    Tries ``obsm['spatial']`` first, then common ``obs`` column pairs.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with spatial information.

    Returns
    -------
    ndarray, shape (n_spots, 2)
        Spatial coordinates.

    Raises
    ------
    ValueError
        If spatial coordinates cannot be found.
    """
    if "spatial" in adata.obsm:
        coords = np.array(adata.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    for cols in (
        ("array_row", "array_col"),
        ("x", "y"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ):
        if all(c in adata.obs.columns for c in cols):
            return adata.obs.loc[:, list(cols)].to_numpy()
    raise ValueError(
        "Spatial coordinates not found in obsm['spatial'] or standard obs columns."
    )
