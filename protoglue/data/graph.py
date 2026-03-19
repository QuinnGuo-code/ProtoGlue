"""Graph construction utilities for ProtoGlue.

Provides functions to build multi-scale spatial graphs and feature-space
graphs from spatial coordinates and PCA embeddings.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import torch


def symmetrize(A: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetrize a sparse adjacency matrix."""
    return ((A + A.T) > 0).astype(np.float32)


def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a sparse matrix (D^{-1} A)."""
    d = np.maximum(np.asarray(A.sum(1)).squeeze(), 1e-12)
    return sp.diags(1.0 / d) @ A


def knn_graph_fixed(X: np.ndarray, k: int) -> sp.csr_matrix:
    """Build a fixed-k KNN graph, symmetrized and row-normalized.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Feature matrix or coordinate matrix.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    csr_matrix
        Row-normalized adjacency matrix with self-loops.
    """
    n = X.shape[0]
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    return row_normalize(symmetrize(A) + sp.eye(n, dtype=np.float32))


def knn_graph_adaptive(
    coords: np.ndarray, k_base=8, k_min=4, k_max=20, alpha=0.5
) -> sp.csr_matrix:
    """Build a density-adaptive KNN graph.

    Spots in denser regions get more neighbors; spots in sparser regions
    get fewer. This adapts the graph structure to local tissue density.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Spatial coordinates.
    k_base : int
        Base number of neighbors.
    k_min, k_max : int
        Minimum and maximum neighbor counts.
    alpha : float
        Density-adaptive exponent.

    Returns
    -------
    csr_matrix
        Row-normalized adaptive adjacency matrix with self-loops.
    """
    n = coords.shape[0]
    dist, idx = NearestNeighbors(n_neighbors=k_max + 1).fit(coords).kneighbors(coords)
    k0 = min(k_base, k_max)
    mean_dist = dist[:, 1 : k0 + 1].mean(axis=1) + 1e-8
    rho = 1.0 / mean_dist
    k_i = np.clip(k_base * (rho / np.median(rho)) ** alpha, k_min, k_max).astype(int)
    rows, cols = [], []
    for i in range(n):
        nbrs = idx[i, 1 : k_i[i] + 1]
        rows.append(np.full(k_i[i], i, dtype=int))
        cols.append(nbrs.astype(int))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    return row_normalize(symmetrize(A) + sp.eye(n, dtype=np.float32))


def scipy_to_torch_sparse(A: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters
    ----------
    A : csr_matrix
        Scipy sparse matrix.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Coalesced sparse tensor on device.
    """
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce().to(device)


def build_graphs(
    coords: np.ndarray,
    pca1: np.ndarray,
    pca2: np.ndarray,
    use_amsg: bool = True,
    k_base: int = 5,
    k_min: int = 3,
    k_max: int = 10,
    density_alpha: float = 0.2,
    spatial_scales: list = None,
    feat_k: int = 15,
):
    """Build multi-scale spatial and feature-space graphs.

    Parameters
    ----------
    coords : ndarray, shape (n_spots, 2)
        Spatial coordinates.
    pca1 : ndarray, shape (n_spots, d1)
        PCA embeddings for modality 1.
    pca2 : ndarray, shape (n_spots, d2)
        PCA embeddings for modality 2.
    use_amsg : bool
        Whether to include a density-adaptive graph as the first scale.
    k_base, k_min, k_max, density_alpha
        Parameters for adaptive KNN construction.
    spatial_scales : list of int
        Fixed-k values for additional spatial scales.
    feat_k : int
        Number of neighbors for feature-space graphs.

    Returns
    -------
    A_sp_list : list of csr_matrix
        Multi-scale spatial adjacency matrices.
    A_f1 : csr_matrix
        Feature graph for modality 1.
    A_f2 : csr_matrix
        Feature graph for modality 2.
    """
    if spatial_scales is None:
        spatial_scales = [3, 8]

    A_sp_list = []
    if use_amsg:
        A_sp_list.append(
            knn_graph_adaptive(coords, k_base, k_min, k_max, density_alpha)
        )
    for k in spatial_scales:
        A_sp_list.append(knn_graph_fixed(coords, int(k)))
    A_f1 = knn_graph_fixed(pca1, feat_k)
    A_f2 = knn_graph_fixed(pca2, feat_k)
    return A_sp_list, A_f1, A_f2


def graphs_to_torch(A_sp_list, A_f1, A_f2, device):
    """Convert all scipy sparse graphs to torch sparse tensors.

    Parameters
    ----------
    A_sp_list : list of csr_matrix
        Spatial graph list.
    A_f1, A_f2 : csr_matrix
        Feature graphs.
    device : torch.device
        Target device.

    Returns
    -------
    A_sp_t : list of torch.Tensor
    A_f1_t, A_f2_t : torch.Tensor
    """
    A_sp_t = [scipy_to_torch_sparse(A, device) for A in A_sp_list]
    A_f1_t = scipy_to_torch_sparse(A_f1, device)
    A_f2_t = scipy_to_torch_sparse(A_f2, device)
    return A_sp_t, A_f1_t, A_f2_t
