"""Data loading, preprocessing, and graph construction for ProtoGlue."""

from .io import (
    find_modalities,
    extract_matrix_from_adata,
    ensure_adata_X,
    make_h5ad_writeable,
    safe_write_h5ad,
)
from .preprocessing import load_and_preprocess
from .graph import (
    symmetrize,
    row_normalize,
    knn_graph_fixed,
    knn_graph_adaptive,
    scipy_to_torch_sparse,
    build_graphs,
    graphs_to_torch,
)
from .utils import (
    fix_seed,
    ensure_dense,
    clr_normalize_per_cell,
    spatial_coords_from_adata,
)

__all__ = [
    "find_modalities",
    "extract_matrix_from_adata",
    "ensure_adata_X",
    "make_h5ad_writeable",
    "safe_write_h5ad",
    "load_and_preprocess",
    "symmetrize",
    "row_normalize",
    "knn_graph_fixed",
    "knn_graph_adaptive",
    "scipy_to_torch_sparse",
    "build_graphs",
    "graphs_to_torch",
    "fix_seed",
    "ensure_dense",
    "clr_normalize_per_cell",
    "spatial_coords_from_adata",
]
