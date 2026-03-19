"""Data loading and preprocessing pipeline for spatial multi-omics."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from .io import find_modalities, ensure_adata_X, extract_matrix_from_adata
from .utils import ensure_dense, clr_normalize_per_cell, spatial_coords_from_adata


def load_and_preprocess(
    data_dir: Union[str, Path],
    rna_file: Optional[str] = None,
    om2_file: Optional[str] = None,
    n_hvg: int = 3000,
    n_pca: int = 50,
    seed: int = 2022,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ad.AnnData, ad.AnnData]:
    """Load and preprocess spatial multi-omics data (RNA + second modality).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .h5ad files and optional annotation.csv.
    rna_file : str, optional
        RNA data filename. Auto-detected if None.
    om2_file : str, optional
        Second modality filename. Auto-detected if None.
    n_hvg : int
        Number of highly variable genes for RNA feature selection.
    n_pca : int
        PCA target dimensionality (adjusted automatically if needed).
    seed : int
        Random seed.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    pca1 : ndarray, shape (n_spots, n_pca_actual)
        RNA PCA embeddings.
    pca2 : ndarray, shape (n_spots, n_pca_actual_om2)
        Second modality PCA embeddings.
    coords : ndarray, shape (n_spots, 2)
        Spatial coordinates.
    adata_rna : AnnData
        RNA AnnData with annotation in ``obs['manual_anno']`` if available.
    adata_om2 : AnnData
        Second modality AnnData after preprocessing.
    """
    data_dir = Path(data_dir)

    # 1. Locate data files
    rna_path, om2_path = find_modalities(data_dir, rna_file, om2_file)
    if verbose:
        print("RNA:", rna_path.name)
        print("Omics2:", om2_path.name)

    # 2. Read data
    adata_rna = sc.read_h5ad(str(rna_path))
    adata_om2 = sc.read_h5ad(str(om2_path))
    adata_rna.var_names_make_unique()
    adata_om2.var_names_make_unique()

    # 3. Ensure expression matrix is in .X
    adata_rna = ensure_adata_X(
        adata_rna, preferred_layers=["counts", "data"], modality_name="RNA"
    )
    adata_om2 = ensure_adata_X(
        adata_om2, preferred_layers=["counts", "data"], modality_name="Omics2"
    )

    # 4. Align spots
    common = adata_rna.obs_names.intersection(adata_om2.obs_names)
    adata_rna = adata_rna[common].copy()
    adata_om2 = adata_om2[common].copy()
    adata_om2 = adata_om2[adata_rna.obs_names].copy()
    if verbose:
        print("Aligned spots:", adata_rna.n_obs)

    # 5. Load ground-truth annotation (for evaluation only, not training)
    _anno_loaded = False
    _OBS_ANNO_CANDIDATES = [
        "RNA_clusters", "ATAC_clusters", "manual_anno", "annotation",
        "celltype", "cell_type", "label", "cluster", "ground_truth",
    ]
    for _col in _OBS_ANNO_CANDIDATES:
        if _col in adata_rna.obs.columns:
            adata_rna.obs["manual_anno"] = (
                adata_rna.obs[_col].astype(str).astype("category")
            )
            if verbose:
                print(
                    f"Annotation loaded from obs['{_col}']: "
                    f"{adata_rna.obs['manual_anno'].nunique()} classes"
                )
            _anno_loaded = True
            break
        if not _anno_loaded and _col in adata_om2.obs.columns:
            adata_rna.obs["manual_anno"] = (
                adata_om2.obs.reindex(adata_rna.obs_names)[_col]
                .astype(str)
                .astype("category")
            )
            if verbose:
                print(
                    f"Annotation loaded from adata_om2.obs['{_col}']: "
                    f"{adata_rna.obs['manual_anno'].nunique()} classes"
                )
            _anno_loaded = True
            break

    # Fallback: annotation.csv
    if not _anno_loaded:
        anno_path = data_dir / "annotation.csv"
        if anno_path.exists():
            ann = pd.read_csv(anno_path)
            if "Barcode" in ann.columns:
                ann = ann.set_index("Barcode")
            elif "barcode" in ann.columns:
                ann = ann.set_index("barcode")
            lab_col = (
                "manual-anno"
                if "manual-anno" in ann.columns
                else ("label" if "label" in ann.columns else ann.columns[0])
            )
            adata_rna.obs["manual_anno"] = ann.reindex(adata_rna.obs_names)[
                lab_col
            ].astype("category")
            missing = int(adata_rna.obs["manual_anno"].isna().sum())
            if verbose:
                print(f"Annotation loaded from annotation.csv. Missing: {missing}")
            _anno_loaded = True

    if not _anno_loaded and verbose:
        print("No annotation found (unsupervised mode; ARI/NMI will be skipped).")

    # 6. RNA preprocessing: HVG -> normalize -> log1p -> scale -> PCA
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    adata_rna.raw = adata_rna.copy()
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_hvg, flavor="seurat_v3")
    sc.pp.scale(adata_rna, max_value=10)
    adata_rna_h = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    n_comps_rna = min(n_pca, adata_rna_h.n_vars - 1, adata_rna_h.n_obs - 1)
    if n_comps_rna < n_pca and verbose:
        print(f"Note: adjusting RNA PCA n_comps {n_pca} -> {n_comps_rna}")
    sc.tl.pca(adata_rna_h, n_comps=n_comps_rna, svd_solver="arpack")
    pca1 = np.array(adata_rna_h.obsm["X_pca"], dtype=np.float32)

    # 7. Second modality preprocessing: CLR + scale + PCA
    X2 = ensure_dense(
        extract_matrix_from_adata(
            adata_om2, preferred_layers=["counts", "data"], modality_name="Omics2"
        )
    ).astype(np.float32)
    adata_om2.X = clr_normalize_per_cell(X2)
    if verbose:
        print("Omics2 preprocessing: CLR normalization")
    sc.pp.scale(adata_om2, max_value=10)

    n_comps_om2 = min(n_pca, adata_om2.n_vars - 1, adata_om2.n_obs - 1)
    if n_comps_om2 < n_pca and verbose:
        print(f"Note: adjusting Omics2 PCA n_comps {n_pca} -> {n_comps_om2}")
    sc.tl.pca(adata_om2, n_comps=n_comps_om2, svd_solver="arpack")
    pca2 = np.array(adata_om2.obsm["X_pca"], dtype=np.float32)

    # 8. Extract spatial coordinates
    coords = spatial_coords_from_adata(adata_rna)

    if verbose:
        print(f"pca1: {pca1.shape}  pca2: {pca2.shape}  coords: {coords.shape}")

    assert pca1.shape[0] == pca2.shape[0], "Spot count mismatch between RNA and Omics2"
    return pca1, pca2, coords, adata_rna, adata_om2
