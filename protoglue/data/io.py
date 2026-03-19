"""File I/O utilities for loading spatial multi-omics data."""

from pathlib import Path
import pandas as pd
import anndata as ad


def extract_matrix_from_adata(adata, preferred_layers=None, modality_name="adata"):
    """Return a usable expression matrix from an AnnData object.

    Priority: adata.X -> preferred_layers -> common layer names -> raw.X

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    preferred_layers : list of str, optional
        Layer names to try first.
    modality_name : str
        Name used in error messages.

    Returns
    -------
    matrix
        Expression matrix (dense or sparse).

    Raises
    ------
    ValueError
        If no usable expression matrix is found.
    """
    if preferred_layers is None:
        preferred_layers = []
    if getattr(adata, "X", None) is not None:
        return adata.X
    candidates = list(preferred_layers) + [
        "counts", "count", "Counts", "matrix", "Matrix",
        "data", "expr", "expression", "log1p", "normalized",
    ]
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in adata.layers and adata.layers[key] is not None:
            print(f"{modality_name}: X is None, using layers['{key}']")
            return adata.layers[key]
    if getattr(adata, "raw", None) is not None and getattr(adata.raw, "X", None) is not None:
        print(f"{modality_name}: X is None, using raw.X")
        return adata.raw.X
    layer_info = list(adata.layers.keys())
    raise ValueError(
        f"{modality_name}: no usable expression matrix found. "
        f"adata.X is None, layers={layer_info}, "
        f"raw={'yes' if getattr(adata, 'raw', None) is not None else 'no'}"
    )


def ensure_adata_X(adata, preferred_layers=None, modality_name="adata"):
    """Ensure adata.X is populated; fill from layers/raw if necessary."""
    if getattr(adata, "X", None) is None:
        adata.X = extract_matrix_from_adata(
            adata, preferred_layers=preferred_layers, modality_name=modality_name
        )
    return adata


def find_modalities(data_dir, rna_file=None, om2_file=None):
    """Locate RNA and second-modality h5ad files in a directory.

    If explicit filenames are given, they are used directly.
    Otherwise falls back to fixed name pairs, then heuristic scoring.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .h5ad files.
    rna_file : str, optional
        Explicit RNA filename.
    om2_file : str, optional
        Explicit second-modality filename.

    Returns
    -------
    tuple of (Path, Path)
        Paths to RNA and second-modality h5ad files.

    Raises
    ------
    FileNotFoundError
        If required files cannot be located.
    """
    data_dir = Path(data_dir)
    if rna_file is not None and om2_file is not None:
        r, a = data_dir / rna_file, data_dir / om2_file
        if not r.exists():
            raise FileNotFoundError(f"RNA file not found: {r}")
        if not a.exists():
            raise FileNotFoundError(f"Second-modality file not found: {a}")
        return r, a

    # Try common naming conventions
    pairs = [
        ("adata_RNA.h5ad", "adata_ADT.h5ad"),
        ("adata_RNA.h5ad", "adata_peaks_normalized.h5ad"),
    ]
    for r_name, a_name in pairs:
        r, a = data_dir / r_name, data_dir / a_name
        if r.exists() and a.exists():
            return r, a

    # Heuristic: score filenames by keyword matches
    h5ads = sorted(data_dir.glob("*.h5ad"))
    if len(h5ads) < 2:
        raise FileNotFoundError(
            "At least 2 .h5ad files required (RNA + second modality)."
        )

    def score(name, keys):
        name = name.lower()
        return sum(k in name for k in keys)

    rna_keys = ["rna", "gene", "gex", "expr"]
    adt_keys = ["adt", "protein", "prot", "cite", "ab", "peak", "peaks", "atac", "chromatin"]
    scores = [(score(p.name, rna_keys), score(p.name, adt_keys), p) for p in h5ads]
    rna = max(scores, key=lambda x: x[0])[2]
    om2 = max(scores, key=lambda x: x[1])[2]
    if rna == om2:
        om2 = sorted(
            [p for p in h5ads if p != rna],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )[0]
    return rna, om2


def make_h5ad_writeable(adata: ad.AnnData) -> ad.AnnData:
    """Sanitize obs/var dtypes to avoid Arrow/nullable string write errors."""
    out = adata.copy()
    pd.options.mode.string_storage = "python"
    ad.settings.allow_write_nullable_strings = True

    def _fix_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.astype(str).astype("object")
        for c in df.columns:
            dt = df[c].dtype
            if isinstance(dt, pd.StringDtype) or str(dt).startswith("string"):
                df[c] = df[c].astype(str).astype("object")
                continue
            try:
                if df[c].array.__class__.__name__ == "ArrowStringArray":
                    df[c] = df[c].astype(str).astype("object")
                    continue
            except Exception:
                pass
            if isinstance(dt, pd.CategoricalDtype):
                df[c] = df[c].astype(str).astype("object")
        return df

    out.obs = _fix_df(out.obs)
    out.var = _fix_df(out.var)
    return out


def safe_write_h5ad(adata: ad.AnnData, out_path):
    """Write AnnData to h5ad with dtype sanitization."""
    adata2 = make_h5ad_writeable(adata)
    adata2.write_h5ad(Path(out_path), convert_strings_to_categoricals=False)
