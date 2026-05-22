# Data Preparation

This repository does not redistribute the raw datasets.
Please obtain the datasets from their original publications or official release pages and organize them as described below.

## Datasets

| Dataset | Modalities | Source | Reference |
|---------|-----------|--------|-----------|
| Human Lymph Node | RNA + ADT | Official public release / original publication | Long et al., *Nature Methods* (2024) |
| Human Breast Cancer | RNA + ADT | Official public release / original publication | Long et al., *Nature Methods* (2024) |

## Minimal Example Data Requirement

The script `examples/minimal_example.py` expects a directory containing at least two `.h5ad` files:

- one RNA modality file
- one second-modality file

Example:

    data/human_lymph_node/
    ├── adata_RNA.h5ad
    └── adata_ADT.h5ad

If fewer than two `.h5ad` files are found in the specified directory, the example script will raise a `FileNotFoundError`.

## Expected Directory Structure

After downloading, organize files as follows:

    data/
    ├── human_lymph_node/
    │   ├── adata_RNA.h5ad
    │   ├── adata_ADT.h5ad
    │   └── annotation.csv
    │
    └── human_breast_cancer/
        ├── adata_RNA.h5ad
        ├── adata_ADT.h5ad
        └── annotation.csv

`annotation.csv` is optional and only used when ground-truth labels are available.

## Required File Format

Each `.h5ad` file should be an [AnnData](https://anndata.readthedocs.io/) object with:

- an expression matrix stored in `.X` or `.layers`
- spatial coordinates stored in `.obsm["spatial"]` (shape: `n_spots × 2`) or in `.obs` columns such as `array_row` and `array_col`
- matching barcodes (`obs_names`) across both modalities

## Optional Ground-Truth Annotation

If available, ground-truth labels can be provided in one of the following ways:

- as a column in `adata_rna.obs`, for example: `manual_anno`, `celltype`, `label`, or `ground_truth`
- as a separate `annotation.csv` file with columns such as `Barcode` and `manual-anno`

Ground-truth labels are **not required** for running ProtoGlue.
They are only used for evaluation metrics such as ARI and NMI, and for automatic cluster-number selection when applicable.

## Notes

- ProtoGlue can auto-detect modality file names; explicit file names can also be provided via the `rna_file` and `om2_file` arguments.
- The second modality can be protein (ADT) or chromatin accessibility (ATAC) data.
- For ADT data, CLR normalization is commonly applied to the second modality.
  For ATAC or other modality types, please follow the corresponding tutorial or dataset-specific preprocessing pipeline.
