# Output Description

ProtoGlue creates output files in the user-specified output directory during runtime.

Typical outputs may include:

- predicted spatial domain labels
- low-dimensional latent embeddings
- clustering and evaluation metrics
- publication-ready visualization figures

This directory is not included in the repository with precomputed results by default.
It will be created automatically when running the example scripts or tutorial workflows.

For example, if the following output path is specified:

`results/human_lymph_node`

ProtoGlue will automatically create the directory if it does not already exist.
