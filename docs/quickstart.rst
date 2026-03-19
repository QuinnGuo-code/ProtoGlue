Quick Start
===========

Data preparation
----------------

ProtoGlue expects a data directory with at least two ``.h5ad`` files
(one per modality) and optionally an ``annotation.csv`` for ground-truth labels:

.. code-block:: text

   data_dir/
   ├── adata_RNA.h5ad                   # RNA expression
   ├── adata_ADT.h5ad                   # Protein (ADT) or ATAC data
   └── annotation.csv                   # Optional ground truth

Each ``.h5ad`` file should contain:

- Expression data in ``.X`` (or ``.layers``)
- Spatial coordinates in ``.obsm["spatial"]`` or ``obs`` columns (e.g., ``array_row``, ``array_col``)
- Matching ``obs_names`` (barcodes) across both modalities

One-line pipeline
-----------------

The simplest way to run ProtoGlue:

.. code-block:: python

   import protoglue as pg

   result = pg.run_pipeline(
       data_dir="data/human_lymph_node",
       output_dir="output/lymph_node",
   )

   # Results
   labels = result["labels"]        # cluster assignments
   z = result["latent"]             # fused latent embedding (n_spots, 64)
   metrics = result["metrics"]      # dict with ARI, NMI, silhouette, etc.
   adata = result["adata_rna"]      # AnnData with labels in obs["protoglue_labels"]

Custom configuration
--------------------

Override any hyperparameter via the ``config`` dict:

.. code-block:: python

   result = pg.run_pipeline(
       data_dir="data/my_data",
       output_dir="output/my_output",
       config={
           "N_CLUSTERS": 10,            # force K=10
           "PRETRAIN_EPOCHS": 500,       # longer pretraining
           "FINETUNE_EPOCHS": 800,
           "LATENT_DIM": 128,            # larger latent space
           "SPATIAL_SCALES": [3, 5, 10], # three spatial scales
           "K_GRID": list(range(6, 12)), # scan K from 6 to 11
       },
       auto_k=True,                     # still auto-select K within K_GRID
   )

Step-by-step usage
------------------

For full control over each stage:

.. code-block:: python

   import protoglue as pg
   import torch

   pg.fix_seed(2022)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Step 1: Load and preprocess
   pca1, pca2, coords, adata_rna, adata_om2 = pg.load_and_preprocess(
       data_dir="data/human_lymph_node",
       n_hvg=3000,
       n_pca=50,
   )

   # Step 2: Build graphs
   A_sp_list, A_f1, A_f2 = pg.build_graphs(coords, pca1, pca2)
   A_sp_t, A_f1_t, A_f2_t = pg.graphs_to_torch(A_sp_list, A_f1, A_f2, device)
   X1_t = torch.tensor(pca1, dtype=torch.float32, device=device)
   X2_t = torch.tensor(pca2, dtype=torch.float32, device=device)

   # Step 3: Create model
   model = pg.ProtoGlue(
       in1=pca1.shape[1],
       in2=pca2.shape[1],
       n_scales=len(A_sp_t),
   ).to(device)

   # Step 4: Pretrain
   from protoglue.losses.total import total_loss

   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
   logs = pg.run_stage(
       stage="pretrain", epochs=300,
       model=model, X1_t=X1_t, X2_t=X2_t,
       A_sp_t=A_sp_t, A_f1_t=A_f1_t, A_f2_t=A_f2_t,
       optimizer=optimizer, loss_fn=total_loss,
   )

   # Step 5: Select K
   model.eval()
   with torch.no_grad():
       z0 = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t)["z"].cpu().numpy()
   best_k, df = pg.scan_select_k(z0, A_sp_list[0])
   print(f"Best K = {best_k}")

   # Step 6: Fine-tune with DEC
   from sklearn.cluster import KMeans

   dec_head = pg.DECHead(best_k, 64).to(device)
   km = KMeans(n_clusters=best_k, n_init=50).fit(z0)
   with torch.no_grad():
       dec_head.mu.copy_(torch.tensor(km.cluster_centers_, device=device))

   optimizer_ft = torch.optim.AdamW(
       list(model.parameters()) + list(dec_head.parameters()), lr=2.5e-4
   )
   logs_ft = pg.run_stage(
       stage="finetune", epochs=600,
       model=model, X1_t=X1_t, X2_t=X2_t,
       A_sp_t=A_sp_t, A_f1_t=A_f1_t, A_f2_t=A_f2_t,
       dec_head=dec_head, optimizer=optimizer_ft, loss_fn=total_loss,
       warm_offset=300,
   )

   # Step 7: Get final results
   model.eval()
   with torch.no_grad():
       out = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t, dec_head=dec_head)
   labels = out["q"].cpu().numpy().argmax(1)
