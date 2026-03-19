API Reference
=============

High-level API
--------------

.. autofunction:: protoglue.run_pipeline

Data Loading & Preprocessing
-----------------------------

.. automodule:: protoglue.data.preprocessing
   :members:
   :undoc-members:

.. automodule:: protoglue.data.io
   :members:
   :undoc-members:

.. automodule:: protoglue.data.utils
   :members:
   :undoc-members:

Graph Construction
------------------

.. automodule:: protoglue.data.graph
   :members:
   :undoc-members:

Model
-----

.. autoclass:: protoglue.models.model.ProtoGlue
   :members:
   :undoc-members:

.. autoclass:: protoglue.models.encoders.MultiScaleGCN
   :members:

.. autoclass:: protoglue.models.encoders.ResidualGCN
   :members:

.. autoclass:: protoglue.models.attention.CrossAttentionGatedFusion
   :members:

.. autoclass:: protoglue.models.attention.TwoTokenGatedMHA
   :members:

.. autoclass:: protoglue.models.decoders.MLPDecoder
   :members:

.. autoclass:: protoglue.models.decoders.DECHead
   :members:

Training
--------

.. autofunction:: protoglue.training.trainer.run_stage

.. autofunction:: protoglue.training.k_selection.scan_select_k

.. autofunction:: protoglue.training.k_selection.eval_labels

.. autofunction:: protoglue.training.k_selection.cluster_gmm_full

Losses
------

.. autofunction:: protoglue.losses.total.total_loss

.. autoclass:: protoglue.losses.total.AdaptiveLossWeights
   :members:

.. autofunction:: protoglue.losses.vicreg.stable_vicreg_loss

.. autofunction:: protoglue.losses.smooth.laplacian_smooth_loss

.. autofunction:: protoglue.losses.clustering.dec_kl

.. autofunction:: protoglue.losses.clustering.balance_kl

Configuration
-------------

All default hyperparameters are defined in ``protoglue.config``.
Key parameters include:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``LATENT_DIM``
     - 64
     - Latent embedding dimension
   * - ``HIDDEN_DIM``
     - 256
     - Hidden layer dimension
   * - ``NUM_LAYERS``
     - 2
     - Number of GCN layers
   * - ``NUM_HEADS``
     - 8
     - Attention heads
   * - ``PRETRAIN_EPOCHS``
     - 300
     - Pretraining epochs
   * - ``FINETUNE_EPOCHS``
     - 600
     - Fine-tuning epochs
   * - ``LR``
     - 5e-4
     - Learning rate
   * - ``N_HVG``
     - 3000
     - Highly variable genes
   * - ``N_PCA``
     - 50
     - PCA dimensions
   * - ``SPATIAL_SCALES``
     - [3, 8]
     - Spatial KNN scales
   * - ``K_GRID``
     - [5, 6, 7, 8]
     - K candidates for auto-selection
