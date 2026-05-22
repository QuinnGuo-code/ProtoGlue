Tutorials
=========

We provide step-by-step tutorial notebooks for three benchmark datasets.
Each tutorial walks through the complete ProtoGlue workflow from data loading
to spatial domain visualization.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Tutorial
     - Modalities
     - Ground Truth
     - Spots
   * - :doc:`tutorials/tutorial_human_lymph_node`
     - RNA + ADT
     - Yes
     - ~4,000
   * - :doc:`tutorials/tutorial_human_breast_cancer`
     - RNA + ADT
     - No
     - ~3,800

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/tutorial_human_lymph_node
   tutorials/tutorial_human_breast_cancer

Running the tutorials
---------------------

1. Install ProtoGlue following the :doc:`installation` guide.
2. Download the datasets following the instructions in ``data/README.md``.
3. Open any tutorial notebook with Jupyter:

   .. code-block:: bash

      jupyter notebook tutorials/tutorial_human_lymph_node.ipynb

4. Run all cells sequentially. GPU is recommended for training steps.

Alternatively, all tutorials can be run as a single function call:

.. code-block:: python

   import protoglue as pg
   result = pg.run_pipeline(data_dir="data/human_lymph_node", output_dir="results/hln")
