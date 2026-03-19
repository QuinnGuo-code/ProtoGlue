Installation
============

Requirements
------------

- Python >= 3.8
- PyTorch >= 1.12.0 (GPU support strongly recommended)
- CUDA toolkit (for GPU acceleration)

Install from GitHub
-------------------

The recommended way to install ProtoGlue:

.. code-block:: bash

   pip install git+https://github.com/QuinnGuo-code/ProtoGlue.git

Install from source (development)
----------------------------------

For development or to modify ProtoGlue:

.. code-block:: bash

   git clone https://github.com/QuinnGuo-code/ProtoGlue.git
   cd ProtoGlue
   pip install -e .

Optional dependencies
---------------------

For PowerPoint report generation:

.. code-block:: bash

   pip install "protoglue[pptx]"

For building documentation locally:

.. code-block:: bash

   pip install "protoglue[docs]"

Verify installation
-------------------

.. code-block:: python

   import protoglue as pg
   print(pg.__version__)

   # Check GPU availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")

PyTorch installation
--------------------

If you need to install PyTorch with CUDA support, visit
`https://pytorch.org/get-started/locally/ <https://pytorch.org/get-started/locally/>`_
for platform-specific instructions. For example:

.. code-block:: bash

   # CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # CPU only
   pip install torch --index-url https://download.pytorch.org/whl/cpu
