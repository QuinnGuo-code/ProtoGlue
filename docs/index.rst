Welcome to ProtoGlue's documentation!
======================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

**ProtoGlue** is a graph-based deep learning framework for spatial multi-omics
integration and spatial domain identification.

It integrates spatial multi-omics data (e.g., RNA + protein/ATAC) through
multi-scale spatial graph construction, asymmetric encode-decode architecture
with cross-reconstruction correspondence, and prototype-based deep embedded
clustering with spatial refinement.

Key Features
------------

- **Multi-scale spatial graphs** with density-adaptive neighbor selection
- **Cross-attention gated fusion** for inter-modality integration
- **Cross-reconstruction correspondence** for enforcing cross-modal consistency
- **Automatic K selection** via composite scoring
- **End-to-end pipeline** from raw data to spatial domains in a single function call

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Citation
--------

If you use ProtoGlue in your research, please cite::

   @misc{protoglue2026,
     title={ProtoGlue: a unified framework for spatial multi-omics
            integration and spatial domain identification},
     author={Xuan Guo, Xuewei Xian, Siyi Wu, Yanqin Wang, Lanyue Zhang and Wei Liu},
     year={2026},
     note={Manuscript under review}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
