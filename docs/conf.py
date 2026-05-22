# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ProtoGlue"
copyright = "2026, Xuan Guo"
author = "Xuan Guo"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
autodoc_mock_imports = ["torch"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# nbsphinx settings
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = None
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}
