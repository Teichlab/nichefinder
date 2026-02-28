# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to sys.path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "nichefinder"
copyright = "2024, Teichlab"
author = "J. Patrick Pett"
release = "0.1.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Generate API docs from docstrings
    "sphinx.ext.napoleon",      # Support NumPy and Google style docstrings
    "sphinx.ext.viewcode",      # Add links to source code
    "sphinx.ext.autosummary",   # Auto-generate summary tables
    "sphinx.ext.intersphinx",   # Link to other projects' docs
    "myst_nb",                  # Include Jupyter notebooks + Markdown (no Pandoc needed)
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Suffixes for source files (myst_nb handles .ipynb and .md)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Napoleon settings ---------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings ----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True

# -- Intersphinx mapping -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# -- myst-nb settings ---------------------------------------------------------
nb_execution_mode = "off"    # Do not re-execute notebooks on build
nb_execution_allow_errors = False

# -- Options for HTML output ---------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
}
