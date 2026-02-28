Installation
============

Requirements
------------

nichefinder requires Python >= 3.5 and < 3.12.

Optional: create a dedicated conda environment first::

    mamba create -n nichefinder "python<3.12"
    mamba activate nichefinder

Install from GitHub
-------------------

::

    pip install git+ssh://git@github.com/Teichlab/nichefinder.git

Or via HTTPS::

    pip install git+https://github.com/Teichlab/nichefinder.git

Install for Development
-----------------------

Clone the repository and install in editable mode::

    git clone https://github.com/Teichlab/nichefinder.git
    cd nichefinder
    pip install -e ".[dev]"

Dependencies
------------

The following packages are installed automatically:

- `numpy <https://numpy.org>`_
- `scanpy <https://scanpy.readthedocs.io>`_
- `scipy <https://scipy.org>`_
- `scikit-learn <https://scikit-learn.org>`_
- `matplotlib <https://matplotlib.org>`_
- `pandas <https://pandas.pydata.org>`_
- `joblib <https://joblib.readthedocs.io>`_
- `networkx <https://networkx.org>`_
- `numpy-groupies <https://github.com/ml31415/numpy-groupies>`_
