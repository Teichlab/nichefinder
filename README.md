# nichefinder

A collection of functions for working with ISS data, including label transfer from a suspension reference and niche analysis.

## System requirements

<details>
<summary><b>show requirements</b></summary>

### Hardware requirements

`nichefinder` can run on a standard computer with enough RAM to hold the used datasets in memory.

### Software requirements

**OS requirements**

The package has been tested on:

- macOS Tahoe 26.3 (25D125)
- Linux: Ubuntu 22.04 jammy

**Python requirements**

A python version `>=3.5` and `<3.12` is required for all dependencies to work. 
Various python libraries are used, listed in `setup.py`, including the python scientific stack, `scikit-learn`, `joblib`, `scanpy` and `networkx`.
`nichefinder` and all dependencies can be installed via `pip` (see below).

</details>

## Installation

Optional: create a new conda environment with Python

```bash
mamba create -n nichefinder "python<3.12"
mamba activate nichefinder
```

### Install from GitHub

To download via ssh, an ssh key must have be [added to your GitHub profile](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

```bash
pip install git+ssh://git@github.com/Teichlab/nichefinder.git
```

Alternatively, clone the repository and install with pip in two steps:

```bash
git clone <preferred way ...>
cd nichefinder
pip nichefinder .
```

*(installation time: <5 min)*

## Usage

Example notebooks:
- [toy example](https://github.com/Teichlab/nichefinder/blob/main/notebooks/example_simulated_data.ipynb)
- [score plotting](https://github.com/Teichlab/nichefinder/blob/main/notebooks/niche_score_mapping.ipynb)

(*running time: <5min*)

## Citation

*TBA*
