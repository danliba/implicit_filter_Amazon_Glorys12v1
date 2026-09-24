#!/bin/bash
set -e
export MAMBA_ROOT_PREFIX=/work/bk1450/b383184/conda
export CONDA_PKGS_DIRS=/work/bk1450/b383184/conda/pkgs
ENV=/work/bk1450/b383184/conda/envs/implicit_filter
micromamba create -y -p $ENV -c conda-forge python=3.11 numpy scipy xarray netcdf4 dask matplotlib cartopy cmocean pandas scikit-learn gsw ipykernel nbformat nbconvert papermill pip "cupy>=13" "cuda-version=12.4" cuda-cudart libcublas libcusparse libcusolver libcurand cuda-nvrtc libcufft cuda-cudart-dev cuda-nvrtc-dev libcusparse-dev libcublas-dev cuda-cccl jax
$ENV/bin/python -m pip install -e /work/bk1450/b383184/Amazon/Mercator/implicit_filter
$ENV/bin/python -m ipykernel install --user --name implicit_filter --display-name implicit_filter
$ENV/bin/python -c "import implicit_filter, cupy, jax; print('ENV OK', implicit_filter.__file__, cupy.__version__)"
