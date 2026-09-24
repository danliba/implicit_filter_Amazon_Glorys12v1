#!/bin/bash
#SBATCH --job-name=impf_val
#SBATCH --partition=gpu
#SBATCH --account=bk1450_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/bk1450/b383184/Amazon/Mercator/implicit_run/logs/val_%j.out
export CUDA_PATH=/work/bk1450/b383184/conda/envs/implicit_filter/targets/x86_64-linux
nvidia-smi --query-gpu=name,memory.total --format=csv
/work/bk1450/b383184/conda/envs/implicit_filter/bin/python -u /work/bk1450/b383184/Amazon/Mercator/implicit_run/tests/validate_filter.py
