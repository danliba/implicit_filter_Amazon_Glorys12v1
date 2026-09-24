#!/bin/bash
# usage: sbatch run_filter.sh CONFIG      (CONFIG in W1.5 W2.0 W2.5 R2Ld R3Ld)
#SBATCH --job-name=impfilt
#SBATCH --partition=gpu
#SBATCH --account=bk1450_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bk1450/b383184/Amazon/Mercator/implicit_run/logs/run_%x_%j.out
export CUDA_PATH=/work/bk1450/b383184/conda/envs/implicit_filter/targets/x86_64-linux
export OMP_NUM_THREADS=16
nvidia-smi --query-gpu=name,memory.total --format=csv
/work/bk1450/b383184/conda/envs/implicit_filter/bin/python -u /work/bk1450/b383184/Amazon/Mercator/implicit_run/run_filter.py "$@"
