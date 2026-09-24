#!/bin/bash
#SBATCH --job-name=gpuprobe
#SBATCH --partition=gpu
#SBATCH --account=bk1450_gpu
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --output=logs/probe_%j.out
nvidia-smi
