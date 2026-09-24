#!/bin/bash
#SBATCH --partition=shared --account=bk1450 --time=02:00:00 --mem=60G --cpus-per-task=4
#SBATCH --job-name=bnd_acc --output=/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/boundaries/logs/acc_%x_%j.out
# usage: sbatch accumulate.sh CFG
cd /work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/boundaries
/work/bk1450/b383184/conda/envs/implicit_filter/bin/python accumulate.py $1
