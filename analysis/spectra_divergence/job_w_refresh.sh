#!/bin/bash
#SBATCH --job-name=specdiv_w
#SBATCH --partition=shared
#SBATCH --account=bk1450
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/spectra_divergence/job_w_refresh_%j.log
# Refresh W-dependent parts after the final W rerun (T-cell operator consistent with the vector filter)
cd /work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/spectra_divergence
PY=/work/bk1450/b383184/conda/envs/implicit_filter/bin/python
export PYTHONWARNINGS=ignore
( $PY spectra.py hann W > log_spectra_W_hann.txt 2>&1; $PY spectra.py tukey W > log_spectra_W_tukey.txt 2>&1 ) &
( $PY residual.py > log_residual.txt 2>&1; $PY residual_extra.py > log_residual_extra.txt 2>&1 ) &
$PY w_stats.py > log_w_stats.txt 2>&1 &
wait
$PY make_metrics.py > log_make_metrics.txt 2>&1
echo ALLDONE
