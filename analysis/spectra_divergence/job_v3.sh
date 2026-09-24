#!/bin/bash
#SBATCH --job-name=specdiv_v3
#SBATCH --partition=shared
#SBATCH --account=bk1450
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/spectra_divergence/job_v3_%j.log
# v3 analysis: spectra (UV, W; Hann and Tukey), continuity residual, W statistics, theory figure
cd /work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/spectra_divergence
test -f ../../output/V3_FIXED || { echo "V3_FIXED missing"; exit 1; }
PY=/work/bk1450/b383184/conda/envs/implicit_filter/bin/python
export PYTHONWARNINGS=ignore
( $PY spectra.py hann UV > log_spectra_UV_hann.txt 2>&1; $PY spectra.py tukey UV > log_spectra_UV_tukey.txt 2>&1 ) &
( $PY spectra.py hann W > log_spectra_W_hann.txt 2>&1; $PY spectra.py tukey W > log_spectra_W_tukey.txt 2>&1 ) &
$PY residual.py > log_residual.txt 2>&1 &
$PY w_stats.py > log_w_stats.txt 2>&1 &
$PY theory_validation.py > log_theory.txt 2>&1 &
wait
echo ALLDONE
