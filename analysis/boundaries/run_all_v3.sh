#!/bin/bash
# Driver for the v3 (fixed) boundary analysis: waits for the rerun, accumulates statistics via SLURM,
# then runs all analysis scripts. ORIG statistics (cache/stats_ORIG.npz) do not depend on the filter.
cd /work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/boundaries
P=/work/bk1450/b383184/conda/envs/implicit_filter/bin/python
until [ -f ../../output/V3_FIXED ] && [ -f ../../output/R3Ld/diag_R3Ld.json ]; do sleep 30; done
echo "outputs ready $(date)"
[ -f cache/stats_ORIG.npz ] || sbatch -J acc_ORIG accumulate.sh ORIG
for c in W1.5 W2.0 W2.5 R2Ld R3Ld; do rm -f cache/stats_$c.npz; sbatch -J acc_$c accumulate.sh $c; done
OMP_NUM_THREADS=8 $P validate_cpu.py > logs/validate.log 2>&1
OMP_NUM_THREADS=8 $P edge_test.py > logs/edge_test.log 2>&1
$P passages.py > logs/passages.log 2>&1
$P mask_check.py > logs/mask.log 2>&1
until [ -f cache/stats_R3Ld.npz ] && [ -f cache/stats_R2Ld.npz ] && [ -f cache/stats_W2.5.npz ] && [ -f cache/stats_W2.0.npz ] && [ -f cache/stats_W1.5.npz ]; do sleep 20; done
sleep 30
$P boundary_stats.py > logs/bstats.log 2>&1
$P maps.py > logs/maps.log 2>&1
$P assemble_metrics.py > logs/assemble.log 2>&1
echo "ALL DONE $(date)"
