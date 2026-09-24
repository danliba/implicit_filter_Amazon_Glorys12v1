#!/bin/bash
# Runs the complete NBC / energy analysis (login node is sufficient: ~6 min, <16 GB).
set -e
PY=/work/bk1450/b383184/conda/envs/implicit_filter/bin/python
cd /work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis/nbc_energy
$PY nbc_sections.py      > log_sections.txt 2>&1
NPROC=8 $PY energy.py    > log_energy.txt 2>&1
$PY energy_analysis.py   > log_energy_analysis.txt 2>&1
$PY large_scale.py       > log_large_scale.txt 2>&1
$PY w_section.py         > log_wsection.txt 2>&1
$PY v1_v3_compare.py     > log_v1_v3.txt 2>&1
$PY make_metrics.py      > log_metrics.txt 2>&1
