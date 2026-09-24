## 8. Code and reproduction
See `implicit_run/README.md` for the layout and exact commands. In brief:
- **`cgrid_filter.py`:** filter library (operators, GPU solver, continuity).
- **`run_filter.py` / `run_filter.sh`:** production, one configuration per GPU job.
- **Validation:**
  - `tests/validate_filter.py`: synthetic transfer function; cross-check with the `implicit_filter` package solver (2×10⁻⁸); symmetry and conservation.
  - `tests/validate_vector.py`: vector-filter commutation with divergence (5×10⁻⁵).
  - `tests/test_e3_independence.py`: response independent of e3 for rotational and divergent waves.
- **Analysis:** `analysis/{nbc_energy,spectra_divergence,boundaries}/`, each with scripts, `metrics.json` and figures.
- **Rossby radius:** `rossby/rossby_radius.py`.
- **Notebook:** `implicit_data/v0.Implicit_filtering_ARP.ipynb`, the interactive version of every step. It was executed end-to-end on an A100 with papermill; the executed copy is `implicit_run/scratch/nb_executed.ipynb`.
