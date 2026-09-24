"""Check that the CPU operators used in edge_test.py reproduce the production (GPU) v3 output on the full
domain for 1 Jan 1993: U,V vector filter at the surface and the W scalar filter at the W level nearest 100 m,
for W2.0 and R2Ld. Writes cache/metrics_validate.json."""
import sys, json, os
import numpy as np
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
from common import mesh, read, available_configs
import cgrid_filter as cf
import run_filter as rf
HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh(); res = {}
KW = int(np.argmin(np.abs(m.gdepw_0 - 100)))
for cfg in [c for c in ["W2.0", "R2Ld"] if c in available_configs()]:
    lT, lF = rf.ell_fields(m, cfg)
    u = read("U", "ORIG", t=0, k=0); v = read("V", "ORIG", t=0, k=0)
    uc, vc, its, n = cf.filter_level_vector(m, 0, u[None], v[None], lT, lF, backend="cpu", tol=1e-10)
    ug = read("U", cfg, t=0, k=0); vg = read("V", cfg, t=0, k=0)
    w = read("W", "ORIG", t=0, k=KW)
    wet = m.tmask[KW] & np.isfinite(w)
    wc = cf.filter_level(m, "T", np.where(wet, w, np.nan)[None], lT, backend="cpu", tol=1e-10)[0][0]
    wg = read("W", cfg, t=0, k=KW)
    res[cfg] = dict(max_abs_diff_U=float(np.nanmax(np.abs(uc[0] - ug))), max_abs_diff_V=float(np.nanmax(np.abs(vc[0] - vg))),
                    rms_U=float(np.sqrt(np.nanmean(ug ** 2))),
                    max_abs_diff_W=float(np.nanmax(np.abs(np.where(wet, wc, np.nan) - np.where(wet, wg, np.nan)))),
                    rms_W=float(np.sqrt(np.nanmean(wg[wet] ** 2))),
                    nan_identical_U=bool((np.isnan(ug) == np.isnan(u)).all()))
    print(cfg, res[cfg], flush=True)
json.dump(res, open(f"{HERE}/cache/metrics_validate.json", "w"), indent=1)
