"""
Pass 1 (heavy, run via SLURM): monthly statistics of U, V and W (v3 outputs) for one configuration on the whole
model domain and all 50 levels, streamed one day at a time.

usage: python accumulate.py CFG      (CFG = ORIG, W1.5, W2.0, W2.5, R2Ld, R3Ld)

Writes cache/stats_<CFG>.npz with float32 (nz, ny, nx) arrays
  umean, vmean      monthly mean
  u2mean, v2mean    monthly mean of the square (-> KE, EKE = u2mean - umean^2)
  du2mean, dv2mean  monthly mean of (filtered - ORIG)^2 of daily fields (0 for ORIG)
  wmean, w2mean, dw2mean  same for W (W levels; ORIG = rigid-lid W_1993-01fc.nc)
and diagnostics
  nan_mismatch_U/V  (nz,) number of (t,j,i) where isnan(filtered) != isnan(ORIG)
  maxp_*            maximum-principle check (not guaranteed for the v3 vector operator; guaranteed for scalar W): per level/day the largest exceedance of the filtered
                    field over the [min, max] of the ORIG field in the same connected wet region
                    (4-connected, i.e. the filter stencil), in m/s.
"""
import sys, os, time
import numpy as np
import netCDF4
from scipy import ndimage

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
from common import path, VNAME  # noqa: E402

cfg = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/cache", exist_ok=True)
MAXP_LEVELS = [0, 10, 20, 25, 30, 34]      # 0.5, 16, 78, 186(ish), 454, 902 m
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
import cgrid_filter as cf  # noqa: E402
TMASK = cf.CMesh().tmask   # W is filtered only where tmask (floor W points keep 0)

out = {}
diag = {}
for var in ["U", "V", "W"]:
    fo = netCDF4.Dataset(path(var, "ORIG"))[VNAME[var]]
    ff = netCDF4.Dataset(path(var, cfg))[VNAME[var]] if cfg != "ORIG" else None
    nt, nz, ny, nx = fo.shape
    s1 = np.zeros((nz, ny, nx)); s2 = np.zeros((nz, ny, nx)); sd = np.zeros((nz, ny, nx))
    mism = np.zeros(nz, dtype=np.int64)
    maxp = np.zeros((nt, len(MAXP_LEVELS))); maxpf = np.zeros((nt, len(MAXP_LEVELS)))
    labels = {}
    t0 = time.time()
    for t in range(nt):
        o = fo[t].filled(np.nan).astype(np.float64)
        f = ff[t].filled(np.nan).astype(np.float64) if ff is not None else o
        mism += (np.isnan(o) != np.isnan(f)).reshape(nz, -1).sum(1)
        f0 = np.nan_to_num(f)
        s1 += f0; s2 += f0 ** 2
        if ff is not None:
            sd += np.nan_to_num(f - o) ** 2
            for n, k in enumerate(MAXP_LEVELS):
                wet = ~np.isnan(o[k])
                if var == "W":
                    wet &= TMASK[k]
                if k == 0 and var == "W":
                    continue
                if k not in labels:
                    labels[k] = ndimage.label(wet)
                lab, nl = labels[k]
                ids = np.arange(1, nl + 1)
                omin = np.array(ndimage.minimum(np.where(wet, o[k], 0), lab, ids))
                omax = np.array(ndimage.maximum(np.where(wet, o[k], 0), lab, ids))
                fk = np.where(wet, f[k], 0)
                lo = np.zeros(nl + 1); hi = np.zeros(nl + 1)
                lo[1:] = omin; hi[1:] = omax
                exc = np.maximum(fk - hi[lab], lo[lab] - fk)
                maxp[t, n] = exc[wet].max()
                # fraction of wet points exceeding the regional range by > 1 mm/s (U,V) or > 1e-6 m/s (W)
                maxpf[t, n] = (exc[wet] > (1e-6 if var == "W" else 1e-3)).mean()
        print(f"[{cfg}] {var} t={t} {time.time()-t0:.0f}s", flush=True)
    wet = ~np.isnan(o)
    for a in (s1, s2, sd):
        a /= nt
        a[~wet] = np.nan
    lv = var.lower()
    out[f"{lv}mean"] = s1.astype(np.float32)
    out[f"{lv}2mean"] = s2.astype(np.float32)
    out[f"d{lv}2mean"] = sd.astype(np.float32)
    out[f"nan_mismatch_{var}"] = mism
    out[f"maxp_{var}"] = maxp
    out[f"maxpfrac_{var}"] = maxpf
out["maxp_levels"] = np.array(MAXP_LEVELS)
np.savez(f"{HERE}/cache/tmp_stats_{cfg}.npz", **out)
os.replace(f"{HERE}/cache/tmp_stats_{cfg}.npz", f"{HERE}/cache/stats_{cfg}.npz")   # atomic: file appears when complete
print("done", cfg)
