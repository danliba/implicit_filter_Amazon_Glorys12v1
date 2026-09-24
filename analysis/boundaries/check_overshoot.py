"""Locate points where the v3 vector-filtered U or V leaves the [min, max] range of the original component in
its connected wet region (the vector div-rot operator has no maximum principle). Day 1, levels 0 and 10,
W2.0 and R3Ld. Lists the largest exceedances with position, local mask geometry and values.
Writes cache/metrics_overshoot.json."""
import sys, os, json
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
from common import mesh, read
HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh(); res = {}
for cfg in ["W2.0", "R3Ld"]:
    for k in [0, 10, 20]:
        for g in ["U", "V"]:
            o = read(g, "ORIG", t=0, k=k); f = read(g, cfg, t=0, k=k)
            wet = np.isfinite(o)
            lab, nl = ndimage.label(wet)
            ids = np.arange(1, nl + 1)
            lo = np.r_[0, ndimage.minimum(np.where(wet, o, 0), lab, ids)]
            hi = np.r_[0, ndimage.maximum(np.where(wet, o, 0), lab, ids)]
            exc = np.where(wet, np.maximum(f - hi[lab], lo[lab] - f), -np.inf)
            size = np.r_[0, ndimage.sum(wet, lab, ids)][lab]
            # also local exceedance vs 5x5 neighbourhood range of the original
            omax5 = ndimage.maximum_filter(np.where(wet, o, -9), 5); omin5 = ndimage.minimum_filter(np.where(wet, o, 9), 5)
            exc5 = np.where(wet, np.maximum(f - omax5, omin5 - f), -np.inf)
            top = np.argsort(exc.ravel())[::-1][:8]
            lst = []
            for idx in top:
                j, i = np.unravel_index(idx, exc.shape)
                if exc[j, i] <= 0:
                    break
                lst.append(dict(j=int(j), i=int(i), lon=float(m.glamt[j, i]), lat=float(m.gphit[j, i]), exceed=float(exc[j, i]),
                                orig=float(o[j, i]), filt=float(f[j, i]), region_size=int(size[j, i]),
                                wet_T_in_3x3=int(m.tmask[k, max(j-1,0):j+2, max(i-1,0):i+2].sum())))
            res[f"{cfg}_{g}_k{k}"] = dict(n_exceed_1mm=int((exc > 1e-3).sum()), n_wet=int(wet.sum()),
                                          n_exceed_local5x5_1cm=int((exc5 > 1e-2).sum()),
                                          n_exceed_in_regions_lt_10pts=int(((exc > 1e-3) & (size < 10)).sum()), top=lst)
            print(cfg, g, k, res[f"{cfg}_{g}_k{k}"]["n_exceed_1mm"], res[f"{cfg}_{g}_k{k}"]["n_exceed_in_regions_lt_10pts"],
                  res[f"{cfg}_{g}_k{k}"]["n_exceed_local5x5_1cm"], lst[:3], flush=True)
json.dump(res, open(f"{HERE}/cache/metrics_overshoot.json", "w"), indent=1)
