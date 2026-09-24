"""Task 3 add-on: RMS of top-down w (w(k)-w(0)) and of vovecrtz_fixed at 100/500/1000 m over 70-30W 5S-30N,
split by distance to land AT THAT LEVEL (<=3, 4-15, >15 cells; the >15 band also excludes lat>29N, the unhaloed model edge),
plus median |w| (robust to wall/edge spikes), days 1-3, ORIG vs configs. Output: metrics_w_split.json"""
import json, sys
import numpy as np, netCDF4
from scipy import ndimage
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C
m = C.mesh()
js, is_ = C.box_slices(-70, -30, -5, 30)
kw = [int(np.argmin(abs(m.gdepw_0 - z))) for z in (100, 500, 1000)]
res = {"levels_m": [float(m.gdepw_0[k]) for k in kw]}
bands = {"le3": (0, 3), "4_15": (3, 15), "gt15": (15, 1e9)}
north = m.gphit[js, is_] > 29.0
dist = {k: ndimage.distance_transform_edt(m.tmask[k])[js, is_] for k in kw}
for cfg in ["ORIG"] + C.available_configs():
    ds = netCDF4.Dataset(C.path("W", cfg))
    w0 = ds["vovecrtz"][0:3, 0, js, is_].filled(np.nan).astype(float)
    r = {}
    for n, k in enumerate(kw):
        a = ds["vovecrtz"][0:3, k, js, is_].filled(np.nan).astype(float)
        af = ds["vovecrtz_fixed"][0:3, k, js, is_].filled(np.nan).astype(float) if cfg != "ORIG" else a
        mk = m.tmask[k, js, is_]
        for b, (lo, hi) in bands.items():
            s = mk & (dist[k] > lo) & (dist[k] <= hi)
            if b == "gt15":
                s &= ~north
            r.setdefault(f"top_down_{b}", []).append(float(np.sqrt(np.nanmean((a - w0)[:, s] ** 2))))
            r.setdefault(f"fixed_{b}", []).append(float(np.sqrt(np.nanmean(af[:, s] ** 2))))
            r.setdefault(f"top_down_median_abs_{b}", []).append(float(np.nanmedian(np.abs(a - w0)[:, s])))
            r.setdefault(f"fixed_median_abs_{b}", []).append(float(np.nanmedian(np.abs(af)[:, s])))
            r.setdefault(f"frac_points_{b}", []).append(float(s.sum() / mk.sum()))
    s = mk & north
    r["top_down_rms_north_edge_gt29N"] = [float(np.sqrt(np.nanmean(((ds["vovecrtz"][0:3, k, js, is_].filled(np.nan) - w0)[:, m.tmask[k, js, is_] & north]) ** 2))) for k in kw]
    res[cfg] = r
    print(cfg, {k: ["%.1e" % x for x in v] for k, v in r.items() if not k.startswith("frac")}, flush=True)
json.dump(res, open("metrics_w_split.json", "w"), indent=1)
