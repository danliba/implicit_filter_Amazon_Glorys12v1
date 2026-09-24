"""Verify that the NaN mask of the ORIGINAL GLORYS U/V equals the C-grid velocity mask derived from mbathy
(umask = tmask(j,i) & tmask(j,i+1), vmask = tmask(j,i) & tmask(j+1,i)) on all 50 levels (day 1), i.e. that
velocity points on a land face (zero normal flow) are NaN and therefore excluded from the filter.
The last column (U) / last row (V) have no i+1 / j+1 neighbour in the regional file and are reported separately.
Also checks that filtered files have the identical NaN mask (day 1 and day 31, all levels).
Output: cache/metrics_mask.json"""
import sys, os, json
import numpy as np, netCDF4
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
from common import mesh, path, VNAME, available_configs
HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh(); tm = m.tmask
um = np.zeros_like(tm); um[:, :, :-1] = tm[:, :, :-1] & tm[:, :, 1:]
vm = np.zeros_like(tm); vm[:, :-1] = tm[:, :-1] & tm[:, 1:]
res = {}
for g, msk in [("U", um), ("V", vm)]:
    o = netCDF4.Dataset(path(g, "ORIG"))[VNAME[g]][0].filled(np.nan)
    wet = np.isfinite(o)
    inner = np.ones_like(wet)
    if g == "U":
        inner[:, :, -1] = False
    else:
        inner[:, -1, :] = False
    res[f"ORIG_{g}"] = dict(wet_but_not_cgrid_mask_interior=int((wet & ~msk & inner).sum()),
                           cgrid_mask_but_nan_interior=int((~wet & msk & inner).sum()),
                           finite_in_last_col_or_row=int((wet & ~inner).sum()),
                           zeros_in_wet=int((o[wet] == 0).sum()))
    for c in available_configs():
        ds = netCDF4.Dataset(path(g, c))[VNAME[g]]
        mism = 0
        for t in (0, ds.shape[0] - 1):
            mism += int((np.isfinite(ds[t].filled(np.nan)) != wet).sum())
        res[f"{c}_{g}_nan_mismatch_days1_31"] = mism
    print(res, flush=True)
json.dump(res, open(f"{HERE}/cache/metrics_mask.json", "w"), indent=1)
