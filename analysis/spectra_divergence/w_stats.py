"""
Task 3 (v3): W statistics. RMS W vs depth for ORIG (rigid-lid W product W_1993-01fc.nc) and each config
(filtered W), (i) whole domain interior wet points from diag_<CFG>.npz (w_rms_*), (ii) 70-30W 5S-30N over
all 31 days at every W level, split into open ocean (>15 cells from land at that level, <29N) and all wet.
Also checks: W exactly 0 at the surface, W at the sea-floor W point, NaN pattern.
Outputs: fig_w_profiles.png, metrics_w.json
"""
import json, sys
import numpy as np
import netCDF4
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C

OUT = C.RUN + "/analysis/spectra_divergence"
m = C.mesh()
CFGS = C.available_configs()
assert len(CFGS) == 5, f"missing configs: {CFGS}"
ALL = ["ORIG"] + CFGS
js, is_ = C.box_slices(-70, -30, -5, 30)
lat = m.gphit[js, is_]
zw = m.gdepw_0
met = {"depthw": zw.tolist(), "region": {}, "whole_domain_npz": {}, "checks": {}}
open_mask = [(ndimage.distance_transform_edt(m.tmask[k])[js, is_] > 15) & (lat < 29) & m.tmask[k, js, is_] for k in range(m.nz)]
for cfg in ALL:
    ds = netCDF4.Dataset(C.path("W", cfg))
    var = ds["vovecrtz"]
    r_all, r_open = [], []
    for k in range(m.nz):
        mk = m.tmask[k, js, is_]
        if mk.sum() < 100:
            r_all.append(np.nan); r_open.append(np.nan); continue
        a = var[:, k, js, is_].filled(np.nan).astype(np.float64)
        r_all.append(float(np.sqrt(np.nanmean(a[:, mk] ** 2))))
        r_open.append(float(np.sqrt(np.nanmean(a[:, open_mask[k]] ** 2))) if open_mask[k].sum() > 100 else np.nan)
        if k == 0:
            met["checks"].setdefault(cfg, {})["surface_max_abs"] = float(np.nanmax(np.abs(a)))
    met["region"][cfg] = dict(all_wet=r_all, open_ocean=r_open)
    # sea-floor W point (k = mbathy) of day 1 over the whole domain
    w1 = var[0].filled(np.nan)
    kb = m.mbathy
    jj, ii = np.nonzero((kb > 0) & (kb < m.nz))
    fl = w1[kb[jj, ii], jj, ii]
    met["checks"][cfg].update(floor_point_max_abs=float(np.nanmax(np.abs(fl))), floor_point_nan_frac=float(np.mean(np.isnan(fl))),
                              wet_nan_count=int(np.isnan(w1[m.tmask]).sum()))
    print(cfg, "100/500/1000 m:", [f"{r_all[k]:.2e}" for k in (22, 31, 35)], met["checks"][cfg], flush=True)
for c in CFGS:
    d = np.load(f"{C.RUN}/output/{c}/diag_{c}.npz")
    met["whole_domain_npz"][c] = d["w_rms_filt"].mean(0).tolist()
    met["whole_domain_npz"]["ORIG"] = d["w_rms_orig"].mean(0).tolist()
kw = [int(np.argmin(abs(zw - zz))) for zz in (100, 500, 1000)]
met["rms_100_500_1000m_region"] = {c: {"levels_m": [float(zw[k]) for k in kw],
                                       "all_wet": [met["region"][c]["all_wet"][k] for k in kw],
                                       "open_ocean": [met["region"][c]["open_ocean"][k] for k in kw],
                                       "ratio_to_ORIG_all": [met["region"][c]["all_wet"][k] / met["region"]["ORIG"]["all_wet"][k] for k in kw]}
                                   for c in ALL}
json.dump(met, open(f"{OUT}/metrics_w.json", "w"), indent=1)

fig, axs = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True, constrained_layout=True)
for cfg in ALL:
    kw_ = dict(color=C.COLORS[cfg], lw=2.2 if cfg == "ORIG" else 1.5, label=cfg)
    axs[0].semilogx(np.array(met["region"][cfg]["all_wet"]) * 86400, zw, **kw_)
    axs[1].semilogx(np.array(met["region"][cfg]["open_ocean"]) * 86400, zw, **kw_)
    axs[2].semilogx(np.array(met["whole_domain_npz"][cfg]) * 86400, zw, **kw_)
for ax, t in zip(axs, ["70–30°W 5°S–30°N, all wet", "70–30°W 5°S–30°N, open ocean (>15 cells from land, <29°N)",
                       "whole domain interior (diag npz)"]):
    ax.set_title(t, fontsize=10); ax.grid(which="both", alpha=0.3); ax.set_xlabel("RMS w [m/day]")
axs[0].set_ylim(5500, 0); axs[0].set_ylabel("depth [m]"); axs[0].legend()
fig.suptitle("RMS vertical velocity vs depth (31 days): rigid-lid GLORYS W vs scalar-filtered W", fontsize=12)
fig.savefig(f"{OUT}/fig_w_profiles.png", dpi=130)
print(json.dumps(met["rms_100_500_1000m_region"], indent=0))
