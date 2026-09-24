"""Add-on to residual.py: (1) RMS R within 20 cells of each regional-grid edge (S, N, W, E), coast > 30 cells,
(2) open-ocean (edge>40, coast>30, step>30) RMS R in 2.5-degree latitude bands together with the band-median |grad l_T|,
to test whether the residual away from walls is tied to spatial variation of l, at k = 0 and 22, 31 days.
Output: metrics_residual_extra.json, fig_R_lat_gradl.png"""
import json, sys, types, time
import numpy as np, netCDF4
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C, cgrid_filter as cf
m = C.mesh(); ny, nx = m.ny, m.nx
exec(open("residual.py").read().split("# ------------------------------------------------------------------ (a) profiles")[0].split("m = C.mesh()")[1].replace('assert len(CFGS) == 5, f"missing configs: {CFGS}"', ""))
src = open("residual.py").read()
exec(src[src.index("def level_R"):src.index("cats = {}")])
LD = netCDF4.Dataset(C.RUN + "/rossby/rossby_radius_T.nc")["Ld"][:].filled(np.nan).astype(float)
jj, ii = np.mgrid[0:ny, 0:nx]
d_edge = np.minimum.reduce([jj, ny - 1 - jj, ii, nx - 1 - ii])
edges = {"S": jj, "N": ny - 1 - jj, "W": ii, "E": nx - 1 - ii}
lat = m.gphit
out = {}
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, k in zip(axs, [0, 22]):
    dc = ndimage.distance_transform_edt(m.tmask[k]); ds = ndimage.distance_transform_edt(~(m.tmask[k] & ~m.tmask[k + 1]))
    wet = m.tmask[k] & (d_edge >= 2)
    clean = wet & (d_edge > 40) & (dc > 30) & (ds > 30)
    lb = np.arange(-10, 30.1, 2.5)
    res = {}
    for cfg in ["ORIG", "W2.0", "R2Ld", "R3Ld"]:
        R = level_R(cfg, k); ms = np.nanmean(R ** 2, 0)
        r = {f"edge_{e}_le20_coast>30": float(np.sqrt(np.nanmean(ms[wet & (d <= 20) & (dc > 30) & (d_edge == d)]))) for e, d in edges.items()}
        band = [float(np.sqrt(np.nanmean(ms[clean & (lat >= a) & (lat < a + 2.5)]))) if (clean & (lat >= a) & (lat < a + 2.5)).sum() > 100 else None for a in lb[:-1]]
        r["clean_by_lat"] = dict(zip([f"{a:.1f}" for a in lb[:-1]], band))
        res[cfg] = r
        ax.semilogy(lb[:-1] + 1.25, [np.nan if b is None else b for b in band], "o-", color=C.COLORS[cfg], label=cfg)
    for cfg, mult in [("R2Ld", 2.0), ("R3Ld", 3.0)]:
        lT = cf.ell_T_F(m, "rossby", ld_T=LD, mult=mult)[0]
        g = np.hypot(np.gradient(lT, axis=1) / m.e1t, np.gradient(lT, axis=0) / m.e2t)
        res[cfg]["median_grad_l_by_lat"] = [float(np.median(g[clean & (lat >= a) & (lat < a + 2.5)])) if (clean & (lat >= a) & (lat < a + 2.5)).sum() > 100 else None for a in lb[:-1]]
    lW = cf.ell_T_F(m, "window", window_deg=2.0)[0]; gW = np.abs(np.gradient(lW, axis=0) / m.e2t)
    res["W2.0"]["median_grad_l_by_lat"] = [float(np.median(gW[clean & (lat >= a) & (lat < a + 2.5)])) if (clean & (lat >= a) & (lat < a + 2.5)).sum() > 100 else None for a in lb[:-1]]
    ax2 = ax.twinx()
    for cfg in ["W2.0", "R2Ld", "R3Ld"]:
        ax2.plot(lb[:-1] + 1.25, [np.nan if b is None else b for b in res[cfg]["median_grad_l_by_lat"]], ":", color=C.COLORS[cfg])
    ax2.set_ylabel("median |∇ℓ| (dotted) [m/m]")
    ax.set_xlabel("latitude"); ax.set_ylabel("RMS R, open ocean (edge>40, coast>30, step>30) [s⁻¹]")
    ax.set_title(f"z = {m.gdept_0[k]:.0f} m"); ax.grid(alpha=0.3); ax.legend()
    out[f"k{k}"] = res
    print(k, json.dumps({c: {kk: v for kk, v in r.items() if kk.startswith("edge")} for c, r in res.items()}), flush=True)
fig.suptitle("Open-ocean continuity residual by latitude vs gradient of ℓ (31 days)", fontsize=11)
fig.savefig("fig_R_lat_gradl.png", dpi=130)
json.dump(out, open("metrics_residual_extra.json", "w"), indent=1)
