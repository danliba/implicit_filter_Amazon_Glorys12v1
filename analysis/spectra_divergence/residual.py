"""
Task 2 (v3): continuity residual R = dw/dz + hdiv(u,v)  [1/s]  (cgrid_filter.continuity_residual)
of the filtered U,V (vector div-rot filter) and W (scalar filter of the rigid-lid W product).

(a) Whole-domain profiles (diag_<CFG>.npz): time-mean RMS R and RMS hdiv, filtered and ORIG.
(b) Surface-layer R maps over 70-30W, 5S-30N: 31-day mean (Rmean_*[0]) and day 1 (Rsurf_*[0]).
(c) Decomposition over the FULL month and all configs at k = 0 (0.5 m, npz daily maps), k = 22 (109.7 m)
    and k = 33 (763 m, computed here from the output files):
      d_coast = distance [cells] to the nearest land T cell at level k
      d_step  = distance to the nearest bottom-step cell (tmask[k] & ~tmask[k+1], i.e. the sea floor is at
                the bottom of this cell)
      d_edge  = distance to the regional-grid edge (min(j, ny-1-j, i, nx-1-i)); cells with d_edge < 2 are
                excluded everywhere (the divergence operator sets row/col 0 to zero)
    RMS over time and space in categories, and RMS vs distance curves with the other two distances large.
(d) Amazon mouth (53-45W, 2S-5N): maps of the 31-day mean surface-layer R, and the box- and depth-integrated
    source Q = sum(R e3t e1t e2t) [m3/s], daily, for ORIG and each config. Also R of the original GLORYS W
    (data/variables/W_1993-01.nc, free-surface W) for comparison.
(e) v1 (component-wise), v2 (vector + barotropic correction), v3: RMS hdiv at the surface on day 1 for W2.0,
    by distance to the coast, 70-30W 5S-30N and whole domain.
Outputs: fig_R_profiles.png, fig_R_map_mean.png, fig_R_map_day1.png, fig_R_distance.png, fig_R_amazon.png,
         fig_hdiv_versions.png, metrics_residual.json
"""
import json, sys, types, time
import numpy as np
import netCDF4
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C
import cgrid_filter as cf

OUT = C.RUN + "/analysis/spectra_divergence"
m = C.mesh()
CFGS = C.available_configs()
assert len(CFGS) == 5, f"missing configs: {CFGS}"
ALL = ["ORIG"] + CFGS
ny, nx = m.ny, m.nx
z = m.gdept_0
met = {"configs": CFGS}
GLORYS_W = "/work/bk1450/b383184/Amazon/Mercator/data/variables/W_1993-01.nc"
NPZ = {c: np.load(f"{C.RUN}/output/{c}/diag_{c}.npz") for c in CFGS}
t00 = time.time()

# ------------------------------------------------------------------ (a) profiles
fig, axs = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True, constrained_layout=True)
d0 = NPZ[CFGS[0]]
wet = d0["hdiv_rms_orig"].mean(0) > 0
axs[0].semilogx(d0["R_rms_orig"].mean(0)[wet], z[wet], "k-", lw=2.2, label="ORIG R")
axs[0].semilogx(d0["hdiv_rms_orig"].mean(0)[wet], z[wet], "k--", lw=2.2, label="ORIG hdiv")
prof = {"depth": z[wet].tolist(), "ORIG": dict(R=d0["R_rms_orig"].mean(0)[wet].tolist(), hdiv=d0["hdiv_rms_orig"].mean(0)[wet].tolist(),
                                              R_max=d0["R_max_orig"].mean(0)[wet].tolist())}
for c in CFGS:
    d = NPZ[c]
    rf, hf = d["R_rms_filt"].mean(0), d["hdiv_rms_filt"].mean(0)
    axs[0].semilogx(rf[wet], z[wet], "-", color=C.COLORS[c], lw=1.5, label=f"{c} R")
    axs[0].semilogx(hf[wet], z[wet], "--", color=C.COLORS[c], lw=1.2, label=f"{c} hdiv")
    axs[1].semilogx((rf / d["hdiv_rms_filt"].mean(0))[wet], z[wet], "-", color=C.COLORS[c], label=c)
    prof[c] = dict(R=rf[wet].tolist(), hdiv=hf[wet].tolist(), R_max=d["R_max_filt"].mean(0)[wet].tolist(),
                   R_over_hdiv_k0=float(rf[0] / hf[0]), R_over_hdiv_k22=float(rf[22] / hf[22]),
                   R_filt_over_R_orig_k0=float(rf[0] / d["R_rms_orig"].mean(0)[0]),
                   R_filt_over_R_orig_k22=float(rf[22] / d["R_rms_orig"].mean(0)[22]))
axs[1].semilogx((d0["R_rms_orig"].mean(0) / d0["hdiv_rms_orig"].mean(0))[wet], z[wet], "k-", lw=2, label="ORIG")
for ax in axs:
    ax.set_yscale("symlog", linthresh=100); ax.grid(which="both", alpha=0.3)
axs[0].set_ylim(6000, 0); axs[0].set_ylabel("depth [m]")
axs[0].set_xlabel("RMS [s⁻¹] (solid: residual R, dashed: hdiv)"); axs[1].set_xlabel("RMS R / RMS hdiv")
axs[0].legend(fontsize=7, ncol=2)
fig.suptitle("Continuity residual vs horizontal divergence, whole domain interior wet points, 31-day mean of daily RMS", fontsize=11)
fig.savefig(f"{OUT}/fig_R_profiles.png", dpi=130)
met["profiles_whole_domain"] = prof

# ------------------------------------------------------------------ (b) surface-layer maps
js, is_ = C.box_slices(-70, -30, -5, 30)
lon, lat = m.glamt[js, is_], m.gphit[js, is_]
for key, idx, fname, ttl in [("Rmean", 0, "fig_R_map_mean.png", "31-day mean"), ("Rsurf", 0, "fig_R_map_day1.png", "1 Jan 1993")]:
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    for ax, c in zip(axs.ravel(), ALL):
        d = NPZ[CFGS[0]] if c == "ORIG" else NPZ[c]
        a = d[f"{key}_{'orig' if c == 'ORIG' else 'filt'}"][idx][js, is_].astype(float)
        a = np.where(m.tmask[0, js, is_], a, np.nan)
        a[-2:, :] = np.nan   # last two rows = northern regional-grid edge (row ny-1 has an undefined divergence)
        pc = ax.pcolormesh(lon, lat, a, norm=SymLogNorm(1e-10, vmin=-1e-6, vmax=1e-6, base=10), cmap="RdBu_r", shading="auto")
        ax.set_facecolor("0.75"); ax.set_aspect(1 / np.cos(np.deg2rad(12)))
        ax.set_title(f"{c}: RMS {np.sqrt(np.nanmean(a ** 2)):.2e} s⁻¹ (excl. last 2 rows)", fontsize=10)
    fig.colorbar(pc, ax=axs, shrink=0.6, label="R = ∂w/∂z + ∇h·u  [s⁻¹] (symlog, linear below 1e-10)")
    fig.suptitle(f"Continuity residual, surface layer (0–1 m), {ttl}", fontsize=12)
    fig.savefig(f"{OUT}/{fname}", dpi=130)

# ------------------------------------------------------------------ (c) decomposition
jj, ii = np.mgrid[0:ny, 0:nx]
d_edge = np.minimum.reduce([jj, ny - 1 - jj, ii, nx - 1 - ii])
LEVS = [0, 22, 33]
DIST = {}
for k in LEVS:
    dc = ndimage.distance_transform_edt(m.tmask[k])
    step = m.tmask[k] & ~m.tmask[k + 1]
    ds = ndimage.distance_transform_edt(~step)
    DIST[k] = (dc, ds)


def level_R(cfg, k):
    """(31, ny, nx) residual at level k for the full month."""
    if k == 0:
        d = NPZ[CFGS[0]] if cfg == "ORIG" else NPZ[cfg]
        return d[f"Rsurf_{'orig' if cfg == 'ORIG' else 'filt'}"].astype(np.float64)
    ns = types.SimpleNamespace(**{v: getattr(m, v) for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v"]})
    for v in ["e3t", "e3u", "e3v", "tmask"]:
        setattr(ns, v, getattr(m, v)[k:k + 2])
    u = C.read("U", cfg, k=k); v = C.read("V", cfg, k=k)
    w = C.read("W", cfg, k=slice(k, k + 2))
    out = np.empty((u.shape[0], ny, nx))
    zero = np.zeros((ny, nx))
    for t in range(u.shape[0]):
        R = cf.continuity_residual(ns, np.stack([u[t], zero]), np.stack([v[t], zero]), w[t])
        out[t] = R[0]
    return out


cats = {}
curves = {}
bins = np.arange(0, 61)
for k in LEVS:
    dc, ds = DIST[k]
    wetk = m.tmask[k] & (d_edge >= 2)
    masks = {
        "all_interior": wetk,
        "clean_edge>40_coast>30_step>30": wetk & (d_edge > 40) & (dc > 30) & (ds > 30),
        "edge>40_coast>30": wetk & (d_edge > 40) & (dc > 30),
        "coast>30_incl_edges": wetk & (dc > 30),
        "edge<=20_coast>30": wetk & (d_edge <= 20) & (dc > 30),
        "coast<=3": wetk & (dc <= 3),
        "coast_4_30": wetk & (dc > 3) & (dc <= 30),
        "step<=3_coast>30_edge>40": wetk & (ds <= 3) & (dc > 30) & (d_edge > 40),
    }
    lev = {"n_points": {n: int(s.sum()) for n, s in masks.items()}}
    cv = {}
    for cfg in ALL:
        t0 = time.time()
        R = level_R(cfg, k)
        ms = np.nanmean(R ** 2, 0)                       # time-mean R^2 per point
        lev[cfg] = {n: float(np.sqrt(np.nanmean(ms[s]))) if s.sum() else None for n, s in masks.items()}
        tot = np.nansum(ms[wetk])
        lev[cfg]["share_R2_coast<=3"] = float(np.nansum(ms[masks["coast<=3"]]) / tot)
        lev[cfg]["share_R2_edge<=20"] = float(np.nansum(ms[wetk & (d_edge <= 20)]) / tot)
        lev[cfg]["share_R2_step<=3_not_coast<=3"] = float(np.nansum(ms[wetk & (ds <= 3) & (dc > 3)]) / tot)
        # curves: RMS vs one distance with the other two large
        cv[cfg] = {}
        for name, dist, other in [("coast", dc, wetk & (d_edge > 40) & (ds > 30)),
                                  ("step", ds, wetk & (d_edge > 40) & (dc > 30)),
                                  ("edge", d_edge, wetk & (dc > 30) & (ds > 30))]:
            r = []
            for b in bins:
                s = other & (np.floor(dist) == b) if b < 60 else other & (dist >= 60)
                r.append(float(np.sqrt(np.nanmean(ms[s]))) if s.sum() > 20 else np.nan)
            cv[cfg][name] = r
        print(f"k={k} {cfg} clean={lev[cfg]['clean_edge>40_coast>30_step>30']:.2e} all={lev[cfg]['all_interior']:.2e} ({time.time()-t0:.0f}s)", flush=True)
    cats[f"k{k}_{z[k]:.1f}m"] = lev
    curves[k] = cv
met["decomposition_full_month"] = cats

fig, axs = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True, sharex=True)
for r, k in enumerate(LEVS):
    for c, (name, lab) in enumerate([("coast", "distance to coast (land at level k) [cells]"),
                                     ("step", "distance to bottom step [cells]"),
                                     ("edge", "distance to regional-grid edge [cells]")]):
        ax = axs[r, c]
        for cfg in ALL:
            ax.semilogy(bins, curves[k][cfg][name], color=C.COLORS[cfg], lw=2.2 if cfg == "ORIG" else 1.4, label=cfg)
        ax.set_title(f"z = {z[k]:.0f} m: RMS R vs {name} (other two distances >30/40)", fontsize=10)
        ax.grid(which="both", alpha=0.3)
        if r == 2:
            ax.set_xlabel(lab)
    axs[r, 0].set_ylabel("RMS R over 31 days [s⁻¹]")
axs[0, 0].legend(fontsize=9, ncol=2)
fig.suptitle("Continuity residual vs distance to coast, bottom steps and regional-grid edges (last bin: ≥60 cells)", fontsize=12)
fig.savefig(f"{OUT}/fig_R_distance.png", dpi=130)
json.dump(met, open(f"{OUT}/metrics_residual.json", "w"), indent=1)

# ------------------------------------------------------------------ (d) Amazon mouth
jA, iA = C.box_slices(-53, -45, -2, 5)
jH, iH = slice(jA.start - 1, jA.stop), slice(iA.start - 1, iA.stop)       # 1-cell S/W halo for divergence
ns = types.SimpleNamespace()
for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v"]:
    setattr(ns, v, getattr(m, v)[jH, iH])
for v in ["e3t", "e3u", "e3v", "tmask"]:
    setattr(ns, v, getattr(m, v)[:, jH, iH])
area = (m.e1t * m.e2t)[jA, iA]
amz = {}
Rmaps = {}
srcs = ALL + ["GLORYS_W_free_surface"]
for cfg in srcs:
    ucfg = "ORIG" if cfg == "GLORYS_W_free_surface" else cfg
    U = C.read("U", ucfg, j=jH, i=iH); V = C.read("V", ucfg, j=jH, i=iH)
    if cfg == "GLORYS_W_free_surface":
        with netCDF4.Dataset(GLORYS_W) as ds_:
            W = ds_["vovecrtz"][:, :, jH, iH].filled(np.nan).astype(float)
    else:
        W = C.read("W", cfg, j=jH, i=iH)
    Q = []; Rs = np.zeros(area.shape); col = np.zeros(area.shape)
    for t in range(U.shape[0]):
        R = cf.continuity_residual(ns, U[t], V[t], W[t])[:, 1:, 1:]
        colR = np.nansum(np.nan_to_num(R) * m.e3t[:, jA, iA], 0)          # m/s
        Q.append(float(np.sum(colR * area)))
        Rs += np.nan_to_num(R[0]) / U.shape[0]; col += colR / U.shape[0]
    amz[cfg] = dict(Q_daily_m3s=Q, Q_mean_m3s=float(np.mean(Q)), Q_std_m3s=float(np.std(Q)),
                    R_surface_mean_max=float(np.nanmax(np.abs(Rs))),
                    R_surface_mean_rms=float(np.sqrt(np.mean(Rs[m.tmask[0, jA, iA]] ** 2))))
    Rmaps[cfg] = (np.where(m.tmask[0, jA, iA], Rs, np.nan), np.where(m.tmask[0, jA, iA], col, np.nan))
    print("amazon", cfg, amz[cfg]["Q_mean_m3s"], amz[cfg]["Q_std_m3s"], flush=True)
met["amazon_mouth_53W45W_2S5N"] = amz
lonA, latA = m.glamt[jA, iA], m.gphit[jA, iA]
show = ["GLORYS_W_free_surface", "ORIG", "W1.5", "W2.5", "R2Ld", "R3Ld"]
show = [s for s in show if s in Rmaps]
fig, axs = plt.subplots(2, len(show), figsize=(3.6 * len(show) + 1.5, 8), constrained_layout=True, squeeze=False)
pcs = [None, None]
for c, cfg in enumerate(show):
    for r, (lab, idx) in enumerate([("surface-layer R [s⁻¹]", 0), ("column ∫R dz [m/s]", 1)]):
        ax = axs[r, c]
        lim = 1e-6 if idx == 0 else 2e-5
        pcs[r] = ax.pcolormesh(lonA, latA, Rmaps[cfg][idx], norm=SymLogNorm(lim * 1e-4, vmin=-lim, vmax=lim, base=10),
                               cmap="RdBu_r", shading="auto")
        ax.set_facecolor("0.75"); ax.set_aspect(1)
        name = "GLORYS W (free surface)" if cfg.startswith("GLORYS") else ("ORIG rigid-lid W" if cfg == "ORIG" else cfg)
        if r == 0:
            ax.set_title(f"{name}\nQ = {amz[cfg]['Q_mean_m3s']/1e3:.0f}±{amz[cfg]['Q_std_m3s']/1e3:.0f} ×10³ m³/s", fontsize=9)
fig.colorbar(pcs[0], ax=axs[0, :].tolist(), shrink=0.8, label="surface-layer R [s⁻¹]")
fig.colorbar(pcs[1], ax=axs[1, :].tolist(), shrink=0.8, label="column ∫R dz [m/s]")
fig.suptitle("Amazon mouth: 31-day mean continuity residual (top: surface layer, bottom: column integral); Q = box integral", fontsize=11)
fig.savefig(f"{OUT}/fig_R_amazon.png", dpi=130)
json.dump(met, open(f"{OUT}/metrics_residual.json", "w"), indent=1)

# ------------------------------------------------------------------ (e) v1 vs v2 vs v3 hdiv at the coast
VERS = {"ORIG": None, "v1_componentwise": "output_v1_componentwise", "v2_btcorr": "output_v2_btcorr", "v3": "output"}
dc0 = DIST[0][0]
ns0 = types.SimpleNamespace(**{v: getattr(m, v) for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v"]})
for v in ["e3u", "e3v", "tmask"]:
    setattr(ns0, v, getattr(m, v)[0:1])
reg = np.zeros((ny, nx), bool); reg[js, is_] = True
wet0 = m.tmask[0] & (d_edge >= 2)
vers = {}
for name, folder in VERS.items():
    if folder is None:
        u = C.read("U", "ORIG", t=0, k=0); v = C.read("V", "ORIG", t=0, k=0)
    else:
        with netCDF4.Dataset(f"{C.RUN}/{folder}/W2.0/U_1993-01c_W2.0.nc") as d_:
            u = d_["vozocrtx"][0, 0].filled(np.nan).astype(float)
        with netCDF4.Dataset(f"{C.RUN}/{folder}/W2.0/V_1993-01c_W2.0.nc") as d_:
            v = d_["vomecrty"][0, 0].filled(np.nan).astype(float)
    hd = cf.horizontal_divergence_e3(ns0, u[None], v[None])[0] / np.where(m.e3t[0] > 0, m.e3t[0], 1)
    r = {}
    for dom, dm in [("region_70W30W_5S30N", reg), ("whole_domain", np.ones((ny, nx), bool))]:
        for lab, s in [("coast<=3", dc0 <= 3), ("coast_4_15", (dc0 > 3) & (dc0 <= 15)), ("coast>15", dc0 > 15), ("first_wet_cell", dc0 <= 1)]:
            sel = wet0 & dm & s
            r[f"{dom}:{lab}"] = float(np.sqrt(np.mean(hd[sel] ** 2)))
    vers[name] = r
    print(name, r, flush=True)
met["hdiv_versions_W2.0_surface_day1"] = vers
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
labs = ["coast<=3", "coast_4_15", "coast>15"]
x = np.arange(len(labs)); wd = 0.2
for n, (name, col) in enumerate(zip(VERS, ["k", "#d95f02", "#7570b3", "#1b9e77"])):
    ax.bar(x + n * wd, [vers[name][f"region_70W30W_5S30N:{l}"] for l in labs], wd, color=col, label=name)
ax.set_xticks(x + 1.5 * wd); ax.set_xticklabels(["≤3 cells from coast", "4–15 cells", ">15 cells"])
ax.set_yscale("log"); ax.set_ylabel("RMS hdiv [s⁻¹]"); ax.grid(axis="y", alpha=0.3); ax.legend()
ax.set_title("W2.0, surface, 1 Jan 1993, 70–30°W 5°S–30°N: RMS horizontal divergence by filter version")
fig.savefig(f"{OUT}/fig_hdiv_versions.png", dpi=130)
met["runtime_s"] = time.time() - t00
json.dump(met, open(f"{OUT}/metrics_residual.json", "w"), indent=1)
print("done", time.time() - t00)
