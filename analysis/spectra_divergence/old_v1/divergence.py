"""
Task 2a/b: horizontal divergence of the filtered velocities.

(a) Profiles (from output/<CFG>/diag_<CFG>.npz, whole-domain interior wet points, 31-day mean):
    RMS hdiv (filtered and original), ratio, max |hdiv|.
(b) Maps over 70-30W, 5S-30N at k=0 (0.5 m) and k=22 (109.7 m): 31-day mean |hdiv| from
    cgrid_filter.horizontal_divergence_e3 / e3t for ORIG and each config.
    * RMS(div_filt)/RMS(div_orig) over all wet points, and split by distance to land
      (<=3 cells "coastal", 4-15 cells "near-coast", >15 cells "open") and excluding the 30N model edge.
    * Anomaly flags: A = mean|div_f| / (31x31 median of mean|div_f|) > 4 and
      mean|div_f| > mean|div_o| smoothed (Gaussian 3 cells), i.e. filtered divergence locally larger than its
      surroundings AND than ORIG. Flagged points are characterised by distance to land, depth, latitude.
    * Zonal-band RMS of div (open ocean, >15 cells from land) vs latitude, with |dl/dy| for the Rossby configs,
      to expose residual divergence where l varies (Ld equatorial transition).
Outputs: fig_div_profiles.png, fig_div_maps_k0.png, fig_div_maps_k22.png, fig_div_ratio_maps.png,
         fig_div_lat_bands.png, metrics_divergence.json
"""
import json, sys, types
import numpy as np
import netCDF4
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C
import cgrid_filter as cf

OUT = C.RUN + "/analysis/spectra_divergence"
m = C.mesh()
CFGS = C.available_configs()
ALL = ["ORIG"] + CFGS
met = {"configs": CFGS}

# ------------------------------------------------------------------ (a) profiles
z = m.gdept_0
fig, axs = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True, constrained_layout=True)
prof = {}
for n, cfg in enumerate(CFGS):
    d = np.load(f"{C.RUN}/output/{cfg}/diag_{cfg}.npz")
    rf, ro = d["hdiv_rms_filt"].mean(0), d["hdiv_rms_orig"].mean(0)
    mf, mo = d["hdiv_max_filt"].mean(0), d["hdiv_max_orig"].mean(0)
    wet = ro > 0
    if n == 0:
        axs[0].semilogx(ro[wet], z[wet], "k-", lw=2, label="ORIG")
        axs[2].semilogx(mo[wet], z[wet], "k-", lw=2, label="ORIG")
    axs[0].semilogx(rf[wet], z[wet], color=C.COLORS[cfg], lw=1.6, label=cfg)
    axs[1].plot(rf[wet] / ro[wet], z[wet], color=C.COLORS[cfg], lw=1.6, label=cfg)
    axs[2].semilogx(mf[wet], z[wet], color=C.COLORS[cfg], lw=1.6, label=cfg)
    prof[cfg] = dict(rms_ratio_k0=float(rf[0] / ro[0]), rms_ratio_k22=float(rf[22] / ro[22]),
                     rms_ratio_depth_range=[float((rf[wet] / ro[wet]).min()), float((rf[wet] / ro[wet]).max())],
                     rms_filt_k0=float(rf[0]), rms_orig_k0=float(ro[0]),
                     max_filt_k0=float(mf[0]), max_orig_k0=float(mo[0]),
                     max_ratio_k0=float(mf[0] / mo[0]), max_ratio_k22=float(mf[22] / mo[22]))
for ax in axs:
    ax.invert_yaxis() if ax is axs[0] else None
    ax.set_yscale("symlog", linthresh=100); ax.grid(which="both", alpha=0.3)
axs[0].set_ylim(6000, 0)
axs[0].set_ylabel("depth [m]")
axs[0].set_xlabel("RMS hdiv [s⁻¹]"); axs[1].set_xlabel("RMS(hdiv filt) / RMS(hdiv orig)"); axs[2].set_xlabel("max |hdiv| [s⁻¹]")
axs[0].legend(fontsize=9)
fig.suptitle("Horizontal divergence vs depth, whole model domain (interior wet points), 31-day mean of daily statistics", fontsize=11)
fig.savefig(f"{OUT}/fig_div_profiles.png", dpi=130)
met["profiles_whole_domain"] = prof

# ------------------------------------------------------------------ (b) maps
js, is_ = C.box_slices(-70, -30, -5, 30, grid="T")
jh, ih = slice(js.start - 1, js.stop), slice(is_.start - 1, is_.stop)   # 1-cell halo to the S/W for differences
sub = types.SimpleNamespace()
for k in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v"]:
    setattr(sub, k, getattr(m, k)[jh, ih])
lon = m.glamt[jh, ih][1:, 1:]; lat = m.gphit[jh, ih][1:, 1:]
dist_land = {}
ROSSBY = netCDF4.Dataset(C.RUN + "/rossby/rossby_radius_T.nc")["Ld"][:].filled(np.nan).astype(float)

maps = {}
for k in [0, 22]:
    sub.e3u = m.e3u[k:k + 1, jh, ih]; sub.e3v = m.e3v[k:k + 1, jh, ih]; sub.tmask = m.tmask[k:k + 1, jh, ih]
    e3t = m.e3t[k, jh, ih]
    wet = m.tmask[k, jh, ih][1:, 1:]
    dl = ndimage.distance_transform_edt(m.tmask[k])[jh, ih][1:, 1:]   # distance to nearest land cell [cells], full domain
    dist_land[k] = dl
    for cfg in ALL:
        u = C.read("U", cfg, k=slice(k, k + 1), j=jh, i=ih)[:, 0]
        v = C.read("V", cfg, k=slice(k, k + 1), j=jh, i=ih)[:, 0]
        acc = np.zeros(wet.shape); sq = np.zeros(wet.shape)
        for t in range(u.shape[0]):
            de3 = cf.horizontal_divergence_e3(sub, u[t][None], v[t][None])[0]
            hd = np.where(sub.tmask[0], de3 / np.where(e3t > 0, e3t, 1), np.nan)[1:, 1:]
            acc += np.abs(np.nan_to_num(hd)); sq += np.nan_to_num(hd) ** 2
        maps[(cfg, k)] = (np.where(wet, acc / u.shape[0], np.nan), np.where(wet, np.sqrt(sq / u.shape[0]), np.nan))
        print(k, cfg, np.nanmean(maps[(cfg, k)][0]), flush=True)


def rms(a, sel):
    return float(np.sqrt(np.nanmean(a[sel] ** 2)))


north_edge = lat > 29.0   # last ~12 rows: no filter halo beyond 30N
res_maps = {}
for k in [0, 22]:
    wet = np.isfinite(maps[("ORIG", k)][0])
    dl = dist_land[k]
    masks = {"all": wet, "all_excl_north_edge": wet & ~north_edge,
             "coastal_le3": wet & (dl <= 3), "near_coast_4_15": wet & (dl > 3) & (dl <= 15),
             "open_gt15": wet & (dl > 15) & ~north_edge, "north_edge_gt29N_open": wet & north_edge & (dl > 15)}
    ro = {n: rms(maps[("ORIG", k)][1], s) for n, s in masks.items()}
    lev = {"rms_orig": ro}
    # ORIG: smoothed mean|div| as the reference
    for cfg in CFGS:
        af, rf = maps[(cfg, k)]
        ao = maps[("ORIG", k)][0]
        d = {"rms_ratio": {n: rms(rf, s) / ro[n] for n, s in masks.items()}}
        # share of domain-integrated div^2 (area-weighted) in the coastal band, filt vs orig
        area = (m.e1t * m.e2t)[js, is_]
        for lab, a2 in [("filt", rf), ("orig", maps[("ORIG", k)][1])]:
            tot = np.nansum(area * a2 ** 2 * wet)
            d[f"coastal_le3_share_of_div2_{lab}"] = float(np.nansum((area * a2 ** 2)[masks["coastal_le3"]]) / tot)
        # anomaly flags
        # 31x31 running median of the filtered field (land filled with the domain median)
        fill = np.where(wet, af, np.nanmedian(af))
        med = ndimage.median_filter(fill, size=31, mode="nearest")
        wsum = ndimage.gaussian_filter(wet.astype(float), 3)
        ao_s = ndimage.gaussian_filter(np.where(wet, ao, 0.0), 3) / np.maximum(wsum, 1e-6)
        A = af / med
        flag = wet & (A > 4) & (af > ao_s)
        lab_, nlab = ndimage.label(flag, structure=np.ones((3, 3)))
        clusters = []
        for c in range(1, nlab + 1):
            s = lab_ == c
            ii = np.argmax(np.where(s, af, -1))
            clusters.append(dict(n=int(s.sum()), lon=float(lon.ravel()[ii]), lat=float(lat.ravel()[ii]),
                                 peak_mean_abs_div=float(af.ravel()[ii]), peak_over_median31=float(A.ravel()[ii]),
                                 peak_over_orig_smoothed=float(af.ravel()[ii] / ao_s.ravel()[ii]),
                                 dist_land_cells=float(dl.ravel()[ii]), H_m=float(m.H[js, is_].ravel()[ii])))
        clusters.sort(key=lambda c: -c["peak_mean_abs_div"] * c["n"])
        d["n_flagged_points"] = int(flag.sum())
        d["flagged_fraction_coastal_le3"] = float((flag & (dl <= 3)).sum() / max(flag.sum(), 1))
        d["flagged_fraction_north_edge"] = float((flag & north_edge).sum() / max(flag.sum(), 1))
        d["top_clusters"] = clusters[:12]
        # max of 31-day-mean |div| and where
        ii = np.nanargmax(af)
        d["max_mean_abs_div"] = dict(value=float(af.ravel()[ii]), lon=float(lon.ravel()[ii]), lat=float(lat.ravel()[ii]),
                                     dist_land_cells=float(dl.ravel()[ii]), orig_there=float(ao.ravel()[ii]),
                                     orig_smoothed_there=float(ao_s.ravel()[ii]))
        d["_flag"] = flag
        lev[cfg] = d
    res_maps[k] = lev

# ---- figures: maps
for k in [0, 22]:
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    for ax, cfg in zip(axs.ravel(), ALL):
        a = maps[(cfg, k)][0]
        pc = ax.pcolormesh(lon, lat, a, norm=LogNorm(1e-8, 3e-5), cmap="magma_r", shading="auto")
        ax.contour(lon, lat, np.isfinite(a).astype(float), [0.5], colors="k", linewidths=0.6)
        ax.set_facecolor("0.75")
        if cfg != "ORIG":
            fl = res_maps[k][cfg]["_flag"]
            ax.plot(lon[fl], lat[fl], ".", color="cyan", ms=2)
            r = res_maps[k][cfg]["rms_ratio"]
            ax.set_title(f"{cfg}: RMS ratio all {r['all']:.3f}, open {r['open_gt15']:.3f}, coast≤3 {r['coastal_le3']:.2f}", fontsize=10)
        else:
            ax.set_title(f"ORIG (RMS {res_maps[k]['rms_orig']['all']:.2e} s⁻¹)", fontsize=10)
        ax.set_aspect(1 / np.cos(np.deg2rad(12)))
    fig.colorbar(pc, ax=axs, shrink=0.6, label="31-day mean |hdiv| [s⁻¹]")
    fig.suptitle(f"31-day mean |horizontal divergence| at z = {z[k]:.1f} m (cyan: flagged local anomalies, A>4 and > smoothed ORIG)", fontsize=12)
    fig.savefig(f"{OUT}/fig_div_maps_k{k}.png", dpi=130)

# ratio maps (smoothed): filtered / original mean|div|, Gaussian sigma 3 cells
fig, axs = plt.subplots(2, len(CFGS), figsize=(4.0 * len(CFGS), 8.5), constrained_layout=True, squeeze=False)
for r, k in enumerate([0, 22]):
    wet = np.isfinite(maps[("ORIG", k)][0]); ws = ndimage.gaussian_filter(wet.astype(float), 3)
    for c, cfg in enumerate(CFGS):
        sm = lambda a: np.where(wet, ndimage.gaussian_filter(np.where(wet, a, 0), 3) / np.maximum(ws, 1e-6), np.nan)
        R = sm(maps[(cfg, k)][0]) / sm(maps[("ORIG", k)][0])
        ax = axs[r, c]
        pc = ax.pcolormesh(lon, lat, R, norm=LogNorm(0.1, 10), cmap="RdBu_r", shading="auto")
        ax.set_facecolor("0.75"); ax.set_title(f"{cfg}, z={z[k]:.0f} m", fontsize=10)
        ax.set_aspect(1 / np.cos(np.deg2rad(12)))
fig.colorbar(pc, ax=axs, shrink=0.6, label="smoothed mean|div_filt| / smoothed mean|div_orig|")
fig.suptitle("Ratio of 31-day mean |hdiv| filtered/ORIG (Gaussian σ=3 cells). Log colour scale centred on 1; red: filtered divergence exceeds ORIG", fontsize=12)
fig.savefig(f"{OUT}/fig_div_ratio_maps.png", dpi=130)

# latitude bands, open ocean
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
lat_edges = np.arange(-5, 30.01, 0.5)
latc = 0.5 * (lat_edges[1:] + lat_edges[:-1])
bands = {}
for ax, k in zip(axs, [0, 22]):
    wet = np.isfinite(maps[("ORIG", k)][0]); sel0 = wet & (dist_land[k] > 15)
    for cfg in ALL:
        rr = []
        for a, b in zip(lat_edges[:-1], lat_edges[1:]):
            s = sel0 & (lat >= a) & (lat < b)
            rr.append(rms(maps[(cfg, k)][1], s) if s.sum() > 50 else np.nan)
        rr = np.array(rr)
        ax.semilogy(latc, rr, color=C.COLORS[cfg], lw=2 if cfg == "ORIG" else 1.5, label=cfg)
        bands[(cfg, k)] = rr
    # |d l/dy| for R2Ld (zonal median over the open-ocean part of the map), secondary axis
    ax2 = ax.twinx()
    ld = ROSSBY[js, is_]
    l2 = 2 * ld / 3.5
    dly = np.gradient(l2, axis=0) / m.e2t[js, is_]
    prof_dl = [np.nanmedian(np.abs(dly)[(lat >= a) & (lat < b) & sel0]) if ((lat >= a) & (lat < b) & sel0).sum() > 50 else np.nan
               for a, b in zip(lat_edges[:-1], lat_edges[1:])]
    ax2.plot(latc, prof_dl, "k:", lw=1.2); ax2.set_ylabel("|dℓ/dy| of R2Ld [m/m] (dotted)")
    ax.set_xlabel("latitude [°N]"); ax.set_ylabel("RMS hdiv, open ocean (>15 cells from land) [s⁻¹]")
    ax.set_title(f"z = {z[k]:.0f} m, 70-30W"); ax.grid(alpha=0.3)
axs[0].legend(fontsize=9, ncol=2)
fig.suptitle("Zonal-band RMS divergence (0.5° bands, 31 days) and meridional gradient of ℓ for R2Ld", fontsize=11)
fig.savefig(f"{OUT}/fig_div_lat_bands.png", dpi=130)

for k in [0, 22]:
    for cfg in CFGS:
        res_maps[k][cfg].pop("_flag")
        rb = bands[(cfg, k)] / bands[("ORIG", k)]
        res_maps[k][cfg]["open_ocean_band_ratio_by_lat"] = {f"{a:.2f}": (None if np.isnan(x) else float(x)) for a, x in zip(latc, rb)}
met["maps_70W30W_5S30N"] = {f"k{k}_{z[k]:.1f}m": res_maps[k] for k in [0, 22]}
json.dump(met, open(f"{OUT}/metrics_divergence.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/div_maps.npz", lon=lon, lat=lat,
                    **{f"{c}_k{k}_meanabs": maps[(c, k)][0].astype(np.float32) for c, k in maps},
                    **{f"{c}_k{k}_rms": maps[(c, k)][1].astype(np.float32) for c, k in maps})
print(json.dumps({k: {c: res_maps[k][c]["rms_ratio"] for c in CFGS} for k in [0, 22]}, indent=1))
