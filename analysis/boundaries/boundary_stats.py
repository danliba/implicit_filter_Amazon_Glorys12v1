"""
Tasks 2a, 2b, 2c and 5 (empirical part), from the monthly statistics in cache/stats_<CFG>.npz
(produced by accumulate.py).

2a  distance from the coast (Euclidean distance transform of each grid's own wet mask, in cells,
    converted to km with the local sqrt(e1*e2)), analysis domain 70-30W, 5S-30N:
      - ratio of bin-mean monthly-mean speed (T points, C-grid average) filtered / ORIG
      - RMS of daily (filtered - ORIG) velocity (U and V points), absolute and relative to the RMS
        of the daily ORIG velocity
      - EKE retained (U and V points)
      - filtered W (v3, scalar filter of the rigid-lid W): RMS of the monthly mean and variance retained
        at the W levels nearest 50 m and 200 m, filtered / ORIG
    at 0.5 m, 110 m, 454 m, 1062 m.
2b  regions by bottom depth (H_u = min of the two adjacent T columns): shelf H<200 m, slope
    200-1000 m, 1000-3000 m, deep H>3000 m. Volume-weighted (e1 e2 e3) KE, mean KE and EKE retained
    fractions; depth-averaged monthly-mean velocity change.
2c  first wet velocity faces normal to the coast (U(j,i) wet with U(j,i+1) or U(j,i-1) land;
    V(j,i) wet with V(j+1,i) or V(j-1,i) land); onshore-positive velocity statistics.
    NaN-mask identity (land faces stay NaN, i.e. zero normal flow through the coast).
5   zonal mean (60-40W) EKE retained vs latitude near the north edge (30N);
    RMS of the daily surface-layer continuity residual R (diag_<CFG>.npz) vs distance from the coast and
    from the north / south / west model-domain edges (residual statistics themselves: spectra_divergence).

Output: cache/metrics_stats.json, fig_coastdist_surface.png, fig_coastdist_depth.png,
        fig_coastal_normal.png, fig_north_edge_lat.png
"""
import sys, os, json
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
from common import mesh, CONFIGS, COLORS, LABELS, uv_to_t  # noqa: E402
import cgrid_filter as cf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh()
nz, ny, nx = m.nz, m.ny, m.nx
cfgs = [c for c in CONFIGS if os.path.exists(f"{HERE}/cache/stats_{c}.npz")]
LEVELS = [0, 22, 30, 35]
KW = {"w50": int(np.argmin(np.abs(m.gdepw_0 - 50))), "w200": int(np.argmin(np.abs(m.gdepw_0 - 200)))}
RUN = "/work/bk1450/b383184/Amazon/Mercator/implicit_run"

# ----------------------------------------------------------------------------- masks, geometry
lon_t, lat_t = m.glamt, m.gphit
dom = (lon_t >= -70) & (lon_t <= -30) & (lat_t >= -5) & (lat_t <= 30)
dom[-1, :] = False; dom[:, -1] = False   # last row/col: no outer neighbour, continuity w meaningless there
tm = m.tmask
umask = np.zeros_like(tm); umask[:, :, :-1] = tm[:, :, :-1] & tm[:, :, 1:]
vmask = np.zeros_like(tm); vmask[:, :-1, :] = tm[:, :-1, :] & tm[:, 1:, :]
Hu = np.zeros((ny, nx)); Hu[:, :-1] = np.minimum(m.H[:, :-1], m.H[:, 1:])
Hv = np.zeros((ny, nx)); Hv[:-1, :] = np.minimum(m.H[:-1, :], m.H[1:, :])
dx_t = np.sqrt(m.e1t * m.e2t) / 1e3; dx_u = np.sqrt(m.e1u * m.e2u) / 1e3; dx_v = np.sqrt(m.e1v * m.e2v) / 1e3

EDGES = np.array([0, 13, 22, 32, 45, 60, 80, 105, 135, 175, 230, 300, 400, 550, 750, 1000, 1500])
NB = len(EDGES) - 1


def dist_km(mask2d, dx):
    """Distance (km) of wet points to the nearest dry point of the same grid (domain edges not dry)."""
    return ndimage.distance_transform_edt(mask2d) * dx


def binsum(x, d, sel):
    b = np.digitize(d[sel], EDGES) - 1
    ok = (b >= 0) & (b < NB)
    return np.bincount(b[ok], weights=x[sel][ok], minlength=NB)


REG = {"shelf_H<200": (0, 200), "slope_200-1000": (200, 1000), "mid_1000-3000": (1000, 3000), "deep_H>3000": (3000, 1e5)}

res = {"levels_depth_m": [float(m.gdept_0[k]) for k in LEVELS], "dist_bin_edges_km": EDGES.tolist(),
       "coastdist": {}, "regions": {}, "coastal_normal": {}, "nan_mask": {}, "max_principle": {},
       "north_edge": {}}

# distances per level and grid (analysis-domain selections)
DIST = {}
for k in LEVELS:
    DIST[k] = dict(T=dist_km(tm[k], dx_t), U=dist_km(umask[k], dx_u), V=dist_km(vmask[k], dx_v))

# coastal-normal faces (surface and full 3D), onshore sign
def coastal_faces():
    su = np.zeros((nz, ny, nx)); sv = np.zeros((nz, ny, nx))
    east = np.zeros_like(umask); east[:, :, :-1] = umask[:, :, :-1] & ~umask[:, :, 1:]
    west = np.zeros_like(umask); west[:, :, 1:] = umask[:, :, 1:] & ~umask[:, :, :-1]
    north = np.zeros_like(vmask); north[:, :-1, :] = vmask[:, :-1, :] & ~vmask[:, 1:, :]
    south = np.zeros_like(vmask); south[:, 1:, :] = vmask[:, 1:, :] & ~vmask[:, :-1, :]
    # exclude the model-domain edge rows/cols (not coast)
    for a in (east, west, north, south):
        a[:, :, :2] = False; a[:, :, -2:] = False; a[:, :2, :] = False; a[:, -2:, :] = False
    su += east; su -= west; sv += north; sv -= south
    return su, sv, (east | west), (north | south)


onsU, onsV, cU, cV = coastal_faces()
domU = dom[None] & cU
domV = dom[None] & cV


def load(cfg):
    z = np.load(f"{HERE}/cache/stats_{cfg}.npz")
    return {k: z[k] for k in z.files}


O = load("ORIG")
for k in ["umean", "vmean", "u2mean", "v2mean"]:
    O[k] = O[k].astype(np.float64)
for k in ["wmean", "w2mean"]:
    O[k] = O[k].astype(np.float64)
Hdepth_u = np.where(umask, m.e3u, 0).sum(0); Hdepth_v = np.where(vmask, m.e3v, 0).sum(0)


def depth_avg(a, grid):
    e3, msk, Hd = (m.e3u, umask, Hdepth_u) if grid == "U" else (m.e3v, vmask, Hdepth_v)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(Hd > 0, np.nansum(np.where(msk, a, 0) * e3, 0) / Hd, np.nan)


ubar_o, vbar_o = depth_avg(O["umean"], "U"), depth_avg(O["vmean"], "V")


def level_curves(F, k):
    """Distance-binned quantities at level k for filtered stats F (dict) vs ORIG."""
    D = DIST[k]
    out = {}
    ut_o, vt_o = uv_to_t(O["umean"][k], O["vmean"][k]); so = np.sqrt(ut_o ** 2 + vt_o ** 2)
    ut_f, vt_f = uv_to_t(F["umean"][k], F["vmean"][k]); sf = np.sqrt(ut_f ** 2 + vt_f ** 2)
    selT = dom & tm[k]
    n_t = binsum(np.ones_like(so), D["T"], selT)
    out["n_T"] = n_t
    out["speed_ratio"] = binsum(sf, D["T"], selT) / binsum(so, D["T"], selT)
    # pointwise ratio > 1.2 where orig speed > 5 cm/s
    big = selT & (so > 0.05)
    out["frac_speed_ratio_gt_1p2"] = binsum((sf > 1.2 * so).astype(float), D["T"], big) / np.maximum(binsum(np.ones_like(so), D["T"], big), 1)
    selU = dom & umask[k]; selV = dom & vmask[k]
    d2 = binsum(np.nan_to_num(F["du2mean"][k]), D["U"], selU) + binsum(np.nan_to_num(F["dv2mean"][k]), D["V"], selV)
    o2 = binsum(np.nan_to_num(O["u2mean"][k]), D["U"], selU) + binsum(np.nan_to_num(O["v2mean"][k]), D["V"], selV)
    nuv = binsum(np.ones((ny, nx)), D["U"], selU) + binsum(np.ones((ny, nx)), D["V"], selV)
    out["rms_daily_diff"] = np.sqrt(d2 / nuv)
    out["rel_rms_daily_diff"] = np.sqrt(d2 / o2)
    eke_o = binsum(np.nan_to_num(O["u2mean"][k] - O["umean"][k] ** 2), D["U"], selU) + \
        binsum(np.nan_to_num(O["v2mean"][k] - O["vmean"][k] ** 2), D["V"], selV)
    eke_f = binsum(np.nan_to_num(F["u2mean"][k] - F["umean"][k] ** 2), D["U"], selU) + \
        binsum(np.nan_to_num(F["v2mean"][k] - F["vmean"][k] ** 2), D["V"], selV)
    out["eke_retained"] = eke_f / eke_o
    return out


def w_curves(F):
    """Filtered W vs ORIG W by distance from the coast at W levels KW (wet = tmask of the level)."""
    out = {}
    for name, kw in KW.items():
        sel = dom & tm[kw]
        D = dist_km(tm[kw], dx_t)
        n = np.maximum(binsum(np.ones((ny, nx)), D, sel), 1)
        a = binsum(np.nan_to_num(F["wmean"][kw]) ** 2, D, sel); b = binsum(np.nan_to_num(O["wmean"][kw]) ** 2, D, sel)
        vf = binsum(np.nan_to_num(F["w2mean"][kw] - F["wmean"][kw] ** 2), D, sel)
        vo = binsum(np.nan_to_num(O["w2mean"][kw] - O["wmean"][kw] ** 2), D, sel)
        dd = binsum(np.nan_to_num(F["dw2mean"][kw]), D, sel)
        out[f"rms_wmean_ratio_{name}"] = np.sqrt(a / b)
        out[f"rms_wmean_filt_{name}"] = np.sqrt(a / n)
        out[f"rms_wmean_orig_{name}"] = np.sqrt(b / n)
        out[f"wvar_retained_{name}"] = vf / vo
        out[f"rel_rms_daily_diff_{name}"] = np.sqrt(dd / binsum(np.nan_to_num(O["w2mean"][kw]), D, sel))
        out[f"dmid_{name}"] = binsum(D, D, sel) / n
    return out


curves = {}
lat_rows = {}
for cfg in cfgs:
    F = load(cfg)
    for key in ["umean", "vmean", "u2mean", "v2mean", "du2mean", "dv2mean", "wmean", "w2mean", "dw2mean"]:
        F[key] = F[key].astype(np.float64)
    res["nan_mask"][cfg] = dict(mismatch_U=int(F["nan_mismatch_U"].sum()), mismatch_V=int(F["nan_mismatch_V"].sum()))
    res["max_principle"][cfg] = dict(levels=[int(x) for x in F["maxp_levels"]],
                                     max_exceedance_W_m_s=float(F["maxp_W"].max()),
                                     frac_points_exceeding_1mm_s_U=float(F["maxpfrac_U"].mean()),
                                     frac_points_exceeding_1mm_s_V=float(F["maxpfrac_V"].mean()),
                                     p_levels_max_exceedance_U=[float(x) for x in F["maxp_U"].max(0)],
                                     max_exceedance_U_m_s=float(F["maxp_U"].max()),
                                     max_exceedance_V_m_s=float(F["maxp_V"].max()))
    curves[cfg] = {k: level_curves(F, k) for k in LEVELS}
    curves[cfg]["w"] = w_curves(F)
    res["coastdist"][cfg] = {f"k{k}": {kk: np.round(vv, 4).tolist() for kk, vv in curves[cfg][k].items()} for k in LEVELS}
    res["coastdist"][cfg]["w"] = {kk: np.round(vv, 4).tolist() for kk, vv in curves[cfg]["w"].items()}

    # ---------------------------------------------------------------- 2b regions
    ubar_f, vbar_f = depth_avg(F["umean"], "U"), depth_avg(F["vmean"], "V")
    rr = {}
    for rname, (h0, h1) in REG.items():
        ru = dom & (Hu >= h0) & (Hu < h1); rv = dom & (Hv >= h0) & (Hv < h1)
        d = {}
        for lab, ksl in [("full_depth", slice(None)), ("surface", slice(0, 1))]:
            wu = (m.e1u * m.e2u)[None] * m.e3u[ksl] * (umask[ksl] & ru[None])
            wv = (m.e1v * m.e2v)[None] * m.e3v[ksl] * (vmask[ksl] & rv[None])
            def S(a_u, a_v):
                return float(np.nansum(np.nan_to_num(a_u[ksl]) * wu) + np.nansum(np.nan_to_num(a_v[ksl]) * wv))
            ke_o = S(O["u2mean"], O["v2mean"]); ke_f = S(F["u2mean"], F["v2mean"])
            mk_o = S(O["umean"] ** 2, O["vmean"] ** 2); mk_f = S(F["umean"] ** 2, F["vmean"] ** 2)
            d[lab] = dict(KE_retained=ke_f / ke_o, MKE_retained=mk_f / mk_o,
                          EKE_retained=(ke_f - mk_f) / (ke_o - mk_o),
                          EKE_fraction_of_KE_orig=(ke_o - mk_o) / ke_o)
        au = m.e1u * m.e2u * ru; av = m.e1v * m.e2v * rv
        okU = np.isfinite(ubar_o) & ru; okV = np.isfinite(vbar_o) & rv
        num = np.nansum(((ubar_f - ubar_o) ** 2 * au)[okU]) + np.nansum(((vbar_f - vbar_o) ** 2 * av)[okV])
        den = np.nansum((ubar_o ** 2 * au)[okU]) + np.nansum((vbar_o ** 2 * av)[okV])
        nf = np.nansum((ubar_f ** 2 * au)[okU]) + np.nansum((vbar_f ** 2 * av)[okV])
        d["depth_avg_mean_flow"] = dict(rel_rms_change=float(np.sqrt(num / den)),
                                        rms_change_m_s=float(np.sqrt(num / (au[okU].sum() + av[okV].sum()))),
                                        rms_orig_m_s=float(np.sqrt(den / (au[okU].sum() + av[okV].sum()))),
                                        KE_retained=float(nf / den))
        d["n_T_columns"] = int((dom & tm[0] & (m.H >= h0) & (m.H < h1)).sum())
        rr[rname] = d
    res["regions"][cfg] = rr

    # ---------------------------------------------------------------- 2c coastal-normal faces
    cn = {}
    for lab, ksl in [("surface", slice(0, 1)), ("all_levels", slice(None))]:
        su, sv = onsU[ksl], onsV[ksl]
        mu, mv = domU[ksl], domV[ksl]
        on_o = np.concatenate([(O["umean"][ksl] * su)[mu], (O["vmean"][ksl] * sv)[mv]])
        on_f = np.concatenate([(F["umean"][ksl] * su)[mu], (F["vmean"][ksl] * sv)[mv]])
        dd = np.concatenate([F["du2mean"][ksl][mu], F["dv2mean"][ksl][mv]])
        oo = np.concatenate([O["u2mean"][ksl][mu], O["v2mean"][ksl][mv]])
        fu = (m.e2u[None] * m.e3u[ksl]); fv = (m.e1v[None] * m.e3v[ksl])
        Q_o = np.nansum((O["umean"][ksl] * su * fu)[mu]) + np.nansum((O["vmean"][ksl] * sv * fv)[mv])
        Q_f = np.nansum((F["umean"][ksl] * su * fu)[mu]) + np.nansum((F["vmean"][ksl] * sv * fv)[mv])
        absQ_o = np.nansum(np.abs(O["umean"][ksl] * su * fu)[mu]) + np.nansum(np.abs(O["vmean"][ksl] * sv * fv)[mv])
        absQ_f = np.nansum(np.abs(F["umean"][ksl] * su * fu)[mu]) + np.nansum(np.abs(F["vmean"][ksl] * sv * fv)[mv])
        cn[lab] = dict(n_faces=int(on_o.size),
                       mean_onshore_orig=float(np.nanmean(on_o)), mean_onshore_filt=float(np.nanmean(on_f)),
                       rms_mean_onshore_orig=float(np.sqrt(np.nanmean(on_o ** 2))),
                       rms_mean_onshore_filt=float(np.sqrt(np.nanmean(on_f ** 2))),
                       rms_mean_onshore_change=float(np.sqrt(np.nanmean((on_f - on_o) ** 2))),
                       corr_mean_onshore=float(np.corrcoef(on_o, on_f)[0, 1]),
                       rms_daily_diff=float(np.sqrt(np.nanmean(dd))), rms_daily_orig=float(np.sqrt(np.nanmean(oo))),
                       net_onshore_transport_Sv_orig=Q_o / 1e6, net_onshore_transport_Sv_filt=Q_f / 1e6,
                       sum_abs_onshore_transport_Sv_orig=absQ_o / 1e6, sum_abs_onshore_transport_Sv_filt=absQ_f / 1e6,
                       p99_abs_onshore_orig=float(np.nanpercentile(np.abs(on_o), 99)),
                       p99_abs_onshore_filt=float(np.nanpercentile(np.abs(on_f), 99)))
        if lab == "surface":
            curves[cfg]["onshore_scatter"] = (on_o, on_f)
    res["coastal_normal"][cfg] = cn

    # ---------------------------------------------------------------- 5 latitude profile near 30N
    band = (lon_t[0] >= -60) & (lon_t[0] <= -40)
    prof = {}
    for lab, k in [("surface", 0), ("454m", 30)]:
        ekeo = np.nansum(np.nan_to_num(O["u2mean"][k] - O["umean"][k] ** 2)[:, band], 1) + \
            np.nansum(np.nan_to_num(O["v2mean"][k] - O["vmean"][k] ** 2)[:, band], 1)
        ekef = np.nansum(np.nan_to_num(F["u2mean"][k] - F["umean"][k] ** 2)[:, band], 1) + \
            np.nansum(np.nan_to_num(F["v2mean"][k] - F["vmean"][k] ** 2)[:, band], 1)
        prof[lab] = (ekeo, ekef)
    lat_rows[cfg] = prof
    latc = lat_t[:, 700]
    ne = {}
    for lab, (eo, ef) in prof.items():
        def band_ratio(a, b):
            s = (latc >= a) & (latc < b)
            return float(ef[s].sum() / eo[s].sum())
        ne[lab] = {"EKE_retained_20-25N": band_ratio(20, 25), "EKE_retained_25-29N": band_ratio(25, 29),
                   "EKE_retained_29-30N": band_ratio(29, 30.1), "EKE_retained_28-30N": band_ratio(28, 30.1)}
    res["north_edge"][cfg] = ne
    # ---------------------------------------------------------------- residual R vs coast / domain edges
    dg = np.load(os.environ.get("DIAG_OVERRIDE", f"{RUN}/output/{cfg}/diag_{cfg}.npz"))  # override: testing only
    rf2 = np.nanmean(dg["Rsurf_filt"].astype(np.float64) ** 2, 0); ro2 = np.nanmean(dg["Rsurf_orig"].astype(np.float64) ** 2, 0)
    wet0 = tm[0] & np.isfinite(rf2)
    Dc = DIST[0]["T"]
    rr = {"coast_dmid_km": [], "coast_rms_filt": [], "coast_rms_orig": []}
    for b in range(NB):
        s_ = dom & wet0 & (Dc >= EDGES[b]) & (Dc < EDGES[b + 1])
        rr["coast_dmid_km"].append(float(Dc[s_].mean()) if s_.any() else np.nan)
        rr["coast_rms_filt"].append(float(np.sqrt(rf2[s_].mean())) if s_.any() else np.nan)
        rr["coast_rms_orig"].append(float(np.sqrt(ro2[s_].mean())) if s_.any() else np.nan)
    far = wet0 & (Dc > 300)          # exclude coastal influence for the edge profiles
    prof = {}
    for ename, (sel2d, dist_rows) in {
            "north_30N_60-20W": ((lon_t >= -60) & (lon_t <= -20), (ny - 1) - np.arange(ny)[:, None] * np.ones((1, nx))),
            "south_10S_30W-5E": ((lon_t >= -30) & (lon_t <= 5), np.arange(ny)[:, None] * np.ones((1, nx))),
            "west_95W_10S-5N": ((lat_t >= -10) & (lat_t <= 5), np.ones((ny, 1)) * np.arange(nx)[None, :])}.items():
        d = dist_rows
        pf, po, dd_ = [], [], []
        for c0 in range(0, 60):
            s_ = sel2d & far & (d == c0)
            if s_.sum() < 20:
                pf.append(np.nan); po.append(np.nan); dd_.append(c0); continue
            pf.append(float(np.sqrt(rf2[s_].mean()))); po.append(float(np.sqrt(ro2[s_].mean()))); dd_.append(c0)
        prof[ename] = dict(cells=dd_, rms_filt=pf, rms_orig=po)
    rr["edges"] = prof
    res.setdefault("residual_boundary", {})[cfg] = rr
    del F, dg
    print("done", cfg, flush=True)

json.dump(res, open(f"{HERE}/cache/metrics_stats.json", "w"), indent=1)
np.savez(f"{HERE}/cache/curves.npz", **{f"{c}_{k}_{q}": v for c in cfgs for k in LEVELS for q, v in curves[c][k].items()})

# ----------------------------------------------------------------------------- figures
# representative bin-centre distance: mean distance of analysis-domain T points in each bin
selT = dom & tm[0]
dmid = binsum(DIST[0]["T"], DIST[0]["T"], selT) / binsum(np.ones((ny, nx)), DIST[0]["T"], selT)

fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
for cfg in cfgs:
    c = curves[cfg][0]
    kw = dict(color=COLORS[cfg], marker="o", ms=3, label=LABELS[cfg])
    axs[0, 0].semilogx(dmid, c["speed_ratio"], **kw)
    axs[0, 1].semilogx(dmid, c["rms_daily_diff"] * 100, **kw)
    axs[1, 0].semilogx(dmid, c["rel_rms_daily_diff"], **kw)
    axs[1, 1].semilogx(dmid, c["eke_retained"], **kw)
axs[0, 0].axhline(1, color="k", lw=0.6)
axs[0, 0].set_ylabel("⟨|ū| filtered⟩ / ⟨|ū| ORIG⟩ (monthly mean)")
axs[0, 1].set_ylabel("RMS daily (filtered − ORIG) [cm/s]")
axs[1, 0].set_ylabel("RMS daily (filtered − ORIG) / RMS daily ORIG")
axs[1, 1].set_ylabel("EKE retained (filtered / ORIG)")
for ax in axs.flat:
    ax.set_xlabel("distance from coast [km] (bin mean)"); ax.grid(alpha=0.3)
axs[0, 0].legend(fontsize=8)
fig.suptitle("Surface (0.5 m), 70–30°W 5°S–30°N, Jan 1993: dependence on distance from the nearest land point")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_coastdist_surface.png", dpi=130)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
for j, k in enumerate(LEVELS[1:]):
    sel = dom & tm[k]
    dm = binsum(DIST[k]["T"], DIST[k]["T"], sel) / np.maximum(binsum(np.ones((ny, nx)), DIST[k]["T"], sel), 1)
    for cfg in cfgs:
        axs[j].semilogx(dm, curves[cfg][k]["speed_ratio"], color=COLORS[cfg], marker="o", ms=3, label=LABELS[cfg])
        axs[j].semilogx(dm, curves[cfg][k]["eke_retained"], color=COLORS[cfg], ls="--", lw=0.8)
    axs[j].axhline(1, color="k", lw=0.6)
    axs[j].set_title(f"{m.gdept_0[k]:.0f} m: speed ratio (solid), EKE retained (dashed)")
    axs[j].set_xlabel("distance from coast at this level [km]"); axs[j].grid(alpha=0.3)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_coastdist_depth.png", dpi=130)

# coastal normal: scatter orig vs filt mean onshore velocity (surface) + w ratio vs distance
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for cfg in cfgs:
    on_o, on_f = curves[cfg]["onshore_scatter"]
    axs[0].scatter(on_o * 100, on_f * 100, s=2, color=COLORS[cfg], alpha=0.4, label=LABELS[cfg])
lim = 60
axs[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.6); axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-lim, lim)
axs[0].set_xlabel("ORIG monthly-mean onshore velocity [cm/s]"); axs[0].set_ylabel("filtered [cm/s]")
axs[0].set_title("First wet face normal to the coast, surface"); axs[0].legend(fontsize=7, markerscale=4)
for cfg in cfgs:
    rr = res["residual_boundary"][cfg]
    axs[1].loglog(rr["coast_dmid_km"], rr["coast_rms_filt"], color=COLORS[cfg], marker="o", ms=2, label=cfg)
rr = res["residual_boundary"][cfgs[0]]
axs[1].loglog(rr["coast_dmid_km"], rr["coast_rms_orig"], color="k", marker="o", ms=2, label="ORIG")
axs[1].set_xlabel("distance from coast [km]"); axs[1].set_ylabel("RMS daily surface-layer residual R [1/s]")
axs[1].set_title("v3 continuity residual R = ∂w/∂z + ∇h·u vs distance from coast", fontsize=9)
axs[1].legend(fontsize=7)
axs[1].grid(alpha=0.3)
x = np.arange(len(cfgs))
axs[2].bar(x - 0.2, [res["coastal_normal"][c]["surface"]["rms_mean_onshore_orig"] * 100 for c in cfgs], 0.4, color="0.6", label="ORIG")
axs[2].bar(x + 0.2, [res["coastal_normal"][c]["surface"]["rms_mean_onshore_filt"] * 100 for c in cfgs], 0.4,
           color=[COLORS[c] for c in cfgs], label="filtered")
axs[2].set_xticks(x); axs[2].set_xticklabels(cfgs); axs[2].set_ylabel("RMS monthly-mean onshore velocity [cm/s]")
axs[2].set_title("First wet coastal faces, surface, 70–30°W 5°S–30°N"); axs[2].legend()
fig.tight_layout(); fig.savefig(f"{HERE}/fig_coastal_normal.png", dpi=130)

latc = lat_t[:, 700]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for j, lab in enumerate(["surface", "454m"]):
    for cfg in cfgs:
        eo, ef = lat_rows[cfg][lab]
        # 3-row running sums to reduce noise
        ker = np.ones(3)
        r = np.convolve(ef, ker, "same") / np.convolve(eo, ker, "same")
        s = latc >= 10
        axs[j].plot(latc[s], r[s], color=COLORS[cfg], label=LABELS[cfg])
    axs[j].axvspan(29, 30.1, color="0.85"); axs[j].set_xlim(10, 30.1)
    axs[j].set_xlabel("latitude [°N]"); axs[j].set_ylabel("EKE retained (zonal sum 60–40°W)")
    axs[j].set_title(f"{lab}: EKE retained vs latitude (north edge = 30°N)"); axs[j].grid(alpha=0.3)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_north_edge_lat.png", dpi=130)
# filtered W vs distance from coast
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for cfg in cfgs:
    w = curves[cfg]["w"]
    for name, ls in [("w50", "-"), ("w200", "--")]:
        axs[0].semilogx(w[f"dmid_{name}"], w[f"rms_wmean_ratio_{name}"], color=COLORS[cfg], ls=ls, marker="o", ms=2,
                        label=cfg if name == "w50" else None)
        axs[1].semilogx(w[f"dmid_{name}"], w[f"wvar_retained_{name}"], color=COLORS[cfg], ls=ls, marker="o", ms=2)
        axs[2].semilogx(w[f"dmid_{name}"], w[f"rel_rms_daily_diff_{name}"], color=COLORS[cfg], ls=ls, marker="o", ms=2)
axs[0].set_ylabel("RMS W̄ filtered / ORIG (monthly mean)"); axs[1].set_ylabel("W daily variance retained")
axs[2].set_ylabel("RMS daily (W filt − ORIG) / RMS daily W ORIG")
for a in axs:
    a.set_xlabel("distance from coast at this level [km]"); a.grid(alpha=0.3); a.axhline(1, color="k", lw=0.5)
axs[0].legend(fontsize=7)
fig.suptitle(f"v3 filtered W (scalar filter), 70–30°W 5°S–30°N: {m.gdepw_0[KW['w50']]:.0f} m solid, {m.gdepw_0[KW['w200']]:.0f} m dashed")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_w_coastdist.png", dpi=130)

# residual vs distance from the model-domain edges
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for j, ename in enumerate(res["residual_boundary"][cfgs[0]]["edges"]):
    for cfg in cfgs:
        e = res["residual_boundary"][cfg]["edges"][ename]
        axs[j].semilogy(e["cells"], e["rms_filt"], color=COLORS[cfg], label=cfg)
    e = res["residual_boundary"][cfgs[0]]["edges"][ename]
    axs[j].semilogy(e["cells"], e["rms_orig"], color="k", label="ORIG")
    axs[j].set_title(f"edge {ename}", fontsize=10); axs[j].set_xlabel("distance from the edge [cells]"); axs[j].grid(alpha=0.3)
axs[0].set_ylabel("RMS daily surface-layer R [1/s] (points >300 km from coast)"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_residual_edges.png", dpi=130)
print("all done")
