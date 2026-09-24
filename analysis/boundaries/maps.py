"""
Tasks 1 and 4: close-up maps near the Amazon shelf / mouth and at the shelf break at depth.

Fig. fig_shelf_mean.png     53-43W 3S-7N, surface, Jan-1993 mean: speed + quivers (top), filtered-ORIG (bottom)
Fig. fig_shelf_daily.png    same for the daily field of 15 Jan 1993
Fig. fig_mouth_mean.png     51.5-47W 1.5S-3N zoom of the monthly mean (Amazon mouth)
Fig. fig_shelf_transects.png monthly-mean u, v, speed along 2N and 5N (lon 53-45W), with H
Fig. fig_depth_slope.png    50-44W 0-8N at 454 m and 1062 m: ORIG, W2.0, R2Ld and differences
Fig. fig_w_shelf_<z>m.png   filtered W (v3 scalar filter of the rigid-lid W), monthly mean, shelf box, ~50 m and ~200 m
Fig. fig_w_shelf_daily_50m.png  same, daily 15 Jan, ~50 m
Fig. fig_w_mouth_<z>m.png   Amazon mouth, monthly mean W at ~20 m and ~50 m
v1 (component-wise filter) monthly means are read from v1/cache/stats_<CFG>.npz for the transect comparison.
Isobaths 200 m (white/black) and 1000 m (grey) from mesh H; land from the T mask of the level.
Monthly means come from cache/stats_<CFG>.npz (accumulate.py).
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
from common import mesh, read, CONFIGS, COLORS, LABELS, uv_to_t, box_slices  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh()
cfgs = ["ORIG"] + [c for c in CONFIGS if os.path.exists(f"{HERE}/cache/stats_{c}.npz")]
TDAY = 14   # 15 Jan


def monthly(cfg, k):
    z = np.load(f"{HERE}/cache/stats_{cfg}.npz")
    return z["umean"][k].astype(float), z["vmean"][k].astype(float)


def to_t(u, v, k):
    ut, vt = uv_to_t(u, v)
    ut[~m.tmask[k]] = np.nan; vt[~m.tmask[k]] = np.nan
    return ut, vt


def panel(ax, js, is_, k, ut, vt, vmax, diff=False, q=3, qscale=None, title=""):
    lon = m.glamt[js, is_]; lat = m.gphit[js, is_]
    s = np.sqrt(ut ** 2 + vt ** 2)[js, is_]
    if diff:
        pc = ax.pcolormesh(lon, lat, s, cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
    else:
        pc = ax.pcolormesh(lon, lat, s, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
    land = (~m.tmask[k][js, is_]).astype(float)
    ax.contourf(lon, lat, land, levels=[0.5, 1.5], colors="0.65")
    ax.contour(lon, lat, m.H[js, is_], levels=[200], colors="w" if not diff else "k", linewidths=0.7)
    ax.contour(lon, lat, m.H[js, is_], levels=[1000], colors="0.5", linewidths=0.7)
    ax.quiver(lon[::q, ::q], lat[::q, ::q], ut[js, is_][::q, ::q], vt[js, is_][::q, ::q],
              scale=qscale, width=0.0025, color="k" if diff else "w")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    return pc


def shelf_figure(fields, fname, suptitle, region=(-53, -43, -3, 7), q=3, vmax=1.5, dvmax=0.5, k=0, show=None):
    js, is_ = box_slices(*region)
    show = show or cfgs
    n = len(show)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 6.6), squeeze=False)
    uo, vo = fields["ORIG"]
    for c, cfg in enumerate(show):
        u, v = fields[cfg]
        pc = panel(axs[0, c], js, is_, k, u, v, vmax, q=q, qscale=vmax * 20, title=LABELS[cfg])
        if cfg == "ORIG":
            axs[1, c].axis("off")
            continue
        pd = panel(axs[1, c], js, is_, k, u - uo, v - vo, dvmax, diff=True, q=q, qscale=dvmax * 20,
                   title=f"{cfg} − ORIG (|Δu|, Δu arrows)")
    fig.colorbar(pc, ax=axs[0, :], shrink=0.8, label="speed [m/s]", pad=0.01)
    fig.colorbar(pd, ax=axs[1, 1:], shrink=0.8, label="|Δu| [m/s]", pad=0.01)
    fig.suptitle(suptitle)
    fig.savefig(f"{HERE}/{fname}", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- surface maps
mean0 = {c: to_t(*monthly(c, 0), 0) for c in cfgs}
shelf_figure(mean0, "fig_shelf_mean.png", "Amazon shelf, surface, Jan-1993 mean. Isobaths: 200 m (white/black), 1000 m (grey)")
shelf_figure(mean0, "fig_mouth_mean.png", "Amazon mouth, surface, Jan-1993 mean", region=(-51.5, -47, -1.5, 3), q=1,
             vmax=1.2, dvmax=0.4)
day = {}
js_r, is_r = box_slices(-55, -41, -5, 9)
for c in cfgs:
    u = np.full((m.ny, m.nx), np.nan); v = np.full((m.ny, m.nx), np.nan)
    u[js_r, is_r] = read("U", c, t=TDAY, k=0, j=js_r, i=is_r)
    v[js_r, is_r] = read("V", c, t=TDAY, k=0, j=js_r, i=is_r)
    day[c] = to_t(u, v, 0)
shelf_figure(day, "fig_shelf_daily.png", "Amazon shelf, surface, 15 Jan 1993 (daily). Isobaths 200/1000 m", vmax=1.8,
             dvmax=0.8)

# ----------------------------------------------------------------------------- W close-ups
def w_monthly(cfg, k):
    z = np.load(f"{HERE}/cache/stats_{cfg}.npz")
    w = z["wmean"][k].astype(float)
    w[~m.tmask[k]] = np.nan
    return w


def w_figure(fields, k, fname, suptitle, region, vmax, dvmax):
    js, is_ = box_slices(*region)
    lon = m.glamt[js, is_]; lat = m.gphit[js, is_]
    n = len(cfgs)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 6.6), squeeze=False)
    wo = fields["ORIG"]
    for c, cfg in enumerate(cfgs):
        for r, (fld, vm, cm) in enumerate([(fields[cfg], vmax, "RdBu_r"), (fields[cfg] - wo, dvmax, "PuOr_r")]):
            ax = axs[r, c]
            if r == 1 and cfg == "ORIG":
                ax.axis("off"); continue
            pc = ax.pcolormesh(lon, lat, fld[js, is_] * 86400, cmap=cm, vmin=-vm, vmax=vm, shading="auto")
            ax.contourf(lon, lat, (~m.tmask[k][js, is_]).astype(float), levels=[0.5, 1.5], colors="0.65")
            ax.contour(lon, lat, m.H[js, is_], levels=[200], colors="k", linewidths=0.6)
            ax.contour(lon, lat, m.H[js, is_], levels=[1000], colors="0.4", linewidths=0.6)
            ax.set_aspect("equal")
            ax.set_title(LABELS[cfg] if r == 0 else f"{cfg} − ORIG", fontsize=9)
            if r == 0:
                p0 = pc
            else:
                p1 = pc
    fig.colorbar(p0, ax=axs[0, :], shrink=0.8, pad=0.01, label="W [m/day]")
    fig.colorbar(p1, ax=axs[1, 1:], shrink=0.8, pad=0.01, label="ΔW [m/day]")
    fig.suptitle(suptitle)
    fig.savefig(f"{HERE}/{fname}", dpi=130, bbox_inches="tight"); plt.close(fig)


wmet = {}
for zt in [20, 50, 200]:
    k = int(np.argmin(np.abs(m.gdepw_0 - zt)))
    wm = {c: w_monthly(c, k) for c in cfgs}
    for region, tag in [((-53, -43, -3, 7), "shelf"), ((-51.5, -47, -1.5, 3), "mouth")]:
        if (tag == "shelf" and zt == 20) or (tag == "mouth" and zt == 200):
            continue
        js, is_ = box_slices(*region)
        vmax = float(np.nanpercentile(np.abs(wm["ORIG"][js, is_]), 98)) * 86400
        w_figure(wm, k, f"fig_w_{tag}_{zt}m.png",
                 f"W (monthly mean) at {m.gdepw_0[k]:.0f} m, {tag}; ORIG = rigid-lid GLORYS W. Isobaths 200/1000 m",
                 region, vmax, vmax)
        # boundary check: RMS W in the first wet cell next to land vs 3-10 cells away (within the box)
        from scipy import ndimage
        dcell = ndimage.distance_transform_cdt(m.tmask[k], metric="taxicab")[js, is_]
        for cfg in cfgs:
            w = wm[cfg][js, is_]
            wmet[f"{tag}_{m.gdepw_0[k]:.0f}m_{cfg}"] = dict(
                rms_first_wet_cell_m_day=float(np.sqrt(np.nanmean(w[dcell == 1] ** 2)) * 86400),
                rms_cells_3to10_m_day=float(np.sqrt(np.nanmean(w[(dcell >= 3) & (dcell <= 10)] ** 2)) * 86400),
                rms_box_m_day=float(np.sqrt(np.nanmean(w ** 2)) * 86400))
k50 = int(np.argmin(np.abs(m.gdepw_0 - 50)))
wd = {}
for c in cfgs:
    w = np.full((m.ny, m.nx), np.nan)
    w[js_r, is_r] = read("W", c, t=TDAY, k=k50, j=js_r, i=is_r)
    w[~m.tmask[k50]] = np.nan
    wd[c] = w
vmax = float(np.nanpercentile(np.abs(wd["ORIG"][box_slices(-53, -43, -3, 7)]), 98)) * 86400
w_figure(wd, k50, "fig_w_shelf_daily_50m.png", f"W (daily, 15 Jan) at {m.gdepw_0[k50]:.0f} m, shelf", (-53, -43, -3, 7),
         vmax, vmax)

# ----------------------------------------------------------------------------- transects
metrics = {}
fig, axs = plt.subplots(4, 2, figsize=(13, 11), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.6]))
for c_, lat0 in enumerate([2.0, 5.0]):
    j = int(np.argmin(np.abs(m.gphit[:, 700] - lat0)))
    ii = np.where((m.glamt[j] >= -53) & (m.glamt[j] <= -45))[0]
    lon = m.glamt[j, ii]
    for cfg in cfgs:
        ut, vt = mean0[cfg]
        axs[0, c_].plot(lon, ut[j, ii], color=COLORS[cfg], label=LABELS[cfg], lw=1.2 if cfg != "ORIG" else 2)
        axs[1, c_].plot(lon, vt[j, ii], color=COLORS[cfg], lw=1.2 if cfg != "ORIG" else 2)
        axs[2, c_].plot(lon, np.hypot(ut[j, ii], vt[j, ii]), color=COLORS[cfg], lw=1.2 if cfg != "ORIG" else 2)
        if cfg in ("W2.0", "R3Ld") and os.path.exists(f"{HERE}/v1/cache/stats_{cfg}.npz"):
            z1 = np.load(f"{HERE}/v1/cache/stats_{cfg}.npz")
            u1, v1_ = to_t(z1["umean"][0].astype(float), z1["vmean"][0].astype(float), 0)
            for a, fld in [(axs[0, c_], u1), (axs[1, c_], v1_), (axs[2, c_], np.hypot(u1, v1_))]:
                a.plot(lon, fld[j, ii], color=COLORS[cfg], lw=1.0, ls="--", label=f"{cfg} v1 (component-wise)" if a is axs[0, c_] else None)
        # shelf-mean speed along the transect (H<200)
        sh = m.tmask[0, j, ii] & (m.H[j, ii] < 200)
        metrics[f"lat{lat0:.0f}N_{cfg}_mean_speed_shelf_H<200"] = float(np.nanmean(np.hypot(ut[j, ii], vt[j, ii])[sh]))
    axs[3, c_].fill_between(lon, -m.H[j, ii], 0, color="0.6"); axs[3, c_].set_ylim(-3000, 0)
    axs[3, c_].axhline(-200, color="k", lw=0.5)
    axs[0, c_].set_title(f"Jan-1993 mean surface velocity along {lat0:.0f}°N (solid v3, dashed v1)")
    axs[0, c_].set_ylabel("u (T pt) [m/s]"); axs[1, c_].set_ylabel("v (T pt) [m/s]")
    axs[2, c_].set_ylabel("speed [m/s]"); axs[3, c_].set_ylabel("−H [m]"); axs[3, c_].set_xlabel("longitude")
    for a in axs[:3, c_]:
        a.axhline(0, color="k", lw=0.4); a.grid(alpha=0.3)
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_shelf_transects.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------- depth levels (task 4)
show = [c for c in ["ORIG", "W2.0", "R2Ld"] if c in cfgs]
region = (-50, -44, 0, 8)
js, is_ = box_slices(*region)
fig, axs = plt.subplots(2, 5, figsize=(19, 8.5), squeeze=False)
for r, k in enumerate([30, 35]):
    f = {c: to_t(*monthly(c, k), k) for c in show}
    uo, vo = f["ORIG"]
    vmax = 0.3 if k == 30 else 0.25
    for c, cfg in enumerate(show):
        pc = panel(axs[r, c], js, is_, k, *f[cfg], vmax, q=3, qscale=vmax * 20, title=f"{LABELS[cfg]}, {m.gdept_0[k]:.0f} m")
    for c, cfg in enumerate(show[1:]):
        pd = panel(axs[r, 3 + c], js, is_, k, f[cfg][0] - uo, f[cfg][1] - vo, vmax / 2, diff=True, q=3,
                   qscale=vmax * 10, title=f"{cfg} − ORIG, {m.gdept_0[k]:.0f} m")
    fig.colorbar(pc, ax=axs[r, :3], shrink=0.8, pad=0.01, label="speed [m/s]")
    fig.colorbar(pd, ax=axs[r, 3:], shrink=0.8, pad=0.01, label="|Δu| [m/s]")
    # quantify: mean speed ratio over the slope (200<H<3000) vs deep (H>3000) in this box
    for cfg in show[1:]:
        so = np.hypot(uo, vo)[js, is_]; sf = np.hypot(*f[cfg])[js, is_]
        H = m.H[js, is_]
        for lab, sel in [("slope_H<3000", (H < 3000)), ("deep_H>3000", H >= 3000)]:
            ok = sel & np.isfinite(so)
            metrics[f"depth{m.gdept_0[k]:.0f}m_{cfg}_{lab}_speed_ratio"] = float(sf[ok].mean() / so[ok].mean())
fig.suptitle("Shelf break / slope off the Amazon (50–44°W, 0–8°N), Jan-1993 mean; land = T mask of the level")
fig.savefig(f"{HERE}/fig_depth_slope.png", dpi=130, bbox_inches="tight"); plt.close(fig)
metrics["W_boundary"] = wmet
json.dump(metrics, open(f"{HERE}/cache/metrics_maps.json", "w"), indent=1)
print(json.dumps(metrics, indent=1))
