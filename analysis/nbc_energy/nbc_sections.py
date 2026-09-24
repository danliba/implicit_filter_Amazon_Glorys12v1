"""
North Brazil Current sections for ORIG and all filter configurations.

Section A (primary): meridional line on the U-grid column closest to 44 W (glamu = -44.04),
    from the first wet U point at the Brazilian coast (~2.4 S) to 6 N.
    Normal velocity = u (positive EASTWARD). The NBC is westward there (u < 0).
    NBC transport is reported POSITIVE WESTWARD:  T_A = - sum_{u<0, NBC region} u e2u e3u   [Sv]
Section B (secondary): zonal line on the V-grid row closest to 5 N (gphiv = 5.035),
    from the first wet V point at the coast (~52.5 W) to 46 W.
    Normal velocity = v (positive NORTHWARD). The NBC flows NW there (v > 0).
    NBC transport is reported POSITIVE NORTHWARD: T_B = + sum_{v>0, NBC region} v e1v e3v [Sv]
    A zonal line of V faces gives the exact C-grid volume flux through the section, which a
    coast-normal (diagonal) line would not (it would need u,v interpolation). The line crosses
    the NW-SE oriented Guiana shelf break at ~45 deg, so distances along the line are ~1.4x the
    coast-normal distances and v is the northward component (~0.7 |U| of a NW jet); the
    transport is unaffected.

NBC region (offshore limit): fixed for all configs and days, taken from the ORIG 31-day mean as the
first point offshore of the maximum of the 0-1000 m integrated westward (A) / northward (B)
transport per unit width where that integral changes sign.  A fixed 300 km limit is also reported.
Vertical integration uses partial-cell thicknesses clipped to [0, zmax] with zmax = 1000 m and 300 m.

Outputs: data/sections.npz, fig_section_A_44W.png, fig_section_B_5N.png, fig_nbc_transport_timeseries.png,
         metrics_sections.json
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{OUT}/data", exist_ok=True)
R = 6371.0e3
KMAX = 36          # levels 0..35 cover > 1000 m (gdepw_0[35] = 980 m, bottom 1151 m)
m = C.mesh()
CFGS = ["ORIG"] + C.available_configs()
zw = m.gdepw_0[:KMAX]
zt = m.gdept_0[:KMAX]


def clip_thickness(e3, zmax):
    """e3 (k, n) partial-cell thicknesses -> thickness inside [0, zmax]."""
    top = zw[:, None]
    return np.clip(np.minimum(top + e3, zmax) - top, 0, None) * (e3 > 0)


def section_geometry(name):
    if name == "A":
        i = int(np.argmin(np.abs(m.glamu[150] + 44.0)))
        jj = np.arange(80, 200)
        jj = jj[m.gphiu[jj, i] <= 6.0]
        e3 = m.e3u[:KMAX, jj, i]
        wet = e3[0] > 0
        jj = jj[np.argmax(wet):]          # start at first wet point from the coast
        e3 = m.e3u[:KMAX, jj, i]
        dx = m.e2u[jj, i]
        coord = m.gphiu[jj, i]
        dist = (coord - coord[0]) * np.pi / 180 * R / 1e3
        sl = dict(j=slice(jj[0], jj[-1] + 1), i=i)
        return dict(var="U", sl=sl, e3=e3, dx=dx, coord=coord, dist=dist, sign=-1.0,
                    lab="latitude", title="Section A: 44°W, u (m/s, >0 eastward)", i=i, jj=jj)
    j = int(np.argmin(np.abs(m.gphiv[:, 520] - 5.0)))
    ii = np.arange(500, 600)
    ii = ii[m.glamv[j, ii] <= -46.0]
    wet = m.e3v[0, j, ii] > 0
    ii = ii[np.argmax(wet):]
    e3 = m.e3v[:KMAX, j, ii]
    dx = m.e1v[j, ii]
    coord = m.glamv[j, ii]
    dist = (coord - coord[0]) * np.pi / 180 * R * np.cos(np.deg2rad(m.gphiv[j, ii[0]])) / 1e3
    sl = dict(j=j, i=slice(ii[0], ii[-1] + 1))
    return dict(var="V", sl=sl, e3=e3, dx=dx, coord=coord, dist=dist, sign=+1.0,
                lab="longitude", title="Section B: 5°N, v (m/s, >0 northward)", j=j, ii=ii)


def read_section(g, cfg):
    x = C.read(g["var"], cfg, k=slice(0, KMAX), **g["sl"])      # (t, k, n)
    x = np.where(g["e3"][None] > 0, x, np.nan)
    return x


def offshore_limit(g, xm):
    """Index (exclusive) of the NBC offshore limit from the ORIG mean section."""
    a = g["dx"] * clip_thickness(g["e3"], 1000.0)
    q = g["sign"] * np.nansum(xm * a / g["dx"], 0)     # m^2/s per unit width, NBC-positive
    icore = int(np.argmax(q))
    neg = np.where(q[icore:] <= 0)[0]
    return icore + (int(neg[0]) if neg.size else q.size - icore), q


def core3(xm, reg, kreg):
    """Robust core: max of the 3-point (along-section, NaN-aware) running mean of the NBC-signed
    monthly-mean velocity, within the NBC region and above 1000 m. Suppresses single-cell extremes
    at bathymetric notches (see section.md)."""
    a = np.where(np.isfinite(xm), xm, 0.0); c = np.isfinite(xm).astype(float)
    num = a.copy(); den = c.copy()
    num[:, 1:] += a[:, :-1]; den[:, 1:] += c[:, :-1]
    num[:, :-1] += a[:, 1:]; den[:, :-1] += c[:, 1:]
    sm = np.where(np.isfinite(xm), num / np.maximum(den, 1), np.nan)
    sub = np.nan_to_num(np.where(kreg[:, None] & reg[None, :], sm, -np.inf), nan=-np.inf)
    k0, n0 = np.unravel_index(np.argmax(sub), sub.shape)
    return float(sub[k0, n0]), float(zt[k0])


def jet_edges(prof, n0, vcore, d):
    """Contiguous range around index n0 where prof > 0.5 vcore; edges linearly interpolated (km)."""
    prof = np.nan_to_num(prof, nan=0.0); half = 0.5 * vcore
    lo = n0
    while lo > 0 and prof[lo - 1] > half:
        lo -= 1
    hi = n0
    while hi < prof.size - 1 and prof[hi + 1] > half:
        hi += 1
    dlo = d[lo] if lo == 0 else d[lo - 1] + (half - prof[lo - 1]) / (prof[lo] - prof[lo - 1]) * (d[lo] - d[lo - 1])
    dhi = d[hi] if hi == prof.size - 1 else d[hi] + (prof[hi] - half) / (prof[hi] - prof[hi + 1]) * (d[hi + 1] - d[hi])
    return dlo, dhi


def diagnostics(g, x, iend, kref=None):
    """x (t,k,n). Returns dict of NBC metrics."""
    s = g["sign"]
    reg = np.zeros(x.shape[-1], bool); reg[:iend] = True
    kreg = zt <= 1000.0
    out = {}
    for zmax in (1000.0, 300.0):
        a = g["dx"][None] * clip_thickness(g["e3"], zmax)       # (k, n) m^2
        xs = s * np.nan_to_num(x)                                  # NBC-positive
        tr_daily = np.sum(np.where(xs > 0, xs, 0) * a[None] * reg[None, None], (1, 2)) / 1e6
        net_daily = np.sum(xs * a[None] * reg[None, None], (1, 2)) / 1e6
        xm = xs.mean(0)
        tr_mean = np.sum(np.where(xm > 0, xm, 0) * a * reg[None]) / 1e6
        reg300 = g["dist"] <= 300.0
        tr300 = np.sum(np.where(xs > 0, xs, 0) * a[None] * reg300[None, None], (1, 2)) / 1e6
        tag = f"0_{int(zmax)}m"
        out[f"T_daily_{tag}"] = tr_daily.tolist()
        out[f"T_mean_of_daily_{tag}"] = float(tr_daily.mean())
        out[f"T_of_monthly_mean_{tag}"] = float(tr_mean)
        out[f"Tnet_mean_{tag}"] = float(net_daily.mean())
        out[f"T_300km_mean_{tag}"] = float(tr300.mean())
        out[f"T_std_daily_{tag}"] = float(tr_daily.std())
    xs = s * x
    xm = np.nanmean(xs, 0)
    sub = np.where(kreg[:, None] & reg[None, :], xm, -np.inf)
    sub = np.nan_to_num(sub, nan=-np.inf)
    k0, n0 = np.unravel_index(np.argmax(sub), sub.shape)
    vcore = float(sub[k0, n0])
    daily = np.nan_to_num(np.where(kreg[None, :, None] & reg[None, None, :], xs, -np.inf), nan=-np.inf)
    dmax = daily.reshape(daily.shape[0], -1).max(1)
    dlo, dhi = jet_edges(xm[k0], n0, vcore, g["dist"])
    d = g["dist"]
    # width at the ORIG core depth (fixed level) around the strongest point of that level in the region
    kr = k0 if kref is None else kref
    prof_r = np.nan_to_num(np.where(reg, xm[kr], -np.inf), nan=-np.inf)
    nr = int(np.argmax(prof_r))
    wlo, whi = jet_edges(xm[kr], nr, float(prof_r[nr]), g["dist"])
    out.update(jet_width_at_ORIG_core_depth_km=float(whi - wlo), vmax_at_ORIG_core_depth=float(prof_r[nr]),
               core_level=int(k0))
    # net transport over the whole section (coast to 6N / 46W), 0-1000 m
    a = g["dx"][None] * clip_thickness(g["e3"], 1000.0)
    out["core3_velocity_monthly_mean"], out["core3_depth_m"] = core3(xm, reg, kreg)
    out["Tnet_full_section_0_1000m"] = float(np.mean(np.sum(s * np.nan_to_num(x) * a[None], (1, 2))) / 1e6)
    out.update(core_velocity_monthly_mean=vcore, core_velocity_mean_daily_max=float(dmax.mean()),
               core_depth_m=float(zt[k0]), core_dist_km=float(d[n0]), core_coord=float(g["coord"][n0]),
               jet_width_km=float(dhi - dlo), jet_edges_km=[float(dlo), float(dhi)])
    return out


def main():
    res, store = {}, {}
    for name in ("A", "B"):
        g = section_geometry(name)
        data = {c: read_section(g, c) for c in CFGS}
        iend, q = offshore_limit(g, np.nanmean(data["ORIG"], 0))
        res[name] = dict(offshore_limit_index=iend, offshore_limit_km=float(g["dist"][iend - 1]),
                         offshore_limit_coord=float(g["coord"][iend - 1]),
                         coast_coord=float(g["coord"][0]), configs={})
        for c in CFGS:
            kref = None if c == "ORIG" else res[name]["configs"]["ORIG"]["core_level"]
            res[name]["configs"][c] = diagnostics(g, data[c], iend, kref)
        o = res[name]["configs"]["ORIG"]
        for c in CFGS:
            d = res[name]["configs"][c]
            d["pct_change"] = {k: 100.0 * (d[k] - o[k]) / o[k] for k in
                               ["core_velocity_monthly_mean", "core3_velocity_monthly_mean", "core_velocity_mean_daily_max", "jet_width_km",
                                "jet_width_at_ORIG_core_depth_km", "vmax_at_ORIG_core_depth", "Tnet_full_section_0_1000m",
                                "T_mean_of_daily_0_1000m", "T_mean_of_daily_0_300m", "T_300km_mean_0_1000m"]}
        store[name] = (g, {c: np.nanmean(data[c], 0) for c in CFGS}, iend)
        np.savez_compressed(f"{OUT}/data/section_{name}.npz", dist=g["dist"], coord=g["coord"], z=zt,
                            **{f"mean_{c}": np.nanmean(data[c], 0) for c in CFGS})
        print(name, "limit", res[name]["offshore_limit_coord"], res[name]["offshore_limit_km"], "km")
    json.dump(res, open(f"{OUT}/metrics_sections.json", "w"), indent=1)
    plot_sections(store, res)
    plot_timeseries(res)
    for name in ("A", "B"):
        print(f"--- section {name}")
        for c in CFGS:
            d = res[name]["configs"][c]
            print(f"{c:5s} vcore={d['core_velocity_monthly_mean']:.3f} dmax={d['core_velocity_mean_daily_max']:.3f} "
                  f"z={d['core_depth_m']:.0f} W={d['jet_width_km']:.0f} T1000={d['T_mean_of_daily_0_1000m']:.2f} "
                  f"T300={d['T_mean_of_daily_0_300m']:.2f} T300km={d['T_300km_mean_0_1000m']:.2f} "
                  f"net={d['Tnet_mean_0_1000m']:.2f} netfull={d['Tnet_full_section_0_1000m']:.2f} "
                  f"core3={d['core3_velocity_monthly_mean']:.3f}@{d['core3_depth_m']:.0f} Wref={d['jet_width_at_ORIG_core_depth_km']:.0f} vref={d['vmax_at_ORIG_core_depth']:.2f} pct={ {k: round(v, 1) for k, v in d['pct_change'].items()} }")


def plot_sections(store, res):
    for name, fname in (("A", "fig_section_A_44W.png"), ("B", "fig_section_B_5N.png")):
        g, means, iend = store[name]
        n = len(CFGS)
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
        lev = np.arange(-0.9, 0.91, 0.1)
        for ax, c in zip(axs.flat, CFGS):
            pc = ax.pcolormesh(g["coord"], zt, means[c], vmin=-0.9, vmax=0.9, cmap="RdBu_r", shading="nearest")
            ax.contour(g["coord"], zt, means[c], levels=[0], colors="k", linewidths=0.6)
            ax.axvline(g["coord"][iend - 1], color="0.3", ls="--", lw=1)
            d = res[name]["configs"][c]
            ax.plot(d["core_coord"], d["core_depth_m"], "k*", ms=9)
            ax.set_title(f"{C.LABELS[c]}\ncore {abs(d['core_velocity_monthly_mean']):.2f} m/s, "
                         f"T$_{{NBC}}$ {d['T_mean_of_daily_0_1000m']:.1f} Sv", fontsize=10)
            ax.set_facecolor("0.6")
        for ax in axs.flat[n:]:
            ax.set_visible(False)
        axs[0, 0].set_ylim(1000, 0)
        for ax in axs[:, 0]:
            ax.set_ylabel("depth (m)")
        for ax in axs[-1]:
            ax.set_xlabel(g["lab"] + " (°)")
        cb = fig.colorbar(pc, ax=axs, shrink=0.8)
        cb.set_label("31-day mean normal velocity (m/s)")
        fig.suptitle(g["title"] + " — Jan 1993 mean; dashed = NBC offshore limit; * = core; grey = land/below bottom")
        fig.savefig(f"{OUT}/{fname}", dpi=130)
        plt.close(fig)


def plot_timeseries(res):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    days = np.arange(1, 32)
    for ax, name, t in zip(axs, ("A", "B"), ("44°W: westward NBC transport (0–1000 m)",
                                            "5°N: northward NBC transport (0–1000 m)")):
        for c in CFGS:
            ax.plot(days, res[name]["configs"][c]["T_daily_0_1000m"], color=C.COLORS[c], label=C.LABELS[c],
                    lw=2 if c == "ORIG" else 1.3)
        ax.set_title(t); ax.set_xlabel("day of January 1993"); ax.set_ylabel("transport (Sv)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.savefig(f"{OUT}/fig_nbc_transport_timeseries.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
