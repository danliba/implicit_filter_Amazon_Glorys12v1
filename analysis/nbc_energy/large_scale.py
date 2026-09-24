"""
Large-scale signal retention (proxy for "seasonal" signal; only January 1993 exists, so no seasonal
cycle can be computed) and NBC ring treatment.

(i)  31-day-mean circulation at 0.5 m (k=0) and 92 m (k=21), T points, for sub-regions
       NBC retroflection  52W-42W, 3N-10N
       NECC               40W-20W, 4N-10N    (outside the 70-30W box to the east, but it is the NECC band)
       NEC                60W-30W, 10N-20N
     Vector pattern correlation (centred, area weighted):
       r = sum w [(u_o-<u_o>)(u_f-<u_f>) + (v_o-<v_o>)(v_f-<v_f>)] / sqrt(sum w |u_o-<u_o>|^2 * sum w |u_f-<u_f>|^2)
     RMS vector difference sqrt(<|u_o-u_f|^2>) and its ratio to the RMS of |u_o|; area-mean u retained.
(ii) Large scales: NaN-aware 121x121-point running box (~10 deg) on the full model domain applied to
     the 31-day means of ORIG and each filter; correlation, relative RMS difference in 70W-30W, 5S-30N.
(iii) NBC rings in ORIG from daily surface relative vorticity zeta = dv/dx - du/dy and Okubo-Weiss
     OW = s_n^2 + s_s^2 - zeta^2 (T points, centred differences). Ring = connected region with
     OW < -0.2 sigma_OW(ORIG, region), zeta < 0, equivalent diameter > 100 km, centroid 5-15N, 62-44W.
     For 3 dates the ORIG ring mask is applied to each config: mean zeta/f, min zeta/f,
     max speed within 1.5 R_eq of the centroid, and whether the filtered field still has an
     OW-detected anticyclone overlapping the ORIG ring (with the ORIG threshold).
Figures: fig_largescale_maps.png, fig_rings.png. Output: metrics_largescale.json
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("OUTDIR", HERE + "/data")
FIG = os.environ.get("FIGDIR", HERE)
m = C.mesh()
M = np.load(f"{DATA}/energy_maps.npz")
CFGS = [c for c in ["ORIG"] + C.CONFIGS if f"UMfull_{c}_k0" in M]
K100 = int(M["K100"])
AREA = m.e1t * m.e2t
LON, LAT = m.glamt, m.gphit
REGIONS = {"retroflection_52W-42W_3N-10N": (-52, -42, 3, 10), "NECC_40W-20W_4N-10N": (-40, -20, 4, 10),
           "NEC_60W-30W_10N-20N": (-60, -30, 10, 20), "box_70W-30W_5S-30N": (-70, -30, -5, 30)}


def sel(reg, k):
    x0, x1, y0, y1 = reg
    return (LON >= x0) & (LON <= x1) & (LAT >= y0) & (LAT <= y1) & m.tmask[k]


def compare(uo, vo, uf, vf, mask):
    w = AREA[mask]; uo, vo, uf, vf = uo[mask], vo[mask], uf[mask], vf[mask]
    ok = np.isfinite(uo) & np.isfinite(uf) & np.isfinite(vo) & np.isfinite(vf)
    w, uo, vo, uf, vf = w[ok], uo[ok], vo[ok], uf[ok], vf[ok]
    W = w.sum()
    a = [x - np.sum(w * x) / W for x in (uo, vo, uf, vf)]
    r = np.sum(w * (a[0] * a[2] + a[1] * a[3])) / np.sqrt(np.sum(w * (a[0] ** 2 + a[1] ** 2)) * np.sum(w * (a[2] ** 2 + a[3] ** 2)))
    rmsd = np.sqrt(np.sum(w * ((uo - uf) ** 2 + (vo - vf) ** 2)) / W)
    rmso = np.sqrt(np.sum(w * (uo ** 2 + vo ** 2)) / W)
    return dict(pattern_corr=float(r), rms_diff_ms=float(rmsd), rel_rms_diff=float(rmsd / rmso),
                mean_u_orig=float(np.sum(w * uo) / W), mean_u_filt=float(np.sum(w * uf) / W),
                mean_v_orig=float(np.sum(w * vo) / W), mean_v_filt=float(np.sum(w * vf) / W))


def box_smooth(x, n=121):
    ok = np.isfinite(x)
    s = ndimage.uniform_filter(np.where(ok, x, 0.0), n, mode="constant")
    c = ndimage.uniform_filter(ok.astype(float), n, mode="constant")
    return np.where(ok & (c > 0.05), s / np.maximum(c, 1e-12), np.nan)


def part_i_ii():
    out = {"circulation": {}, "large_scale_10deg": {}}
    for k in (0, K100):
        tag = f"z{m.gdept_0[k]:.0f}m"
        uo, vo = M[f"UMfull_ORIG_k{k}"].astype(float), M[f"VMfull_ORIG_k{k}"].astype(float)
        uos, vos = box_smooth(uo), box_smooth(vo)
        for c in CFGS[1:]:
            uf, vf = M[f"UMfull_{c}_k{k}"].astype(float), M[f"VMfull_{c}_k{k}"].astype(float)
            out["circulation"].setdefault(tag, {})[c] = {rn: compare(uo, vo, uf, vf, sel(r, k)) for rn, r in REGIONS.items()}
            ufs, vfs = box_smooth(uf), box_smooth(vf)
            out["large_scale_10deg"].setdefault(tag, {})[c] = compare(uos, vos, ufs, vfs, sel(REGIONS["box_70W-30W_5S-30N"], k))
            # how much of the filtered-out mean flow is large scale
            du, dv = uo - uf, vo - vf
            dus, dvs = box_smooth(du), box_smooth(dv)
            mk = sel(REGIONS["box_70W-30W_5S-30N"], k) & np.isfinite(dus)
            w = AREA[mk]
            out["large_scale_10deg"][tag][c]["removed_mean_KE_in_large_scales_frac"] = float(
                np.sum(w * (dus[mk] ** 2 + dvs[mk] ** 2)) / np.sum(w * (np.nan_to_num(du[mk]) ** 2 + np.nan_to_num(dv[mk]) ** 2)))
        if k == 0:
            smooth0 = (uos, vos, {c: (box_smooth(M[f"UMfull_{c}_k0"].astype(float)), box_smooth(M[f"VMfull_{c}_k0"].astype(float))) for c in CFGS[1:]})
    return out, smooth0


def vort_ow(u, v, J0, I0, ny, nx):
    """u,v native C-grid (t, ny+1, nx+1) incl. SW halo -> zeta, OW at T points of the box."""
    ut, vt = C.uv_to_t(u, v)
    ut = ut[..., 1:, 1:]; vt = vt[..., 1:, 1:]
    e1 = m.e1t[J0:J0 + ny, I0:I0 + nx]; e2 = m.e2t[J0:J0 + ny, I0:I0 + nx]
    dudx = np.gradient(ut, axis=-1) / e1; dudy = np.gradient(ut, axis=-2) / e2
    dvdx = np.gradient(vt, axis=-1) / e1; dvdy = np.gradient(vt, axis=-2) / e2
    z = dvdx - dudy
    ow = (dudx - dvdy) ** 2 + (dvdx + dudy) ** 2 - z ** 2
    return ut, vt, z, ow


def part_iii():
    D = np.load(f"{DATA}/surface_daily_uv.npz")
    J0, J1, I0, I1 = D["box"]; ny, nx = J1 - J0, I1 - I0
    lon = LON[J0:J1, I0:I1]; lat = LAT[J0:J1, I0:I1]
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(lat))
    wet = m.tmask[0, J0:J1, I0:I1]
    area = AREA[J0:J1, I0:I1]
    fields = {}
    for c in CFGS:
        ut, vt, z, ow = vort_ow(D[f"U0_{c}_k0"].astype(float), D[f"V0_{c}_k0"].astype(float), J0, I0, ny, nx)
        fields[c] = (ut, vt, np.where(wet, z, np.nan), np.where(wet, ow, np.nan))
    reg = (lon >= -62) & (lon <= -44) & (lat >= 5) & (lat <= 15) & wet
    ow0 = fields["ORIG"][3]
    thr = -0.2 * np.nanstd(ow0[:, reg])

    def detect(z, ow, t):
        lab, n = ndimage.label(np.nan_to_num(ow[t], nan=0) < thr)
        rings = []
        for ic in range(1, n + 1):
            mk = lab == ic
            if not np.any(mk & reg):
                continue
            A = area[mk].sum(); deq = 2 * np.sqrt(A / np.pi) / 1e3
            zz = z[t][mk]
            if deq < 100 or np.nanmean(zz) >= 0:
                continue
            cx = np.sum(lon[mk] * area[mk]) / A; cy = np.sum(lat[mk] * area[mk]) / A
            if not (-62 <= cx <= -44 and 5 <= cy <= 15):
                continue
            rings.append(dict(mask=mk, deq_km=deq, lon=cx, lat=cy))
        return rings

    nt = fields["ORIG"][0].shape[0]
    count = {c: [len(detect(fields[c][2], fields[c][3], t)) for t in range(nt)] for c in CFGS}
    # pick 3 dates with the largest ORIG rings, spread through the month
    # dates and first-guess ring positions chosen from the ORIG vorticity maps (a ring recently shed
    # and drifting NW, the same ring two weeks later, and a new ring forming at the retroflection)
    dates = [0, 14, 29]
    guess = {0: (-54.0, 9.0), 14: (-57.0, 9.5), 29: (-49.0, 7.0)}
    res = {"OW_threshold_s-2": float(thr), "daily_ring_count_mean": {c: float(np.mean(v)) for c, v in count.items()},
           "dates": {}}
    picks = []
    for t in dates:
        rs = [r for r in detect(fields["ORIG"][2], fields["ORIG"][3], t) if r["deq_km"] > 150]
        print("ORIG rings day", t + 1, [(round(r["lon"], 1), round(r["lat"], 1), round(r["deq_km"])) for r in rs])
        rs = sorted(rs, key=lambda r: np.hypot(r["lon"] - guess[t][0], r["lat"] - guess[t][1]))
        if not rs:
            continue
        r0 = rs[0]
        picks.append((t, r0))
        dist = np.hypot((lon - r0["lon"]) * np.cos(np.deg2rad(r0["lat"])), lat - r0["lat"]) * 111.2
        circ = (dist <= 0.75 * r0["deq_km"]) & wet
        entry = dict(center_lon=r0["lon"], center_lat=r0["lat"], deq_km=r0["deq_km"], configs={})
        for c in CFGS:
            ut, vt, z, ow = fields[c]
            zf = (z[t] / f)
            filt_rings = detect(z, ow, t)
            overl = [fr for fr in filt_rings if np.sum(fr["mask"] & r0["mask"]) > 0.3 * r0["mask"].sum()]
            entry["configs"][c] = dict(mean_zeta_over_f_in_ring=float(np.nanmean(zf[r0["mask"]])),
                                       min_zeta_over_f=float(np.nanmin(zf[r0["mask"]])),
                                       max_speed_ms=float(np.nanmax(np.hypot(ut[t], vt[t])[circ])),
                                       KE_in_circle=float(np.nansum(0.5 * (ut[t] ** 2 + vt[t] ** 2)[circ] * area[circ])),
                                       detected=bool(overl),
                                       detected_deq_km=float(overl[0]["deq_km"]) if overl else None)
        o = entry["configs"]["ORIG"]
        for c in CFGS:
            e = entry["configs"][c]
            e["mean_zeta_ratio"] = e["mean_zeta_over_f_in_ring"] / o["mean_zeta_over_f_in_ring"]
            e["max_speed_ratio"] = e["max_speed_ms"] / o["max_speed_ms"]
            e["KE_circle_ratio"] = e["KE_in_circle"] / o["KE_in_circle"]
        res["dates"][f"1993-01-{t + 1:02d}"] = entry
    # figure: rows = dates, cols = configs
    fig, axs = plt.subplots(len(picks), len(CFGS), figsize=(3.2 * len(CFGS), 3.1 * len(picks)),
                            sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for row, (t, r0) in enumerate(picks):
        for col, c in enumerate(CFGS):
            ax = axs[row, col]
            ut, vt, z, ow = fields[c]
            pc = ax.pcolormesh(lon, lat, z[t] / f, vmin=-0.6, vmax=0.6, cmap="RdBu_r", shading="auto")
            ax.contour(lon, lat, r0["mask"], [0.5], colors="k", linewidths=1.0)
            ax.contour(lon, lat, np.nan_to_num(ow[t], nan=0) < thr, [0.5], colors="lime", linewidths=0.6)
            ax.contourf(lon, lat, ~wet, [0.5, 1.5], colors="0.7")
            sk = 12
            ax.quiver(lon[::sk, ::sk], lat[::sk, ::sk], ut[t, ::sk, ::sk], vt[t, ::sk, ::sk], scale=15, width=0.003)
            e = res["dates"][f"1993-01-{t + 1:02d}"]["configs"][c]
            ax.set_title(f"{c} day {t + 1}\nζ/f ×{e['mean_zeta_ratio']:.2f}, KE ×{e['KE_circle_ratio']:.2f}, "
                         f"{'det' if e['detected'] else 'lost'}", fontsize=8)
            ax.set_xlim(-62, -40); ax.set_ylim(2, 16); ax.set_aspect("equal")
    cb = fig.colorbar(pc, ax=axs, shrink=0.6); cb.set_label("surface ζ/f")
    fig.suptitle("NBC rings: daily surface ζ/f; black = ORIG ring (OW<−0.2σ, ζ<0); green = OW-detected cores in each field; "
                 "ratios relative to ORIG", fontsize=10)
    fig.savefig(f"{FIG}/fig_rings.png", dpi=130)
    plt.close(fig)
    return res


def maps_figure(smooth0):
    uos, vos, sm = smooth0
    rows = [("ORIG", M["UMfull_ORIG_k0"], M["VMfull_ORIG_k0"])] + [(c, M[f"UMfull_{c}_k0"], M[f"VMfull_{c}_k0"]) for c in CFGS[1:]]
    fig, axs = plt.subplots(2, len(rows), figsize=(3.3 * len(rows), 7), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    mk = ~m.tmask[0]
    for col, (c, u, v) in enumerate(rows):
        ax = axs[0, col]
        pc = ax.pcolormesh(LON, LAT, u, vmin=-0.8, vmax=0.8, cmap="RdBu_r", shading="auto")
        ax.contourf(LON, LAT, mk, [0.5, 1.5], colors="0.7")
        ax.set_title(f"{c}: mean u, 0.5 m", fontsize=9)
        us = uos if c == "ORIG" else sm[c][0]
        ax = axs[1, col]
        pc2 = ax.pcolormesh(LON, LAT, us, vmin=-0.25, vmax=0.25, cmap="RdBu_r", shading="auto")
        ax.contourf(LON, LAT, mk, [0.5, 1.5], colors="0.7")
        ax.set_title(f"{c}: 10° box-smoothed u", fontsize=9)
    for ax in axs.flat:
        ax.set_xlim(-70, -20); ax.set_ylim(-5, 30); ax.set_aspect("equal")
    fig.colorbar(pc, ax=axs[0], shrink=0.8, label="u (m/s)")
    fig.colorbar(pc2, ax=axs[1], shrink=0.8, label="u (m/s)")
    fig.suptitle("31-day mean zonal surface velocity (top: NBC retroflection, NECC, NEC) and its >~10° part (bottom)")
    fig.savefig(f"{FIG}/fig_largescale_maps.png", dpi=130)
    plt.close(fig)


def main():
    out, smooth0 = part_i_ii()
    out["rings"] = part_iii()
    out["note"] = ("Only January 1993 is available: no seasonal cycle can be computed; the 31-day mean and "
                   "10-degree smoothed fields are used as proxies for the large-scale / low-frequency signal.")
    json.dump(out, open(f"{DATA if 'OUTDIR' in os.environ else HERE}/metrics_largescale.json", "w"), indent=1)
    maps_figure(smooth0)
    for tag, d in out["circulation"].items():
        for c, dd in d.items():
            print(tag, c, {rn: (round(v["pattern_corr"], 3), round(v["rel_rms_diff"], 3)) for rn, v in dd.items()})
    for tag, d in out["large_scale_10deg"].items():
        for c, dd in d.items():
            print("LS", tag, c, {k: round(v, 3) for k, v in dd.items()})
    print("ring counts", out["rings"]["daily_ring_count_mean"])
    for dt, e in out["rings"]["dates"].items():
        print(dt, round(e["center_lon"], 1), round(e["center_lat"], 1), round(e["deq_km"]),
              {c: (round(v["mean_zeta_ratio"], 2), round(v["KE_circle_ratio"], 2), v["detected"]) for c, v in e["configs"].items()})


if __name__ == "__main__":
    main()
