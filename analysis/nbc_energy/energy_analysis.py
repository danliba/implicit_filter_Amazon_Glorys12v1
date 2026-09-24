"""
Post-processing of data/energy_profiles.npz and data/energy_maps.npz (written by energy.py).

Depth ranges 0-200 m, 0-1000 m and full depth are formed from the level sums with a 1-D clipping
fraction per level, f_k = (thickness of [gdepw_k, gdepw_k + e3t_0,k] inside the range) / e3t_0,k.
(Exact except for partial bottom cells inside the one straddling level; negligible.)

Metrics (per range, per config; o = ORIG, f = filtered):
  KE_ratio            = KE_tot,f / KE_tot,o             KE_removed = 1 - KE_ratio
  EKE_removed         = 1 - EKE_f / EKE_o                (EKE = KE of daily anomalies from 31-day mean)
  KEmean_removed      = 1 - KE_mean,f / KE_mean,o         (KE of the 31-day mean flow)
  SS_frac             = 0.5<(u_o-u_f)^2+(v_o-v_f)^2> / KE_tot,o   (KE of the filtered-out residual)
  SSmean_frac         = residual KE of the monthly means / KE_mean,o
  SSeddy_frac         = (SS - SSmean) / EKE_o
  Note KE_removed != SS_frac: KE_o = KE_f + SS + <u_f.(u_o-u_f)>*2; the cross term vanishes only for
  a projection filter.
Also a sensitivity for the band 5S-25N (away from the halo-free northern edge at 30N).

Figures: fig_energy_profiles.png, fig_surface_speed_maps.png, fig_surface_eke_maps.png
Output: metrics_energy.json
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("OUTDIR", HERE + "/data")
FIG = os.environ.get("FIGDIR", HERE)
m = C.mesh()
P = np.load(f"{DATA}/energy_profiles.npz")
CFGS = list(P["cfgs"])
zt = m.gdept_0; zw = m.gdepw_0
e3t0 = np.diff(np.append(zw, zw[-1] + 2 * (zt[-1] - zw[-1])))


def frac(zmax):
    return np.clip(np.minimum(zw + e3t0, zmax) - zw, 0, None) / e3t0


def integrate(arr, zmax, rows):
    """arr (ncfg, nz, ny) -> (ncfg,) summed over rows and depth range."""
    return np.einsum("ckj,k->c", arr[..., rows], frac(zmax))


def metrics(rows):
    out = {}
    for rng, zmax in (("0-200m", 200.0), ("0-1000m", 1000.0), ("full", 1e5)):
        KE = integrate(P["KEtot"], zmax, rows); KM = integrate(P["KEmean"], zmax, rows)
        SS = integrate(P["SS"], zmax, rows); SM = integrate(P["SSmean"], zmax, rows)
        V = np.einsum("kj,k->", P["vol"][:, rows], frac(zmax))
        EKE = KE - KM
        d = {"ORIG_mean_KE_m2s2": float(KE[0] / V), "ORIG_mean_EKE_m2s2": float(EKE[0] / V),
             "ORIG_mean_KEmean_m2s2": float(KM[0] / V), "ORIG_EKE_over_KE": float(EKE[0] / KE[0])}
        for ic, c in enumerate(CFGS):
            if c == "ORIG":
                continue
            d[c] = dict(KE_ratio=float(KE[ic] / KE[0]), KE_removed=float(1 - KE[ic] / KE[0]),
                        EKE_ratio=float(EKE[ic] / EKE[0]), EKE_removed=float(1 - EKE[ic] / EKE[0]),
                        KEmean_removed=float(1 - KM[ic] / KM[0]),
                        SS_frac=float(SS[ic] / KE[0]), SSmean_frac=float(SM[ic] / KM[0]),
                        SSeddy_frac=float((SS[ic] - SM[ic]) / EKE[0]))
        out[rng] = d
    return out


def profiles_plot():
    rows = slice(None)
    KE = P["KEtot"].sum(-1); KM = P["KEmean"].sum(-1); SS = P["SS"].sum(-1)
    EKE = KE - KM
    ok = P["vol"].sum(-1) > 0
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, constrained_layout=True)
    V = P["vol"].sum(-1)
    axs[0].semilogx(KE[0][ok] / V[ok], zt[ok], "k", label="KE total")
    axs[0].semilogx(EKE[0][ok] / V[ok], zt[ok], "k--", label="EKE (daily anomalies)")
    axs[0].semilogx(KM[0][ok] / V[ok], zt[ok], "k:", label="KE of 31-day mean")
    axs[0].set_title("ORIG box-mean KE (m$^2$ s$^{-2}$)"); axs[0].legend(fontsize=8)
    for ic, c in enumerate(CFGS):
        if c == "ORIG":
            continue
        kw = dict(color=C.COLORS[c], label=C.LABELS[c])
        axs[1].plot(1 - KE[ic][ok] / KE[0][ok], zt[ok], **kw)
        axs[2].plot(1 - EKE[ic][ok] / EKE[0][ok], zt[ok], **kw)
        axs[3].plot(1 - KM[ic][ok] / KM[0][ok], zt[ok], **kw)
        axs[1].plot(SS[ic][ok] / KE[0][ok], zt[ok], ls=":", color=C.COLORS[c])
    axs[1].set_title("KE removed: 1 − KE$_f$/KE$_o$\n(dotted: residual KE / KE$_o$)")
    axs[2].set_title("EKE removed: 1 − EKE$_f$/EKE$_o$")
    axs[3].set_title("KE of 31-day mean removed")
    for ax in axs[1:]:
        ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.set_xlabel("fraction")
    axs[0].grid(alpha=0.3)
    axs[0].set_yscale("symlog", linthresh=100); axs[0].set_ylim(zt[ok].max() + 200, 0)
    axs[0].set_ylabel("depth (m)  [linear to 100 m, log below]")
    axs[2].legend(fontsize=8, loc="lower left")
    fig.suptitle("Kinetic energy removed by the filters vs depth, 70°W–30°W, 5°S–30°N, Jan 1993 (volume-weighted per level)")
    fig.savefig(f"{FIG}/fig_energy_profiles.png", dpi=130)
    plt.close(fig)


def maps_plot():
    M = np.load(f"{DATA}/energy_maps.npz")
    J0, J1, I0, I1 = M["box"]
    lon = m.glamt[J0:J1, I0:I1]; lat = m.gphit[J0:J1, I0:I1]
    land = ~m.tmask[0, J0:J1, I0:I1]
    cf = [c for c in CFGS if f"UMfull_{c}_k0" in M]
    # speed + quivers
    fig, axs = plt.subplots(2, 3, figsize=(17, 11), sharex=True, sharey=True, constrained_layout=True)
    sk = 18
    for ax, c in zip(axs.flat, cf):
        u = M[f"UMfull_{c}_k0"][J0:J1, I0:I1]; v = M[f"VMfull_{c}_k0"][J0:J1, I0:I1]
        sp = np.hypot(u, v)
        pc = ax.pcolormesh(lon, lat, sp, vmin=0, vmax=1.2, cmap="viridis", shading="auto")
        ax.contourf(lon, lat, land, [0.5, 1.5], colors="0.75")
        ax.quiver(lon[::sk, ::sk], lat[::sk, ::sk], u[::sk, ::sk], v[::sk, ::sk], scale=12, width=0.0022, color="w")
        ax.set_title(C.LABELS[c]); ax.set_aspect("equal")
    for ax in axs[:, 0]:
        ax.set_ylabel("latitude")
    for ax in axs[-1]:
        ax.set_xlabel("longitude")
    cb = fig.colorbar(pc, ax=axs, shrink=0.7); cb.set_label("speed of 31-day mean surface velocity (m/s)")
    fig.suptitle("Surface (0.5 m) 31-day mean speed, January 1993; arrows every 1.5°")
    fig.savefig(f"{FIG}/fig_surface_speed_maps.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(17, 11), sharex=True, sharey=True, constrained_layout=True)
    for ax, c in zip(axs.flat, cf):
        e = M[f"EKEmap_{c}_k0"]
        pc = ax.pcolormesh(lon, lat, np.clip(e, 1e-4, None), norm=LogNorm(1e-3, 0.3), cmap="magma", shading="auto")
        ax.contourf(lon, lat, land, [0.5, 1.5], colors="0.75")
        eo = M["EKEmap_ORIG_k0"]
        w = (m.e1t * m.e2t)[J0:J1, I0:I1] * ~land
        ratio = np.nansum(e * w) / np.nansum(eo * w)
        ax.set_title(f"{C.LABELS[c]}   (box EKE / ORIG = {ratio:.2f})"); ax.set_aspect("equal")
    cb = fig.colorbar(pc, ax=axs, shrink=0.7); cb.set_label("surface EKE of daily anomalies (m$^2$ s$^{-2}$)")
    fig.suptitle("Surface EKE, 0.5·<u'²+v'²>, u' = daily minus 31-day mean, January 1993")
    fig.savefig(f"{FIG}/fig_surface_eke_maps.png", dpi=130)
    plt.close(fig)


def main():
    lat = P["lat"]
    res = {"box_70W-30W_5S-30N": metrics(slice(None)),
           "band_5S-25N_sensitivity": metrics(np.where(lat <= 25.0)[0])}
    json.dump(res, open(f"{DATA if 'OUTDIR' in os.environ else HERE}/metrics_energy.json", "w"), indent=1)
    for rng, d in res["box_70W-30W_5S-30N"].items():
        print(rng, {k: v for k, v in d.items() if k.startswith("ORIG")})
        for c in CFGS[1:]:
            print("  ", c, {k: round(v, 3) for k, v in d[c].items()})
    profiles_plot()
    maps_plot()


if __name__ == "__main__":
    main()
