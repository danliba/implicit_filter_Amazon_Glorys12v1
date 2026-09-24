"""
Task 1 (v3 outputs): empirical isotropic wavenumber spectra and transfer functions T(K) = E_filt/E_orig
for ORIG and the five filter configurations, in three land-free open-ocean boxes.
  set "UV": KE of U,V (vector div-rot filter) at T levels k=0 (0.5 m) and k=22 (109.7 m)
  set "W" : variance of W (scalar filter of the rigid-lid W) at W levels k=22 (100.6 m) and k=31 (495.7 m);
            ORIG W = data/variables_c/UVW/W_1993-01fc.nc
Run: python spectra.py [hann|tukey] [UV|W]

Method (per box, level, config, day):
  * U taken on its native U points, V on its native V points (no averaging to T).
  * Box = index rectangle (grid is lat-lon aligned here); dx = box-mean e1u (e1v), dy = box-mean e2u (e2v).
  * Remove a least-squares plane (mean + linear trend in i and j).
  * Multiply by a 2D Hann window (outer product), FFT, |F|^2 normalised by sum(w^2) so that
    the integral of the spectrum equals the (windowed) variance.
  * KE density 0.5(|u^|^2+|v^|^2) binned in annuli of width dK = max(dkx, dky) -> E(K) [m^3 s^-2].
  * Days averaged (31) before forming T(K) = <E_filt>/<E_orig>.
Theory: G(K) = 1/(1 + 0.5 l^2 K^2), power transfer G^2, l from the box mean (W: box-mean latitude,
R: box-mean Ld on T points). A discrete-operator version (annulus mean of 1/(1+0.5 l^2 Kd^2)^2 with
Kd^2 = (2/dx sin(kx dx/2))^2 + (2/dy sin(ky dy/2))^2, 5-point Laplacian symbol) is also stored.
Half-power wavelength: T = 0.5; theory lambda_1/2 = 2 pi l / sqrt(2(sqrt2-1)) = 6.90 l.

Run: python spectra.py  (login node, ~5-10 min). Output: spectra.npz, fig_ke_spectra.png, fig_transfer.png,
metrics_spectra.json.
"""
import json, sys
import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C
import cgrid_filter as cf

OUT = C.RUN + "/analysis/spectra_divergence"
BOXES = {  # name: (lon0, lon1, lat0, lat1)
    "WTA": (-50, -35, 5, 14),    # western tropical Atlantic (NBC retroflection / rings region, offshore)
    "STA": (-55, -35, 18, 28),   # subtropical gyre
    "EQA": (-28, -10, -4, 4),    # equatorial (east of Fernando de Noronha, which is land at 100 m)
}
import os
C.CONFIGS = os.environ.get("TEST_CFGS", ",".join(C.CONFIGS)).split(",")  # test override
CFGS = ["ORIG"] + C.CONFIGS
WIN = sys.argv[1] if len(sys.argv) > 1 else "hann"
SET = sys.argv[2] if len(sys.argv) > 2 else "UV"
VARS = {"UV": ["U", "V"], "W": ["W"]}[SET]
LEVELS = {"UV": [0, 22], "W": [22, 31]}[SET]
ZLEV = m_z = None  # set after mesh
HALF = np.sqrt(2 * (np.sqrt(2) - 1))   # K l at G^2 = 1/2

m = C.mesh()
LD = netCDF4.Dataset(C.RUN + "/rossby/rossby_radius_T.nc")["Ld"][:].filled(np.nan).astype(float)
ZLEV = m.gdept_0 if SET == "UV" else m.gdepw_0
QTY = "KE" if SET == "UV" else "W variance"
TAG = "" if SET == "UV" else "_W"
LT = {}   # l at T points exactly as in run_filter.py v3 (Rossby: Gaussian-smoothed, sigma = 6 cells)
for c in C.CONFIGS:
    if c.startswith("W"):
        LT[c] = cf.ell_T_F(m, "window", window_deg=float(c[1:]))[0]
    else:
        LT[c] = cf.ell_T_F(m, "rossby", ld_T=LD, mult=float(c[1]), smooth_sigma_cells=6.0)[0]


def window2d(ny, nx, kind):
    if kind == "hann":
        wy, wx = np.hanning(ny), np.hanning(nx)
    else:  # tukey alpha=0.5
        from scipy.signal.windows import tukey
        wy, wx = tukey(ny, 0.5), tukey(nx, 0.5)
    return np.outer(wy, wx)


def detrend_plane(f):
    """Remove least-squares plane a + b*i + c*j from each (t, ny, nx) slice."""
    nt, ny, nx = f.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    A = np.stack([np.ones(ny * nx), ii.ravel(), jj.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, f.reshape(nt, -1).T, rcond=None)
    return f - (A @ coef).T.reshape(nt, ny, nx)


def power2d(f, dx, dy, w):
    """|FFT|^2 density (t, ny, nx) with sum(P)*dkx*dky = mean(w^2 f^2)/mean(w^2) (Parseval)."""
    nt, ny, nx = f.shape
    F = np.fft.fft2(f * w[None], axes=(1, 2))
    dkx, dky = 2 * np.pi / (nx * dx), 2 * np.pi / (ny * dy)
    return np.abs(F) ** 2 / (nx * ny) ** 2 / np.mean(w ** 2) / (dkx * dky)


def radial(P, kx, ky, edges):
    """Annulus integral of density P over (kx,ky): E(K) = sum P dkx dky / dK."""
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    dkx, dky = abs(kx[1] - kx[0]), abs(ky[1] - ky[0])
    ib = np.digitize(K.ravel(), edges) - 1
    nb = len(edges) - 1
    ok = (ib >= 0) & (ib < nb)
    E = np.zeros(P.shape[:-2] + (nb,))
    Pf = P.reshape(P.shape[:-2] + (-1,))
    for b in range(nb):
        sel = ok & (ib == b)
        E[..., b] = Pf[..., sel].sum(-1) * dkx * dky / (edges[b + 1] - edges[b])
    return E


def ell_box(cfg, js, is_):
    """Box-mean, min, max of l [m] over the box (T points)."""
    l = LT[cfg][js, is_]
    return float(l.mean()), float(l.min()), float(l.max())


def half_power_lambda(lam, T):
    """Largest wavelength (scanning from long to short) where T crosses 0.5 (log-interp)."""
    o = np.argsort(lam)[::-1]
    lam, T = lam[o], T[o]
    for n in range(len(T) - 1):
        if T[n] >= 0.5 > T[n + 1]:
            x0, x1 = np.log(lam[n]), np.log(lam[n + 1])
            return float(np.exp(x0 + (0.5 - T[n]) * (x1 - x0) / (T[n + 1] - T[n])))
    return np.nan


res = {}
metrics = {}
for bname, box in BOXES.items():
    js, is_ = C.box_slices(*box, grid="T")
    ny, nx = int(js.stop - js.start), int(is_.stop - is_.start)
    dx = float(np.mean(0.5 * (m.e1u[js, is_] + m.e1v[js, is_])))
    dy = float(np.mean(0.5 * (m.e2u[js, is_] + m.e2v[js, is_])))
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx); ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    dK = min(abs(kx[1]), abs(ky[1]))            # fine bins (resolution of the longer box side)
    Kmax = min(np.pi / dx, np.pi / dy)
    edges = np.arange(0.5 * dK, Kmax + 1e-12, dK)
    while True:   # merge empty annuli (low K, anisotropic sampling) with their outer neighbour
        cnt = radial(np.ones((1, ny, nx)), kx, ky, edges)[0]
        if (cnt > 0).all():
            break
        edges = edges[np.r_[True, cnt > 0]] if cnt[-1] > 0 else edges[:-1]
    Kc = 0.5 * (edges[1:] + edges[:-1])
    w = window2d(ny, nx, WIN)
    Lmin = min(nx * dx, ny * dy)
    lat_mean = float(m.gphit[js, is_].mean())
    wet_ok = {k: bool(m.tmask[k, js.start - 1:js.stop + 1, is_.start - 1:is_.stop + 1].all()) for k in LEVELS}
    assert all(wet_ok.values()), (bname, wet_ok)
    metrics[bname] = dict(box_lonlat=box, ny=ny, nx=nx, dx_km=dx / 1e3, dy_km=dy / 1e3,
                          Lx_km=nx * dx / 1e3, Ly_km=ny * dy / 1e3, lat_mean=lat_mean,
                          wet_all_levels_checked=wet_ok, dy_range_km=[float(m.e2t[js, is_].min() / 1e3), float(m.e2t[js, is_].max() / 1e3)],
                          levels={})
    print(bname, ny, nx, dx, dy, wet_ok, flush=True)
    for k in LEVELS:
        E = {}
        for cfg in CFGS:
            Pk = 0.0
            for var in VARS:
                f = C.read(var, cfg, k=k, j=js, i=is_)
                assert np.isfinite(f).all(), (bname, k, cfg, var, "land/NaN in box")
                Pk = Pk + 0.5 * power2d(detrend_plane(f), dx, dy, w)
            E[cfg] = radial(Pk, kx, ky, edges).mean(0)   # 31-day mean
            print(f"  {bname} k={k} {cfg} KE(int)={np.sum(E[cfg] * np.diff(edges)):.4e}", flush=True)
        lev = {}
        dKb = np.diff(edges)
        KX, KY = np.meshgrid(kx, ky)
        Kd2 = (2 / dx * np.sin(KX * dx / 2)) ** 2 + (2 / dy * np.sin(KY * dy / 2)) ** 2
        for cfg in C.CONFIGS:
            lm, lmin, lmax = ell_box(cfg, js, is_)
            T = E[cfg] / E["ORIG"]
            Tth = 1.0 / (1 + 0.5 * lm ** 2 * Kc ** 2) ** 2
            Tdisc = radial((1.0 / (1 + 0.5 * lm ** 2 * Kd2) ** 2)[None], kx, ky, edges)[0] / \
                radial(np.ones((1, ny, nx)), kx, ky, edges)[0]
            lam = 2 * np.pi / Kc
            lh_emp = half_power_lambda(lam, T)
            lh_th = 2 * np.pi * lm / HALF
            # ratio emp/theory at lambda = 2 pi l
            i2 = np.argmin(abs(Kc - 1 / lm))
            # least-squares effective l: T_emp ~ 1/(1+l^2K^2/2)^2 over 60 km < lambda < Lmin/2
            from scipy.optimize import minimize_scalar
            sel = (lam > 60e3) & (lam < Lmin / 2)
            lfit = minimize_scalar(lambda L: np.sum((T[sel] - 1 / (1 + 0.5 * L ** 2 * Kc[sel] ** 2) ** 2) ** 2),
                                   bounds=(1e3, 500e3), method="bounded").x
            # KE retained fraction
            lev[cfg] = dict(l_mean_km=lm / 1e3, l_min_km=lmin / 1e3, l_max_km=lmax / 1e3,
                            lambda_2pil_km=2 * np.pi * lm / 1e3, box_3p5l_km=3.5 * lm / 1e3,
                            lambda_half_emp_km=lh_emp / 1e3, lambda_half_theory_km=lh_th / 1e3,
                            lambda_half_rel_dev=(lh_emp - lh_th) / lh_th,
                            max_abs_Tdisc_minus_Tth=float(np.abs(Tdisc - Tth)[lam < Lmin / 2].max()),
                            l_eff_fit_km=lfit / 1e3, l_eff_over_l=lfit / lm,
                            lambda_half_fit_km=2 * np.pi * lfit / HALF / 1e3,
                            T_emp_at_2pil=float(T[i2]), T_theory_at_2pil=float(Tth[i2]),
                            lambda_half_theory_binned_km=half_power_lambda(lam, Tth) / 1e3,
                            KE_fraction_retained=float((E[cfg] * dKb).sum() / (E["ORIG"] * dKb).sum()),
                            KE_fraction_expected_from_G2=float((E["ORIG"] * Tth * dKb).sum() / (E["ORIG"] * dKb).sum()))
            res[f"{bname}_{k}_{cfg}_T"] = T; res[f"{bname}_{k}_{cfg}_Tth"] = Tth; res[f"{bname}_{k}_{cfg}_Tdisc"] = Tdisc
        for cfg in CFGS:
            res[f"{bname}_{k}_{cfg}_E"] = E[cfg]
        res[f"{bname}_K"] = Kc
        metrics[bname]["levels"][f"k{k}_{ZLEV[k]:.1f}m"] = lev

np.savez(f"{OUT}/spectra{TAG}_{WIN}.npz", **res)
json.dump(metrics, open(f"{OUT}/metrics_spectra{TAG}_{WIN}.json", "w"), indent=1)

# ------------------------------------------------------------------ figures
if WIN != "hann":
    sys.exit()
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
for c, bname in enumerate(BOXES):
    Kc = res[f"{bname}_K"]
    for r, k in enumerate(LEVELS):
        ax = axs[r, c]
        for cfg in CFGS:
            ax.loglog(Kc / (2 * np.pi) * 1e3, res[f"{bname}_{k}_{cfg}_E"] * 2 * np.pi / 1e3,
                      color=C.COLORS[cfg], lw=2 if cfg == "ORIG" else 1.4, label=C.LABELS[cfg])
        if SET == "UV":
            # reference slopes anchored on ORIG at K ~ (150 km)^-1
            kk = Kc / (2 * np.pi) * 1e3
            E0 = res[f"{bname}_{k}_ORIG_E"] * 2 * np.pi / 1e3
            i0 = np.argmin(abs(kk - 1 / 150.))
            xr = np.array([kk[i0], kk[-1]])
            ax.loglog(xr, 3 * E0[i0] * (xr / xr[0]) ** (-3), "k--", lw=0.9)
            ax.loglog(xr, 3 * E0[i0] * (xr / xr[0]) ** (-5 / 3), "k:", lw=0.9)
            ax.text(xr[1], 3 * E0[i0] * (xr[1] / xr[0]) ** (-3), " $K^{-3}$", fontsize=9, va="center")
            ax.text(xr[1], 3 * E0[i0] * (xr[1] / xr[0]) ** (-5 / 3), " $K^{-5/3}$", fontsize=9, va="center")
        for cfg in C.CONFIGS:
            lm = metrics[bname]["levels"][f"k{k}_{ZLEV[k]:.1f}m"][cfg]["l_mean_km"]
            ax.axvline(1 / (2 * np.pi * lm), color=C.COLORS[cfg], lw=0.8, ls="--", alpha=0.7)
        ax.set_title(f"{bname} {BOXES[bname]}  z={ZLEV[k]:.0f} m", fontsize=11)
        ax.set_xlabel("1/λ  [cycles per km]"); ax.set_ylabel(f"E  [m² s⁻² / (cycle km⁻¹)]")
        ax.grid(which="both", alpha=0.25)
        sec = ax.secondary_xaxis("top", functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9)))
        sec.set_xlabel("λ [km]")
axs[0, 0].legend(fontsize=9, loc="lower left")
fig.suptitle(f"31-day-mean isotropic {QTY} spectra ({'U,V native points' if SET == 'UV' else 'W at W/T points'}, plane-detrended, 2D Hann). Dashed verticals: λ = 2πℓ (box-mean ℓ)", fontsize=12)
fig.savefig(f"{OUT}/fig{TAG}_spectra.png" if TAG else f"{OUT}/fig_ke_spectra.png", dpi=130)

fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True, sharey=True)
for c, bname in enumerate(BOXES):
    Kc = res[f"{bname}_K"]; lam = 2 * np.pi / Kc / 1e3
    for r, k in enumerate(LEVELS):
        ax = axs[r, c]
        levm = metrics[bname]["levels"][f"k{k}_{ZLEV[k]:.1f}m"]
        for cfg in C.CONFIGS:
            col = C.COLORS[cfg]
            ax.semilogx(lam, res[f"{bname}_{k}_{cfg}_T"], color=col, lw=1.8, label=f"{cfg} empirical")
            ax.semilogx(lam, res[f"{bname}_{k}_{cfg}_Tth"], color=col, lw=1.1, ls="--")
            ax.plot(levm[cfg]["lambda_2pil_km"], 1.03, marker="v", color=col, ms=8, clip_on=False)
            ax.plot(levm[cfg]["box_3p5l_km"], -0.03, marker="^", color=col, ms=8, clip_on=False)
        ax.axhline(0.5, color="grey", lw=0.6)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(lam.min(), lam.max())
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000]); ax.set_xticklabels(["20", "50", "100", "200", "500", "1000", "2000"])
        ax.set_title(f"{bname} {BOXES[bname]}  z={ZLEV[k]:.0f} m", fontsize=11)
        ax.set_xlabel("wavelength λ [km]"); ax.set_ylabel("T(K) = E_filt / E_orig")
        ax.grid(which="both", alpha=0.25)
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"{SET}: empirical power transfer (solid) vs theory G² with box-mean ℓ (dashed).  "
             "▼ λ = 2πℓ,  ▲ ℓ_box = 3.5ℓ", fontsize=12)
fig.savefig(f"{OUT}/fig{TAG}_transfer.png" if TAG else f"{OUT}/fig_transfer.png", dpi=130)
print("done")
