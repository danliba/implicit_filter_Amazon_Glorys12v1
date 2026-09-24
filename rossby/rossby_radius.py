"""First-baroclinic Rossby deformation radius on the GLORYS12v1 regional T-grid.

Usage: python rossby_radius.py
See README.md for the method.
"""
import os
import time
import numpy as np
import netCDF4
import xarray as xr
from scipy.ndimage import gaussian_filter, distance_transform_edt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal

BASE = "/work/bk1450/b383184/Amazon/Mercator"
TFILE = f"{BASE}/implicit_data/T_1993-01c.nc"
SFILE = f"{BASE}/implicit_data/S_1993-01c.nc"
HGR = f"{BASE}/data/Hgr_cmesh.nc"
ZGR = f"{BASE}/data/Zgr_cmesh2.nc"
OUTDIR = f"{BASE}/implicit_run/rossby"
OUT = f"{OUTDIR}/rossby_radius_T.nc"
MEANFILE = f"{OUTDIR}/TS_mean_1993-01.npz"   # cache of monthly means

G = 9.81
RHO0 = 1026.0
OMEGA = 7.292115e-5
REARTH = 6371e3
N2MIN = 1e-8
HMIN = 1000.0
SMOOTH_SIGMA = 5.0


# ----------------------------------------------------------------------------
# Equation of state: Jackett & McDougall (1995), potential temperature,
# practical salinity, depth in m (~dbar), as implemented in NEMO (eos_insitu).
# ----------------------------------------------------------------------------
def rho_jmd95(t, s, z):
    t = np.asarray(t, np.float64)
    s = np.asarray(s, np.float64)
    z = np.asarray(z, np.float64)
    sr = np.sqrt(np.abs(s))
    r1 = ((((6.536332e-9 * t - 1.120083e-6) * t + 1.001685e-4) * t
           - 9.095290e-3) * t + 6.793952e-2) * t + 999.842594
    r2 = (((5.3875e-9 * t - 8.2467e-7) * t + 7.6438e-5) * t - 4.0899e-3) * t + 0.824493
    r3 = (-1.6546e-6 * t + 1.0227e-4) * t - 5.72466e-3
    r4 = 4.8314e-4
    rhop = (r4 * s + r3 * sr + r2) * s + r1
    e = (-3.508914e-8 * t - 1.248266e-8) * t - 2.595994e-6
    bw = (1.296821e-6 * t - 5.782165e-9) * t + 1.045941e-4
    b = bw + e * s
    d = -2.042967e-2
    c = (-7.267926e-5 * t + 2.598241e-3) * t + 0.1571896
    aw = ((5.939910e-6 * t + 2.512549e-3) * t - 0.1028859) * t - 4.721788
    a = (d * sr + c) * s + aw
    b1 = (-0.1909078 * t + 7.390729) * t - 55.87545
    a1 = ((2.326469e-3 * t + 1.553190) * t - 65.00517) * t + 1044.077
    kw = (((-1.361629e-4 * t - 1.852732e-2) * t - 30.41638) * t + 2098.925) * t + 190925.6
    k0 = (b1 * sr + a1) * s + kw
    return rhop / (1.0 - z / (k0 - z * (a - z * b)))


def laplace_fill(field, unknown):
    """Solve del^2 u = 0 on unknown points with known values as Dirichlet
    data and zero-flux at the domain boundary (5-point stencil, sparse direct)."""
    ny, nx = field.shape
    idx = -np.ones((ny, nx), np.int64)
    uj, ui = np.nonzero(unknown)
    n = uj.size
    idx[uj, ui] = np.arange(n)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    diag = np.zeros(n)
    for dj, di in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nj, ni = uj + dj, ui + di
        inside = (nj >= 0) & (nj < ny) & (ni >= 0) & (ni < nx)
        diag[inside] += 1.0
        p = np.nonzero(inside)[0]
        nbr = idx[nj[p], ni[p]]
        unk = nbr >= 0
        rows.append(p[unk]); cols.append(nbr[unk]); vals.append(-np.ones(unk.sum()))
        rhs[p[~unk]] += field[nj[p[~unk]], ni[p[~unk]]]
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    out = field.copy()
    out[uj, ui] = spla.spsolve(A, rhs)
    return out


def monthly_mean(fname, var):
    nc = netCDF4.Dataset(fname)
    v = nc.variables[var]
    v.set_auto_mask(False)
    nt = v.shape[0]
    acc = np.zeros(v.shape[1:], np.float64)
    for it in range(nt):
        acc += v[it].astype(np.float64)
        print(f"  {var} day {it + 1}/{nt}", flush=True)
    nc.close()
    return (acc / nt).astype(np.float32)


def main():
    t0 = time.time()
    # check value JMD95: S=35.5, theta=3, p=3000 dbar -> 1041.83267
    print("EOS check (expect ~1041.83):", rho_jmd95(3.0, 35.5, 3000.0))

    # --- meshes
    hg = xr.open_dataset(HGR).squeeze("t")
    zg = xr.open_dataset(ZGR).squeeze("t")
    glamt = hg.glamt.values.astype(np.float64)
    gphit = hg.gphit.values.astype(np.float64)
    mb = zg.mbathy.values.astype(np.int64)
    gdept0 = zg.gdept_0.values
    gdepw0 = zg.gdepw_0.values
    e3t0 = zg.e3t_0.values
    e3tps = zg.e3t_ps.values
    gdept_bot = zg.deptht.values            # depth of bottom (partial) T point
    nz = gdept0.size
    ny, nx = mb.shape
    k = np.arange(nz)[:, None, None]
    wetT = k < mb[None]                      # (z,y,x)
    kb = np.clip(mb - 1, 0, nz - 1)
    e3t3 = np.where(k < (mb - 1)[None], e3t0[:, None, None], 0.0).sum(0)
    H = np.where(mb > 0, e3t3 + e3tps, 0.0)   # H = sum of wet e3t
    print("max |H - (gdepw0[mb-1]+e3t_ps)| =",
          np.abs(H - np.where(mb > 0, gdepw0[kb] + e3tps, 0.0)).max())

    # T-point depths (3D, partial bottom)
    gdept3 = np.broadcast_to(gdept0[:, None, None], (nz, ny, nx)).copy()
    sel = mb > 0
    jj, ii = np.nonzero(sel)
    gdept3[kb[sel], jj, ii] = gdept_bot[sel]

    # --- monthly means
    if os.path.exists(MEANFILE):
        d = np.load(MEANFILE)
        T, S = d["T"], d["S"]
    else:
        T = monthly_mean(TFILE, "votemper")
        S = monthly_mean(SFILE, "vosaline")
        np.savez(MEANFILE, T=T, S=S)
    print(f"means done {time.time() - t0:.0f}s")
    assert np.all(np.isfinite(T[wetT])) and np.all(np.isfinite(S[wetT]))

    # --- N^2 at W levels k=1..nz-1 (between T levels k-1 and k),
    # densities of both parcels evaluated at the w-level depth (local reference)
    N2 = np.full((nz, ny, nx), np.nan)
    dzw = np.full((nz, ny, nx), np.nan)
    for kk in range(1, nz):
        wet = kk < mb                        # both T levels wet
        if not wet.any():
            continue
        zw = gdepw0[kk]
        ra = rho_jmd95(T[kk - 1][wet], S[kk - 1][wet], zw)
        rb = rho_jmd95(T[kk][wet], S[kk][wet], zw)
        dz = gdept3[kk][wet] - gdept3[kk - 1][wet]
        n2 = G / RHO0 * (rb - ra) / dz
        N2[kk][wet] = np.maximum(n2, N2MIN)
        dzw[kk][wet] = dz
    print(f"N2 done {time.time() - t0:.0f}s")

    # --- WKB: c1 = (1/pi) int_{-H}^0 N dz
    # N at interior w levels integrated over dz between T points, plus the
    # top half-cell (0..gdept[0]) with N(w1) and bottom (gdept_bot..H) with N(w_mb-1)
    Nw = np.sqrt(N2)
    integ = np.nansum(Nw * dzw, axis=0)
    ok = mb >= 2
    N_top = Nw[1]
    kbw = np.clip(mb - 1, 1, nz - 1)
    N_bot = np.take_along_axis(Nw, kbw[None], 0)[0]
    integ = integ + np.where(ok, N_top * gdept0[0] + N_bot * np.maximum(H - gdept_bot, 0.0), 0.0)
    c1 = np.where(ok, integ / np.pi, np.nan)

    # --- Rossby radius (Chelton et al. 1998 blend)
    lat = np.deg2rad(gphit)
    f = 2 * OMEGA * np.sin(lat)
    beta = 2 * OMEGA * np.cos(lat) / REARTH

    def ld_from_c(c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.fmin(c / np.abs(f), np.sqrt(c / (2 * beta)))

    deep = ok & (H >= HMIN)
    Ld_raw = np.where(deep, ld_from_c(c1), np.nan)

    # --- fill shelf/land: Laplacian (harmonic) extrapolation of c1 from deep
    # water (Neumann at domain edges), Gaussian smoothing of the filled field,
    # blended into the raw field over 2*sigma cells; Ld then from filled c1
    # and local latitude (keeps the f/beta latitude structure over land).
    c_fill = laplace_fill(np.where(deep, c1, np.nan), ~deep)
    c_sm = gaussian_filter(c_fill, SMOOTH_SIGMA, mode="nearest")
    dist = distance_transform_edt(deep)      # cells to nearest masked point
    w = np.clip(1.0 - (dist - 1.0) / (2 * SMOOTH_SIGMA), 0.0, 1.0)
    w = np.where(deep, w, 1.0)
    c_final = (1 - w) * c_fill + w * c_sm
    Ld = ld_from_c(c_final)
    assert np.all(np.isfinite(Ld)) and np.all(Ld > 0)
    print(f"Ld range {Ld.min() / 1e3:.1f} - {Ld.max() / 1e3:.1f} km, "
          f"filled c1 range {c_final.min():.2f} - {c_final.max():.2f} m/s")

    # --- sanity table: Atlantic zonal medians
    lines = ["lat   median Ld_raw [km]  median Ld [km]  median c1 [m/s]  (lon -60..-20)"]
    atl = (glamt >= -60) & (glamt <= -20)
    for la in [0, 5, 10, 15, 20, 25, 30]:
        band = atl & (np.abs(gphit - la) <= 0.5)
        lr = np.nanmedian(np.where(band, Ld_raw, np.nan)) / 1e3
        lf = np.median(Ld[band]) / 1e3
        cc = np.nanmedian(np.where(band, c1, np.nan))
        lines.append(f"{la:3d}N  {lr:10.1f}  {lf:14.1f}  {cc:12.2f}")
    table = "\n".join(lines)
    print(table)

    # --- WKB vs Sturm-Liouville eigen check on random deep columns
    # w'' + (N^2/c^2) w = 0, w=0 at z=0 and z=-H; unknowns at w levels 1..mb-1
    rng = np.random.default_rng(0)
    jd, id_ = np.nonzero(deep)
    pick = rng.choice(jd.size, size=min(3000, jd.size), replace=False)
    ratios, lats_s = [], []
    for p in pick:
        j, i = jd[p], id_[p]
        m = mb[j, i]
        n2 = N2[1:m, j, i]
        zt = gdept3[:m, j, i]
        dzt = np.diff(zt)                       # spacing between T points (=dzw)
        # w-level depths
        zw = np.concatenate([[0.0], gdepw0[1:m], [H[j, i]]])
        ht = np.diff(zw)                        # T-cell thicknesses (m of them)
        # second difference at w level kk (1..m-1): neighbours w_{kk-1}, w_{kk+1}
        # (w_{kk+1}-w_kk)/ht[kk] - (w_kk-w_{kk-1})/ht[kk-1] = -lam*N2*dzt
        diag = 1.0 / ht[1:m] + 1.0 / ht[0:m - 1]
        off = -1.0 / ht[1:m - 1]
        Md = n2 * dzt
        s = 1.0 / np.sqrt(Md)
        a = diag * s * s
        b = off * s[:-1] * s[1:]
        lam = eigh_tridiagonal(a, b, eigvals_only=True, select="i",
                               select_range=(0, 0))[0]
        ce = 1.0 / np.sqrt(lam)
        ratios.append(c1[j, i] / ce)
        lats_s.append(gphit[j, i])
    ratios = np.array(ratios)
    lats_s = np.array(lats_s)
    eig_txt = (f"WKB/eigen c1 ratio over {ratios.size} random deep columns: "
               f"median {np.median(ratios):.3f}, mean {ratios.mean():.3f}, "
               f"5-95% [{np.percentile(ratios, 5):.3f}, {np.percentile(ratios, 95):.3f}]")
    print(eig_txt)
    with open(f"{OUTDIR}/sanity_check.txt", "w") as fh:
        fh.write(table + "\n\n" + eig_txt + "\n")

    # --- save NetCDF
    ds = xr.Dataset(
        {
            "Ld": (("y", "x"), Ld.astype(np.float32),
                   {"units": "m", "long_name": "first baroclinic Rossby radius (filled, smoothed)"}),
            "c1": (("y", "x"), c1.astype(np.float32),
                   {"units": "m s-1", "long_name": "first baroclinic gravity wave speed (WKB), NaN where mbathy<2"}),
            "Ld_raw": (("y", "x"), Ld_raw.astype(np.float32),
                       {"units": "m", "long_name": "Rossby radius before fill, NaN where H<1000 m or land"}),
            "H": (("y", "x"), H.astype(np.float32),
                  {"units": "m", "long_name": "bottom depth incl. partial cell"}),
            "glamt": (("y", "x"), glamt.astype(np.float32), {"units": "degrees_east"}),
            "gphit": (("y", "x"), gphit.astype(np.float32), {"units": "degrees_north"}),
        },
        attrs={
            "title": "First-baroclinic Rossby deformation radius, GLORYS12v1 ORCA12 regional T-grid",
            "source": "monthly mean of daily votemper/vosaline, January 1993",
            "method_eos": "Jackett & McDougall (1995) EOS (NEMO form), theta/SP, depth as pressure; "
                          "N2 at W levels from locally referenced density, clipped to >= 1e-8 s-2",
            "method_c1": "WKB c1 = (1/pi) int_{-H}^0 N dz with partial bottom cell",
            "method_Ld": "Ld = min(c1/|f|, sqrt(c1/(2 beta))) (Chelton et al. 1998)",
            "method_fill": f"c1 at points with H<{HMIN:.0f} m and land filled by Laplacian extrapolation from "
                           f"deep water, Gaussian smoothed (sigma={SMOOTH_SIGMA} cells) and blended into raw c1 "
                           f"over {2 * SMOOTH_SIGMA:.0f} cells next to the mask; Ld recomputed from filled c1 "
                           "with local f, beta",
            "wkb_vs_eigen": eig_txt,
            "script": f"{OUTDIR}/rossby_radius.py",
        },
    )
    if os.path.exists(OUT):
        os.remove(OUT)
    ds.to_netcdf(OUT)
    print("wrote", OUT)

    # --- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 5.5))
    pc = ax.pcolormesh(glamt, gphit, Ld / 1e3, shading="auto", cmap="viridis",
                       vmin=0, vmax=250)
    ax.contour(glamt, gphit, mb > 0, levels=[0.5], colors="w", linewidths=0.5)
    fig.colorbar(pc, ax=ax, label="Ld [km]")
    ax.set_title("First-baroclinic Rossby radius (WKB, filled), GLORYS12 Jan 1993")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.savefig(f"{OUTDIR}/rossby_radius_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    lat_bins = np.arange(-10, 30.01, 0.5)
    lc = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    med_raw, med_fill = [], []
    for a0, a1 in zip(lat_bins[:-1], lat_bins[1:]):
        band = atl & (gphit >= a0) & (gphit < a1)
        v = Ld_raw[band]
        v = v[np.isfinite(v)]
        med_raw.append(np.median(v) / 1e3 if v.size else np.nan)
        med_fill.append(np.median(Ld[band]) / 1e3 if band.any() else np.nan)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lc, med_raw, label="Ld_raw (deep, H>=1000 m)")
    ax.plot(lc, med_fill, "--", label="Ld (filled)")
    ax.set_yscale("log")
    ax.set_xlabel("latitude"); ax.set_ylabel("zonal median Ld [km], 60W-20W")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.savefig(f"{OUTDIR}/rossby_radius_zonal_median.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
