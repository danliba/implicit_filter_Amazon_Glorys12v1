"""
Production run (version 3): filter GLORYS12v1 (U, V) jointly on the C-grid with the vector
div-rot implicit filter, and filter W (Fix_W rigid-lid product W_YYYY-MMfc.nc) as a scalar on the
W/T points with the same scale, all 50 levels x 31 days. No barotropic correction; W is NOT
recomputed from divergence (NEMO's continuity contains runoff / E-P / free-surface sources).
Diagnostics: continuity residual R = dw/dz + hdiv(u,v) for original and filtered fields.

usage: python run_filter.py CONFIG [--month 1993-01]
CONFIG in {W1.5, W2.0, W2.5, R2Ld, R3Ld}

Outputs (NEMO-style, same dims/coords as inputs) in output/CONFIG/:
  U_1993-01c_CONFIG.nc   vozocrtx (time_counter, deptht, y, x)
  V_1993-01c_CONFIG.nc   vomecrty (time_counter, deptht, y, x)
  W_1993-01c_CONFIG.nc   vovecrtz (time_counter, depthw, y, x)  filtered W (rigid lid, as input)
  diag_CONFIG.json       CG iterations, timings
  diag_CONFIG.npz        residual statistics per (t, level) and time-mean residual maps
"""
import argparse, os, sys, time, json
import numpy as np
import netCDF4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgrid_filter as cf

RUN = "/work/bk1450/b383184/Amazon/Mercator/implicit_run"
IN = "/work/bk1450/b383184/Amazon/Mercator/implicit_data"
W_IN = "/work/bk1450/b383184/Amazon/Mercator/data/variables_c/UVW/W_{month}fc.nc"
ROSSBY = RUN + "/rossby/rossby_radius_T.nc"

CONFIGS = {
    "W1.5": dict(kind="window", window_deg=1.5),
    "W2.0": dict(kind="window", window_deg=2.0),
    "W2.5": dict(kind="window", window_deg=2.5),
    "R2Ld": dict(kind="rossby", mult=2.0),
    "R3Ld": dict(kind="rossby", mult=3.0),
}


def ell_fields(mesh, cfg):
    c = CONFIGS[cfg]
    if c["kind"] == "window":
        return cf.ell_T_F(mesh, "window", window_deg=c["window_deg"])
    ld = netCDF4.Dataset(ROSSBY)["Ld"][:].filled(np.nan).astype(np.float64)
    assert np.isfinite(ld).all() and (ld > 0).all()
    return cf.ell_T_F(mesh, "rossby", ld_T=ld, mult=c["mult"], smooth_sigma_cells=6.0)


def global_attrs(cfg, var, btcorr):
    c = CONFIGS[cfg]
    a = {
        "title": f"GLORYS12v1 {var} implicitly filtered ({cfg})",
        "filter": "implicit filter (1 + gamma*M) u_bar = u, gamma=1/2, n=1, M = -grad(l^2 div) + curl(l^2 zeta) "
                  "(C-grid vector Laplacian, div-rot form); Danilov et al. 2023 JAMES; Nowak et al. 2025 GMD",
        "filter_version": "v3: U,V vector div-rot; W scalar",
        "filter_gamma": cf.GAMMA, "filter_order_n": 1, "box_factor_l_box_over_l": cf.BOX_FACTOR,
        "transfer_function": "G(K) = 1/(1 + l^2 K^2 / 2)",
        "staggering": "U and V filtered jointly on their native U/V points, level by level, no interpolation",
        "boundary_condition": "zero normal flow through land faces, free slip (zeta=0 at coastal F points); open edges of the regional grid treated as walls",
        "discretisation": "finite volume on NEMO scale factors incl. partial steps; symmetric positive definite system, batched Jacobi-PCG on GPU (CuPy), rel tol 1e-8",
        "w_treatment": "W (Fix_W rigid-lid product) filtered as a scalar on W/T points with the same l, using the T-cell operator "
                       "implied by the level-k vector filter: vol*w_bar + gamma*F W^-1 F^T (l^2 w_bar) = vol*w (no-flux at land); wet mask = tmask[k]; "
                       "sea-floor W points keep their original value 0; not recomputed from divergence",
        "config": cfg,
        "equivalent_implicit_filter_package_k": "k = 2/l (package solves I + 2(-L/k^2))",
        "code": RUN,
    }
    if c["kind"] == "window":
        a["window_deg"] = c["window_deg"]
        a["l_definition"] = "l = window_deg*pi/180*R*cos(lat)/3.5, R=6371 km"
    else:
        a["rossby_multiplier"] = c["mult"]
        a["l_definition"] = (f"l = {c['mult']}*Ld/3.5, Ld first-baroclinic Rossby radius (WKB) from GLORYS Jan-1993 T/S "
                             "(rossby/rossby_radius_T.nc), Gaussian-smoothed (sigma = 6 cells)")
    return a


def copy_coords(src, dst, zname, zout=None, zvalues=None):
    """Copy time/depth/x/y/nav_lon/nav_lat; optionally write the depth axis under a new name
    (netCDF4/HDF5 renameVariable on coordinate variables loses the data, so never rename)."""
    zout = zout or zname
    for d in ["time_counter", zname, "y", "x"]:
        dst.createDimension(zout if d == zname else d, None if d == "time_counter" else len(src.dimensions[d]))
    for v in ["time_counter", zname, "x", "y", "nav_lon", "nav_lat"]:
        if v in src.variables:
            s = src[v]
            dims = tuple(zout if dd == zname else dd for dd in s.dimensions)
            o = dst.createVariable(zout if v == zname else v, s.dtype, dims)
            o.setncatts({k: s.getncattr(k) for k in s.ncattrs() if k != "_FillValue"})
            o[:] = zvalues if (v == zname and zvalues is not None) else s[:]


def create_var(dst, name, dims, ny, nx, like=None, **atts):
    # One chunk per (time, level). The write loop below fills exactly one level at a
    # time, so a chunk spanning several levels forced netCDF to decompress, modify and
    # recompress the same chunk once per level: write cost ramped 6.3 -> 17.1 s with
    # period 5, matching the old chunksizes=(1, 5, ny, nx). One level per chunk makes
    # every write a single whole-chunk write, and also suits per-level reads in Parcels.
    v = dst.createVariable(name, "f4", dims, zlib=True, complevel=1, fill_value=np.float32(np.nan),
                           chunksizes=(1, 1, ny, nx))
    if like is not None:
        v.setncatts({k: like.getncattr(k) for k in like.ncattrs()
                     if k not in ("_FillValue", "_ChunkSizes", "missing_value")})
    v.setncatts(atts)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", choices=list(CONFIGS))
    p.add_argument("--month", default="1993-01")
    p.add_argument("--outdir", default=f"{RUN}/output")
    p.add_argument("--w-only", action="store_true", help="keep existing U,V outputs, (re)filter W and recompute diagnostics")
    a = p.parse_args()
    cfg = a.config
    od = f"{a.outdir}/{cfg}"
    os.makedirs(od, exist_ok=True)
    mesh = cf.CMesh()
    nz, ny, nx = mesh.nz, mesh.ny, mesh.nx
    lT, lF = ell_fields(mesh, cfg)
    T0 = time.time()
    if a.w_only and os.path.exists(f"{od}/diag_{cfg}.json"):
        old = json.load(open(f"{od}/diag_{cfg}.json"))
        os.replace(f"{od}/diag_{cfg}.json", f"{od}/diag_{cfg}_uvrun.json")
    diag = {"config": cfg, "version": "v3 UV vector div-rot, W with the consistent T-cell operator", "levels": {},
            "l_T_min_m": float(lT.min()), "l_T_max_m": float(lT.max())}

    su = netCDF4.Dataset(f"{IN}/U_{a.month}c.nc"); sv = netCDF4.Dataset(f"{IN}/V_{a.month}c.nc")
    sw = netCDF4.Dataset(W_IN.format(month=a.month))
    vu, vv, vw = su["vozocrtx"], sv["vomecrty"], sw["vovecrtz"]
    nt = vu.shape[0]
    assert np.allclose(su["time_counter"][:], sw["time_counter"][:]), "U and W time axes differ"

    # ------------------------------------------------------------ output files
    outs = {}
    if not a.w_only:
        for var, vname, src in [("U", "vozocrtx", su), ("V", "vomecrty", sv)]:
            ds = netCDF4.Dataset(f"{od}/{var}_{a.month}c_{cfg}.nc", "w", format="NETCDF4")
            copy_coords(src, ds, "deptht")
            ds.setncatts(global_attrs(cfg, vname, False))
            outs[var] = (ds, create_var(ds, vname, src[vname].dimensions, ny, nx, like=src[vname]))
    wpath = f"{od}/W_{a.month}c_{cfg}.nc"
    wtmp = wpath + ".tmp"
    dw = netCDF4.Dataset(wtmp, "w", format="NETCDF4")
    copy_coords(su, dw, "deptht", zout="depthw", zvalues=sw["depthw"][:])
    dw["depthw"].long_name = "Vertical W levels"
    dw.setncatts(global_attrs(cfg, "vovecrtz", False))
    wout = create_var(dw, "vovecrtz", ("time_counter", "depthw", "y", "x"), ny, nx, units="m s-1",
                      long_name="Vertical velocity (rigid lid, Fix_W), implicitly filtered with the T-cell operator of the U,V vector filter")

    # ------------------------------------------------------------ 1. U,V (vector) and W (scalar), level by level
    for k in range(nz):
        t0 = time.time()
        if a.w_only:
            u = vu[0:1, k].filled(np.nan).astype(np.float32); v = vv[0:1, k].filled(np.nan).astype(np.float32)
        else:
            u = vu[:, k].filled(np.nan).astype(np.float32); v = vv[:, k].filled(np.nan).astype(np.float32)
        w = vw[:, k].filled(np.nan).astype(np.float32)
        tr = time.time() - t0; t0 = time.time()
        its, n = 0, 0
        if not a.w_only:
            uo, vo, its, n = cf.filter_level_vector(mesh, k, u, v, lT, lF, backend="gpu", tol=1e-8)
        tuv = time.time() - t0; t0 = time.time()
        # W at the top of T-cell k: filtered with the T-cell operator implied by the level-k vector
        # filter (same no-flux walls, e3 weights, l^2 placement, edge treatment). Wet = tmask[k];
        # other points (sea floor W = 0, land NaN) keep their input value. Level 0 is 0 (rigid lid).
        wf = w.copy()
        itw = 0
        if k > 0 and mesh.tmask[k].any():
            wo_, itw = cf.filter_level_w(mesh, k, np.nan_to_num(w), ~np.isnan(u[0]), ~np.isnan(v[0]), lT, backend="gpu", tol=1e-8)
            wf = np.where(mesh.tmask[k][None], wo_, w)
        tw = time.time() - t0; t0 = time.time()
        if not a.w_only:
            outs["U"][1][:, k] = uo; outs["V"][1][:, k] = vo
        wout[:, k] = wf
        twr = time.time() - t0
        diag["levels"][str(k)] = dict(its_uv=int(its), its_w=int(itw), nwet_uv=int(n), read=tr, filt_uv=tuv, filt_w=tw, write=twr)
        print(f"[{cfg}] level {k:2d} nwet_uv={n:7d} its_uv={its:4d} its_w={itw:4d} read={tr:5.1f}s uv={tuv:6.2f}s w={tw:5.2f}s write={twr:5.1f}s", flush=True)
    for ds, _ in outs.values():
        ds.close()
    dw.close()
    os.replace(wtmp, wpath)
    diag["filter_time"] = time.time() - T0

    # ------------------------------------------------------------ 2. continuity residual diagnostics
    t_r = time.time()
    fu = netCDF4.Dataset(f"{od}/U_{a.month}c_{cfg}.nc")["vozocrtx"]
    fv = netCDF4.Dataset(f"{od}/V_{a.month}c_{cfg}.nc")["vomecrty"]
    fw = netCDF4.Dataset(f"{od}/W_{a.month}c_{cfg}.nc")["vovecrtz"]
    keys = ["R_rms_filt", "R_rms_orig", "R_max_filt", "R_max_orig", "hdiv_rms_filt", "hdiv_rms_orig",
            "dwdz_rms_filt", "dwdz_rms_orig", "w_rms_filt", "w_rms_orig"]
    stats = {kk: np.zeros((nt, nz)) for kk in keys}
    kmaps = [0, 10, 22, 33]
    Rmean_f = np.zeros((len(kmaps), ny, nx)); Rmean_o = np.zeros((len(kmaps), ny, nx))
    Rsurf_f = np.zeros((nt, ny, nx), np.float32); Rsurf_o = np.zeros((nt, ny, nx), np.float32)
    inner = np.zeros((ny, nx), bool); inner[2:-2, 2:-2] = True
    for t in range(nt):
        res = {}
        for tag, (U_, V_, W_) in {"filt": (fu[t], fv[t], fw[t]), "orig": (vu[t], vv[t], vw[t])}.items():
            U_ = U_.filled(np.nan).astype(np.float64); V_ = V_.filled(np.nan).astype(np.float64); W_ = W_.filled(np.nan).astype(np.float64)
            R = cf.continuity_residual(mesh, U_, V_, W_)
            hd = cf.horizontal_divergence_e3(mesh, U_, V_) / np.where(mesh.e3t > 0, mesh.e3t, 1.0)
            dwdz = R - hd
            for k in range(nz):
                m = mesh.tmask[k] & inner
                if m.sum() == 0:
                    continue
                stats[f"R_rms_{tag}"][t, k] = np.sqrt(np.mean(R[k][m] ** 2))
                stats[f"R_max_{tag}"][t, k] = np.abs(R[k][m]).max()
                stats[f"hdiv_rms_{tag}"][t, k] = np.sqrt(np.mean(hd[k][m] ** 2))
                stats[f"dwdz_rms_{tag}"][t, k] = np.sqrt(np.mean(dwdz[k][m] ** 2))
                stats[f"w_rms_{tag}"][t, k] = np.sqrt(np.nanmean(W_[k][m] ** 2))
            res[tag] = R
        Rsurf_f[t] = res["filt"][0]; Rsurf_o[t] = res["orig"][0]
        for i, k in enumerate(kmaps):
            Rmean_f[i] += np.nan_to_num(res["filt"][k]) / nt; Rmean_o[i] += np.nan_to_num(res["orig"][k]) / nt
        print(f"[{cfg}] t={t:2d} residual rms (surface layer) filt/orig = {stats['R_rms_filt'][t,0]:.2e}/{stats['R_rms_orig'][t,0]:.2e}"
              f"  (hdiv rms {stats['hdiv_rms_filt'][t,0]:.2e}/{stats['hdiv_rms_orig'][t,0]:.2e});"
              f" 100 m R filt/orig {stats['R_rms_filt'][t,10]:.2e}/{stats['R_rms_orig'][t,10]:.2e}", flush=True)
    diag["residual_time"] = time.time() - t_r
    diag["total_time"] = time.time() - T0
    np.savez(f"{od}/diag_{cfg}.npz", **stats, kmaps=np.array(kmaps), Rmean_filt=Rmean_f.astype(np.float32),
             Rmean_orig=Rmean_o.astype(np.float32), Rsurf_filt=Rsurf_f, Rsurf_orig=Rsurf_o)
    json.dump(diag, open(f"{od}/diag_{cfg}.json", "w"), indent=1)
    print(f"[{cfg}] DONE total {diag['total_time']/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
