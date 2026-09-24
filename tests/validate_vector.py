"""Validate the div-rot vector filter (v2): transfer function, commutation with divergence,
vertical velocity for one full 3D day, with/without barotropic correction, vs component-wise v1."""
import sys, time, json
import numpy as np, scipy.sparse as sp, netCDF4
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
import cgrid_filter as cf
mesh = cf.CMesh(); ny, nx, nz = mesh.ny, mesh.nx, mesh.nz
res = {}

if False:
    # 1. transfer function: sinusoids in u and v, all wet, no land, constant l
    lT = np.full((ny, nx), 222e3 / 3.5); lF = lT.copy()
    wet = np.ones((ny, nx), bool)
    x = np.cumsum(mesh.e1u, 1); y = np.cumsum(mesh.e2v, 0)
    lams = [50e3, 100e3, 200e3, 400e3, 800e3]
    U = np.stack([np.cos(2*np.pi*x/l) for l in lams] + [np.cos(2*np.pi*y/l) for l in lams])
    V = np.stack([np.cos(2*np.pi*y/l) for l in lams] + [np.cos(2*np.pi*x/l) for l in lams])
    t0 = time.time()
    import copy
    flat = copy.copy(mesh); flat.tmask = np.ones_like(mesh.tmask); flat.e3t = np.ones_like(mesh.e3t)
    flat.e3u = np.ones_like(mesh.e3u); flat.e3v = np.ones_like(mesh.e3v)
    uo, vo, its, n = cf.filter_level_vector(flat, 0, U, V, lT, lF, tol=1e-10)
    jc, ic = slice(150, 350), slice(300, 960)
    gains = [float((uo[i][jc, ic]*U[i][jc, ic]).sum()/(U[i][jc, ic]**2).sum()) for i in range(10)]
    gv = [float((vo[i][jc, ic]*V[i][jc, ic]).sum()/(V[i][jc, ic]**2).sum()) for i in range(10)]
    th = [1/(1+0.5*(2*np.pi/l)**2*(222e3/3.5)**2) for l in lams]
    print("transfer its", its, "time", time.time()-t0)
    for i, l in enumerate(lams):
        print(f"  lam {l/1e3:4.0f} km: u(x-wave) {gains[i]:.3f} u(y-wave) {gains[5+i]:.3f} v(y) {gv[i]:.3f} v(x) {gv[5+i]:.3f} theory {th[i]:.3f}")
    res["transfer"] = dict(lams=lams, u_xwave=gains[:5], u_ywave=gains[5:], v_ywave=gv[:5], v_xwave=gv[5:], theory=th)
    
# 2. full 3D day (t=10), W2.0 window, vector filter all levels
dsu = netCDF4.Dataset(cf.MESH_DIR.replace("data/", "implicit_data/") + "U_1993-01c.nc")["vozocrtx"]
dsv = netCDF4.Dataset(cf.MESH_DIR.replace("data/", "implicit_data/") + "V_1993-01c.nc")["vomecrty"]
t = 10
u3 = dsu[t].filled(np.nan).astype(np.float64); v3 = dsv[t].filled(np.nan).astype(np.float64)
lT, lF = cf.ell_T_F(mesh, "window", window_deg=2.0)
uf = np.full_like(u3, np.nan); vf = np.full_like(v3, np.nan)
comm = []
t0 = time.time()
for k in range(nz):
    out = cf.filter_level_vector(mesh, k, u3[k][None], v3[k][None], lT, lF, tol=1e-10, return_ops=True)
    if out[3] == 0:
        continue
    uo, vo, its, n, ops = out
    uf[k] = uo[0]; vf[k] = vo[0]
    iu, iv, Wd, K, FT, it = ops
    if k in (0, 22):
        xo = np.concatenate([u3[k].ravel()[iu], v3[k].ravel()[iv]])
        xf = np.concatenate([uf[k].ravel()[iu], vf[k].ravel()[iv]])
        vol = (mesh.e1t*mesh.e2t*np.where(mesh.e3t[k] > 0, mesh.e3t[k], 1)).ravel()[it]
        Do = (FT @ xo)/vol; Df = (FT @ xf)/vol
        lam = (lT**2/(mesh.e1t*mesh.e2t*np.where(mesh.e3t[k] > 0, mesh.e3t[k], 1))).ravel()[it]
        # (I + gamma D W^-1 F^T diag(lamT*vol)/... ) : check  Df + gamma*(1/vol) F W^-1 F^T (lam*vol*Df) = Do
        lhs = Df + 0.5*(FT @ ((FT.T @ (lam*vol*Df)) / Wd))/vol
        # interior points only (drop regional-edge T cells where a face is outside the domain)
        J, I = np.unravel_index(it, (ny, nx)); inner = (J > 0) & (J < ny-1) & (I > 0) & (I < nx-1)
        rel = np.sqrt(((lhs-Do)[inner]**2).sum()/(Do[inner]**2).sum())
        comm.append((k, float(rel), float(np.sqrt((Df[inner]**2).mean())/np.sqrt((Do[inner]**2).mean()))))
        print(f"level {k}: commutation residual {rel:.2e}, rms div filt/orig {comm[-1][2]:.3f}, its {its}")
print("3D filter time", time.time()-t0)
res["commutation"] = comm

# 3. vertical velocity: orig, v1 (component-wise output), v2, v2 + barotropic correction
v1u = netCDF4.Dataset("/work/bk1450/b383184/Amazon/Mercator/implicit_run/output_v1_componentwise/W2.0/U_1993-01c_W2.0.nc")["vozocrtx"][t].filled(np.nan)
v1v = netCDF4.Dataset("/work/bk1450/b383184/Amazon/Mercator/implicit_run/output_v1_componentwise/W2.0/V_1993-01c_W2.0.nc")["vomecrty"][t].filled(np.nan)
de3o = cf.horizontal_divergence_e3(mesh, u3, v3)
colflux_o = de3o.sum(0) * mesh.e1t * mesh.e2t
t0 = time.time()
uw3 = ~np.isnan(u3); vw3 = ~np.isnan(v3)
cc = cf.ColumnConsistency(mesh, uw3, vw3, lT, backend="gpu")
ufc, vfc, phi, target, info = cc.apply(uf, vf, u3, v3)
print("barotropic correction time", time.time()-t0, info)
ducorr = np.nanmax(np.abs(ufc-uf)); dvcorr = np.nanmax(np.abs(vfc-vf))
rmscorr = float(np.sqrt(np.nanmean((ufc[0]-uf[0])**2)))
for nm, dd in [("du", np.abs(ufc[0]-uf[0])), ("dv", np.abs(vfc[0]-vf[0]))]:
    f = np.nanargmax(dd); j, i = np.unravel_index(f, dd.shape)
    print(nm, "max at", j, i, mesh.glamt[j, i], mesh.gphit[j, i], "H", mesh.H[j, i], float(dd[j, i]), "p99.9", float(np.nanpercentile(dd, 99.9)))
print(f"correction max |du| {ducorr:.3e} |dv| {dvcorr:.3e} rms du surface {rmscorr:.3e}")
wo = cf.w_from_continuity(mesh, u3, v3)[0]
# coast distance
from scipy.ndimage import distance_transform_edt
dist = distance_transform_edt(mesh.tmask[0])
open_ = (dist > 15); coast = (dist <= 3)
lev = {"surf": 0, "100m": int(np.argmin(abs(mesh.gdepw_0-100))), "500m": int(np.argmin(abs(mesh.gdepw_0-500))), "1000m": int(np.argmin(abs(mesh.gdepw_0-1000)))}
js, is_ = slice(np.argmin(abs(mesh.gphit[:, 600]+5)), np.argmin(abs(mesh.gphit[:, 600]-29))), slice(np.argmin(abs(mesh.glamt[200]+70)), np.argmin(abs(mesh.glamt[200]+30)))
wres = {}
for name, (uu, vv) in {"ORIG": (u3, v3), "v1_componentwise": (v1u, v1v), "v2_vector": (uf, vf), "v2_vector_btcorr": (ufc, vfc)}.items():
    w = cf.w_from_continuity(mesh, uu, vv)[0]
    d = {}
    for ln, k in lev.items():
        m = mesh.tmask[k][js, is_]
        for reg, rm in [("all", m), ("open", m & open_[js, is_]), ("coast", m & coast[js, is_])]:
            d[f"{ln}_{reg}"] = float(np.sqrt(np.nanmean(w[k][js, is_][rm]**2)))
    wres[name] = d
    print(name, {k: f"{v:.2e}" for k, v in d.items()})
res["w_rms"] = wres
res["btcorr"] = dict(max_du=float(ducorr), max_dv=float(dvcorr), rms_du_surf=rmscorr)
np.savez_compressed("/work/bk1450/b383184/Amazon/Mercator/implicit_run/tests/validate_vector_day10.npz",
                    w_orig=wo[[0, lev["100m"], lev["1000m"]]].astype(np.float32),
                    w_v2=cf.w_from_continuity(mesh, uf, vf)[0][[0, lev["100m"], lev["1000m"]]].astype(np.float32),
                    w_v2c=cf.w_from_continuity(mesh, ufc, vfc)[0][[0, lev["100m"], lev["1000m"]]].astype(np.float32),
                    w_v1=cf.w_from_continuity(mesh, v1u, v1v)[0][[0, lev["100m"], lev["1000m"]]].astype(np.float32),
                    phi=phi.astype(np.float32), levs=np.array(list(lev.values())))
json.dump(res, open("/work/bk1450/b383184/Amazon/Mercator/implicit_run/tests/validate_vector.json", "w"), indent=1)
