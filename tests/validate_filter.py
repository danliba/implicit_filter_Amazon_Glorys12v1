"""GPU validation: synthetic sinusoids, package cross-check, symmetry/conservation, timing."""
import sys, time, math, json
import numpy as np
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
import cgrid_filter as cf
import netCDF4

mesh = cf.CMesh()
res = {}
L_box = {"W1.5": 1.5, "W2.0": 2.0, "W2.5": 2.5}

# ---------- 1. synthetic sinusoids on the real U-grid metrics, no land, constant l
# distance along x (zonal) and along y (meridional) from scale factors
ny, nx = mesh.ny, mesh.nx
x = np.cumsum(mesh.e1u, axis=1); y = np.cumsum(mesh.e2u, axis=0)
lams = [50e3, 100e3, 200e3, 400e3, 800e3]
wet = np.ones((ny, nx), bool)
jc = slice(150, 350); ic = slice(300, 960)   # interior away from open edges
out = {}
for Lbox_km in [167.0, 222.0, 278.0]:
    ell = np.full((ny, nx), Lbox_km * 1e3 / cf.BOX_FACTOR)
    idx, area, K = cf.build_operator(mesh, "U", wet, ell)
    S = cf.ImplicitSolver(area, K, backend="gpu", tol=1e-10)
    fields = []
    for lam in lams:
        fields.append(np.cos(2 * np.pi * x / lam))
    for lam in lams:
        fields.append(np.cos(2 * np.pi * y / lam))
    B = np.stack([f.ravel() for f in fields], 1)
    t0 = time.time(); X, it = S.solve(B); X = S.xp.asnumpy(X); dt = time.time() - t0
    gains = []
    for n, f in enumerate(fields):
        xo = X[:, n].reshape(ny, nx)[jc, ic]; fo = f[jc, ic]
        gains.append(float((xo * fo).sum() / (fo * fo).sum()))
    ellv = Lbox_km * 1e3 / cf.BOX_FACTOR
    theory = [1 / (1 + cf.GAMMA * (2 * np.pi / l) ** 2 * ellv ** 2) for l in lams]
    boxg = [float(np.sinc(Lbox_km * 1e3 / l)) for l in lams]  # 1D box transfer sin(kL/2)/(kL/2)
    out[Lbox_km] = dict(zonal=gains[:5], merid=gains[5:], theory=theory, box1d=boxg, its=it, secs=dt)
    print(f"L_box={Lbox_km} km l={ellv/1e3:.1f} km its={it} t={dt:.2f}s")
    for i, l in enumerate(lams):
        print(f"   lambda={l/1e3:5.0f} km  gain_x={gains[i]:.3f} gain_y={gains[5+i]:.3f} theory={theory[i]:.3f} box1d={boxg[i]:.3f}")
res["sinusoids"] = {str(k): v for k, v in out.items()}

# ---------- 2. cross-check with implicit_filter package solver (constant l, real surface mask)
from implicit_filter import LatLonFilter
u0 = netCDF4.Dataset("/work/bk1450/b383184/Amazon/Mercator/implicit_data/U_1993-01c.nc")["vozocrtx"][0, 0].filled(np.nan).astype(float)
wet = ~np.isnan(u0)
ellc = 222e3 / cf.BOX_FACTOR
ell = np.full((ny, nx), ellc)
idx, area, K = cf.build_operator(mesh, "U", wet, ell / ellc)   # K with l=1 -> pure metric fluxes
Lmat = -(K.multiply(1.0 / area[:, None])).tocoo()                 # package convention: L = -A^-1 K
f = LatLonFilter()
f._ss, f._ii, f._jj = Lmat.data, Lmat.row, Lmat.col
f._e2d = idx.size; f._nx = idx.size; f._ny = 1
f.set_backend("gpu")
from cupyx.scipy.sparse.linalg import cg as _cg
# implicit_filter 1.3.1 passes tol= to cupyx cg, renamed rtol in CuPy>=14 -> patch for the cross-check
f.cg = lambda A, b, x0, tol, maxiter, M: _cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M) if False else _cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
data = u0.ravel()[idx]
t0 = time.time(); pk = f._compute(1, float(2.0 / ellc), data, tol=1e-10); tpk = time.time() - t0
mine, it, _ = cf.filter_level(mesh, "U", u0[None], ell, tol=1e-10)
mine = mine[0].ravel()[idx]
rel = np.abs(pk - mine).max() / np.abs(data).max()
print(f"package cross-check (k=2/l): max|diff|/max|u| = {rel:.2e}; package cupyx cg {tpk:.2f}s")
res["package_crosscheck_maxreldiff"] = float(rel)

# ---------- 3. symmetry, integral conservation, uniform-field preservation with variable l
lat = mesh.gphiu
ellv = cf.ell_from_window_deg(mesh, "U", 2.0)
idx, area, K = cf.build_operator(mesh, "U", wet, ellv)
print("K symmetric:", abs(K - K.T).max())
S = cf.ImplicitSolver(area, K, backend="gpu", tol=1e-12)
X, _ = S.solve(np.stack([data, np.ones_like(data)], 1)); X = S.xp.asnumpy(X)
print("integral before/after:", (area * data).sum(), (area * X[:, 0]).sum())
print("uniform field max deviation:", np.abs(X[:, 1] - 1).max())
res["uniform_dev"] = float(np.abs(X[:, 1] - 1).max())
res["integral_rel"] = float(abs((area * data).sum() - (area * X[:, 0]).sum()) / abs((area * np.abs(data)).sum()))

# ---------- 4. timing: one full level x 31 days, largest scale
U = netCDF4.Dataset("/work/bk1450/b383184/Amazon/Mercator/implicit_data/U_1993-01c.nc")["vozocrtx"]
t0 = time.time(); d = U[:, 0].filled(np.nan).astype(np.float32); tread = time.time() - t0
t0 = time.time(); o, it, nw = cf.filter_level(mesh, "U", d, cf.ell_from_window_deg(mesh, "U", 2.5), tol=1e-8); tf = time.time() - t0
print(f"read level {tread:.1f}s; filter 31 days W2.5 level0: {tf:.2f}s its={it} nwet={nw}")
res["timing_level0_W2.5"] = dict(read=tread, filt=tf, its=it, nwet=nw)
json.dump(res, open("/work/bk1450/b383184/Amazon/Mercator/implicit_run/tests/validate_filter.json", "w"), indent=1)
