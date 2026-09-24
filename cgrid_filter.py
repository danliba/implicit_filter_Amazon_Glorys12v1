"""
Implicit filter (Danilov et al. 2023; Nowak et al. 2025) on the NEMO ORCA12 C-grid.

Filters U on the U-grid and V on the V-grid, level by level, solving

    (1 + gamma * (-div(l^2 grad)))  phi_bar = phi ,      gamma = 1/2, n = 1

in finite-volume form on each staggered sub-mesh:

    (A + gamma * K_l) phi_bar = A phi

A   = diag(cell area) of the U- (e1u*e2u) or V-cell (e1v*e2v)
K_l = symmetric positive semi-definite flux matrix; every wet face between two wet
      cells i,j contributes  c_ij = l^2_face * (face width / centre distance)
      to K[i,i], K[j,j] and -c_ij to K[i,j], K[j,i].
      Faces touching land or the open edge of the regional domain are omitted
      (no-flux / Neumann boundary condition).

Face metrics (NEMO scale factors, indices (j, i), U(j,i) sits between T(j,i) and T(j,i+1),
V(j,i) between T(j,i) and T(j+1,i), F(j,i) at the NE corner of T(j,i)):
  U-grid  east/west face through T(j,i+1): width e2t(j,i+1), distance e1t(j,i+1)
          north/south face through F(j,i): width e1f(j,i),   distance e2f(j,i)
  V-grid  east/west face through F(j,i):   width e2f(j,i),   distance e1f(j,i)
          north/south face through T(j+1,i): width e1t(j+1,i), distance e2t(j+1,i)

For a spatially constant l this is exactly the operator of implicit_filter's
LatLonFilter/NemoFilter `_compute` (Smat = I + 2*(-L/k^2)) with k = 2/l, but
(i) it is built on the correct staggered metrics and (ii) with variable l it uses the
divergence form div(l^2 grad), which keeps the system symmetric (CG is valid), preserves
the area integral AND preserves spatially uniform fields.  The package's column scaling
L*diag(1/k^2) preserves the integral but not a uniform field.

The solve is a batched Jacobi-preconditioned conjugate gradient on the GPU (CuPy) that
treats all time steps of one level at once; a SciPy/NumPy fallback is used on CPU.
"""
from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
import xarray as xr

R_EARTH = 6371.0e3  # m
GAMMA = 0.5
BOX_FACTOR = 3.5    # l_box / l for gamma = 1/2 (Nowak et al. 2025, GMD)

MESH_DIR = "/work/bk1450/b383184/Amazon/Mercator/data/"


# ----------------------------------------------------------------------------- mesh
class CMesh:
    """Horizontal/vertical NEMO metrics of the regional domain (float64, (y, x))."""

    def __init__(self, hgr=MESH_DIR + "Hgr_cmesh.nc", zgr=MESH_DIR + "Zgr_cmesh2.nc"):
        h = xr.open_dataset(hgr).squeeze()
        z = xr.open_dataset(zgr).squeeze()
        for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
                  "glamt", "gphit", "glamu", "gphiu", "glamv", "gphiv", "glamf", "gphif"]:
            setattr(self, v, h[v].values.astype(np.float64))
        self.nav_lon = h.nav_lon.values
        self.nav_lat = h.nav_lat.values
        self.ny, self.nx = self.e1t.shape
        self.nz = z.sizes["z"]
        self.mbathy = z.mbathy.values.astype(int)
        self.gdept_0 = z.gdept_0.values.astype(np.float64)
        self.gdepw_0 = z.gdepw_0.values.astype(np.float64)
        self.e3t_0 = z.e3t_0.values.astype(np.float64)
        k = np.arange(self.nz)[:, None, None]
        self.tmask = k < self.mbathy[None]
        e3t = np.where(k < self.mbathy[None] - 1, z.e3t_0.values[:, None, None],
                       z.e3t_ps.values[None])
        self.e3t = np.where(self.tmask, e3t, 0.0)
        # NEMO partial steps: e3u = min(e3t(i), e3t(i+1)), e3v = min(e3t(j), e3t(j+1))
        self.e3u = np.zeros_like(self.e3t)
        self.e3u[:, :, :-1] = np.minimum(self.e3t[:, :, :-1], self.e3t[:, :, 1:])
        self.e3v = np.zeros_like(self.e3t)
        self.e3v[:, :-1, :] = np.minimum(self.e3t[:, :-1, :], self.e3t[:, 1:, :])
        self.H = self.e3t.sum(0)
        # depth of the top of each cell (W-level depth incl. partial bottom cell), as in Fix_W
        self.depthw_ps = np.concatenate([np.zeros((1, self.ny, self.nx)),
                                         np.cumsum(self.e3t, 0)[:-1]], 0)

    def lat(self, grid):
        return {"U": self.gphiu, "V": self.gphiv, "T": self.gphit}[grid]

    def lon(self, grid):
        return {"U": self.glamu, "V": self.glamv, "T": self.glamt}[grid]


# ----------------------------------------------------------------------------- scales
def ell_from_window_deg(mesh: CMesh, grid: str, window_deg: float) -> np.ndarray:
    """l(x,y) = dtheta * R * cos(phi) / 3.5 at the U/V points [m]."""
    L = np.deg2rad(window_deg) * R_EARTH * np.cos(np.deg2rad(mesh.lat(grid)))
    return L / BOX_FACTOR


def t_to_grid(field_t: np.ndarray, grid: str) -> np.ndarray:
    """Average a T-point field (finite everywhere) to U or V points."""
    out = field_t.copy()
    if grid == "U":
        out[:, :-1] = 0.5 * (field_t[:, :-1] + field_t[:, 1:])
    elif grid == "V":
        out[:-1, :] = 0.5 * (field_t[:-1, :] + field_t[1:, :])
    return out


# ----------------------------------------------------------------------------- operator
def build_operator(mesh: CMesh, grid: str, wet2d: np.ndarray, ell: np.ndarray):
    """
    Assemble A (vector) and K_l (CSR, symmetric) on the wet points of one level.

    Parameters
    ----------
    grid  : "U" or "V"
    wet2d : bool (ny, nx), True where the velocity point is wet (taken from the data)
    ell   : (ny, nx) filter length l [m] at the velocity points (finite everywhere)

    Returns
    -------
    idx  : flat indices (C order) of wet points
    area : (nw,) cell areas
    K    : scipy CSR (nw, nw)
    """
    ny, nx = wet2d.shape
    flat = np.full(ny * nx, -1, dtype=np.int64)
    idx = np.flatnonzero(wet2d)
    flat[idx] = np.arange(idx.size)
    num = flat.reshape(ny, nx)
    l2 = ell ** 2

    if grid == "U":
        area = (mesh.e1u * mesh.e2u)
        # zonal faces between U(j,i) and U(j,i+1): through T(j,i+1)
        cx = mesh.e2t[:, 1:] / mesh.e1t[:, 1:]
        # meridional faces between U(j,i) and U(j+1,i): through F(j,i)
        cy = mesh.e1f[:-1, :] / mesh.e2f[:-1, :]
    elif grid == "T":
        # T/W points: zonal faces through U(j,i), meridional faces through V(j,i)
        area = (mesh.e1t * mesh.e2t)
        cx = mesh.e2u[:, :-1] / mesh.e1u[:, :-1]
        cy = mesh.e1v[:-1, :] / mesh.e2v[:-1, :]
    elif grid == "V":
        area = (mesh.e1v * mesh.e2v)
        # zonal faces between V(j,i) and V(j,i+1): through F(j,i)
        cx = mesh.e2f[:, :-1] / mesh.e1f[:, :-1]
        # meridional faces between V(j,i) and V(j+1,i): through T(j+1,i)
        cy = mesh.e1t[1:, :] / mesh.e2t[1:, :]
    else:
        raise ValueError(grid)

    # x faces: pairs (j,i)-(j,i+1)
    wx = wet2d[:, :-1] & wet2d[:, 1:]
    ax = (cx * 0.5 * (l2[:, :-1] + l2[:, 1:]))[wx]
    ix, jx = num[:, :-1][wx], num[:, 1:][wx]
    # y faces: pairs (j,i)-(j+1,i)
    wy = wet2d[:-1, :] & wet2d[1:, :]
    ay = (cy * 0.5 * (l2[:-1, :] + l2[1:, :]))[wy]
    iy, jy = num[:-1, :][wy], num[1:, :][wy]

    ii = np.concatenate([ix, jx, iy, jy])
    jj = np.concatenate([jx, ix, jy, iy])
    vv = -np.concatenate([ax, ax, ay, ay])
    nw = idx.size
    diag = np.zeros(nw)
    np.add.at(diag, ix, ax); np.add.at(diag, jx, ax)
    np.add.at(diag, iy, ay); np.add.at(diag, jy, ay)
    K = sp.csr_matrix((np.concatenate([vv, diag]),
                       (np.concatenate([ii, np.arange(nw)]), np.concatenate([jj, np.arange(nw)]))),
                      shape=(nw, nw))
    return idx, area.ravel()[idx], K


# ----------------------------------------------------------------------------- solver
class ImplicitSolver:
    """Batched Jacobi-PCG for (A + gamma K) X = A B, columns of B = time steps."""

    def __init__(self, area, K, gamma=GAMMA, backend="gpu", tol=1e-8, maxiter=20000):
        self.backend = backend
        if backend == "gpu":
            import cupy as cp
            import cupyx.scipy.sparse as csp
            self.xp = cp
            S = sp.diags(area) + gamma * K
            self.S = csp.csr_matrix(S.tocsr())
        else:
            self.xp = np
            self.S = (sp.diags(area) + gamma * K).tocsr()
        xp = self.xp
        self.area = xp.asarray(area)
        self.minv = 1.0 / xp.asarray(self.S.diagonal())
        self.tol, self.maxiter = tol, maxiter

    def solve(self, B):
        """B: (nw, nt) array (host or device). Returns (X on device/host, iterations)."""
        xp = self.xp
        B = xp.asarray(B, dtype=xp.float64)
        if B.ndim == 1:
            B = B[:, None]
        rhs = self.area[:, None] * B
        X = B.copy()                          # initial guess: unfiltered field
        Rr = rhs - self.S @ X
        Z = self.minv[:, None] * Rr
        P = Z.copy()
        rz = (Rr * Z).sum(0)
        bnorm = xp.sqrt((rhs * rhs).sum(0))
        bnorm = xp.where(bnorm == 0, 1.0, bnorm)
        active = xp.ones(B.shape[1], dtype=bool)
        it = 0
        for it in range(1, self.maxiter + 1):
            SP = self.S @ P
            pSp = (P * SP).sum(0)
            alpha = xp.where(active, rz / xp.where(pSp == 0, 1.0, pSp), 0.0)
            X += alpha * P
            Rr -= alpha * SP
            res = xp.sqrt((Rr * Rr).sum(0)) / bnorm
            active = res > self.tol
            if not bool(active.any()):
                break
            Z = self.minv[:, None] * Rr
            rz_new = (Rr * Z).sum(0)
            beta = xp.where(active, rz_new / xp.where(rz == 0, 1.0, rz), 0.0)
            P = Z + beta * P
            rz = rz_new
        else:
            raise RuntimeError(f"PCG did not converge in {self.maxiter} its, max res {float(res.max())}")
        return X, it


def filter_level(mesh, grid, data_tyx, ell, backend="gpu", tol=1e-8):
    """
    Filter one level. data_tyx: (nt, ny, nx) with NaN on land. Returns same shape (NaN on land),
    number of CG iterations, number of wet points.
    """
    wet = ~np.isnan(data_tyx[0])
    out = np.full(data_tyx.shape, np.nan, dtype=np.float32)
    if wet.sum() == 0:
        return out, 0, 0
    idx, area, K = build_operator(mesh, grid, wet, ell)
    solver = ImplicitSolver(area, K, backend=backend, tol=tol)
    nt = data_tyx.shape[0]
    B = data_tyx.reshape(nt, -1)[:, idx].T.astype(np.float64)
    X, it = solver.solve(B)
    if backend == "gpu":
        X = solver.xp.asnumpy(X)
    o = out.reshape(nt, -1)
    o[:, idx] = X.T
    return out, it, idx.size


# ----------------------------------------------------------------------------- continuity
def horizontal_divergence_e3(mesh: CMesh, u_zyx, v_zyx):
    """
    NEMO C-grid flux divergence times e3t (units m/s) on T points:
    [ d_i(e2u e3u u) + d_j(e1v e3v v) ] / (e1t e2t). NaN->0 in inputs. First row/col set to 0.
    """
    u = np.nan_to_num(u_zyx.astype(np.float64))
    v = np.nan_to_num(v_zyx.astype(np.float64))
    fu = mesh.e2u[None] * mesh.e3u * u
    fv = mesh.e1v[None] * mesh.e3v * v
    d = np.zeros_like(fu)
    d[:, 1:, 1:] = (fu[:, 1:, 1:] - fu[:, 1:, :-1] + fv[:, 1:, 1:] - fv[:, :-1, 1:])
    d /= (mesh.e1t * mesh.e2t)[None]
    return np.where(mesh.tmask, d, 0.0)


def w_from_continuity(mesh: CMesh, u_zyx, v_zyx):
    """
    w at W-levels (top of each T-cell), positive upward, integrated from the bottom (w=0 at the
    sea floor, NEMO: w(k) = w(k+1) - e3t(k) hdiv(k)). Returns (w, w_fixed, hdiv) where w_fixed
    removes the linear free-surface part exactly as in Fix_W.ipynb:
        w_fixed = w - (H - depthw)/H * w(surface).
    """
    de3 = horizontal_divergence_e3(mesh, u_zyx, v_zyx)
    w = np.cumsum(de3[::-1], 0)[::-1] * -1.0   # w(k) = -sum_{k'>=k} e3t*hdiv
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(mesh.H[None] > 0, (mesh.H[None] - mesh.depthw_ps) / mesh.H[None], 0.0)
    w_fixed = w - frac * w[0][None]
    wmask = mesh.tmask
    w = np.where(wmask, w, np.nan)
    w_fixed = np.where(wmask, w_fixed, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        hdiv = np.where(mesh.tmask, de3 / np.where(mesh.e3t > 0, mesh.e3t, 1.0), np.nan)
    return w, w_fixed, hdiv


# =============================================================================
# Vector (div-rot) filter on the C-grid  -- version 2
# =============================================================================
"""
Component-wise scalar filtering with no-flux walls for each component (version 1) breaks
the discrete continuity at walls: it leaves a one-cell convergence line along every coast at
every level, and the depth-integrated error gives |w| ~ 1e-3 m/s. Version 2 replaces
-div(l^2 grad) acting on each component by the C-grid vector operator in div-rot form

    M u = - grad( l_T^2 div u ) + curl( l_F^2 zeta(u) )      ( = -l^2 Laplacian(u) for constant l )

    div u   = [d_i(e2u e3u u) + d_j(e1v e3v v)] / (e1t e2t e3t)     (NEMO hdiv, T points)
    zeta(u) = [d_i(e2v v) - d_j(e1u u)] / (e1f e2f)                  (F points)

and solves (W + gamma*(Fᵀ diag(l_T²/(e1t e2t e3t)) F + Cᵀ diag(l_F²/(e1f e2f)) C)) ū = W u,
with W = diag(e1u e2u e3u, e1v e2v e3v), F the T-cell flux-difference matrix and C the F-point
circulation matrix. The system is symmetric positive definite. The boundary conditions are:
- zero normal flow through land faces (they are not unknowns)
- free slip (zeta = 0 at F points that touch a land face)
Discrete identities, which are exact for any l and with partial steps:
- div(curl ·) = 0
- div ū = (I + gamma * D W^-1 Fᵀ diag(l_T²))^-1 div u
So the filtered divergence is the no-flux scalar filter of the original divergence, level by
level, and there are no wall artefacts. In open ocean with constant l the operator equals the
component-wise Laplacian (up to spherical metric terms < 0.1 %), so the transfer function
G(K) = 1/(1 + gamma l² K²) is unchanged.
"""


def ell_T_F(mesh, cfg_kind, window_deg=None, ld_T=None, mult=None, smooth_sigma_cells=6.0):
    """l at T and F points [m] for a window (deg) or Rossby (mult*Ld/3.5) configuration."""
    if cfg_kind == "window":
        k = np.deg2rad(window_deg) * R_EARTH / BOX_FACTOR
        return k * np.cos(np.deg2rad(mesh.gphit)), k * np.cos(np.deg2rad(mesh.gphif))
    from scipy.ndimage import gaussian_filter
    lT = gaussian_filter(mult * ld_T / BOX_FACTOR, smooth_sigma_cells, mode="nearest")
    lF = lT.copy()
    lF[:-1, :-1] = 0.25 * (lT[:-1, :-1] + lT[1:, :-1] + lT[:-1, 1:] + lT[1:, 1:])
    return lT, lF


def build_vector_operator(mesh: CMesh, k: int, uwet: np.ndarray, vwet: np.ndarray,
                          lT: np.ndarray, lF: np.ndarray):
    """
    Assemble the div-rot filter operator for level k.

    Returns
    -------
    iu, iv : flat indices of wet U and V points (unknown order: u then v)
    Wd     : (n,) diagonal mass  e1u e2u e3u | e1v e2v e3v
    K      : CSR (n, n) symmetric PSD, K = Fᵀ Λ_T F + Cᵀ Λ_F C
    FT, it : T flux-difference matrix (nT, n) and flat indices of its T rows (for diagnostics)
    """
    ny, nx = uwet.shape
    iu = np.flatnonzero(uwet); iv = np.flatnonzero(vwet)
    nu, nv = iu.size, iv.size
    n = nu + nv
    numu = np.full(ny * nx, -1, np.int64); numu[iu] = np.arange(nu); numu = numu.reshape(ny, nx)
    numv = np.full(ny * nx, -1, np.int64); numv[iv] = nu + np.arange(nv); numv = numv.reshape(ny, nx)
    e3t, e3u, e3v = mesh.e3t[k], mesh.e3u[k], mesh.e3v[k]
    # partial-step thickness can be zero at the regional edge for "wet" data points -> use e3t
    e3ref = mesh.e3t_0[k]
    e3u = np.where(e3u > 0, e3u, np.where(e3t > 0, e3t, e3ref))
    e3v = np.where(e3v > 0, e3v, np.where(e3t > 0, e3t, e3ref))

    Wd = np.concatenate([(mesh.e1u * mesh.e2u * e3u).ravel()[iu],
                         (mesh.e1v * mesh.e2v * e3v).ravel()[iv]])

    # ---- T flux-difference matrix F (rows: wet T cells)
    twet = mesh.tmask[k]
    it = np.flatnonzero(twet)
    numt = np.full(ny * nx, -1, np.int64); numt[it] = np.arange(it.size); numt = numt.reshape(ny, nx)
    rows, cols, vals = [], [], []
    J, I = np.nonzero(uwet)
    fu = (mesh.e2u * e3u)[J, I]; cu = numu[J, I]
    r = numt[J, I]; m = r >= 0
    rows.append(r[m]); cols.append(cu[m]); vals.append(fu[m])                  # east face of T(j,i)
    ok = I + 1 < nx
    r = np.full(J.size, -1); r[ok] = numt[J[ok], I[ok] + 1]; m = r >= 0
    rows.append(r[m]); cols.append(cu[m]); vals.append(-fu[m])                 # west face of T(j,i+1)
    J, I = np.nonzero(vwet)
    fv = (mesh.e1v * e3v)[J, I]; cv = numv[J, I]
    r = numt[J, I]; m = r >= 0
    rows.append(r[m]); cols.append(cv[m]); vals.append(fv[m])                  # north face of T(j,i)
    ok = J + 1 < ny
    r = np.full(J.size, -1); r[ok] = numt[J[ok] + 1, I[ok]]; m = r >= 0
    rows.append(r[m]); cols.append(cv[m]); vals.append(-fv[m])                 # south face of T(j+1,i)
    FT = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(it.size, n))
    lamT = (lT ** 2 / (mesh.e1t * mesh.e2t * np.where(e3t > 0, e3t, 1.0))).ravel()[it]

    # ---- F circulation matrix C (rows: F points whose 4 surrounding faces are wet)
    fw = np.zeros((ny, nx), bool)
    fw[:-1, :-1] = uwet[:-1, :-1] & uwet[1:, :-1] & vwet[:-1, :-1] & vwet[:-1, 1:]
    J, I = np.nonzero(fw)
    nf = J.size
    rid = np.arange(nf)
    rows = np.concatenate([rid, rid, rid, rid])
    cols = np.concatenate([numv[J, I + 1], numv[J, I], numu[J + 1, I], numu[J, I]])
    vals = np.concatenate([mesh.e2v[J, I + 1], -mesh.e2v[J, I], -mesh.e1u[J + 1, I], mesh.e1u[J, I]])
    C = sp.csr_matrix((vals, (rows, cols)), shape=(nf, n))
    # e3 at F points (NEMO partial steps: min of the surrounding e3u/e3v). The mass matrix carries
    # e3u/e3v, so the vorticity term needs e3f to scale like the divergence term (which carries
    # e3u/e3v in F and 1/e3t in lamT); without it the rotational part is ~1/e3 too weak at depth.
    # div(curl) = 0 holds for any lamF, so the divergence identity is unaffected.
    e3f = np.minimum.reduce([e3u[J, I], e3u[J + 1, I], e3v[J, I], e3v[J, I + 1]])
    lamF = lF[J, I] ** 2 * e3f / (mesh.e1f[J, I] * mesh.e2f[J, I])

    K = (FT.T @ sp.diags(lamT) @ FT + C.T @ sp.diags(lamF) @ C).tocsr()
    return iu, iv, Wd, K, FT, it


def filter_level_vector(mesh, k, u_tyx, v_tyx, lT, lF, backend="gpu", tol=1e-8, return_ops=False):
    """Filter (u, v) of one level jointly. Inputs (nt, ny, nx) with NaN on land."""
    uwet = ~np.isnan(u_tyx[0]); vwet = ~np.isnan(v_tyx[0])
    nt = u_tyx.shape[0]
    uo = np.full(u_tyx.shape, np.nan, np.float32); vo = np.full(v_tyx.shape, np.nan, np.float32)
    if uwet.sum() + vwet.sum() == 0:
        return uo, vo, 0, 0
    iu, iv, Wd, K, FT, it = build_vector_operator(mesh, k, uwet, vwet, lT, lF)
    solver = ImplicitSolver(Wd, K, backend=backend, tol=tol)
    B = np.concatenate([u_tyx.reshape(nt, -1)[:, iu], v_tyx.reshape(nt, -1)[:, iv]], 1).T.astype(np.float64)
    X, its = solver.solve(B)
    if backend == "gpu":
        X = solver.xp.asnumpy(X)
    uo.reshape(nt, -1)[:, iu] = X[:iu.size].T
    vo.reshape(nt, -1)[:, iv] = X[iu.size:].T
    if return_ops:
        return uo, vo, its, iu.size + iv.size, (iu, iv, Wd, K, FT, it)
    return uo, vo, its, iu.size + iv.size


def barotropic_correction(mesh, u_zyx, v_zyx, target_colflux=None, backend="cpu", tol=1e-10):
    """
    Remove the depth-integrated flux divergence of (u, v) relative to `target_colflux`
    (T points, m^3/s per cell; default 0) with a depth-uniform, irrotational velocity correction
    u' = -d_i(phi)/e1u, v' = -d_j(phi)/e2v applied on all wet levels. phi solves
        d_i(e2u Hu/e1u d_i phi) + d_j(e1v Hv/e2v d_j phi) = sum_k d(e3 u) - target
    with no-flux at land and phi = 0 on the open edges of the regional domain (Dirichlet), so the
    net volume imbalance can leave through the open boundaries. Returns corrected u, v and phi.
    """
    ny, nx = mesh.ny, mesh.nx
    u0 = np.nan_to_num(u_zyx); v0 = np.nan_to_num(v_zyx)
    uw = ~np.isnan(u_zyx); vw = ~np.isnan(v_zyx)
    e3u = np.where(uw, np.where(mesh.e3u > 0, mesh.e3u, mesh.e3t), 0.0)
    e3v = np.where(vw, np.where(mesh.e3v > 0, mesh.e3v, mesh.e3t), 0.0)
    Ub = (e3u * u0).sum(0) * mesh.e2u     # column volume flux through U faces
    Vb = (e3v * v0).sum(0) * mesh.e1v
    Hu = e3u.sum(0); Hv = e3v.sum(0)
    div = np.zeros((ny, nx))
    div += Ub; div[:, 1:] -= Ub[:, :-1]
    div += Vb; div[1:, :] -= Vb[:-1, :]
    if target_colflux is not None:
        div -= target_colflux
    twet = mesh.tmask[0]
    edge = np.zeros((ny, nx), bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    unk = twet & ~edge
    idx = np.flatnonzero(unk)
    num = np.full(ny * nx, -1, np.int64); num[idx] = np.arange(idx.size); num = num.reshape(ny, nx)
    cx = mesh.e2u * Hu / mesh.e1u          # face (j,i)-(j,i+1)
    cy = mesh.e1v * Hv / mesh.e2v          # face (j,i)-(j+1,i)
    rows, cols, vals = [], [], []
    diag = np.zeros(idx.size)
    for c, (a, b) in [(cx[:, :-1], ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))),
                      (cy[:-1, :], ((slice(None, -1), slice(None)), (slice(1, None), slice(None))))]:
        na, nb = num[a], num[b]
        w = c > 0
        # contributions to unknown rows; neighbours that are edge cells are Dirichlet 0
        m = w & (na >= 0)
        np.add.at(diag, na[m], c[m])
        m2 = m & (nb >= 0)
        rows.append(na[m2]); cols.append(nb[m2]); vals.append(-c[m2])
        m = w & (nb >= 0)
        np.add.at(diag, nb[m], c[m])
        m2 = m & (na >= 0)
        rows.append(nb[m2]); cols.append(na[m2]); vals.append(-c[m2])
    diag = np.maximum(diag, 1e-12 * diag.max())   # isolated points
    L = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(idx.size)]),
                       np.concatenate(cols + [np.arange(idx.size)]))), shape=(idx.size, idx.size))
    rhs = -div.ravel()[idx]                  # L phi = -div  => d(H grad(-phi)) removes div
    if backend == "gpu":
        import cupy as cp, cupyx.scipy.sparse as csp
        from cupyx.scipy.sparse.linalg import cg
        Lg = csp.csr_matrix(L); M = csp.diags(1.0 / cp.asarray(L.diagonal()))
        x, info = cg(Lg, cp.asarray(rhs), rtol=tol, maxiter=200000, M=M)
        x = cp.asnumpy(x)
    else:
        from scipy.sparse.linalg import cg
        x, info = cg(L, rhs, rtol=tol, maxiter=200000, M=sp.diags(1.0 / L.diagonal()))
    phi = np.zeros(ny * nx); phi[idx] = x; phi = phi.reshape(ny, nx)
    du = np.zeros((ny, nx)); du[:, :-1] = -(phi[:, 1:] - phi[:, :-1]) / mesh.e1u[:, :-1]
    dv = np.zeros((ny, nx)); dv[:-1, :] = -(phi[1:, :] - phi[:-1, :]) / mesh.e2v[:-1, :]
    uc = np.where(uw, u_zyx + du[None], np.nan)
    vc = np.where(vw, v_zyx + dv[None], np.nan)
    return uc, vc, phi, info


def filter_column_flux(mesh, colflux, lT, gamma=GAMMA, backend="gpu", tol=1e-10):
    """
    Smooth the depth-integrated flux divergence (m^3/s per T cell, e.g. -A*d(eta)/dt + runoff)
    with the same implicit filter, H-weighted and no-flux at land:
        A H d_bar + gamma * K_H (l^2 d_bar) = A H d ,   d = colflux/(A H),  K_H faces c = H_face e2u/e1u, e1v H/e2v
    solved in the symmetric variable y = l^2 d_bar. Conserves sum(colflux). Returns A H d_bar.
    This is the column analogue of the level-wise identity div(u_bar) = S_T^-1 div(u) of the
    div-rot filter, and is the target of `barotropic_correction`.
    """
    ny, nx = mesh.ny, mesh.nx
    H = mesh.H; A = mesh.e1t * mesh.e2t
    wet = H > 0
    idx = np.flatnonzero(wet)
    num = np.full(ny * nx, -1, np.int64); num[idx] = np.arange(idx.size); num = num.reshape(ny, nx)
    Hu = np.minimum(H[:, :-1], H[:, 1:]); Hv = np.minimum(H[:-1, :], H[1:, :])
    cx = Hu * mesh.e2u[:, :-1] / mesh.e1u[:, :-1]
    cy = Hv * mesh.e1v[:-1, :] / mesh.e2v[:-1, :]
    ii, jj, vv = [], [], []
    diag = np.zeros(idx.size)
    for c, na, nb in [(cx, num[:, :-1], num[:, 1:]), (cy, num[:-1, :], num[1:, :])]:
        m = (na >= 0) & (nb >= 0) & (c > 0)
        a, b, w = na[m], nb[m], c[m]
        ii += [a, b]; jj += [b, a]; vv += [-w, -w]
        np.add.at(diag, a, w); np.add.at(diag, b, w)
    l2 = lT.ravel()[idx] ** 2
    AH = (A * H).ravel()[idx]
    S = sp.csr_matrix((np.concatenate(vv + [AH / l2 / gamma + diag]),
                       (np.concatenate(ii + [np.arange(idx.size)]), np.concatenate(jj + [np.arange(idx.size)]))),
                      shape=(idx.size, idx.size))
    rhs = colflux.ravel()[idx] / gamma          # = A H d / gamma
    if backend == "gpu":
        import cupy as cp, cupyx.scipy.sparse as csp
        from cupyx.scipy.sparse.linalg import cg
        y, info = cg(csp.csr_matrix(S), cp.asarray(rhs), rtol=tol, maxiter=200000,
                     M=csp.diags(1.0 / cp.asarray(S.diagonal())))
        y = cp.asnumpy(y)
    else:
        from scipy.sparse.linalg import cg
        y, info = cg(S, rhs, rtol=tol, maxiter=200000, M=sp.diags(1.0 / S.diagonal()))
    out = np.zeros(ny * nx); out[idx] = AH * y / l2
    return out.reshape(ny, nx)


class ColumnConsistency:
    """
    Cached version of `filter_column_flux` + `barotropic_correction` for many time steps
    with fixed masks. Usage:
        cc = ColumnConsistency(mesh, uwet3d, vwet3d, lT, backend="gpu")
        u_c, v_c, info = cc.apply(u_filt, v_filt, u_orig, v_orig)
    """

    def __init__(self, mesh, uwet, vwet, lT, gamma=GAMMA, backend="gpu", tol=1e-10):
        self.mesh, self.backend, self.tol = mesh, backend, tol
        ny, nx = mesh.ny, mesh.nx
        self.uwet, self.vwet = uwet, vwet
        self.e3u = np.where(uwet, np.where(mesh.e3u > 0, mesh.e3u, mesh.e3t), 0.0)
        self.e3v = np.where(vwet, np.where(mesh.e3v > 0, mesh.e3v, mesh.e3t), 0.0)
        A = mesh.e1t * mesh.e2t
        # --- smoothing matrix for the column flux (see filter_column_flux)
        H = mesh.H
        wet = H > 0
        self.cidx = np.flatnonzero(wet)
        num = np.full(ny * nx, -1, np.int64); num[self.cidx] = np.arange(self.cidx.size); num = num.reshape(ny, nx)
        Hu = np.minimum(H[:, :-1], H[:, 1:]); Hv = np.minimum(H[:-1, :], H[1:, :])
        S = self._lap(num, Hu * mesh.e2u[:, :-1] / mesh.e1u[:, :-1], Hv * mesh.e1v[:-1, :] / mesh.e2v[:-1, :], self.cidx.size)
        self.l2 = lT.ravel()[self.cidx] ** 2
        self.AH = (A * H).ravel()[self.cidx]
        self.gamma = gamma
        self.Sc = (S + sp.diags(self.AH / self.l2 / gamma)).tocsr()
        # --- barotropic Poisson matrix: unknowns = wet surface T cells not on the open edge
        twet = mesh.tmask[0]
        edge = np.zeros((ny, nx), bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        self.edge = edge
        self.pidx = np.flatnonzero(twet & ~edge)
        pnum = np.full(ny * nx, -1, np.int64); pnum[self.pidx] = np.arange(self.pidx.size); pnum = pnum.reshape(ny, nx)
        Hub = self.e3u.sum(0); Hvb = self.e3v.sum(0)
        cx = (mesh.e2u * Hub / mesh.e1u)[:, :-1]; cy = (mesh.e1v * Hvb / mesh.e2v)[:-1, :]
        self.Lp = self._lap(pnum, cx, cy, self.pidx.size, dirichlet_neighbours=True)
        self.Hub, self.Hvb = Hub, Hvb
        if backend == "gpu":
            import cupy as cp, cupyx.scipy.sparse as csp
            self.cp, self.csp = cp, csp
            self.Sc_g = csp.csr_matrix(self.Sc); self.Mc_g = csp.diags(1.0 / cp.asarray(self.Sc.diagonal()))
            self.Lp_g = csp.csr_matrix(self.Lp); self.Mp_g = csp.diags(1.0 / cp.asarray(self.Lp.diagonal()))

    @staticmethod
    def _lap(num, cx, cy, n, dirichlet_neighbours=False):
        ii, jj, vv = [], [], []
        diag = np.zeros(n)
        for c, na, nb in [(cx, num[:, :-1], num[:, 1:]), (cy, num[:-1, :], num[1:, :])]:
            if dirichlet_neighbours:
                w = c > 0
                for a_, b_ in [(na, nb), (nb, na)]:
                    m = w & (a_ >= 0)
                    np.add.at(diag, a_[m], c[m])
                    m2 = m & (b_ >= 0)
                    ii.append(a_[m2]); jj.append(b_[m2]); vv.append(-c[m2])
            else:
                m = (na >= 0) & (nb >= 0) & (c > 0)
                a, b, w = na[m], nb[m], c[m]
                ii += [a, b]; jj += [b, a]; vv += [-w, -w]
                np.add.at(diag, a, w); np.add.at(diag, b, w)
        diag = np.maximum(diag, 1e-12 * max(diag.max(), 1.0))
        return sp.csr_matrix((np.concatenate(vv + [diag]), (np.concatenate(ii + [np.arange(n)]),
                              np.concatenate(jj + [np.arange(n)]))), shape=(n, n))

    def _cg(self, which, rhs):
        if self.backend == "gpu":
            from cupyx.scipy.sparse.linalg import cg
            A, M = (self.Sc_g, self.Mc_g) if which == "c" else (self.Lp_g, self.Mp_g)
            x, info = cg(A, self.cp.asarray(rhs), rtol=self.tol, maxiter=200000, M=M)
            return self.cp.asnumpy(x), info
        from scipy.sparse.linalg import cg
        A = self.Sc if which == "c" else self.Lp
        return cg(A, rhs, rtol=self.tol, maxiter=200000, M=sp.diags(1.0 / A.diagonal()))

    def colflux(self, u, v):
        """Depth-integrated volume flux divergence per T cell [m^3/s]."""
        m = self.mesh
        Ub = (self.e3u * np.nan_to_num(u)).sum(0) * m.e2u
        Vb = (self.e3v * np.nan_to_num(v)).sum(0) * m.e1v
        d = Ub.copy(); d[:, 1:] -= Ub[:, :-1]
        d += Vb; d[1:, :] -= Vb[:-1, :]
        return d

    def apply(self, uf, vf, uo, vo):
        m = self.mesh
        ny, nx = m.ny, m.nx
        # target: smoothed original column flux divergence
        # edge cells of the regional cut see only part of their faces -> their "divergence" is the
        # net inflow through the open boundary, not a physical column divergence: exclude them
        co = self.colflux(uo, vo); co[self.edge] = 0.0
        y, i1 = self._cg("c", co.ravel()[self.cidx] / self.gamma)
        target = np.zeros(ny * nx); target[self.cidx] = self.AH * y / self.l2; target = target.reshape(ny, nx)
        res = self.colflux(uf, vf) - target
        res[self.edge] = 0.0
        x, i2 = self._cg("p", -res.ravel()[self.pidx])
        phi = np.zeros(ny * nx); phi[self.pidx] = x; phi = phi.reshape(ny, nx)
        du = np.zeros((ny, nx)); du[:, :-1] = -(phi[:, 1:] - phi[:, :-1]) / m.e1u[:, :-1]
        dv = np.zeros((ny, nx)); dv[:-1, :] = -(phi[1:, :] - phi[:-1, :]) / m.e2v[:-1, :]
        uc = np.where(self.uwet, uf + du[None], np.nan)
        vc = np.where(self.vwet, vf + dv[None], np.nan)
        uw0 = self.uwet[0]; vw0 = self.vwet[0]
        info = dict(cg_info=(int(i1), int(i2)),
                    du_rms=float(np.sqrt(np.mean(du[uw0] ** 2))), du_max=float(np.abs(du[uw0]).max()),
                    dv_rms=float(np.sqrt(np.mean(dv[vw0] ** 2))), dv_max=float(np.abs(dv[vw0]).max()),
                    res_rms_before=float(np.sqrt(np.mean((res[m.tmask[0]] / (m.e1t * m.e2t)[m.tmask[0]]) ** 2))))
        return uc, vc, phi, target, info


def continuity_residual(mesh, u_zyx, v_zyx, w_zyx):
    """
    R = dw/dz + hdiv(u,v) on T cells [1/s], w positive upward at the top of each cell (W levels),
    R_k = (w_k - w_{k+1}) / e3t_k + hdiv_k ; w below the last level and NaN floor values -> 0.
    NaN on land.
    """
    de3 = horizontal_divergence_e3(mesh, u_zyx, v_zyx)          # e3t*hdiv
    w = np.nan_to_num(w_zyx.astype(np.float64))
    wb = np.concatenate([w[1:], np.zeros((1,) + w.shape[1:])], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (w - wb + de3) / np.where(mesh.e3t > 0, mesh.e3t, 1.0)
    return np.where(mesh.tmask, R, np.nan)


def filter_level_w(mesh, k, w_tyx, uwet, vwet, lT, backend="gpu", tol=1e-8):
    """
    Filter W at level k (top of T-cell k) with EXACTLY the T-cell operator implied by the vector
    div-rot filter of level k:
        vol x_bar + gamma * F W^-1 F^T (l_T^2 x_bar) = vol x ,   vol = e1t e2t e3t
    so that  d_z(w_bar) and div(u_bar) are filtered by the same operator wherever levels k and
    k+1 share the same wet mask/thickness (no-flux at land, same ℓ² placement, same regional-edge
    treatment). Solved in the symmetric variable y = l_T^2 x_bar:
        (vol / l_T^2) y + gamma * (F W^-1 F^T) y = vol x.
    Only wet T cells (tmask) are filtered; other points keep their input value (e.g. 0 at the floor).
    """
    nt = w_tyx.shape[0]
    out = w_tyx.astype(np.float32).copy()
    if not mesh.tmask[k].any():
        return out, 0
    iu, iv, Wd, K, FT, it = build_vector_operator(mesh, k, uwet, vwet, lT, lT)
    L = (FT @ sp.diags(1.0 / Wd) @ FT.T).tocsr()
    e3 = np.where(mesh.e3t[k] > 0, mesh.e3t[k], 1.0)
    vol = (mesh.e1t * mesh.e2t * e3).ravel()[it]
    l2 = (lT ** 2).ravel()[it]
    S = (sp.diags(vol / l2) + GAMMA * L).tocsr()
    B = np.nan_to_num(w_tyx.reshape(nt, -1)[:, it].T.astype(np.float64))
    if backend == "gpu":
        import cupy as xp
        import cupyx.scipy.sparse as csp
        Sg = csp.csr_matrix(S)
    else:
        xp = np; Sg = S
    minv = 1.0 / xp.asarray(S.diagonal())
    vol_d = xp.asarray(vol); l2_d = xp.asarray(l2)
    Bd = xp.asarray(B)
    rhs = vol_d[:, None] * Bd
    Y = l2_d[:, None] * Bd
    Rr = rhs - Sg @ Y
    Z = minv[:, None] * Rr; P = Z.copy(); rz = (Rr * Z).sum(0)
    bn = xp.sqrt((rhs * rhs).sum(0)); bn = xp.where(bn == 0, 1.0, bn)
    active = xp.ones(nt, dtype=bool); its = 0
    for its in range(1, 20001):
        SP = Sg @ P; pSp = (P * SP).sum(0)
        alpha = xp.where(active, rz / xp.where(pSp == 0, 1.0, pSp), 0.0)
        Y += alpha * P; Rr -= alpha * SP
        active = xp.sqrt((Rr * Rr).sum(0)) / bn > tol
        if not bool(active.any()):
            break
        Z = minv[:, None] * Rr; rzn = (Rr * Z).sum(0)
        beta = xp.where(active, rzn / xp.where(rz == 0, 1.0, rz), 0.0)
        P = Z + beta * P; rz = rzn
    X = Y / l2_d[:, None]
    if backend == "gpu":
        X = xp.asnumpy(X)
    o = out.reshape(nt, -1)
    o[:, it] = X.T
    return out, its
