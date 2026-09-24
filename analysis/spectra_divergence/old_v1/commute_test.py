"""
Task 2c: does filtering commute with the horizontal divergence?  CPU test on an open-ocean sub-box.

  D_a = hdiv( F_U u , F_V v )     (what the production run does: U and V filtered separately)
  D_b = F_T( hdiv(u, v) )         (filter applied to the original divergence, on the T grid)

F_U, F_V: cgrid_filter.build_operator on the sub-box (no-flux at the sub-box edge), direct solve
(scipy.sparse.linalg.spsolve) of (A + K/2) x = A b. F_T: the same assembly on the T sub-mesh
(faces through U points for x, V points for y) obtained by passing build_operator a relabelled metric
namespace for grid "U" (e1u,e2u <- e1t,e2t; e1t,e2t(i+1) <- e1u,e2u(i); e1f,e2f <- e1v,e2v).
Compared in the interior (margin M cells from the sub-box edge, sweeps M).

Cases (surface, day 1):
  W2.0 true l(lat)     box 54-37W, 12-28N (205x205)
  W2.0 constant l      same box, l = box mean
  R2Ld true l          box 48-31W, 0-15N (183x205), spans the Ld equatorial transition
  R2Ld constant l      same box
Also: F_U u in the interior vs the production GPU output (full domain) -> sub-box edge effect.
Output: metrics_commute.json, fig_commute.png
"""
import json, sys, types
import numpy as np
import netCDF4
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C
import cgrid_filter as cf

OUT = C.RUN + "/analysis/spectra_divergence"
m = C.mesh()
LD = netCDF4.Dataset(C.RUN + "/rossby/rossby_radius_T.nc")["Ld"][:].filled(np.nan).astype(float)
K0, T0 = 0, 0


def submesh(js, is_, k):
    s = types.SimpleNamespace()
    for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"]:
        setattr(s, v, getattr(m, v)[js, is_])
    s.e3u = m.e3u[k:k + 1, js, is_]; s.e3v = m.e3v[k:k + 1, js, is_]; s.tmask = m.tmask[k:k + 1, js, is_]
    s.e3t = m.e3t[k, js, is_]
    return s


def tmesh(s):
    """Relabel metrics so that build_operator(grid='U') assembles the T-grid operator."""
    t = types.SimpleNamespace()
    t.e1u, t.e2u = s.e1t, s.e2t                        # cell area
    t.e1t = s.e1t.copy(); t.e2t = s.e2t.copy()
    t.e1t[:, 1:] = s.e1u[:, :-1]; t.e2t[:, 1:] = s.e2u[:, :-1]   # x faces through U(j,i)
    t.e1f, t.e2f = s.e1v, s.e2v                        # y faces through V(j,i)
    return t


def filt(mesh_like, grid, field, ell):
    wet = np.ones(field.shape, bool)
    idx, area, K = cf.build_operator(mesh_like, grid, wet, ell)
    S = (sp.diags(area) + cf.GAMMA * K).tocsc()
    return spsolve(S, area * field.ravel()).reshape(field.shape)


def hdiv(s, u, v):
    de3 = cf.horizontal_divergence_e3(s, u[None], v[None])[0]
    return de3 / s.e3t


def run_case(name, box, cfg, const):
    js, is_ = C.box_slices(*box, grid="T")
    s = submesh(js, is_, K0)
    assert s.tmask.all(), name
    u = C.read("U", "ORIG", t=T0, k=K0, j=js, i=is_); v = C.read("V", "ORIG", t=T0, k=K0, j=js, i=is_)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    if cfg == "W2.0":
        lU = cf.ell_from_window_deg(m, "U", 2.0)[js, is_]; lV = cf.ell_from_window_deg(m, "V", 2.0)[js, is_]
        lT = cf.ell_from_window_deg(m, "T", 2.0)[js, is_]
    else:
        lt = 2 * LD / cf.BOX_FACTOR
        lU = cf.t_to_grid(lt, "U")[js, is_]; lV = cf.t_to_grid(lt, "V")[js, is_]; lT = lt[js, is_]
    if const:
        c = lT.mean(); lU = lV = lT = np.full(lT.shape, c)
    uf = filt(s, "U", u, lU); vf = filt(s, "V", v, lV)
    d0 = hdiv(s, u, v)
    # row 0 / col 0 of the divergence are undefined (set to 0); replace with neighbours before filtering
    d0f = d0.copy(); d0f[0, :] = d0f[1, :]; d0f[:, 0] = d0f[:, 1]
    Da = hdiv(s, uf, vf)
    Db = filt(tmesh(s), "U", d0f, lT)
    out = dict(box=box, shape=list(u.shape), l_T_km=[float(lT.min() / 1e3), float(lT.max() / 1e3)], const_l=const, margins={})
    for M in [5, 10, 20, 40, 60]:
        sl = (slice(M, -M), slice(M, -M))
        e = Da[sl] - Db[sl]
        out["margins"][M] = dict(rms_Da=float(np.sqrt(np.mean(Da[sl] ** 2))), rms_Db=float(np.sqrt(np.mean(Db[sl] ** 2))),
                                 rms_div_orig=float(np.sqrt(np.mean(d0[sl] ** 2))),
                                 rel_rms_diff=float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(Db[sl] ** 2))),
                                 corr=float(np.corrcoef(Da[sl].ravel(), Db[sl].ravel())[0, 1]),
                                 ratio_rms_Da_over_orig=float(np.sqrt(np.mean(Da[sl] ** 2)) / np.sqrt(np.mean(d0[sl] ** 2))))
    if not const:
        # compare sub-box CPU filter with production (full domain GPU) output
        uo = C.read("U", cfg, t=T0, k=K0, j=js, i=is_)
        for M in [5, 20, 40, 60]:
            sl = (slice(M, -M), slice(M, -M))
            out["margins"][M]["u_subbox_vs_production_rel_rms"] = float(np.sqrt(np.mean((uf - uo)[sl] ** 2)) / np.sqrt(np.mean(uo[sl] ** 2)))
    # latitude profile of the commutation error (zonal RMS over interior columns)
    M = 40
    lat = m.gphit[js, is_][:, 0]
    err_lat = np.sqrt(np.mean((Da - Db)[:, M:-M] ** 2, 1)); dref = np.sqrt(np.mean(Db[:, M:-M] ** 2, 1))
    return out, dict(Da=Da, Db=Db, lat=lat, err_lat=err_lat, dref=dref, lon=m.glamt[js, is_][0], M=M)


cases = [("W2.0_true_l", (-54, -37, 12, 28), "W2.0", False), ("W2.0_const_l", (-54, -37, 12, 28), "W2.0", True),
         ("R2Ld_true_l", (-48, -31, 0, 15), "R2Ld", False), ("R2Ld_const_l", (-48, -31, 0, 15), "R2Ld", True)]
res, fields = {}, {}
for name, box, cfg, const in cases:
    res[name], fields[name] = run_case(name, box, cfg, const)
    print(name, json.dumps(res[name]["margins"][40]), flush=True)
json.dump(res, open(f"{OUT}/metrics_commute.json", "w"), indent=1)

fig, axs = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
for c, name in enumerate(["W2.0_true_l", "R2Ld_true_l"]):
    F = fields[name]; lat = F["lat"]; lon = F["lon"]
    vmax = np.percentile(np.abs(F["Db"]), 99)
    for r, (key, ttl) in enumerate([("Da", "hdiv(F u, F v)"), ("Db", "F(hdiv(u,v))")]):
        ax = axs[r, 2 * c]
        pc = ax.pcolormesh(lon, lat, F[key], vmin=-vmax, vmax=vmax, cmap="RdBu_r", shading="auto")
        ax.set_title(f"{name}: {ttl}", fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.8)
    ax = axs[0, 2 * c + 1]
    d = F["Da"] - F["Db"]
    pc = ax.pcolormesh(lon, lat, d, vmin=-vmax / 5, vmax=vmax / 5, cmap="RdBu_r", shading="auto")
    M = F["M"]
    ax.add_patch(plt.Rectangle((lon[M], lat[M]), lon[-M] - lon[M], lat[-M] - lat[M], fill=False, ls="--"))
    ax.set_title(f"{name}: difference (colour range /5), dashed = interior", fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.8)
    ax = axs[1, 2 * c + 1]
    ax.plot(lat, F["err_lat"] / F["dref"], "r-", label="true ℓ")
    Fc = fields[name.replace("true", "const")]
    ax.plot(Fc["lat"], Fc["err_lat"] / Fc["dref"], "b-", label="constant ℓ")
    ax.axvspan(lat[0], lat[M], color="0.9"); ax.axvspan(lat[-M], lat[-1], color="0.9")
    ax.set_yscale("log"); ax.set_xlabel("latitude"); ax.set_ylabel("zonal RMS(D_a − D_b) / RMS(D_b)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("commutation error by latitude (interior columns)", fontsize=10)
fig.suptitle("Commutation of the implicit filter with horizontal divergence, surface, 1 Jan 1993 (CPU direct solve on sub-box)", fontsize=12)
fig.savefig(f"{OUT}/fig_commute.png", dpi=130)
