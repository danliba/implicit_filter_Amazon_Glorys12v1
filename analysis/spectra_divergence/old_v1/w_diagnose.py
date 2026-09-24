"""
Task 3 follow-up: why is the continuity W of the filtered fields (vovecrtz) ~1000x larger than GLORYS W at the
surface and several times larger at depth?  CPU re-filtering of all 50 levels on sub-boxes (W2.0, 1 Jan 1993).

Variants (each level filtered independently with no-flux at land faces and at the sub-box edge):
  A "production": wet set = finite data. NOTE: GLORYS U/V are NaN only where the T point itself is land, so
                  U/V points lying ON a coastal/topographic wall (umask=0, e3u=0) are finite and enter the filter.
  B "umask":      wet set = true C-grid velocity mask (tmask(i)&tmask(i+1) for U, tmask(j)&tmask(j+1) for V);
                  wall points excluded (their value is irrelevant for fluxes since e3u=0).
  C "transport":  as B but filter the layer transport e3u*u (e3v*v) and divide by e3u (e3v) afterwards
                  (partial bottom cells: velocity filtering does not conserve transport where e3u varies).
  D "normal Dirichlet": as B, but the velocity component normal to a land face is set to 0 on that face
                  (u=0 across east/west land faces, v=0 across north/south land faces; tangential faces stay no-flux).
  E "no-slip":    as B with zero Dirichlet across all land faces.
Diagnostics in the sub-box interior (margin 40 cells): RMS of w at the surface (column-integrated divergence),
at 100/500/1000 m, for ORIG (continuity of original U,V) and each variant; A is compared to the production output.
Output: metrics_w_diagnose.json
"""
import json, sys, types, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C
import cgrid_filter as cf

OUT = C.RUN + "/analysis/spectra_divergence"
m = C.mesh()
T0 = 0
MARGIN = 40
BOXES = {"open_subtropical_54W37W_12N28N": (-54, -37, 12, 28),
         "shelf_NBC_52W40W_0N12N": (-52, -40, 0, 12)}
kw = [int(np.argmin(abs(m.gdepw_0 - z))) for z in (100, 500, 1000)]


def sub(js, is_):
    s = types.SimpleNamespace()
    for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"]:
        setattr(s, v, getattr(m, v)[js, is_])
    for v in ["e3t", "e3u", "e3v", "tmask", "depthw_ps"]:
        setattr(s, v, getattr(m, v)[:, js, is_])
    s.H = m.H[js, is_]
    return s


def filt(s, grid, f, wet, ell, dirichlet=None):
    """dirichlet: None (no-flux at land), 'normal' or 'all' (zero value beyond land faces)."""
    out = np.zeros_like(f)
    if wet.sum() == 0:
        return out
    idx, area, K = cf.build_operator(s, grid, wet, ell)
    if dirichlet is not None:
        s2 = types.SimpleNamespace(**vars(s))
        if dirichlet == "normal":   # suppress tangential faces
            if grid == "U":
                s2.e1f = np.zeros_like(s.e1f)
            else:
                s2.e2f = np.zeros_like(s.e2f)
        allw = np.ones_like(wet)
        _, _, Ka = cf.build_operator(s2, grid, allw, ell)
        _, _, Kw = cf.build_operator(s2, grid, wet, ell)
        extra = Ka.diagonal()[idx] - Kw.diagonal()   # coupling to land neighbours (value 0)
        K = K + sp.diags(extra)
    S = (sp.diags(area) + cf.GAMMA * K).tocsc()
    out.ravel()[idx] = splu(S).solve(area * f.ravel()[idx])
    return out


res = {}
for bname, box in BOXES.items():
    t0 = time.time()
    js0, is0 = C.box_slices(*box, grid="T")
    s = sub(js0, is0)
    uo = C.read("U", "ORIG", t=T0, j=js0, i=is0); vo = C.read("V", "ORIG", t=T0, j=js0, i=is0)
    up = C.read("U", "W2.0", t=T0, j=js0, i=is0); vp = C.read("V", "W2.0", t=T0, j=js0, i=is0)
    lU = cf.ell_from_window_deg(m, "U", 2.0)[js0, is0]; lV = cf.ell_from_window_deg(m, "V", 2.0)[js0, is0]
    tm = m.tmask
    jx, ix = js0, slice(is0.start, is0.stop + 1); jy = slice(js0.start, js0.stop + 1)
    umask = (tm[:, js0, is0] & tm[:, js0, is0.start + 1:is0.stop + 1])
    vmask = (tm[:, js0, is0] & tm[:, js0.start + 1:js0.stop + 1, is0])
    V = {k: (np.zeros_like(uo), np.zeros_like(vo)) for k in "ABCDE"}
    for k in range(m.nz):
        wa_u = np.isfinite(uo[k]); wa_v = np.isfinite(vo[k])
        if wa_u.sum() == 0:
            continue
        u0 = np.nan_to_num(uo[k]); v0 = np.nan_to_num(vo[k])
        V["A"][0][k] = filt(s, "U", u0, wa_u, lU); V["A"][1][k] = filt(s, "V", v0, wa_v, lV)
        V["B"][0][k] = filt(s, "U", u0, umask[k], lU); V["B"][1][k] = filt(s, "V", v0, vmask[k], lV)
        V["D"][0][k] = filt(s, "U", u0, umask[k], lU, "normal"); V["D"][1][k] = filt(s, "V", v0, vmask[k], lV, "normal")
        V["E"][0][k] = filt(s, "U", u0, umask[k], lU, "all"); V["E"][1][k] = filt(s, "V", v0, vmask[k], lV, "all")
        e3u, e3v = s.e3u[k], s.e3v[k]
        tu = filt(s, "U", e3u * u0, umask[k], lU); tv = filt(s, "V", e3v * v0, vmask[k], lV)
        V["C"][0][k] = np.where(e3u > 0, tu / np.where(e3u > 0, e3u, 1), 0)
        V["C"][1][k] = np.where(e3v > 0, tv / np.where(e3v > 0, e3v, 1), 0)
    inner = (slice(MARGIN, -MARGIN), slice(MARGIN, -MARGIN))
    out = {"box": box, "shape": list(uo.shape[1:]), "W_levels_m": [float(m.gdepw_0[k]) for k in [0] + kw]}
    cases = {"ORIG": (uo, vo), "production_output": (up, vp), "A_production_like": V["A"],
             "B_umask": V["B"], "C_transport_umask": V["C"],
             "D_normal_dirichlet": V["D"], "E_noslip_dirichlet": V["E"]}
    wA = None
    for name, (u, v) in cases.items():
        w, wfix, hd = cf.w_from_continuity(s, u, v)
        r = {}
        for lab, ww in [("w", w), ("w_fixed", wfix)]:
            r[lab] = [float(np.sqrt(np.nanmean(ww[k][inner][s.tmask[k][inner]] ** 2))) for k in [0] + kw]
        r["w_top_down"] = [float(np.sqrt(np.nanmean((w[k] - w[0])[inner][s.tmask[k][inner]] ** 2))) for k in [0] + kw]
        r["KE_ratio_surface"] = float(np.nanmean((u[0] ** 2)[inner]) + np.nanmean((v[0] ** 2)[inner]))
        r["hdiv_rms_surface"] = float(np.sqrt(np.nanmean(hd[0][inner] ** 2)))
        out[name] = r
        if name != "ORIG":
            r["KE_ratio_surface"] /= out["ORIG"]["KE_ratio_surface_abs"]
        else:
            r["KE_ratio_surface_abs"] = r["KE_ratio_surface"]
        if name == "production_output":
            wP = w
        if name == "A_production_like":
            wA = w
    out["A_vs_production_w_surface_rel_rms_diff"] = float(np.sqrt(np.nanmean((wA[0] - wP[0])[inner] ** 2)) /
                                                         np.sqrt(np.nanmean(wP[0][inner] ** 2)))
    out["n_wall_points_U_finite_not_umask"] = int((np.isfinite(uo) & ~umask).sum())
    out["n_wall_points_V_finite_not_vmask"] = int((np.isfinite(vo) & ~vmask).sum())
    out["secs"] = time.time() - t0
    res[bname] = out
    print(bname, json.dumps(out, indent=1), flush=True)
json.dump(res, open(f"{OUT}/metrics_w_diagnose.json", "w"), indent=1)
