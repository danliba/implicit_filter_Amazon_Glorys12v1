"""
Task 5 (v3): how far does an artificial no-flux wall (the cut of the regional domain) influence the
filtered field?  CPU re-run of the production operators on sub-boxes, one day (1 Jan 1993):
  * U,V  jointly with cgrid_filter.filter_level_vector (div-rot operator, zero normal flow + free slip), surface
  * W    with the scalar T-point filter (cgrid_filter.filter_level, grid "T"), W level nearest 100 m
Same l fields as production (run_filter.ell_fields).

Two experiments on the longitude band 80W-20W (identical E/W boundaries in both runs):
  NORTH : wall at 25N (box 10N-25N)   vs reference with halo up to the real edge 30N (box 10N-30N)
  SOUTH : wall at 0N  (box 0N-20N)    vs reference with halo down to the real edge 10S (box 10S-20N)
Metric per row: RMS(wall - reference) / RMS(reference filtered field); for U,V the two components are
pooled (vector RMS). The 1-D Neumann Green's function predicts decay exp(-d/lambda), lambda = l/sqrt(2).

Output: fig_edge_test.png, cache/metrics_edge.json
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
from common import mesh, read, CONFIGS, COLORS  # noqa: E402
import cgrid_filter as cf  # noqa: E402
import run_filter as rf  # noqa: E402  (only ell_fields is used)

HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh()
T = 0
KW = int(np.argmin(np.abs(m.gdepw_0 - 100)))
LON0, LON1 = -80.0, -20.0
ic = np.where((m.glamt[250] >= LON0) & (m.glamt[250] <= LON1))[0]
isl = slice(ic.min(), ic.max() + 1)
latrow = m.gphit[:, (ic.min() + ic.max()) // 2]
ELL = {c: rf.ell_fields(m, c) for c in CONFIGS}


def row(lat):
    return int(np.argmin(np.abs(latrow - lat)))


class Sub:
    """Sliced mesh view with the attributes the operators use."""
    def __init__(self, jsl):
        for v in ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"]:
            setattr(self, v, getattr(m, v)[jsl, isl])
        for v in ["e3t", "e3u", "e3v", "tmask"]:
            setattr(self, v, getattr(m, v)[:, jsl, isl])
        self.e3t_0 = m.e3t_0


def filt_uv(u, v, cfg, jsl):
    lT, lF = ELL[cfg]
    uo, vo, its, n = cf.filter_level_vector(Sub(jsl), 0, u[jsl, isl][None], v[jsl, isl][None],
                                            lT[jsl, isl], lF[jsl, isl], backend="cpu", tol=1e-10)
    return uo[0].astype(float), vo[0].astype(float)


def filt_w(w, cfg, jsl):
    lT, _ = ELL[cfg]
    wet = m.tmask[KW][jsl, isl] & np.isfinite(w[jsl, isl])
    wm = np.where(wet, w[jsl, isl], np.nan)[None]
    out = cf.filter_level(Sub(jsl), "T", wm, lT[jsl, isl], backend="cpu", tol=1e-10)[0][0]
    return np.where(wet, out, np.nan).astype(float)


exps = {"NORTH": dict(wall=row(25.0), ref=(row(10.0), row(30.0) + 1), box=(row(10.0), row(25.0)), ll=25.0),
        "SOUTH": dict(wall=row(0.0), ref=(0, row(20.0) + 1), box=(row(0.0), row(20.0) + 1), ll=0.0)}
u0 = read("U", "ORIG", t=T, k=0); v0 = read("V", "ORIG", t=T, k=0); w0 = read("W", "ORIG", t=T, k=KW)
res = {}
fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
for r, (name, e) in enumerate(exps.items()):
    jr = slice(*e["ref"]); jb = slice(*e["box"])
    j0 = jb.start - jr.start
    nrow = jb.stop - jb.start
    dkm = np.abs(latrow[jb] - latrow[e["wall"]]) * 111.2
    order = np.argsort(dkm); dk = dkm[order]
    for c, var in enumerate(["UV", "W"]):
        ax = axs[r, c]
        for cfg in CONFIGS:
            if var == "UV":
                ur, vr = filt_uv(u0, v0, cfg, jr); ub, vb = filt_uv(u0, v0, cfg, jb)
                ur, vr = ur[j0:j0 + nrow], vr[j0:j0 + nrow]
                num = np.nanmean((ub - ur) ** 2, 1) + np.nanmean((vb - vr) ** 2, 1)
                den = np.nanmean(ur ** 2, 1) + np.nanmean(vr ** 2, 1)
            else:
                wr = filt_w(w0, cfg, jr)[j0:j0 + nrow]; wb = filt_w(w0, cfg, jb)
                num = np.nanmean((wb - wr) ** 2, 1); den = np.nanmean(wr ** 2, 1)
            rl = np.sqrt(num / den)[order]
            lT = ELL[cfg][0][jb, isl]
            ell_wall = float(np.median(lT[np.argmin(dkm)])) / 1e3

            def first_below(x, thr):
                ok = np.where(np.maximum.accumulate(x[::-1])[::-1] < thr)[0]
                return float(dk[ok[0]]) if ok.size else float("nan")
            res[f"{name}_{var}_{cfg}"] = dict(ell_at_wall_km=ell_wall, lambda_km=ell_wall / np.sqrt(2),
                                             rel_err_first_row=float(rl[0]),
                                             dist_km_rel_err_below_10pct=first_below(rl, 0.10),
                                             dist_km_rel_err_below_5pct=first_below(rl, 0.05),
                                             dist_km_rel_err_below_1pct=first_below(rl, 0.01))
            ax.semilogy(dk, rl, color=COLORS[cfg], label=f"{cfg} (l={ell_wall:.0f} km)")
            ax.semilogy(dk, rl[0] * np.exp(-(dk - dk[0]) / (ell_wall / np.sqrt(2))), color=COLORS[cfg], ls=":", lw=0.8)
            print(name, var, cfg, res[f"{name}_{var}_{cfg}"], flush=True)
        ax.axhline(0.05, color="0.5", lw=0.6); ax.axhline(0.01, color="0.5", lw=0.6, ls="--")
        ax.set_ylim(1e-4, 2); ax.set_xlim(0, 1100)
        lab = "U,V vector filter, surface" if var == "UV" else f"W scalar filter, {m.gdepw_0[KW]:.0f} m"
        ax.set_title(f"{name} wall ({e['ll']:.0f}°N), {lab}", fontsize=10)
        ax.set_xlabel("distance from artificial wall [km]")
        if c == 0:
            ax.set_ylabel("RMS(wall − halo) / RMS(filtered, halo)")
        ax.legend(fontsize=7)
fig.suptitle("v3: effect of a no-flux domain cut on the filtered fields, 1 Jan 1993 (dotted: exp(−d·√2/l))")
fig.tight_layout()
fig.savefig(f"{HERE}/fig_edge_test.png", dpi=130)
json.dump(res, open(f"{HERE}/cache/metrics_edge.json", "w"), indent=1)
