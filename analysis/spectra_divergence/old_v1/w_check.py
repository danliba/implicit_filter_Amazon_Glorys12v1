"""
Task 3: kinematic consistency of the output W.
(a) Config W2.0 (argv[1]), days 1-3: recompute w, w_fixed from the stored filtered U,V with
    cgrid_filter.w_from_continuity and compare with vovecrtz / vovecrtz_fixed (float32 round-off expected).
    Checks: max|diff|; w_floor = w(top of bottom cell) + e3t*hdiv(bottom cell) == 0; surface w statistics;
    vovecrtz_fixed == 0 at the surface; residual of continuity for w_fixed (it violates continuity by w0/H).
(b) RMS of W at W levels nearest 100, 500, 1000 m over 70-30W, 5S-30N (wet points, 31 days) for ORIG
    (original GLORYS W file) and each config (vovecrtz and vovecrtz_fixed).
Output: metrics_w.json, fig_w_rms.png
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
m = C.mesh()
CFG = sys.argv[1] if len(sys.argv) > 1 else "W2.0"
met = {"consistency_config": CFG, "days": {}}

wf = netCDF4.Dataset(C.path("W", CFG))
kb = m.mbathy - 1          # index of bottom wet T cell
wetcol = m.mbathy > 0
jj, ii = np.nonzero(wetcol)
for t in range(3):
    u = C.read("U", CFG, t=t); v = C.read("V", CFG, t=t)
    w, wfix, hd = cf.w_from_continuity(m, u, v)
    ws = wf["vovecrtz"][t].filled(np.nan).astype(np.float64)
    wsf = wf["vovecrtz_fixed"][t].filled(np.nan).astype(np.float64)
    mk = m.tmask
    d = np.abs(ws - w)[mk]; dfx = np.abs(wsf - wfix)[mk]
    # stored field float32: expected error ~ 6e-8 * |w|
    rel = d / np.maximum(np.abs(w[mk]), 1e-12)
    # sea floor: w_floor = w(kb) + e3t(kb) hdiv(kb)  (NEMO: w(kb) = w(kb+1) - e3t hdiv)
    wb = ws[kb[wetcol], jj, ii]
    e3b = m.e3t[kb[wetcol], jj, ii]; hb = hd[kb[wetcol], jj, ii]
    wfloor = wb + e3b * hb
    # continuity residual for stored w: dw/dz_k*e3t + e3t*hdiv  -> (w(k) - w(k+1)) + e3t hdiv = 0
    wk1 = np.concatenate([ws[1:], np.zeros((1,) + ws.shape[1:])], 0)
    wk1 = np.where(np.concatenate([mk[1:], np.zeros((1,) + mk.shape[1:], bool)], 0), wk1, 0.0)
    resid = ((ws - wk1) + m.e3t * np.nan_to_num(hd))[mk]
    wfk1 = np.concatenate([wsf[1:], np.zeros((1,) + ws.shape[1:])], 0)
    wfk1 = np.where(np.concatenate([mk[1:], np.zeros((1,) + mk.shape[1:], bool)], 0), wfk1, 0.0)
    resid_fix = ((wsf - wfk1) + m.e3t * np.nan_to_num(hd))[mk]
    s0 = ws[0][m.tmask[0]]; s0f = wsf[0][m.tmask[0]]
    met["days"][t] = dict(
        max_abs_diff_vovecrtz=float(d.max()), rms_diff_vovecrtz=float(np.sqrt(np.mean(d ** 2))),
        median_rel_diff=float(np.median(rel)), p999_rel_diff=float(np.percentile(rel, 99.9)),
        max_abs_diff_vovecrtz_fixed=float(dfx.max()),
        n_nan_stored_in_wet=int(np.isnan(ws[mk]).sum()), n_finite_stored_on_land=int(np.isfinite(ws[~mk]).sum()),
        w_floor_max_abs=float(np.abs(wfloor).max()), w_floor_rms=float(np.sqrt(np.mean(wfloor ** 2))),
        w_bottom_cell_top_rms=float(np.sqrt(np.mean(wb ** 2))),
        continuity_residual_max_abs_vovecrtz=float(np.abs(resid).max()),
        continuity_residual_rms_vovecrtz_fixed=float(np.sqrt(np.mean(resid_fix ** 2))),
        surface_w_rms=float(np.sqrt(np.mean(s0 ** 2))), surface_w_maxabs=float(np.abs(s0).max()),
        surface_w_mean=float(s0.mean()),
        surface_w_fixed_maxabs=float(np.abs(s0f).max()),
    )
    print(t, json.dumps(met["days"][t]), flush=True)

# ORIG surface w for context
wo = netCDF4.Dataset(C.path("W", "ORIG"))["vovecrtz"]
s0o = wo[0, 0].filled(np.nan)[m.tmask[0]]
met["orig_surface_w_day1"] = dict(rms=float(np.sqrt(np.nanmean(s0o ** 2))), maxabs=float(np.nanmax(np.abs(s0o))))

# ------------------------------------------------------------------ (b) RMS W at 100/500/1000 m
js, is_ = C.box_slices(-70, -30, -5, 30, grid="T")
kw = [int(np.argmin(abs(m.gdepw_0 - zz))) for zz in (100, 500, 1000)]
met["rms_w_70W30W_5S30N"] = {"levels_m": [float(m.gdepw_0[k]) for k in kw]}
cfgs = C.available_configs()
for cfg in ["ORIG"] + cfgs:
    ds = netCDF4.Dataset(C.path("W", cfg))
    out = {}
    for name in (["vovecrtz"] if cfg == "ORIG" else ["vovecrtz", "vovecrtz_fixed"]):
        vals, vtop = [], []
        a0 = ds[name][:, 0, js, is_].filled(np.nan).astype(np.float64)
        for k in kw:
            a = ds[name][:, k, js, is_].filled(np.nan).astype(np.float64)
            mk = m.tmask[k, js, is_]
            vals.append(float(np.sqrt(np.nanmean(a[:, mk] ** 2))))
            vtop.append(float(np.sqrt(np.nanmean((a - a0)[:, mk] ** 2))))
        out[name] = vals
        if name == "vovecrtz":
            # top-down continuity (w=0 at the surface, integrated downward): w_top(k) = w(k) - w(0)
            out["w_top_down"] = vtop
    met["rms_w_70W30W_5S30N"][cfg] = out
    print(cfg, out, flush=True)

json.dump(met, open(f"{OUT}/metrics_w.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
x = np.arange(3); wd = 0.8 / (1 + len(cfgs))
r = met["rms_w_70W30W_5S30N"]
for n, cfg in enumerate(["ORIG"] + cfgs):
    ax.bar(x + n * wd, np.array(r[cfg]["vovecrtz"]) * 86400, wd, color=C.COLORS[cfg], label=cfg, alpha=0.45)
    ax.bar(x + n * wd, np.array(r[cfg]["w_top_down"]) * 86400, wd * 0.5, color=C.COLORS[cfg], hatch="//", edgecolor="k", lw=0.3)
ax.set_xticks(x + 0.4 - wd / 2); ax.set_xticklabels([f"{z:.0f} m" for z in r["levels_m"]])
ax.set_ylabel("RMS w [m/day]"); ax.set_yscale("log"); ax.legend(fontsize=9, ncol=2); ax.grid(axis="y", alpha=0.3)
ax.set_title("RMS w, 70-30W 5S-30N, 31 days: vovecrtz (bars), top-down w (hatched)")
fig.savefig(f"{OUT}/fig_w_rms.png", dpi=130)

# ------------------------------------------------------------------ (c) RMS w profiles (day 1, 70-30W 5S-30N)
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True, constrained_layout=True)
zw = m.gdepw_0
prof = {}
wo1 = netCDF4.Dataset(C.path("W", "ORIG"))["vovecrtz"][0, :, js, is_].filled(np.nan).astype(float)
mk3 = m.tmask[:, js, is_]
rmsz = lambda a: np.array([np.sqrt(np.nanmean(a[k][mk3[k]] ** 2)) if mk3[k].sum() > 100 else np.nan for k in range(m.nz)])
po = rmsz(wo1)
for ax in axs:
    ax.semilogx(po * 86400, zw, "k-", lw=2.2, label="ORIG (GLORYS W)")
for cfg in cfgs:
    ds = netCDF4.Dataset(C.path("W", cfg))
    a = ds["vovecrtz"][0, :, js, is_].filled(np.nan).astype(float)
    af = ds["vovecrtz_fixed"][0, :, js, is_].filled(np.nan).astype(float)
    p1, p2, p3 = rmsz(a), rmsz(af), rmsz(a - a[0][None])
    axs[0].semilogx(p1 * 86400, zw, color=C.COLORS[cfg], label=cfg)
    axs[1].semilogx(p2 * 86400, zw, color=C.COLORS[cfg], label=cfg)
    axs[2].semilogx(p3 * 86400, zw, color=C.COLORS[cfg], label=cfg)
    prof[cfg] = dict(vovecrtz=p1.tolist(), vovecrtz_fixed=p2.tolist(), top_down=p3.tolist())
prof["ORIG"] = po.tolist(); prof["depthw"] = zw.tolist()
for ax, t in zip(axs, ["vovecrtz (bottom-up continuity, w=0 at floor)", "vovecrtz_fixed (linear rigid-lid correction)",
                       "top-down continuity w(k)-w(0) (w=0 at surface)"]):
    ax.set_title(t, fontsize=10); ax.grid(which="both", alpha=0.3); ax.set_xlabel("RMS w [m/day]")
axs[0].set_ylim(5500, 0); axs[0].set_ylabel("depth [m]"); axs[0].legend(fontsize=9)
fig.suptitle("RMS vertical velocity vs depth, 70-30W 5S-30N, 1 Jan 1993", fontsize=12)
fig.savefig(f"{OUT}/fig_w_profiles.png", dpi=130)
met["rms_w_profile_day1_70W30W_5S30N"] = prof
json.dump(met, open(f"{OUT}/metrics_w.json", "w"), indent=1)
