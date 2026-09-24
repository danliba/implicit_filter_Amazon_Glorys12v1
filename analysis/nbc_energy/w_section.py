"""
Vertical velocity on the NBC section at 44 W (T-point column nearest 44.0 W, coast to 6 N, 0-1000 m).
ORIG = rigid-lid GLORYS W (data/variables_c/UVW/W_1993-01fc.nc); filtered = output/<CFG>/W_* (scalar
implicit filter of that product at the same l_T, v3). W is on W levels (top of T cells, gdepw_0),
positive upward. Plots the 31-day mean in m/day for ORIG, W2.0, R2Ld.
Output: fig_w_section_44W.png, metrics_wsection.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
m = C.mesh()
KMAX = 37
i = int(np.argmin(np.abs(m.glamt[150] + 44.0)))
jj = np.arange(80, 200)
jj = jj[m.gphit[jj, i] <= 6.0]
jj = jj[np.argmax(m.tmask[0, jj, i]):]
lat = m.gphit[jj, i]
zw = m.gdepw_0[:KMAX]
CF = [c for c in ["ORIG", "W2.0", "R2Ld"] if c == "ORIG" or os.path.exists(f"{C.RUN}/output/{c}/diag_{c}.json")]
wet = m.tmask[:KMAX, jj, i]
mean, daily_rms = {}, {}
for c in CF:
    w = C.read("W", c, k=slice(0, KMAX), j=slice(jj[0], jj[-1] + 1), i=i)
    w = np.where(wet[None], w, np.nan) * 86400.0            # m/day
    mean[c] = np.nanmean(w, 0)
    daily_rms[c] = float(np.sqrt(np.nanmean(w ** 2)))
res = {"lon": float(m.glamt[150, i]), "units": "m/day, positive upward"}
o = mean["ORIG"]; ok = np.isfinite(o)
for c in CF:
    x = mean[c]; okc = ok & np.isfinite(x)
    res[c] = dict(rms_mean=float(np.sqrt(np.mean(x[okc] ** 2))), rms_daily=daily_rms[c],
                  corr_with_ORIG=float(np.corrcoef(o[okc], x[okc])[0, 1]),
                  max_up=float(np.nanmax(x)), max_down=float(np.nanmin(x)))
json.dump(res, open(f"{H}/metrics_wsection.json", "w"), indent=1)
fig, axs = plt.subplots(1, len(CF), figsize=(5.2 * len(CF), 4.6), sharey=True, constrained_layout=True)
vmax = 20.0
for ax, c in zip(np.atleast_1d(axs), CF):
    pc = ax.pcolormesh(lat, zw, mean[c], vmin=-vmax, vmax=vmax, cmap="RdBu_r", shading="nearest")
    ax.set_facecolor("0.6"); ax.set_ylim(1000, 0); ax.set_xlabel("latitude (°)")
    ax.set_title(f"{C.LABELS[c]}\nrms {res[c]['rms_mean']:.1f} m/d, r(ORIG) {res[c]['corr_with_ORIG']:.2f}", fontsize=10)
np.atleast_1d(axs)[0].set_ylabel("depth (m)")
fig.colorbar(pc, ax=axs, shrink=0.85, label="31-day mean w (m/day, >0 upward)")
fig.suptitle(f"Vertical velocity at {abs(res['lon']):.2f}°W (rigid-lid W; filtered W = scalar filter, v3)")
fig.savefig(f"{H}/fig_w_section_44W.png", dpi=130)
print(json.dumps(res, indent=1))
