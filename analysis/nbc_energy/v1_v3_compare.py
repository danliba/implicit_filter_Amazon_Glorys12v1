"""
Compact comparison of the component-wise v1 filter (output_v1_componentwise/, results saved in v1/)
with the v3 vector div-rot filter (output/) for W2.0 and R2Ld.
 - NBC core speed and transport at 44W and 5N (from v1/metrics_sections.json, metrics_sections.json)
 - KE / EKE removed (0-200 m, 0-1000 m, full) in 70W-30W, 5S-30N
 - surface (0.5 m) 31-day-mean speed at T points within 50 km of the coast (box 70W-30W, 5S-30N),
   area-weighted mean, from the stored full-domain mean u,v (v1/energy_maps.npz, data/energy_maps.npz).
   Distance to coast = Euclidean distance transform on the level-0 land mask with local e1t, e2t.
   (T-point averaging counts land faces as 0, identically for ORIG, v1 and v3.)
 - per-level fraction of KE removed and residual KE for v1 vs v3 (figure fig_v1_v3_profiles.png)
Output: metrics_v1_v3.json
"""
import json, os, sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
m = C.mesh()
CF = os.environ.get("CMP_CFGS", "W2.0,R2Ld").split(",")
S1, S3 = json.load(open(f"{H}/v1/metrics_sections.json")), json.load(open(f"{H}/metrics_sections.json"))
E1, E3 = json.load(open(f"{H}/v1/metrics_energy.json")), json.load(open(f"{H}/metrics_energy.json"))
M1, M3 = np.load(f"{H}/v1/energy_maps.npz"), np.load(f"{H}/data/energy_maps.npz")

land = ~m.tmask[0]
# distance in km using local grid spacing (grid nearly isotropic Mercator)
dist_km = ndimage.distance_transform_edt(~land) * np.sqrt(m.e1t * m.e2t) / 1e3
box = (m.glamt >= -70) & (m.glamt <= -30) & (m.gphit >= -5) & (m.gphit <= 30) & ~land
coast = box & (dist_km <= 50.0)
A = m.e1t * m.e2t


def coast_speed(M, c):
    sp = np.hypot(M[f"UMfull_{c}_k0"], M[f"VMfull_{c}_k0"]).astype(float)
    return float(np.nansum(sp[coast] * A[coast]) / np.sum(A[coast] * np.isfinite(sp[coast])))


out = {"coastal_band": "wet T points within 50 km of land, 70W-30W 5S-30N, level 0 (0.5 m)",
       "ORIG_coastal_mean_speed_ms": coast_speed(M3, "ORIG")}
for c in CF:
    d = {}
    for ver, S, E, M in (("v1", S1, E1, M1), ("v3", S3, E3, M3)):
        a, b = S["A"]["configs"][c], S["B"]["configs"][c]
        e = E["box_70W-30W_5S-30N"]
        core3 = {}
        for sec in "AB":
            if "core3_velocity_monthly_mean" in S[sec]["configs"][c]:
                core3[sec] = (S[sec]["configs"][c]["core3_velocity_monthly_mean"], S[sec]["configs"]["ORIG"]["core3_velocity_monthly_mean"])
            else:   # v1 metrics predate core3: recompute from the stored v1 monthly-mean sections
                sys.path.insert(0, H); import nbc_sections as NS
                z = np.load(f"{H}/v1/section_{sec}.npz"); sg = -1.0 if sec == "A" else 1.0
                iend = S[sec]["offshore_limit_index"]; reg = np.zeros(z["dist"].size, bool); reg[:iend] = True
                kreg = NS.zt <= 1000.0
                core3[sec] = (NS.core3(sg * z[f"mean_{c}"], reg, kreg)[0], NS.core3(sg * z["mean_ORIG"], reg, kreg)[0])
        d[ver] = {"NBC44W_core_ms": a["core_velocity_monthly_mean"], "NBC44W_core_pct": a["pct_change"]["core_velocity_monthly_mean"],
                  "NBC44W_T0_1000_Sv": a["T_mean_of_daily_0_1000m"], "NBC44W_T_pct": a["pct_change"]["T_mean_of_daily_0_1000m"],
                  "NBC44W_net_full_section_pct": a["pct_change"]["Tnet_full_section_0_1000m"],
                  "NBC44W_core3pt_pct": 100 * (core3["A"][0] / core3["A"][1] - 1),
                  "NBC5N_core3pt_pct": 100 * (core3["B"][0] / core3["B"][1] - 1),
                  "NBC5N_core_ms": b["core_velocity_monthly_mean"], "NBC5N_core_pct": b["pct_change"]["core_velocity_monthly_mean"],
                  "NBC5N_T0_1000_Sv": b["T_mean_of_daily_0_1000m"], "NBC5N_T_pct": b["pct_change"]["T_mean_of_daily_0_1000m"],
                  "KE_removed_0-200m": e["0-200m"][c]["KE_removed"], "KE_removed_0-1000m": e["0-1000m"][c]["KE_removed"],
                  "KE_removed_full": e["full"][c]["KE_removed"],
                  "EKE_removed_0-200m": e["0-200m"][c]["EKE_removed"], "EKE_removed_0-1000m": e["0-1000m"][c]["EKE_removed"],
                  "EKE_removed_full": e["full"][c]["EKE_removed"],
                  "coastal50km_surface_speed_ms": coast_speed(M, c)}
        d[ver]["coastal50km_speed_ratio_to_ORIG"] = d[ver]["coastal50km_surface_speed_ms"] / out["ORIG_coastal_mean_speed_ms"]
    out[c] = d
json.dump(out, open(f"{H}/metrics_v1_v3.json", "w"), indent=1)
for c in CF:
    for ver in ("v1", "v3"):
        print(c, ver, {k: round(v, 3) for k, v in out[c][ver].items()})

# per-level profiles
P1, P3 = np.load(f"{H}/v1/energy_profiles.npz"), np.load(f"{H}/data/energy_profiles.npz")
zt = m.gdept_0
fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True, constrained_layout=True)
for P, ls, ver in ((P1, "--", "v1 component-wise"), (P3, "-", "v3 vector")):
    cf = list(P["cfgs"])
    KE = P["KEtot"].sum(-1); KM = P["KEmean"].sum(-1); SS = P["SS"].sum(-1)
    ok = P["vol"].sum(-1) > 0
    for c in CF:
        ic = cf.index(c)
        kw = dict(color=C.COLORS[c], ls=ls, label=f"{c} {ver}")
        axs[0].plot(1 - KE[ic][ok] / KE[0][ok], zt[ok], **kw)
        axs[1].plot(1 - (KE[ic] - KM[ic])[ok] / (KE[0] - KM[0])[ok], zt[ok], **kw)
        axs[2].plot(SS[ic][ok] / KE[0][ok], zt[ok], **kw)
for ax, t in zip(axs, ["KE removed 1−KE$_f$/KE$_o$", "EKE removed", "residual KE ½<|u$_o$−u$_f$|²> / KE$_o$"]):
    ax.set_title(t); ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.set_xlabel("fraction")
axs[0].set_yscale("symlog", linthresh=100); axs[0].set_ylim(5500, 0); axs[0].set_ylabel("depth (m)")
axs[0].legend(fontsize=8)
fig.suptitle("v1 (component-wise) vs v3 (vector div–rot): energy removed per level, 70°W–30°W, 5°S–30°N")
fig.savefig(f"{H}/fig_v1_v3_profiles.png", dpi=130)
