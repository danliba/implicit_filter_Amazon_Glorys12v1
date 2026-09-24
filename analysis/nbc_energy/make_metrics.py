"""Merge metrics_sections.json, metrics_energy.json, metrics_largescale.json into metrics.json with a summary table."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{H}/metrics_sections.json"))
E = json.load(open(f"{H}/metrics_energy.json"))
L = json.load(open(f"{H}/metrics_largescale.json"))
V13 = json.load(open(f"{H}/metrics_v1_v3.json"))
WS = json.load(open(f"{H}/metrics_wsection.json"))
CF = ["W1.5", "W2.0", "W2.5", "R2Ld", "R3Ld"]
summary = {}
for c in CF:
    e = E["box_70W-30W_5S-30N"]
    rings = [d["configs"][c] for d in L["rings"]["dates"].values()]
    summary[c] = {
        "KE_removed_0-200m": e["0-200m"][c]["KE_removed"], "KE_removed_0-1000m": e["0-1000m"][c]["KE_removed"],
        "KE_removed_full": e["full"][c]["KE_removed"],
        "EKE_removed_0-200m": e["0-200m"][c]["EKE_removed"], "EKE_removed_0-1000m": e["0-1000m"][c]["EKE_removed"],
        "EKE_removed_full": e["full"][c]["EKE_removed"],
        "KEmean_removed_0-1000m": e["0-1000m"][c]["KEmean_removed"],
        "residual_KE_frac_0-1000m": e["0-1000m"][c]["SS_frac"],
        "NBC44W_core_velocity_change_pct": S["A"]["configs"][c]["pct_change"]["core_velocity_monthly_mean"],
        "NBC44W_core_dailymax_change_pct": S["A"]["configs"][c]["pct_change"]["core_velocity_mean_daily_max"],
        "NBC44W_transport_0-1000m_change_pct": S["A"]["configs"][c]["pct_change"]["T_mean_of_daily_0_1000m"],
        "NBC44W_transport_0-300m_change_pct": S["A"]["configs"][c]["pct_change"]["T_mean_of_daily_0_300m"],
        "NBC44W_net_full_section_change_pct": S["A"]["configs"][c]["pct_change"]["Tnet_full_section_0_1000m"],
        "NBC5N_core_velocity_change_pct": S["B"]["configs"][c]["pct_change"]["core_velocity_monthly_mean"],
        "NBC5N_transport_0-1000m_change_pct": S["B"]["configs"][c]["pct_change"]["T_mean_of_daily_0_1000m"],
        "retroflection_pattern_corr_surface": L["circulation"]["z0m"][c]["retroflection_52W-42W_3N-10N"]["pattern_corr"],
        "retroflection_pattern_corr_92m": L["circulation"]["z92m"][c]["retroflection_52W-42W_3N-10N"]["pattern_corr"],
        "NECC_pattern_corr_surface": L["circulation"]["z0m"][c]["NECC_40W-20W_4N-10N"]["pattern_corr"],
        "NEC_pattern_corr_surface": L["circulation"]["z0m"][c]["NEC_60W-30W_10N-20N"]["pattern_corr"],
        "largescale10deg_rel_rms_diff_surface": L["large_scale_10deg"]["z0m"][c]["rel_rms_diff"],
        "largescale10deg_rel_rms_diff_92m": L["large_scale_10deg"]["z92m"][c]["rel_rms_diff"],
        "ring_KE_ratio_mean_3dates": sum(r["KE_circle_ratio"] for r in rings) / len(rings),
        "ring_zeta_ratio_mean_3dates": sum(r["mean_zeta_ratio"] for r in rings) / len(rings),
    }
out = {"conventions": {
    "section_A": "44W (glamu=-44.04), U-grid, NBC transport = -sum(u<0) e2u e3u from coast (2.42S) to 1.67N (ORIG mean zero crossing of 0-1000 m integrated u), positive westward, Sv",
    "section_B": "5.035N, V-grid, NBC transport = +sum(v>0) e1v e3v from coast (52.5W) to 47.92W, positive northward, Sv",
    "energy": "T-point KE, weights e1t*e2t*e3t, box 70W-30W 5S-30N; EKE = KE of daily anomalies from 31-day mean",
    "note": L["note"]},
    "filter_version": "v3 fixed (vector div-rot U,V with e3f in the curl term; scalar-filtered rigid-lid W)",
    "summary": summary, "v1_vs_v3": V13, "w_section_44W": WS, "sections": S, "energy": E, "large_scale": L}
json.dump(out, open(f"{H}/metrics.json", "w"), indent=1)
for c in CF:
    print(c, {k: round(v, 3) for k, v in summary[c].items()})
