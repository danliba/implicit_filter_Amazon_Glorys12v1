"""Merge the per-task metric files (v3 analysis) into metrics.json with a summary block."""
import json
P = lambda f: json.load(open(f))
sp, spt = P("metrics_spectra_hann.json"), P("metrics_spectra_tukey.json")
spw, spwt = P("metrics_spectra_W_hann.json"), P("metrics_spectra_W_tukey.json")
tv, rs, wm = P("metrics_theory_validation.json"), P("metrics_residual.json"), P("metrics_w.json")
rx = P("metrics_residual_extra.json")
for S in (sp, spt, spw, spwt):
    for b, d in S.items():
        Lmin = min(d["Lx_km"], d["Ly_km"])
        for lev in d["levels"].values():
            for c, x in lev.items():
                x["resolved_by_box"] = bool(x["lambda_half_theory_km"] < Lmin / 2)
leff = lambda S: {b: {l: {c: round(x["l_eff_over_l"], 3) for c, x in lev.items()} for l, lev in d["levels"].items()} for b, d in S.items()}
summ = {
    "filter_version": "v3 (fixed): U,V vector div-rot filter, W scalar filter of rigid-lid W",
    "UV_spectra_l_eff_over_l_hann": leff(sp),
    "W_spectra_l_eff_over_l_hann": leff(spw),
    "synthetic_max_abs_gain_error": tv["synthetic_validation"]["max_abs_gain_error"],
    "residual_decomposition": rs["decomposition_full_month"],
    "amazon_Q_mean_m3s": {k: v["Q_mean_m3s"] for k, v in rs["amazon_mouth_53W45W_2S5N"].items()},
    "hdiv_versions_W2.0_surface_day1": rs["hdiv_versions_W2.0_surface_day1"],
    "W_rms_100_500_1000m_region": wm["rms_100_500_1000m_region"],
}
json.dump({"summary": summ, "spectra_UV_hann": sp, "spectra_UV_tukey": spt, "spectra_W_hann": spw, "spectra_W_tukey": spwt,
           "theory_validation": tv, "continuity_residual": rs, "continuity_residual_edges_and_lat": rx, "w_statistics": wm},
          open("metrics.json", "w"), indent=1)
print("ok")
