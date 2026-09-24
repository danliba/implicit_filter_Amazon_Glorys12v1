"""Collect all cache/metrics_*.json into metrics.json and derive the domain-edge margin rule."""
import json, glob, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
M = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/cache/metrics_*.json"))}
e = M["edge"]
r1 = [v["dist_km_rel_err_below_1pct"] / v["lambda_km"] for v in e.values()]
r5 = [v["dist_km_rel_err_below_5pct"] / v["lambda_km"] for v in e.values()]
first = [v["rel_err_first_row"] for v in e.values()]
Ld = {"30N": 33.0, "10S": 98.0}
ell = {"W1.5": {"30N": 1.5, "10S": 1.5}, "W2.0": {}, "W2.5": {}}
R = 6371.0
def l_at(cfg, lat, ld):
    if cfg.startswith("W"):
        return math.radians(float(cfg[1:])) * R * math.cos(math.radians(lat)) / 3.5
    return float(cfg[1]) * ld / 3.5
margin = {}
k1 = max(r1); k5 = max(r5)
for cfg in ["W1.5", "W2.0", "W2.5", "R2Ld", "R3Ld"]:
    for edge, lat in [("30N", 30.0), ("10S", -10.0)]:
        l = l_at(cfg, lat, Ld[edge]); lam = l / math.sqrt(2)
        margin[f"{cfg}_{edge}"] = dict(l_km=l, lambda_km=lam, margin_5pct_km=k5 * lam, margin_1pct_km=k1 * lam,
                                        margin_1pct_deg_lat=k1 * lam / 111.2)
M["edge_summary"] = dict(rel_err_at_wall_range=[min(first), max(first)],
                         d1pct_over_lambda_range=[min(r1), max(r1)], d5pct_over_lambda_range=[min(r5), max(r5)],
                         note="lambda = l/sqrt(2); margins use the largest d/lambda of the CPU test and l at the edge "
                              "(Ld at the 10S edge ~98 km, at 30N ~33 km; Ld grows equatorward so 10S margins for R configs are lower bounds)",
                         margins=margin)
json.dump(M, open(f"{HERE}/metrics.json", "w"), indent=1)
print(json.dumps(M["edge_summary"], indent=1))

# ----------------------------------------------------------------------------- v1 (component-wise) vs v3 comparison
V1 = json.load(open(f"{HERE}/v1/metrics.json"))
CF = ["W1.5", "W2.0", "W2.5", "R2Ld", "R3Ld"]
cmp = {}


def both(fn):
    out = {}
    for c in CF:
        try:
            out[c] = {"v1": fn(V1, c, "v1"), "v3": fn(M, c, "v3")}
        except (KeyError, TypeError, IndexError) as ex:
            out[c] = {"error": repr(ex)}
    return out


cmp["coastal_speed_ratio_0-45km_bins"] = both(lambda d, c, v: d["stats"]["coastdist"][c]["k0"]["speed_ratio"][:4])
cmp["coastal_speed_ratio_max_0-45km"] = both(lambda d, c, v: max(d["stats"]["coastdist"][c]["k0"]["speed_ratio"][:4]))
cmp["first_wet_faces_rms_mean_onshore_cm_s(orig 12.35)"] = both(
    lambda d, c, v: 100 * d["stats"]["coastal_normal"][c]["surface"]["rms_mean_onshore_filt"])
cmp["first_wet_faces_corr_with_orig"] = both(lambda d, c, v: d["stats"]["coastal_normal"][c]["surface"]["corr_mean_onshore"])
cmp["first_wet_faces_sum_abs_transport_Sv(orig 0.99)"] = both(
    lambda d, c, v: d["stats"]["coastal_normal"][c]["surface"]["sum_abs_onshore_transport_Sv_filt"])
for n in ["St Vincent Passage", "St Lucia Channel", "Grenada Passage (Tobago–Grenada)", "Guadeloupe Passage (Dominica–Guadeloupe)",
          "Dragon's Mouth"]:
    cmp[f"transport_Sv_{n}"] = both(lambda d, c, v, n=n: d["passages"]["sections"][n][c]["mean_Sv"])
    cmp[f"transport_Sv_{n}"]["ORIG"] = M["passages"]["sections"][n]["ORIG"]["mean_Sv"]
cmp["transport_Sv_arc_sum"] = both(lambda d, c, v: d["passages"]["sections"]["sum_arc_Galleons_to_Guadeloupe"][c]["mean_Sv"])
cmp["transport_Sv_arc_sum"]["ORIG"] = M["passages"]["sections"]["sum_arc_Galleons_to_Guadeloupe"]["ORIG"]["mean_Sv"]
cmp["shelf_depthmean_flow_KE_retained"] = both(lambda d, c, v: d["stats"]["regions"][c]["shelf_H<200"]["depth_avg_mean_flow"]["KE_retained"])
cmp["shelf_fulldepth_MKE_retained"] = both(lambda d, c, v: d["stats"]["regions"][c]["shelf_H<200"]["full_depth"]["MKE_retained"])
cmp["deep_fulldepth_EKE_retained"] = both(lambda d, c, v: d["stats"]["regions"][c]["deep_H>3000"]["full_depth"]["EKE_retained"])
cmp["slope_vs_deep_speed_ratio_1062m"] = {c: {v: [d["maps"].get(f"depth1062m_{c}_slope_H<3000_speed_ratio"),
                                                  d["maps"].get(f"depth1062m_{c}_deep_H>3000_speed_ratio")]
                                              for v, d in [("v1", V1), ("v3", M)]} for c in ["W2.0", "R2Ld"]}


def edge(d, c, v):
    if v == "v1":
        ks = [f"{w}_{g}_{c}" for w in ["NORTH", "SOUTH"] for g in ["U", "V"]]
    else:
        ks = [f"{w}_UV_{c}" for w in ["NORTH", "SOUTH"]]
    e = d["edge"]
    return dict(first_row=max(e[k]["rel_err_first_row"] for k in ks),
                north_d5_km=max(e[k]["dist_km_rel_err_below_5pct"] for k in ks if k.startswith("NORTH")),
                north_d1_km=max(e[k]["dist_km_rel_err_below_1pct"] for k in ks if k.startswith("NORTH")),
                south_d1_km=max(e[k]["dist_km_rel_err_below_1pct"] for k in ks if k.startswith("SOUTH")))


cmp["edge_UV"] = both(edge)
cmp["edge_W_v3"] = {c: {k: M["edge"][f"{w}_W_{c}"][k] for w in ["NORTH", "SOUTH"] for k in ["rel_err_first_row"]}
                    | {f"{w}_d1_km": M["edge"][f"{w}_W_{c}"]["dist_km_rel_err_below_1pct"] for w in ["NORTH", "SOUTH"]} for c in CF}
# residual R near the model-domain edges: first distance (cells) beyond which the RMS stays within
# +-20 % of the interior level (median of cells 40-59)
import numpy as np
redge = {}
for c, rr in M["stats"].get("residual_boundary", {}).items():
    for en, e in rr["edges"].items():
        f = np.array(e["rms_filt"], float); o = np.array(e["rms_orig"], float)
        ref = np.nanmedian(f[40:60])
        bad = np.where(~(np.abs(f / ref - 1) <= 0.2))[0]
        bad = bad[bad < 40]
        hd = float(np.load(f"/work/bk1450/b383184/Amazon/Mercator/implicit_run/output/{c}/diag_{c}.npz")["hdiv_rms_orig"][:, 0].mean())
        dxkm = {"north_30N_60-20W": 8.0, "south_10S_30W-5E": 9.1, "west_95W_10S-5N": 9.2}[en]
        def ncell(thr):
            above = np.where(~(f < thr))[0]
            return int(above.max() + 1) if above.size else 0
        redge.setdefault(f"{c}_{en}", {}).update(
            hdiv_rms_orig_surface=hd, cells_until_R_below_1pct_hdiv=ncell(0.01 * hd), cells_until_R_below_0p1pct_hdiv=ncell(0.001 * hd),
            km_until_R_below_1pct_hdiv=ncell(0.01 * hd) * dxkm, km_until_R_below_0p1pct_hdiv=ncell(0.001 * hd) * dxkm)
        redge[f"{c}_{en}"].update(rms_first_cell=float(f[0]), rms_interior=float(ref), ratio_first_cell=float(f[0] / ref),
                                  cells_until_within_20pct=int(bad.max() + 1) if bad.size else 0,
                                  orig_rms_first_cell=float(o[0]), orig_rms_interior=float(np.nanmedian(o[40:60])))
cmp["residual_R_edges_v3"] = redge
M["v1_vs_v3"] = cmp
json.dump(M, open(f"{HERE}/metrics.json", "w"), indent=1)
print(json.dumps(cmp, indent=1))
