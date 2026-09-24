"""
Task 3: islands and narrow passages (Lesser Antilles, Trinidad / Gulf of Paria).

* Full-depth volume transports through passage sections, daily, ORIG vs configs.
  A section is a staircase of cell edges between two land T-cells (corner points F(j,i));
  a meridional step crosses a U face (U(j,i) = east face of T(j,i)), a zonal step crosses a V face
  (V(j,i) = north face of T(j,i)). Sign: positive = to the right of the walking direction
  (walking north -> eastward positive; all passages below are walked south->north or west->east
  so that positive = eastward (Atlantic-ward) for the arc and northward for Dragon's Mouth).
  Transport = sum e2u*e3u*u (U faces) +- e1v*e3v*v (V faces), NaN -> 0 (land faces carry no flux).
* Mean surface speed in the semi-enclosed Gulf of Paria.
* Maps of the monthly-mean surface speed ORIG / W2.0 / R3Ld with the sections.

Also checks Fernando de Noronha and St Peter & St Paul rocks in the mask.
Outputs: fig_passages_map.png, fig_passages_transport.png, cache/metrics_passages.json
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
from common import mesh, read, CONFIGS, COLORS, LABELS, available_configs, ij_nearest, path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
m = mesh()
R_EARTH = 6371e3
KW50 = int(np.argmin(np.abs(m.gdepw_0 - 50)))   # W level shown in the lower map row

# land T cells (j, i) at both ends (from the surface tmask, see section.md); corners F(j,i)
SECTIONS = {
    "Galleons (Trinidad–Tobago)": [((250, 408), (255, 411))],
    "Grenada Passage (Tobago–Grenada)": [((256, 411), (266, 399))],
    "Grenadines (Grenada–St Vincent)": [((267, 400), (274, 404)), ((274, 404), (280, 405))],
    "St Vincent Passage": [((281, 406), (287, 408))],
    "St Lucia Channel": [((290, 409), (296, 408))],
    "Dominica Passage (Martinique–Dominica)": [((300, 407), (306, 404))],
    "Guadeloupe Passage (Dominica–Guadeloupe)": [((309, 403), (314, 405))],
    "Dragon's Mouth": [((249, 396), (249, 400))],
    "Serpent's Mouth": [((238, 400), (242, 400))],
}
ARC = [k for k in SECTIONS if "Mouth" not in k]
J0, J1, I0, I1 = 225, 335, 365, 445       # read window


def staircase(a, b):
    """Corner path from F(a) to F(b); returns list of (kind, j, i, sign)."""
    (ja, ia), (jb, ib) = a, b
    n = abs(jb - ja) + abs(ib - ia)
    sj = np.sign(jb - ja); si = np.sign(ib - ia)
    j, i = ja, ia
    faces = []
    for s in range(n):
        # choose the step that keeps the path closest to the straight line
        dj_left = abs(jb - j); di_left = abs(ib - i)
        if di_left == 0 or (dj_left > 0 and dj_left / max(abs(jb - ja), 1) >= di_left / max(abs(ib - ia), 1)):
            jn = j + sj
            # edge F(j,i)-F(jn,i): east face of T(max(j,jn), i) -> U(max, i); walking north => +u
            faces.append(("U", max(j, jn), i, +1 if sj > 0 else -1))
            j = jn
        else:
            inn = i + si
            # edge F(j,i)-F(j,inn): north face of T(j, max(i,inn)) -> V; walking east => right = south => -v
            faces.append(("V", j, max(i, inn), -1 if si > 0 else +1))
            i = inn
    return faces


def faces_of(name):
    f = []
    for a, b in SECTIONS[name]:
        f += staircase(a, b)
    return f


def orient(name):
    """Overall sign so that positive = eastward (arc passages) / northward (Dragon's Mouth)."""
    return -1 if name == "Dragon's Mouth" else +1


def transport(u, v, name):
    """u,v (t,k,J,I) in the read window -> daily transport [Sv]."""
    tr = np.zeros(u.shape[0])
    for kind, j, i, s in faces_of(name):
        jj, ii = j - J0, i - I0
        if kind == "U":
            w = m.e2u[j, i] * m.e3u[:, j, i]
            tr += s * np.nansum(u[:, :, jj, ii] * w[None], 1)
        else:
            w = m.e1v[j, i] * m.e3v[:, j, i]
            tr += s * np.nansum(v[:, :, jj, ii] * w[None], 1)
    return orient(name) * tr / 1e6


def width_km(name, u0, v0):
    """Wet-face length of the section at the surface [km] and number of wet faces."""
    L = 0.0; n = 0
    for kind, j, i, s in faces_of(name):
        if kind == "U" and np.isfinite(u0[j - J0, i - I0]):
            L += m.e2u[j, i]; n += 1
        if kind == "V" and np.isfinite(v0[j - J0, i - I0]):
            L += m.e1v[j, i]; n += 1
    return L / 1e3, n


cfgs = ["ORIG"] + available_configs()
res = {"sections": {}, "gulf_of_paria": {}, "small_islands": {}}
T = {}
spd = {}
wsurf = {}
for cfg in cfgs:
    u = read("U", cfg, j=slice(J0, J1), i=slice(I0, I1))
    v = read("V", cfg, j=slice(J0, J1), i=slice(I0, I1))
    if cfg == "ORIG":
        u0, v0 = u[0, 0], v[0, 0]
    T[cfg] = {n: transport(u, v, n) for n in SECTIONS}
    um, vm = np.nanmean(u[:, 0], 0), np.nanmean(v[:, 0], 0)
    ut = np.full_like(um, np.nan); vt = np.full_like(vm, np.nan)
    ut[:, 1:] = 0.5 * (np.nan_to_num(um[:, 1:]) + np.nan_to_num(um[:, :-1]))
    vt[1:, :] = 0.5 * (np.nan_to_num(vm[1:, :]) + np.nan_to_num(vm[:-1, :]))
    s = np.sqrt(ut ** 2 + vt ** 2)
    s[~m.tmask[0, J0:J1, I0:I1]] = np.nan
    spd[cfg] = (s, ut, vt)
    import netCDF4
    with netCDF4.Dataset(path("W", cfg)) as ds:
        w0 = ds["vovecrtz"][:, KW50, J0:J1, I0:I1].filled(np.nan).astype(float).mean(0)
        w0[~m.tmask[KW50, J0:J1, I0:I1]] = np.nan
    wsurf[cfg] = w0
    print(cfg, {n: round(float(T[cfg][n].mean()), 3) for n in SECTIONS}, flush=True)

# Gulf of Paria: wet T cells of rows 242-248, i 391-401 (enclosed by Trinidad and Paria peninsula)
gp = np.zeros_like(m.tmask[0]); gp[242:249, 391:402] = True
gp &= m.tmask[0]
gpw = gp[J0:J1, I0:I1]
for cfg in cfgs:
    s = spd[cfg][0]
    res["gulf_of_paria"][cfg] = dict(mean_surface_speed_of_monthly_mean=float(np.nanmean(s[gpw])))
for cfg in cfgs:
    res["gulf_of_paria"][cfg]["ratio_to_orig"] = res["gulf_of_paria"][cfg]["mean_surface_speed_of_monthly_mean"] / \
        res["gulf_of_paria"]["ORIG"]["mean_surface_speed_of_monthly_mean"]

for n in SECTIONS:
    w, nf = width_km(n, u0, v0)
    d = {"surface_wet_width_km": w, "n_wet_faces_surface": nf}
    for cfg in cfgs:
        d[cfg] = dict(mean_Sv=float(T[cfg][n].mean()), std_daily_Sv=float(T[cfg][n].std()))
        if cfg != "ORIG":
            d[cfg]["mean_ratio"] = float(T[cfg][n].mean() / T["ORIG"][n].mean())
            d[cfg]["rms_daily_diff_Sv"] = float(np.sqrt(np.mean((T[cfg][n] - T["ORIG"][n]) ** 2)))
    res["sections"][n] = d
arc = {cfg: sum(T[cfg][n] for n in ARC) for cfg in cfgs}
res["sections"]["sum_arc_Galleons_to_Guadeloupe"] = {cfg: dict(mean_Sv=float(arc[cfg].mean()),
                                                          std_daily_Sv=float(arc[cfg].std())) for cfg in cfgs}

# small islands resolved?
for name, (lon, lat) in {"Fernando de Noronha": (-32.42, -3.85), "St Peter and St Paul": (-29.35, 0.92)}.items():
    j, i = ij_nearest(lon, lat)
    box = m.tmask[0, j - 3:j + 4, i - 3:i + 4]
    res["small_islands"][name] = dict(land_T_cells_within_3=int((~box).sum()), H_min_m=float(m.H[j-3:j+4, i-3:i+4].min()))

wet = m.tmask[0, J0:J1, I0:I1]
lonw = m.glamt[J0:J1, I0:I1]; latw = m.gphit[J0:J1, I0:I1]
arcbox = wet & (lonw > -62.5) & (lonw < -59.5) & (latw > 10.5) & (latw < 16.5)
res["w50m_monthly_mean_rms_m_per_day_arc_box_62.5-59.5W_10.5-16.5N"] = {
    c: float(np.sqrt(np.nanmean(wsurf[c][arcbox] ** 2)) * 86400) for c in cfgs}
json.dump(res, open(f"{HERE}/cache/metrics_passages.json", "w"), indent=1)

# ------------------------------------------------------------------ figures
lon = m.glamt[J0:J1, I0:I1]; lat = m.gphit[J0:J1, I0:I1]
show = [c for c in ["ORIG", "W2.0", "R3Ld"] if c in cfgs]
fig, axs = plt.subplots(2, len(show), figsize=(5.2 * len(show), 13), squeeze=False)
for ax, cfg in zip(axs[1], show):
    pw = ax.pcolormesh(lon, lat, wsurf[cfg] * 86400, vmin=-10, vmax=10, cmap="RdBu_r", shading="auto")
    ax.contourf(lon, lat, (~m.tmask[KW50, J0:J1, I0:I1]).astype(float), levels=[0.5, 1.5], colors="0.6")
    ax.set_title(f"{LABELS[cfg]}: mean W at {m.gdepw_0[KW50]:.0f} m [m/day]", fontsize=9)
    ax.set_xlim(-63.5, -58.5); ax.set_ylim(9.0, 16.8); ax.set_aspect("equal")
fig.colorbar(pw, ax=axs[1], shrink=0.7, label="W [m/day] (ORIG: rigid-lid GLORYS W; filtered: scalar filter)")
for ax, cfg in zip(axs[0], show):
    s, ut, vt = spd[cfg]
    pc = ax.pcolormesh(lon, lat, s, vmin=0, vmax=1.0, cmap="viridis", shading="auto")
    ax.contourf(lon, lat, (~m.tmask[0, J0:J1, I0:I1]).astype(float), levels=[0.5, 1.5], colors="0.6")
    ax.contour(lon, lat, m.H[J0:J1, I0:I1], levels=[200, 1000], colors=["w", "0.8"], linewidths=0.6)
    q = 2
    ax.quiver(lon[::q, ::q], lat[::q, ::q], ut[::q, ::q], vt[::q, ::q], scale=12, width=0.002, color="k")
    for n in SECTIONS:
        for kind, j, i, sgn in faces_of(n):
            if kind == "U":
                x = m.glamu[j, i]; y0, y1 = m.gphif[j - 1, i], m.gphif[j, i]
                ax.plot([x, x], [y0, y1], "r-", lw=1.5)
            else:
                y = m.gphiv[j, i]; x0, x1 = m.glamf[j, i - 1], m.glamf[j, i]
                ax.plot([x0, x1], [y, y], "r-", lw=1.5)
    ax.set_title(f"{LABELS[cfg]}: Jan-1993 mean surface speed", fontsize=9)
    ax.set_xlim(-63.5, -58.5); ax.set_ylim(9.0, 16.8); ax.set_aspect("equal")
fig.colorbar(pc, ax=axs[0], shrink=0.7, label="speed [m/s]")
fig.savefig(f"{HERE}/fig_passages_map.png", dpi=130, bbox_inches="tight")

names = list(SECTIONS) + ["Arc sum"]
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(names)); wbar = 0.8 / len(cfgs)
for n_c, cfg in enumerate(cfgs):
    mv = [T[cfg][n].mean() for n in SECTIONS] + [arc[cfg].mean()]
    sd = [T[cfg][n].std() for n in SECTIONS] + [arc[cfg].std()]
    ax.bar(x + (n_c - len(cfgs) / 2 + 0.5) * wbar, mv, wbar, yerr=sd, color=COLORS[cfg], label=LABELS[cfg],
           error_kw=dict(lw=0.6))
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({res['sections'][n]['surface_wet_width_km']:.0f} km)" if n in SECTIONS
                                      else n for n in names], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("full-depth transport [Sv]\n(+ east; Dragon's Mouth + north)")
ax.set_title("Jan-1993 mean passage transport (bars) ± daily std; surface wet width in brackets")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_passages_transport.png", dpi=130)
