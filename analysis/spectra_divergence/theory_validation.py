"""
(a) Theoretical transfer G(lambda) = 1/(1 + 0.5 l^2 (2 pi/lambda)^2) for all configs at 0, 10, 20, 30 N.
    W configs: l = deg*pi/180*R*cos(lat)/3.5. Rossby configs: l = mult*Ld/3.5 with Ld = zonal median of the
    Gaussian-smoothed (sigma=6 cells, as in v3) rossby_radius_T.nc over 60W-20W at the nearest T row.
(b) Synthetic validation (tests/validate_filter.json): amplitude gains of zonal/meridional sinusoids
    filtered on the real U-grid metrics (no land, constant l, GPU solver) vs theory G and a 1D box filter.
Outputs: fig_G_theory_lat.png, fig_validation_sinusoids.png, metrics_theory_validation.json
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
ds = netCDF4.Dataset(C.RUN + "/rossby/rossby_radius_T.nc")
LD = ds["Ld"][:].filled(np.nan).astype(float)
from scipy.ndimage import gaussian_filter
LD = gaussian_filter(LD, 6.0, mode="nearest")   # v3: Rossby l fields are Gaussian-smoothed (sigma = 6 cells)
lat_row = m.gphit[:, 700]
lon_row = m.glamt[250]
isel = (lon_row >= -60) & (lon_row <= -20)
HALF = np.sqrt(2 * (np.sqrt(2) - 1))

lam = np.logspace(np.log10(20e3), np.log10(3000e3), 400)
K = 2 * np.pi / lam
lats = [0, 10, 20, 30]
met = {"ell_km": {}, "lambda_half_power_G2_km": {}, "lambda_G_half_km": {}}
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True, constrained_layout=True)
for ax, lat in zip(axs, lats):
    j = int(np.argmin(abs(lat_row - (lat if lat < 30 else 29.9))))
    ld = float(np.median(LD[j, isel]))
    for cfg in C.CONFIGS:
        if cfg.startswith("W"):
            l = np.deg2rad(float(cfg[1:])) * cf.R_EARTH * np.cos(np.deg2rad(lat)) / cf.BOX_FACTOR
        else:
            l = float(cfg[1]) * ld / cf.BOX_FACTOR
        G = 1 / (1 + 0.5 * l ** 2 * K ** 2)
        ax.semilogx(lam / 1e3, G, color=C.COLORS[cfg], lw=1.8, label=f"{cfg}: ℓ={l/1e3:.0f} km")
        ax.plot(2 * np.pi * l / 1e3, 1 / 1.5, "o", color=C.COLORS[cfg], ms=5)
        met["ell_km"].setdefault(cfg, {})[lat] = l / 1e3
        met["lambda_G_half_km"].setdefault(cfg, {})[lat] = 2 * np.pi * l / np.sqrt(2) / 1e3   # G = 1/2
        met["lambda_half_power_G2_km"].setdefault(cfg, {})[lat] = 2 * np.pi * l / HALF / 1e3
    met.setdefault("Ld_median_60W20W_km", {})[lat] = ld / 1e3
    ax.axhline(0.5, color="grey", lw=0.6)
    ax.set_title(f"{lat}°N  (Ld = {ld/1e3:.0f} km)")
    ax.set_xlabel("wavelength λ [km]")
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
axs[0].set_ylabel("amplitude gain G")
fig.suptitle("Theoretical implicit-filter response G = 1/(1+½ℓ²K²) by latitude (dots: λ = 2πℓ, G = 2/3). "
             "Rossby configs are strongest at the equator, weakest at 30°N", fontsize=11)
fig.savefig(f"{OUT}/fig_G_theory_lat.png", dpi=130)

# ---------------------------------------------------------------- synthetic validation
val = json.load(open(C.RUN + "/tests/validate_filter.json"))
lams = np.array([50, 100, 200, 400, 800])
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, constrained_layout=True)
tab = []
for ax, (Lb, d) in zip(axs, val["sinusoids"].items()):
    ax.semilogx(lams, d["theory"], "k-", lw=2, label="theory G")
    ax.semilogx(lams, d["zonal"], "o", color="#d95f02", ms=8, label="implicit, zonal")
    ax.semilogx(lams, d["merid"], "s", mfc="none", color="#1b9e77", ms=10, label="implicit, meridional")
    ax.semilogx(lams, d["box1d"], "^--", color="grey", label="1D box filter (sinc)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"ℓ_box = {float(Lb):.0f} km (ℓ = {float(Lb)/3.5:.1f} km), CG its = {d['its']}")
    ax.set_xlabel("wavelength λ [km]")
    ax.set_xticks(lams); ax.set_xticklabels(lams)
    ax.grid(alpha=0.3)
    for n, L in enumerate(lams):
        tab.append(dict(L_box_km=float(Lb), lambda_km=int(L), zonal=d["zonal"][n], merid=d["merid"][n],
                        theory=d["theory"][n], box1d=d["box1d"][n]))
axs[0].set_ylabel("amplitude gain")
axs[0].legend(fontsize=9)
fig.suptitle("Synthetic sinusoids on the real U-grid metrics (tests/validate_filter.json): implicit filter vs theory vs box filter", fontsize=11)
fig.savefig(f"{OUT}/fig_validation_sinusoids.png", dpi=130)
err = [max(abs(t["zonal"] - t["theory"]), abs(t["merid"] - t["theory"])) for t in tab]
met["synthetic_validation"] = dict(table=tab, max_abs_gain_error=float(max(err)),
                                   package_crosscheck_maxreldiff=val["package_crosscheck_maxreldiff"],
                                   uniform_field_dev=val["uniform_dev"], integral_rel_err=val["integral_rel"])
json.dump(met, open(f"{OUT}/metrics_theory_validation.json", "w"), indent=1)
for t in tab:
    print(t)
print(json.dumps({k: met[k] for k in ["ell_km", "Ld_median_60W20W_km"]}, indent=0))
