# Context for the analysis agents (read fully before starting)

## What was done (VERSION 3 — final; supersedes v1 and v2)
GLORYS12v1 (NEMO ORCA12 C-grid) daily U, V, W for January 1993 (31 days, 50 levels) on a regional grid
(y=499, x=1260; 95°W–10°E, 10°S–30°N) were filtered with the implicit filter, level by level.

**U, V: vector div–rot form.**

    (1 + γ M) ū = u ,  M u = −∇(ℓ_T² ∇·u) + ∇×(ℓ_F² ζ(u)),  γ = 1/2, n = 1,  ℓ_box ≈ 3.5 ℓ   (Nowak et al. 2025 GMD)

- U and V are solved jointly on their native points, using NEMO's discrete divergence (with e3u, e3v) and vorticity.
- Boundary conditions: zero normal flow through land, free slip. Open edges of the regional grid are walls.
- Open ocean with constant ℓ: equals the component-wise Laplacian, G(K) = 1/(1 + ½ ℓ² K²) (verified to <1%).
- The filtered divergence is exactly the scalar no-flux filter of the original divergence, at every level (verified to 5e-5).

**W: scalar filter, same ℓ.** W is filtered on the W/T points at the same ℓ_T with exactly the T-cell operator implied by the level-k vector filter: vol·w̄ + γ F W⁻¹ Fᵀ(ℓ² w̄) = vol·w, with no-flux at land, the same e3 weights, the same ℓ² placement and the same edge treatment (`cgrid_filter.filter_level_w`). This is the final W, rerun at ~11:15 and replacing the earlier `filter_level(...,'T')` version. It makes R vanish where levels k and k+1 share a mask. No barotropic correction is applied, and W is NOT recomputed from divergence.
- Input is the user's rigid-lid W product `/work/bk1450/b383184/Amazon/Mercator/data/variables_c/UVW/W_1993-01fc.nc` (vovecrtz, from data/Fix_W.ipynb). It is 0 at the surface and 0 at the sea-floor W points, which are kept.
- NEMO's continuity contains source terms: river runoff (Amazon!), E−P and the free surface.

**Validation quantity:** the continuity residual R = ∂w/∂z + ∇h·u (1/s, T cells, `cgrid_filter.continuity_residual`). It should be small and smooth. Near the Amazon mouth it is not zero: there it represents the (filtered) source term.

**Rossby ℓ fields:** Gaussian-smoothed (σ = 6 cells).

**Earlier versions, for comparison:**
- `output_v1_componentwise/`: U on the U-grid and V on the V-grid as independent scalars with their own no-flux walls, and W recomputed from divergence. Its U,V have a one-cell convergence line along every wall.
- `output_v2_btcorr/`: vector U,V plus a barotropic correction, with W from divergence.
- In both v1 and v2 W files the `depthw` coordinate is empty (fill values), because of a netCDF rename bug. Use mesh gdepw_0 if you read them.

Configurations (all in `/work/bk1450/b383184/Amazon/Mercator/implicit_run/output/<CFG>/`):
| CFG  | ℓ definition |
|------|--------------|
| W1.5 | ℓ = 1.5°·(π/180)·R·cos(lat)/3.5  (≈47.6 km at the equator, ≈41 km at 30°N) |
| W2.0 | same with 2.0° (≈63.5 km at the equator) |
| W2.5 | same with 2.5° (≈79.4 km at the equator) |
| R2Ld | ℓ = 2·Ld/3.5, where Ld is the first-baroclinic Rossby radius (237 km at the equator, 106 km at 10°N, 52 km at 20°N, 33 km at 30°N) |
| R3Ld | ℓ = 3·Ld/3.5 |

The Ld field is in `implicit_run/rossby/rossby_radius_T.nc` (variable Ld [m] on T points; see that folder's README.md).

## Files
- Original: `/work/bk1450/b383184/Amazon/Mercator/implicit_data/U_1993-01c.nc` (vozocrtx), `V_1993-01c.nc` (vomecrty),
  dims (time_counter=31, deptht=50, y, x), NaN on land, float32 with compression, chunks (1,5,340,481).
  The original GLORYS W (NEMO convention, incl. free-surface part) is `/work/bk1450/b383184/Amazon/Mercator/data/variables/W_1993-01.nc` (vovecrtz).
- Filtered (v3): `output/<CFG>/U_1993-01c_<CFG>.nc` (vozocrtx), `V_...` (vomecrty), and `W_...` (vovecrtz = filtered rigid-lid W, depthw axis).
  All have the same dims and coords as the inputs.
  - `diag_<CFG>.json`: CG iterations and timings per level.
  - `diag_<CFG>.npz`: (nt, nz) arrays R_rms_filt/orig, R_max_filt/orig, hdiv_rms_filt/orig, dwdz_rms_filt/orig and w_rms_filt/orig (interior wet points of the whole domain).
    It also holds time-mean residual maps Rmean_filt/orig at levels kmaps=[0,10,22,33], and daily surface-layer residual maps Rsurf_filt/orig (31, ny, nx).
  - The ORIGINAL W to compare against is the rigid-lid file above (`W_1993-01fc.nc`), not data/variables/W_1993-01.nc.
  **A config is finished when `diag_<CFG>.json` exists.** v3 jobs started ~10:00 (Levante time) and take ~20-35 min each.
  Wait with `until [ -f .../diag_R3Ld.json ]; do sleep 30; done` in a background command (not chained sleeps). Meanwhile, develop and test
  your scripts on ORIG plus whatever configs already exist.
- Mesh: `/work/bk1450/b383184/Amazon/Mercator/data/Hgr_cmesh.nc` and `Zgr_cmesh2.nc`. Use the helper `cgrid_filter.CMesh()` (in `implicit_run/cgrid_filter.py`),
  which provides e1t,e2t,e1u,e2u,e1v,e2v,e1f,e2f, glam*/gphi* (t,u,v,f), mbathy, tmask (nz,ny,nx), e3t/e3u/e3v (partial steps),
  H (bottom depth), gdept_0, and gdepw_0. `cgrid_filter.horizontal_divergence_e3` and `w_from_continuity` implement NEMO continuity.
- Shared helpers: `implicit_run/analysis/common.py` (paths, read hyperslabs, uv_to_t, nearest index, lon/lat box slices, labels/colors).
- Grid staggering: U(j,i) sits between T(j,i) and T(j,i+1). V(j,i) sits between T(j,i) and T(j+1,i). Index order is (t, k, j, i).
  Latitude rows are nearly constant along x (Mercator part of ORCA). Longitudes are in [-180, 180].

## Environment and compute
- Python: `/work/bk1450/b383184/conda/envs/implicit_filter/bin/python` (numpy, scipy, xarray, netCDF4, dask, matplotlib, cartopy, cmocean).
  cartopy may lack Natural Earth data offline; if coastlines fail to download, draw the land mask from mbathy==0 instead. Never block on it.
- The login node is shared. Keep interactive runs under ~15 min and ~30 GB RAM. For heavier jobs use SLURM:
  `#SBATCH --partition=compute --account=bk1450 --time=02:00:00 --mem=0 --exclusive` (256 GB nodes) or `--partition=shared --mem=100G`.
  Read only what you need (hyperslabs by level/region). One full 4D variable is ~20 GB as float32 when uncompressed.
- Analysis domain of interest (Amazon plume / NBC): 70°W–30°W, 5°S–30°N. The larger model domain provides the halo.
  The north edge of the model domain is 30°N, so the filter has no halo there. Keep that in mind when interpreting results near 30°N.
- Do NOT modify anything in `output/`, `implicit_data/`, `data/`, `cgrid_filter.py` or `run_filter.py`.
  Write only inside your own `analysis/<topic>/` folder.

## Deliverables per agent
In `analysis/<topic>/`:
- scripts (commented, runnable)
- figures (PNG, dpi ~130, legible; one message per figure)
- `metrics.json` with the numbers
- `section.md`: a Markdown report section (English, concise, quantitative) that embeds figures with relative paths `analysis/<topic>/<fig>.png`
  (the report sits in `implicit_run/`), states the method and interprets results physically.

**Be honest:** flag anything that looks wrong or artefactual rather than smoothing it over.
