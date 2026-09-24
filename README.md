# Implicit filtering of GLORYS12v1 on the native NEMO C-grid

Removing the mesoscale from GLORYS12v1 `U`, `V`, `W` over the Amazon plume and
North Brazil Current, **on the native ORCA12 C-grid**, for Lagrangian tracking
with [OceanParcels](https://oceanparcels.org/). January 1993, 31 days,
50 levels, 5 filter scales.

The method is the implicit filter of Danilov et al. (2023) and Nowak et al.
(2025). What is new here is that `U` and `V` are filtered **jointly as a
vector** in div–rot form on their own C-grid points, so that the filtered field
stays discretely consistent with NEMO's continuity equation — filtering the
components separately produces a spurious one-cell convergence line along every
coastline (see [REPORT.md](REPORT.md) §"v1").

**Results, all 43 figures and the configuration recommendation: [REPORT.md](REPORT.md).**
Start with [`notebooks/v0.Implicit_filtering_ARP.ipynb`](notebooks/v0.Implicit_filtering_ARP.ipynb)
for a walkthrough of the method and the GPU solver.

## Method

Implicit filter with γ = ½, n = 1:

$$(1 + \tfrac{1}{2}\ell^2(-\Delta))\,\bar\varphi = \varphi,
\qquad G(K) = \frac{1}{1 + \tfrac{1}{2}\ell^2K^2},
\qquad \ell = L_\mathrm{box}/3.5$$

- **U, V** — filtered jointly as a vector, level by level, with
  $M\mathbf{u} = -\nabla(\ell_T^2\,\nabla\cdot\mathbf{u}) + \nabla\times(\ell_F^2\,\zeta)$,
  using NEMO's discrete divergence (with `e3u`, `e3v`) and vorticity. Zero
  normal flow through land, free slip. The operator is symmetric positive
  definite and is solved with a batched Jacobi-preconditioned CG on GPU
  (rtol 1e-8). Exact property: `div(ū)` equals the scalar no-flux filter of
  `div(u)`, at every level.
- **W** — the rigid-lid `W` product, filtered **as a scalar** at the same ℓ with
  the T-cell operator implied by the level-*k* vector filter. It is *not*
  recomputed from divergence: NEMO's continuity also carries runoff, E−P and
  free-surface sources, which matters in an Amazon plume configuration.
- **Scales** — `W1.5`/`W2.0`/`W2.5` use a fixed box width in degrees,
  ℓ = Δθ·(π/180)·R·cos φ / 3.5. `R2Ld`/`R3Ld` use ℓ = m·L_d/3.5 with the WKB
  first-baroclinic Rossby radius from the same month's T/S, Gaussian-smoothed
  (σ = 6 cells).

## Results

| | W1.5 | **W2.0 (recommended)** | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|
| box width at 0° / 15°N / 30°N [km] | 167 / 161 / 145 | **222 / 215 / 193** | 278 / 269 / 241 | 474 / 144 / 66 | 711 / 217 / 99 |
| EKE removed (0–1000 m) | 53 % | **64 %** | 73 % | 66 % | 81 % |
| KE removed (0–1000 m) | 38 % | **49 %** | 57 % | 53 % | 68 % |
| NBC transport change 44°W / 5°N | −11 / −11 % | **−16 / −15 %** | −21 / −19 % | −37 / −25 % | −49 / −35 % |
| retroflection mean-pattern correlation | 0.99 | **0.99** | 0.98 | 0.98 | 0.95 |
| NBC-ring KE kept | ~78 % | **~68 %** | ~60 % | ~65 % | ~50 % |
| wall-clock, 1 A100 (U,V + W) | 21 + 13 min | 22 + 13 min | 24 + 13 min | 30 + 14 min | 39 + 16 min |

**W2.0 is the recommended configuration.** The Rossby-radius configurations are
backwards for this region: they are widest at the equator, where the NBC is, and
narrowest in the subtropics, where the eddies are.

Known behaviour: coastal speeds are 13–29 % higher within ~45 km of land; shelf
depth-mean KE is 22–29 % higher; Lesser Antilles passage transports are
conserved to < 1 %; NBC rings (300–450 km) are only partly removed by any
configuration.

## Relation to `FESOM/implicit_filter`

The production code here is **independent** of the
[`implicit_filter`](https://github.com/FESOM/implicit_filter) package. The
package cannot filter `U`/`V` on their own C-grid points, which is the whole
point of this repository; it was used only as a cross-check, and it agrees with
`cgrid_filter.py` to 2e-8 for constant ℓ (`tests/validate_filter.py`).

Three things to know if you use the package directly:

- It solves `(I + 2(-L/k²))`, so the γ = ½ filter at scale ℓ needs **`k = 2/ℓ`**,
  not `1/ℓ`.
- `k` must be a Python `float`; `np.float64` or `int` raises inside `np.repeat`.
- Its GPU path breaks with CuPy ≥ 14 (`tol` was renamed `rtol` in
  `cupyx.scipy.sparse.linalg.cg`), and its variable-ℓ scaling is not symmetric.

## Layout

| Path | Content |
|---|---|
| [`REPORT.md`](REPORT.md) | The full report: method, validation, all five configurations, 43 figures. |
| [`cgrid_filter.py`](cgrid_filter.py) | Core library. `CMesh` (NEMO metrics, partial steps, masks); `ell_T_F` (filter scales); `build_vector_operator` / `filter_level_vector` (U,V div–rot vector filter); `filter_level_w` (W with the consistent T-cell operator); `ImplicitSolver` (batched Jacobi-PCG, GPU/CPU); `horizontal_divergence_e3`, `continuity_residual`. The v1 component-wise `filter_level` and the v2 `ColumnConsistency` / `barotropic_correction` are kept for reference only. |
| `run_filter.py`, `run_filter.sh` | Production for one configuration: U,V (vector) and W, all levels and days, plus residual diagnostics. `sbatch -J impf_W2.0 run_filter.sh W2.0`. `--w-only` refilters W alone. |
| `notebooks/` | `v0` method and GPU walkthrough, `v1` what the filter removes, `v2` Green's function and transfer for W2.0. All executed, kernel `implicit_filter`. |
| `rossby/` | First-baroclinic Rossby radius from GLORYS T/S (WKB): `rossby_radius_T.nc`, script, README. |
| `tests/` | `validate_filter.py` (sinusoid transfer, package cross-check, symmetry/conservation), `validate_vector.py` (vector filter commutes with divergence), `test_e3_independence.py` (shear and divergent waves at e3 = 1/50/200 m). |
| `analysis/` | `nbc_energy/`, `spectra_divergence/`, `boundaries/`, `greens_function/`: scripts, figures, metrics, report sections; shared `common.py`, `ANALYSIS_CONTEXT.md`. |
| `report_parts/` | Report sources, assembled into `REPORT.md`. |
| `build_env.sh` | Builds the GPU environment and the Jupyter kernel. |

## Environment

```bash
./build_env.sh     # conda env (Python 3.11, CuPy 14 + CUDA 12.4, JAX, xarray,
                   # cartopy, papermill) + Jupyter kernel `implicit_filter`
```

GPU jobs run on the DKRZ Levante `gpu` partition and need
`export CUDA_PATH=$ENV/targets/x86_64-linux`.

## Data availability

The filtered fields are ~101 GB and are **not** in this repository. On DKRZ
Levante they are at:

```
/work/bk1450/b383184/Amazon/Mercator/implicit_run/output/<CFG>/{U,V,W}_1993-01c_<CFG>.nc
CFG ∈ {W1.5, W2.0, W2.5, R2Ld, R3Ld}
```

`output_v1_componentwise/` and `output_v2_btcorr/` hold the two superseded runs,
kept only for the comparison figures in the report. Their `W` files carry an
empty `depthw` coordinate, from a `renameVariable` bug — do not use them for
tracking.

To regenerate from scratch you need the GLORYS12v1 extraction
(`{U,V,T,S}_1993-01c.nc`), the rigid-lid `W` product (`W_1993-01fc.nc`) and the
mesh (`Hgr_cmesh.nc`, `Zgr_cmesh2.nc`), then:

```bash
sbatch tests/validate.sh                       # GPU validation, ~2 min
(cd rossby && python rossby_radius.py)         # only for the R*Ld configurations
for c in W1.5 W2.0 W2.5 R2Ld R3Ld; do sbatch -J impf_$c run_filter.sh $c; done
bash  analysis/nbc_energy/run_all.sh           # ~6 min, login node
sbatch analysis/spectra_divergence/job_v3.sh
bash  analysis/boundaries/run_all_v3.sh
```

**Another month:** `python run_filter.py <CFG> --month YYYY-MM`, same file
naming. The rigid-lid `W` for that month must exist first. Only January 1993 has
been processed, so the seasonal cycle is untested.

## Using the output in OceanParcels

The files keep the original NEMO variable names, dimensions and
`nav_lon`/`nav_lat`, so `FieldSet.from_nemo` works exactly as for the unfiltered
data, with the same mesh file (`glamf`/`gphif`):

- `U` → `vozocrtx(time_counter, deptht, y, x)`
- `V` → `vomecrty(time_counter, deptht, y, x)`
- `W` → `vovecrtz(time_counter, depthw, y, x)`, rigid lid (0 at the surface)

Two caveats:

- These `W` files do not carry the extra `z(depthw, y, x)` and `H(y, x)`
  variables of the original `W` product; take those from the original if needed.
- Keep particles at least 3.5 ℓ (≈ 1.7° for W2.0) away from the 30°N cut, and
  ignore the last grid row and column.

## References

- Danilov, S., et al. (2023). *Journal of Advances in Modeling Earth Systems*.
  [doi:10.1029/2023MS003946](https://doi.org/10.1029/2023MS003946) — mathematical
  formulation of the implicit filter.
- Nowak, K., et al. (2025). *Geoscientific Model Development* **18**, 6541.
  [doi:10.5194/gmd-18-6541-2025](https://doi.org/10.5194/gmd-18-6541-2025) —
  the `implicit_filter` package.
