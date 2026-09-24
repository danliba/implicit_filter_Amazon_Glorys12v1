# Implicit filtering of GLORYS12v1 (NEMO ORCA12 C-grid): Amazon plume / NBC region, January 1993

*Filtered fields:* `implicit_run/output/<CFG>/{U,V,W}_1993-01c_<CFG>.nc`. *Code:* `implicit_run/` (see §8). *Interactive walkthrough:* `implicit_data/v0.Implicit_filtering_ARP.ipynb`.

## Executive summary
**Data.** GLORYS12v1 daily U, V, W for January 1993 (31 days × 50 levels) on the regional ORCA12 cut (95°W–10°E, 10°S–30°N). They were implicitly filtered on the native C-grid with five scale configurations:
- windows of 1.5°, 2.0° and 2.5° (box-equivalent 167, 222 and 278 km at the equator)
- 2·L_d and 3·L_d, with L_d the first-baroclinic Rossby radius

Everything ran on one A100 GPU per configuration: 20–40 min for U,V plus ~14 min for W.

**Final product ("v3")**
- **U,V:** filtered jointly with the C-grid vector Laplacian in div–rot form, (1 + ½·M_ℓ)ū = u.
  - Zero normal flow and free slip at coasts.
  - For constant ℓ in open ocean it equals the component-wise scalar filter; the spherical metric terms are < 0.1 %, as in the vector addendum.
  - The filtered divergence is exactly the scalar-filtered original divergence at every level.
- **W:** the rigid-lid W product (`W_1993-01fc.nc`), filtered as a scalar at the same ℓ with the T-cell operator implied by the U,V filter. It is not recomputed from divergence (W correction). No barotropic correction is applied.
- **Continuity residual** R = ∂w̄/∂z + ∇h·ū:
  - Open ocean: the same as the unfiltered rigid-lid data, 3–9×10⁻¹¹ s⁻¹ at 110 m for the window configs, against a filtered divergence of ~3×10⁻⁷.
  - It is larger only at bottom steps and slopes next to coasts (~8×10⁻⁷ within 3 cells of land at 110 m) and near the western (95°W) and southern (10°S) cuts of the regional grid. Both cuts lie outside the analysis domain.
  - At the Amazon mouth, the surface-layer R is a smooth plume: the filtered river/free-surface source.

**How we got there.** Two approaches were run and rejected, and two bugs were found along the way.
- **Component-wise filtering of U and V (v1), as the task sheet specified.** Each component's no-flux walls produced a one-cell convergence line along every coast, and W recomputed from divergence reached ~10⁻³ m/s.
- **v2** (vector filter + barotropic correction + W from divergence) was superseded by the W correction.
- **Bug 1:** a missing e3 weight in the vorticity term (first v3 build).
- **Bug 2:** an operator mismatch between the U,V and W filters (first W run).

All are documented in §2 and were fixed and rerun.

**Parameter convention.** The `implicit_filter` package's `compute(n, k, data)` solves (1 + 2(−Δ)/k²). The γ = ½ filter at scale ℓ therefore needs **k = 2/ℓ, not 1/ℓ**.

**Recommendation.** **W2.0** (ℓ = 2°·R·cosφ/3.5 ≈ 63 km, box scale ≈ 222 km) for the full-dataset processing:
- removes 64–66 % of intra-monthly EKE
- keeps the NBC transport within −16 % (44°W) and −15 % (5°N)
- keeps the retroflection/NECC mean pattern (r = 0.99) and the >10° circulation (1.5 % RMS difference at the surface)

The Rossby-radius filters are unsuitable here: they are strongest exactly where the NBC is (equator, L_d ≈ 240 km) and weakest where the eddy field is (north of 15°N). See §7 for details and caveats: no filter at these scales removes NBC rings (300–450 km) without also degrading the NBC.
