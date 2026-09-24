## Spectral response and kinematic consistency of the filtered U, V, W (v3)

This section covers the final v3 filter, with the e3f-weighted vorticity fix applied. U and V are filtered together with the vector div–rot operator.
W is the rigid-lid GLORYS W (`W_1993-01fc.nc`) filtered as a scalar with the same ℓ_T, using the T-cell operator implied by the vector filter (`filter_level_w`). January 1993, all 5 configurations.

*Update:* the W, residual and W-statistics parts were recomputed after W was refiltered with the consistent operator. The U,V spectra are unchanged.

Scripts in `analysis/spectra_divergence/`:
- `spectra.py [hann|tukey] [UV|W]`: spectra and transfer functions.
- `theory_validation.py`: theoretical G(λ) by latitude and the synthetic test.
- `residual.py` and `residual_extra.py`: continuity residual.
- `w_stats.py`: W statistics.
- `make_metrics.py`: builds `metrics.json`.

The superseded v1 analysis (component-wise filter, W from divergence) is kept in `old_v1/`.

### 1. Filter response

**Theory.** G(K) = 1/(1+½ℓ²K²). The power transfer G² equals ½ at λ½ = 6.90ℓ.
- The W configs barely depend on latitude: ℓ is 48/64/79 km at the equator and 41/55/69 km at 30°N.
- The Rossby configs vary by a factor of 7 over the domain. v3 uses Ld smoothed with σ = 6 cells. For R2Ld, ℓ is 136, 61, 30 and 19 km at 0°, 10°, 20° and 30°N (zonal-median Ld over 60–20°W: 237, 106, 52, 33 km).
- As a result, R3Ld filters much more strongly than W2.5 at the equator and more weakly poleward of about 12°N.

![Theoretical G](analysis/spectra_divergence/fig_G_theory_lat.png)

**Synthetic validation** (`tests/validate_filter.json`). Cosines on the real U-grid metrics, constant ℓ, no land. The maximum gain error against G is 0.008. A 1D box filter of the same ℓ_box has negative side lobes.

| ℓ_box (ℓ) | λ [km] | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| 167 km (47.7) | implicit zonal / merid | 0.056 / 0.058 | 0.181 / 0.186 | 0.464 / 0.473 | 0.775 / 0.781 | 0.932 / 0.934 |
| | theory G / 1D box | 0.053 / −0.084 | 0.182 / −0.164 | 0.471 / 0.189 | 0.781 / 0.737 | 0.934 / 0.930 |
| 222 km (63.4) | implicit zonal / merid | 0.033 / 0.034 | 0.111 / 0.115 | 0.328 / 0.337 | 0.661 / 0.669 | 0.886 / 0.890 |
| | theory G / 1D box | 0.031 / 0.070 | 0.112 / 0.091 | 0.335 / −0.097 | 0.668 / 0.565 | 0.890 / 0.878 |
| 278 km (79.4) | implicit zonal / merid | 0.021 / 0.022 | 0.074 / 0.076 | 0.238 / 0.244 | 0.554 / 0.563 | 0.832 / 0.837 |
| | theory G / 1D box | 0.020 / −0.056 | 0.074 / 0.073 | 0.243 / −0.215 | 0.562 / 0.375 | 0.837 / 0.813 |

![Synthetic validation](analysis/spectra_divergence/fig_validation_sinusoids.png)

**Empirical spectra: method.**
- Three index-rectangle boxes, each checked to be wet including a one-cell halo at every level used:
  - WTA: 50–35°W, 5–14°N, 1652×995 km.
  - STA: 55–35°W, 18–28°N, 2052×1115 km.
  - EQA: 28–10°W, 4°S–4°N, 2009×898 km.
- The suggested boxes 55–35°W 2–14°N and 35–15°W 3°S–3°N contain land (the Brazilian shelf, and Fernando de Noronha at 100 m).
- Fields: U and V on native points at 0.5 m and 110 m; W at the W levels 100.6 m and 495.7 m.
- Processing: remove a least-squares plane, apply a 2D Hann window, FFT, and bin radially.
- Average the 31 daily spectra, then take T = E_filt/E_orig.
- Theory curves use the box-mean ℓ_T, exactly as in the filter.
- ℓ_eff is a least-squares fit of G²(ℓ_eff) to T over 60 km < λ < L_min/2.
- A Tukey(0.5) window changes ℓ_eff/ℓ by ≤0.05 (`metrics_spectra*_tukey.json`).

![KE spectra](analysis/spectra_divergence/fig_ke_spectra.png)
![U,V transfer](analysis/spectra_divergence/fig_transfer.png)
![W spectra](analysis/spectra_divergence/fig_W_spectra.png)
![W transfer](analysis/spectra_divergence/fig_W_transfer.png)

ℓ_eff/ℓ (Hann):

| box | field, z | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|---|
| STA | U,V 0.5 / 110 m | 0.99 / 0.99 | 0.99 / 0.99 | 0.98 / 0.99 | 0.93 / 0.92 | 0.93 / 0.93 |
| STA | W 100 / 496 m | 0.98 / 1.02 | 0.97 / 1.03 | 0.95 / 1.02 | 0.94 / 0.93 | 0.93 / 0.95 |
| WTA | U,V 0.5 / 110 m | 0.92 / 0.96 | 0.90 / 0.95 | 0.88 / 0.93 | 0.88 / 0.92 | 0.89* / 0.89* |
| WTA | W 100 / 496 m | 0.97 / 0.99 | 0.96 / 0.99 | 0.95 / 0.99 | 0.94 / 0.94 | 0.92* / 0.92* |
| EQA | U,V 0.5 / 110 m | 0.91 / 0.93 | 0.88 / 0.92 | 0.85 / 0.90 | 0.76* / 0.85* | 0.69* / 0.81* |
| EQA | W 100 / 496 m | 0.95 / 0.97 | 0.93 / 0.96 | 0.92 / 0.94 | 0.88* / 0.88* | 0.86* / 0.84* |

\* The box does not resolve λ½: theoretical λ½ exceeds half the short box side. In EQA, ℓ = 141/211 km for R2Ld/R3Ld.

Half-power wavelength, empirical / theory [km], at the surface: STA W2.0 414/403, WTA W2.0 380/432, EQA W2.0 368/438. At 100 m for W: STA 380/403, WTA 423/432, EQA 402/438.
In every case the retained variance fraction matches ∫E_orig G² / ∫E_orig within 0.02; the only exception is R2Ld/R3Ld in WTA (0.49 vs 0.55).

**Interpretation.**
- **Scalar W filter.** It has the same G(K) as the vector U,V filter. For the W configs, ℓ_eff/ℓ = 0.92–1.03, and it is closest to 1 at 500 m and in the subtropics.
- **Subtropics.** The U,V response matches theory essentially exactly (0.98–0.99). Only 16–35% of KE survives the W filters there, because the eddy energy sits at λ < 400 km.
- **Tropical boxes.** The weaker empirical response at the surface (0.85–0.92) is not explained by ℓ, which is constant to 1% in these boxes. The likely causes are Hann smearing of a very red, anisotropic equatorial spectrum over ~900–1000 km short sides, and, for WTA, the nearby Brazilian coast.
- **Rossby configs.** The 5–8% lower ℓ_eff in STA and WTA reflects the ℓ range inside the box (STA R2Ld: 18–38 km; WTA: 40–123 km), since G² is convex in ℓ.
- **v3 vs v1.** The vector filter keeps the scalar G(K) in open ocean: for the W configs, v1 and v3 give identical ℓ_eff/ℓ to ±0.01 for U,V.

### 2. Continuity residual R = ∂w̄/∂z + ∇h·ū

![R vs hdiv profiles](analysis/spectra_divergence/fig_R_profiles.png)

**What R means for ORIG.**
- The rigid-lid W is w_fix = w − (H−z)/H·w(0). The original GLORYS W satisfies bottom-up continuity with U,V to round-off: R computed with it has RMS 8×10⁻¹³ and max 6×10⁻¹¹ s⁻¹.
- Consequently R_orig = −w(0)/H, the depth-uniform spreading of the surface w that Fix_W removed. This holds to 1.5×10⁻⁴ relative RMS over all levels (day 1) and 5×10⁻³ at 110 m.
- R_orig is 1.5×10⁻⁸ s⁻¹ (whole-domain RMS) at the surface, 2×10⁻¹⁰ at 110 m and 6×10⁻¹¹ at 760 m. It is largest on shallow shelves, where H is small.

**Filtered, whole-domain RMS** (diag npz, rows/columns 0–1 and the last two excluded).

| RMS [s⁻¹] | surface: R / hdiv | 110 m: R / hdiv | 760 m: R / hdiv |
|---|---|---|---|
| ORIG | 1.5e-8 / 2.3e-6 | 2.1e-10 / 1.8e-6 | 5.6e-11 / 7.5e-7 |
| W1.5 | 1.9e-7 / 6.6e-7 | 2.4e-7 / 4.1e-7 | 7.6e-8 / 1.9e-7 |
| W2.0 | 1.8e-7 / 5.7e-7 | 2.2e-7 / 3.3e-7 | 6.8e-8 / 1.5e-7 |
| W2.5 | 1.7e-7 / 5.0e-7 | 1.9e-7 / 2.8e-7 | 6.0e-8 / 1.2e-7 |
| R2Ld | 1.7e-7 / 5.8e-7 | 2.1e-7 / 3.2e-7 | 7.0e-8 / 1.9e-7 |
| R3Ld | 1.6e-7 / 4.3e-7 | 1.9e-7 / 2.2e-7 | 6.5e-8 / 1.4e-7 |

Domain-wide, R reaches 30–86% of the filtered RMS hdiv. It is not uniformly small. The decomposition below shows that it comes almost entirely from edges, coasts and the ℓ gradient of the Rossby configs.

![R surface mean](analysis/spectra_divergence/fig_R_map_mean.png)
![R surface day 1](analysis/spectra_divergence/fig_R_map_day1.png)

**Surface-layer maps, 70–30°W 5°S–30°N.** RMS excludes the last two grid rows (30°N). On day 1 it is 2.0×10⁻⁸ for ORIG and 1.7–2.1×10⁻⁸ for the configs.
- **ORIG:** R is concentrated on the North Brazil / Amazon shelf, including runoff-like blocky patches (see below), and is ~10⁻¹¹ offshore.
- **W configs:** R offshore is a smooth eddy-scale pattern of ±10⁻¹⁰–10⁻⁹. The shelf signal is smoothed into a plume around the mouth. A band of 10⁻⁸–10⁻⁷ lies along the northern edge (30°N).
- **Rossby configs:** R offshore is 10⁻⁸–10⁻⁷ in zonal bands at 0–12°N, where ℓ changes fastest (see below).

![R vs distance](analysis/spectra_divergence/fig_R_distance.png)

**Decomposition over the full month** (all 31 days, all configs; distances in grid cells; `metrics_residual.json`). The categories are:
- **clean:** >40 cells from an edge, >30 from the coast and >30 from a bottom step;
- **edge ≤20:** within 20 cells of an edge, >30 from the coast;
- **coast ≤3:** within 3 cells of the coast.

| RMS R [s⁻¹] | level | ORIG | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|---|---|
| clean | 0.5 m | 3.3e-10 | 2.4e-10 | 3.0e-10 | 4.2e-10 | 8.9e-10 | 2.4e-9 |
| clean | 110 m | 3.2e-11 | 2.9e-11 | 3.7e-11 | 9.1e-11 | 4.4e-10 | 1.5e-9 |
| clean | 763 m | 3.1e-11 | 3.9e-11 | 4.0e-11 | 4.6e-11 | 1.2e-10 | 3.7e-10 |
| coast >30 incl. edges | 110 m | 3.3e-11 | 1.6e-7 | 1.5e-7 | 1.3e-7 | 1.1e-7 | 9.1e-8 |
| edge ≤20, coast >30 | 0.5 m | 4.7e-11 | 6.3e-7 | 5.9e-7 | 5.4e-7 | 5.5e-7 | 4.9e-7 |
| edge ≤20, coast >30 | 110 m | 3.7e-11 | 4.7e-7 | 4.3e-7 | 3.9e-7 | 3.3e-7 | 2.7e-7 |
| coast ≤3 | 0.5 m | 5.6e-8 | 1.5e-7 | 1.4e-7 | 1.4e-7 | 1.3e-7 | 1.4e-7 |
| coast ≤3 | 110 m | 9.5e-10 | 9.1e-7 | 8.1e-7 | 7.2e-7 | 9.0e-7 | 7.9e-7 |
| coast ≤3 | 763 m | 1.6e-10 | 2.8e-7 | 2.5e-7 | 2.2e-7 | 2.9e-7 | 2.7e-7 |

Share of domain ΣR² for the filtered fields:
- **Surface:** 96–100% from edge ≤20, 2–3% from coast ≤3.
- **110 m:** 42–51% from edge ≤20 and 57–70% from coast ≤3 (the two overlap where coasts meet an edge).
- **763 m:** 38–46% from edge ≤20 and 67–79% from coast ≤3.

**Confirming or refuting the coordinator's single-day test (W2.0, 110 m).** Both parts are confirmed over the full month.
- **Away from edges and land:** 1.1×10⁻¹⁰ (coordinator: 4×10⁻¹¹ for one day). ORIG has 3.2×10⁻¹¹ there, so the filtered fields are within a factor of ~3.5 of the rigid-lid ORIG.
- **Including cells near the edges:** 1.5×10⁻⁷ (coordinator: 7×10⁻⁸).
- **Coasts:** 7.9×10⁻⁷ within 3 cells.
- **Bottom steps:** in the open ocean at 763 m, R rises from 5×10⁻¹¹ to 3×10⁻⁹ within about 15 cells of a step, but only a few cells are both near a step and far from the coast. Most steps coincide with coasts and slopes at that level, so the coastal number carries most of the step effect.
- **By edge** (110 m, W2.0, within 20 cells, after the W update): western (95°W) 1.2×10⁻⁶, southern (10°S) 3.5×10⁻⁷, eastern 4.8×10⁻⁸, northern (30°N) 3×10⁻¹¹.
- **Decay from the edge:** R drops to the clean level at about 40 (W1.5), 50 (W2.0) and 60 cells (W2.5), i.e. about 2ℓ_box. For the Rossby configs it now reaches ≤1.5×10⁻⁹ (before the W update: ~5×10⁻⁹).
- **Recommendation:** treat W as inconsistent with U,V within ~50 cells (~450 km) of the western (95°W) and southern (10°S) edges of the regional grid. Both lie outside the 70–30°W, 5°S–30°N analysis domain. After the W update the 30°N edge shows no residual.

![R vs latitude and grad l](analysis/spectra_divergence/fig_R_lat_gradl.png)

**Rossby configs: residual from the spatial variation of ℓ, now fixed.**
- In the first W run, the open-ocean R of R2Ld/R3Ld was 40–60× that of the W configs, 1–2×10⁻⁸ s⁻¹. It peaked at 5°N and 7.5°S, where |∇ℓ| peaks.
- The cause was the operator form: the vector filter smooths the divergence as D̄ − γ∇·∇(ℓ²D̄) = D, while the scalar W filter used W̄ − γ∇·(ℓ²∇W̄) = W.
- W is now filtered with exactly the T-cell operator of the vector filter (`filter_level_w`). Open-ocean ("clean") R dropped by 14–17× for R2Ld (6.2e-9 → 4.4e-10 at 110 m) and 4× for R3Ld. For the W configs it is now at the rigid-lid ORIG level (3–9×10⁻¹¹ at 110 m).
- The small remaining R3Ld value (1.5e-9 at 110 m) comes from the grid-scale e3 and ℓ variations of the smoothed Ld field. It is below 1 % of the filtered divergence.
- **Edges after the update** (110 m, W2.0, within 20 cells): 95°W 1.2×10⁻⁶, 10°S 3.5×10⁻⁷, 10°E 4.8×10⁻⁸ and **30°N 3×10⁻¹¹ (clean)**. The residual at the W and S edges comes from the regional cut: the edge cells' discrete divergence misses the flux through the cut. It is far outside the 70–30°W analysis domain.

![Amazon mouth](analysis/spectra_divergence/fig_R_amazon.png)

**Amazon mouth (53–45°W, 2°S–5°N).**
- **GLORYS's own free-surface W:** R ≈ 0, with RMS 4×10⁻¹³ s⁻¹ and box-integrated source Q = 0.07 m³/s. The original W is fully determined by U,V through bottom-up continuity. The river and E−P source is therefore *not* a residual in GLORYS. It is already inside W, as the non-zero surface value w(0) = −∫∇h·u dz, which includes the divergence of the river outflow.
- **Rigid-lid ORIG:** Fix_W removes w(0) linearly, so the source reappears in R as −w(0)/H, uniform over the column. The box- and depth-integrated source is Q = ∫R dV = 2.5×10⁵ ± 0.5×10⁵ m³/s (31-day mean ± daily std). This is the order of the Amazon + Pará discharge plus P−E over the box (Amazon mean ≈ 2×10⁵ m³/s), with daily ∂η/∂t on top.
- **The source appears as blocky, depth-uniform patches** of 10⁻⁷–5×10⁻⁷ s⁻¹ at the mouth. These most likely reflect GLORYS's runoff distribution. That the source is inside W (as w(0)) and not in R is established above; the attribution of the patches specifically to runoff is an interpretation.
- **Filtered surface layer:** R is a smooth plume, the filtered source. Its maximum is 2.6–3.2×10⁻⁷ s⁻¹ (ORIG 4.7×10⁻⁷) and its box RMS 0.8–1.0×10⁻⁷ (ORIG 1.1×10⁻⁷). So, following the user's instructions, R at the mouth can be read as the filtered river source *in the surface layer*.
- **But not as a volume budget.** The filtered box- and depth-integrated Q is 3.1–4.8×10⁵ m³/s with a daily std of 5–7×10⁵ m³/s, against ±0.5×10⁵ for ORIG. The column integral of R is dominated by ±10⁻⁵ m/s structures along the shelf break and slope at deeper levels, which are the coastal and step residual (table above). A reliable source estimate is only possible for ORIG.

![hdiv versions](analysis/spectra_divergence/fig_hdiv_versions.png)

**v1 vs v2 vs v3: coastal divergence** (W2.0, surface, 1 Jan, 70–30°W 5°S–30°N). RMS hdiv [s⁻¹]:

| | ORIG | v1 component-wise | v2 vector + bt-corr | v3 vector |
|---|---|---|---|---|
| first wet cell | 6.8e-6 | **2.0e-5** | 7.6e-7 | 8.7e-7 |
| ≤3 cells from coast | 7.5e-6 | **1.1e-5** | 9.0e-7 | 7.9e-7 |
| 4–15 cells | 4.6e-6 | 5.8e-7 | 7.2e-7 | 5.9e-7 |
| >15 cells | 1.7e-6 | 4.2e-7 | 4.2e-7 | 4.2e-7 |

- v1's one-cell convergence line along walls (3× ORIG in the first wet cell) is gone in v2 and v3.
- In v3 the coastal divergence is smoothly reduced: the ≤3-cell band is 0.11× ORIG, while offshore stays at 0.24×.
- v3 and v2 agree offshore to within 1%. v3 has slightly lower divergence in the coastal band, though slightly higher in the first wet cell.

### 3. W statistics

![W profiles](analysis/spectra_divergence/fig_w_profiles.png)

- **Sanity checks:** filtered W is exactly 0 at the surface and at every sea-floor W point, with no NaN in wet cells (day 1, all configs).
- **RMS W over 70–30°W 5°S–30°N, 31 days:**

| m/s (ratio to ORIG) | 100 m | 500 m | 1000 m |
|---|---|---|---|
| ORIG (rigid-lid) | 4.9e-5 | 7.9e-5 | 7.5e-5 |
| W1.5 | 1.9e-5 (0.38) | 3.0e-5 (0.38) | 3.4e-5 (0.46) |
| W2.0 | 1.6e-5 (0.33) | 2.5e-5 (0.32) | 2.9e-5 (0.39) |
| W2.5 | 1.4e-5 (0.28) | 2.1e-5 (0.27) | 2.5e-5 (0.34) |
| R2Ld | 1.4e-5 (0.28) | 2.2e-5 (0.28) | 2.7e-5 (0.36) |
| R3Ld | 1.1e-5 (0.21) | 1.7e-5 (0.21) | 2.1e-5 (0.28) |

- Open ocean only (>15 cells from land, <29°N): ORIG 3.0/5.6/5.8×10⁻⁵ m/s; W2.0 1.5/2.3/2.6×10⁻⁵ (ratios 0.50/0.41/0.46). The ratios are higher than for all wet points because ORIG W is enhanced near topography.
- The filtered W keeps 21–46% of the RMS, consistent with W variance being concentrated at small scales: the retained W variance in the spectral boxes is 4–50%. The v1 problem, with W at 10× ORIG, is gone.

### Problems / caveats
1. **Regional-grid edges.**
   - The residual is 3–10×10⁻⁷ s⁻¹ within 20 cells of the western and southern edges, and still 10⁻⁸ within 20 cells of 30°N. It decays to background only about 2ℓ_box (35–55 cells) inside.
   - After the W update the 30°N edge (inside the analysis domain) is clean; the western and southern edges are outside it.
2. **Coasts and steep topography.**
   - The residual within 3 cells of land is 1–9×10⁻⁷ s⁻¹ at 0.5–760 m, 10³× ORIG at depth. It dominates the domain R² below the surface.
   - For Lagrangian tracking near the shelf break and slope, U,V and W are not mutually consistent at these levels.
3. **Rossby configs.** The open-ocean R from the different operator forms was fixed by filtering W with the vector filter's T-cell operator. Remaining clean-region R is ≤1.5×10⁻⁹ s⁻¹.
4. **Amazon.**
   - The filtered surface-layer R is a plausible filtered source.
   - The box- and depth-integrated filtered source is not usable: 3–5×10⁵ ± 5–7×10⁵ m³/s vs ORIG 2.5×10⁵ ± 0.5×10⁵.
   - In GLORYS itself the source lives in W (w(0)), not in R.
5. **Tropical spectra.** The U,V spectral response is 10–24% weaker than G² in λ½ at the surface (W: 0–16%). This is attributed to window, box size and the nearby coast. KE and W variance retention still match G² within 0.02.
6. **Run history.** An intermediate v3 run (vorticity term without e3f) was discarded. All numbers here are from the fixed rerun (output/V3_FIXED).
