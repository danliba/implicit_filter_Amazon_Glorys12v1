## Boundary behaviour of the filter (coasts, shelf, islands, passages, domain edges): v3, compared with v1

**Verdict (v3).** The v3 boundaries are numerically clean. They are also physically much better than v1 where it matters most: passages and coast-normal flow.

What is clean:
- Land faces stay NaN (0 mismatches), so no flow crosses the coast.
- The CPU re-implementation reproduces the GPU output (U,V 6·10⁻⁸ m/s; W 1.5·10⁻¹¹ m/s).
- W obeys the maximum principle.

What v3 fixed:
- Lesser Antilles transport is now conserved: arc sum −20.63 to −20.65 Sv vs ORIG −20.82 Sv. v1 lost 33–53 % of it.
- The spurious amplification of flow normal to the coast at the first wet faces is roughly halved.

Four boundary effects remain. They are intrinsic to a horizontal, level-by-level smoother with a free-slip wall, not bugs:
(i) the mean speed within ~45 km of the coast rises by +12…+29 %, unchanged from v1;
(ii) KE of the depth-mean flow on the shelf grows by 22–29 %, because the filter mixes across the shelf break;
(iii) the slope current at ~1000 m is damped more than the flat deep ocean;
(iv) a domain cut perturbs the first row by 45–100 % of the filtered signal for U,V (v1: 38–45 %). The error decays as exp(−d·√2/ℓ), and **a 1 % margin is ≈ 5 λ ≈ 3.5 ℓ, about one filter box width.** At the 30°N edge that is 0.6–2.2° depending on the config.

W (scalar filter of the rigid-lid W) shows **no boundary artefact**. It removes the ±20–30 m/day grid-scale noise that ORIG W carries along the coast and the shelf break.

Scripts: `accumulate.py` (SLURM, monthly U/V/W statistics), `boundary_stats.py`, `maps.py`, `passages.py`, `edge_test.py`, `validate_cpu.py`, `mask_check.py`, `check_overshoot.py`, `assemble_metrics.py`, driver `run_all_v3.sh`. All numbers, including the block `v1_vs_v3`, are in `analysis/boundaries/metrics.json`. The earlier v1 section, figures and metrics are archived in `analysis/boundaries/v1/`.

### Method (unchanged from v1 unless noted)
- **Monthly statistics.** Per config, streamed over 31 days × 50 levels × the whole domain: mean, mean square and mean squared daily (filtered − ORIG) difference for U, V and (new) W.
  - ORIG W is the rigid-lid `W_1993-01fc.nc`.
  - NaN-mask identity check.
  - Range check: filtered values must stay within [min, max] of ORIG per connected wet region. It is exact for the scalar W filter. For the coupled U,V vector operator it is only a diagnostic.
- **Distance from the coast.** Euclidean distance transform of each grid's own wet mask at the level. The analysis domain is 70–30°W, 5°S–30°N, without the last row and column.
- **Regions.** By bottom depth: shelf H < 200 m, slope 200–1000 m, deep H > 3000 m. Weights e1·e2·e3.
- **First wet faces normal to the coast.** A wet U point whose U(i±1) neighbour is land; a wet V point whose V(j±1) neighbour is land. Onshore is positive.
- **Passage transports.** Staircase sections along cell edges between land cells, full depth, daily.
- **Domain edges.**
  - CPU re-run of the production operators on 80–20°W, 1 Jan: `filter_level_vector` for U,V at the surface, the scalar T filter for W at 101 m. Each box is filtered twice, with an artificial wall (at 25°N or 0°) and with the real halo beyond it.
  - New: the daily surface-layer continuity residual R from `diag_<CFG>.npz`, profiled against distance from the 30°N, 10°S and 95°W edges (points > 300 km from any coast). The residual statistics themselves are in the spectra/divergence section; only the boundary dependence is shown here.

### Summary: v1 (component-wise, Neumann) vs v3 (vector div–rot, free slip)

| Problem found in v1 | v1 | v3 | Status |
|---|---|---|---|
| Max coastal speed ratio in 0–45 km (W1.5 / W2.0 / W2.5 / R2Ld / R3Ld) | 1.12 / 1.16 / 1.20 / 1.24 / 1.31 | 1.13 / 1.18 / 1.22 / 1.24 / 1.29 | **not fixed** (intrinsic) |
| Speed ratio, first bin 0–13 km | 1.06–1.22 | 1.12–1.24 | slightly higher (free slip) |
| RMS mean onshore velocity at first wet faces (ORIG 12.4 cm/s) | 17.8–21.1 (+44…+71 %) | 14.6–16.8 (+18…+36 %) | **halved** |
| Correlation with ORIG at those faces | 0.80–0.85 | 0.87–0.89 | better |
| Σ\|onshore transport\| at first wet faces (ORIG 0.99 Sv) | 1.48–1.83 Sv | 1.10–1.26 Sv | better |
| Lesser Antilles arc sum (ORIG −20.82 Sv) | −9.8 … −13.9 | −20.63 … −20.65 | **fixed** (−0.9 %) |
| St Vincent Passage (ORIG −5.84 Sv) | −1.9 … −3.1 | −5.84 … −5.86 | **fixed** |
| St Lucia Channel (ORIG −5.56 Sv) | −1.5 … −3.1 | −5.54 | **fixed** |
| Guadeloupe Passage (ORIG −0.040 Sv) | −0.19 … −0.52 | −0.040 … −0.048 | **fixed** |
| Dragon's Mouth (ORIG +0.123 Sv) | +0.05 … +0.06 | +0.14 … +0.15 | better; now +16…+25 % |
| Shelf depth-mean-flow KE retained | 1.23–1.34 | 1.22–1.29 | **not fixed** (cross-shelf-break mixing) |
| Slope vs deep speed ratio at 1062 m, W2.0 | 0.66 vs 0.74 | 0.62 vs 0.74 | slightly worse |
| Slope vs deep speed ratio at 1062 m, R2Ld | 0.43 vs 0.54 | 0.38 vs 0.54 | slightly worse |
| Edge: relative error in the first row (U,V) | 0.38–0.45 | 0.45–1.03 | worse at the wall |
| Edge: 1 % distance, north wall (U,V) | 76–211 km | 76–203 km | unchanged |
| Edge: d(1 %)/λ, all tests | 2.5–4.7 | 3.0–4.9 | unchanged |
| North-edge EKE-retained uptick, 29–30°N minus 25–29°N, surface | +0.07 (W2.0, R2Ld) | +0.15 (W2.0), +0.09 (R2Ld) | larger |

### 1. Masks, no-flux and range checks
- **NaN masks.** The ORIG U/V NaN mask equals the C-grid mask on all levels. The filtered U, V, W masks are identical for all 5 configs (0 mismatches). There is exactly zero flow through the coast.
- **W range check.** Exceedance 0.0: the scalar filter is a positive averaging operator.
- **U,V range check.**
  - The vector operator has no maximum principle. Values leave the component-wise regional range by more than 1 mm/s at ~1·10⁻⁴ of wet points.
  - The largest cases (0.3–0.5 m/s) are isolated single-face U or V pockets (1–2 wet points of one component, connected only through the other component). The coupled operator damps them almost to zero.
    - Examples: Serpent's Mouth, 62.0°W 10.0°N at 16 m, u −0.41 → −0.01 m/s; 44.3°W 1.3°S; 54.9°W 6.2°N (`metrics.json → overshoot`). This is strong local damping in 1-cell inlets, not ringing.
  - One real overshoot sits in a large connected region: the Yucatán/Cozumel channel, 86.9°W 20.7°N at 78 m, u 0.93 → 1.45 m/s. It is outside the analysis domain.
  - V in the last column (9.9°E) is also amplified (0.03 → 0.14 m/s). This is an edge effect, covered in section 9.

### 2. Coast-normal flow and the residual near the coast (task 2c)

| | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|
| RMS monthly-mean onshore velocity, first wet faces [cm/s] (ORIG 12.4) | 14.6 | 15.5 | 16.1 | 16.1 | 16.8 |
| Correlation with ORIG | 0.89 | 0.88 | 0.87 | 0.89 | 0.87 |
| RMS daily change [cm/s] (ORIG daily RMS 13.0) | 7.1 | 8.0 | 8.7 | 8.2 | 9.1 |
| Net onshore transport [Sv] (ORIG −0.040) | −0.065 | −0.068 | −0.070 | −0.078 | −0.078 |

![Coast-normal flow and residual](analysis/boundaries/fig_coastal_normal.png)

- **v1 → v3.** The one-cell convergence line along every wall in v1 is gone. The filtered divergence is the no-flux scalar filter of the original divergence. Normal flow at the first wet faces is still somewhat amplified, +18…+36 %, because the smoothing kernel is one-sided at the wall. That is half of v1.
- **Residual near the coast.** The continuity residual R of v3 is ~1·10⁻⁷ s⁻¹ within 100 km of the coast, *the same as ORIG*. There it contains river runoff, E−P and the rigid-lid correction.
  - Offshore of 300 km, R is 10⁻¹⁰–10⁻⁸ s⁻¹ (ORIG 3·10⁻¹¹). **The filter adds no coastal residual.**

### 3. Distance from the coast (task 2a)

![Distance from coast, surface](analysis/boundaries/fig_coastdist_surface.png)

- **Mean speed at the surface.** The ratio of monthly-mean speed is **> 1 within ~45–60 km of every coast** in v3, as in v1:
  - 0–13 km: 1.12 / 1.16 / 1.19 / 1.20 / 1.24 (W1.5 / W2.0 / W2.5 / R2Ld / R3Ld)
  - 13–22 km (peak): 1.13–1.29
  - 45–60 km: 1.00–1.05
  - beyond 750 km: 0.76–0.92

  The monotonicity is the same as in v1. 38–54 % of coastal points with ORIG speed > 5 cm/s gain more than 20 %.
- **Why.** The fast NBC and Guiana Current are averaged into the slow coastal band. With free slip the tangential velocity is not pulled to zero at the wall. So the effect is intrinsic to a free-slip horizontal smoother and was not caused by the v1 Neumann walls.
- **Daily differences.** Relative RMS daily difference is 0.42–0.54 at 0–13 km (v1 0.52–0.64), falling to 0.20–0.35 at 100–300 km.
- **EKE retained.** 0.64–0.76 at 0–13 km (v1 0.40–0.62) vs 0.33–0.47 at 45–80 km. It is still non-monotonic, and higher at the coast than in v1.
- **At depth.** Near the coast v3 keeps more mean speed than v1:
  - 110 m, first two bins: 0.90–1.04 (v1 0.77–0.99)
  - 454 m, first bin: 0.73–0.83 (v1 0.58–0.71)

![Distance from coast, depth](analysis/boundaries/fig_coastdist_depth.png)

### 4. Shelf, slope and deep ocean (task 2b)

| v3 | KE retained, full depth (shelf / slope / deep) | EKE retained (shelf / slope / deep) | Depth-mean flow: rel. RMS change (shelf / slope / deep) | Depth-mean KE retained, shelf (v1) |
|---|---|---|---|---|
| W1.5 | 0.99 / 0.75 / 0.56 | 0.58 / 0.47 / 0.46 | 0.29 / 0.26 / 0.35 | **1.22** (1.23) |
| W2.0 | 0.97 / 0.68 / 0.45 | 0.51 / 0.38 / 0.34 | 0.33 / 0.31 / 0.46 | **1.27** (1.30) |
| W2.5 | 0.95 / 0.61 / 0.36 | 0.46 / 0.31 / 0.27 | 0.37 / 0.35 / 0.54 | **1.29** (1.34) |
| R2Ld | 0.89 / 0.60 / 0.44 | 0.49 / 0.37 / 0.33 | 0.37 / 0.37 / 0.39 | **1.27** (1.33) |
| R3Ld | 0.78 / 0.47 / 0.29 | 0.38 / 0.24 / 0.19 | 0.42 / 0.46 / 0.53 | **1.22** (1.33) |

- **Shelf and slope KE.** Slope KE is better retained than in v1 (e.g. W2.0 0.68 vs 0.59). Deep-ocean EKE is unchanged (±0.005).
- **Shelf depth-mean flow still gains 22–29 % KE.** Level by level, the upper layers of the shelf are connected to the NBC over the slope, and the filter spreads NBC momentum onto the shelf.
  - Along 5°N the shelf-mean surface speed barely changes (ORIG 0.76; filtered 0.71–0.76 m/s).
  - The NBC core at the shelf break drops from 1.12 to 0.77–0.96 m/s, and the innermost shelf points gain speed.

![Shelf transects, v3 solid, v1 dashed](analysis/boundaries/fig_shelf_transects.png)

### 5. Amazon shelf and mouth maps (task 1)

![Shelf mean](analysis/boundaries/fig_shelf_mean.png)
![Shelf daily](analysis/boundaries/fig_shelf_daily.png)
![Amazon mouth](analysis/boundaries/fig_mouth_mean.png)

- **No wall artefacts.** No sign alternation, wall-parallel jets or checkerboards in U,V at the coast or in the estuary channels.
- **Where the differences are.** They concentrate on:
  - the NBC core along the 200 m isobath: |Δu| 0.3–0.5 m/s in the monthly mean, up to 0.8 m/s daily (R3Ld);
  - the small mid-shelf jets, which are removed;
  - for R2Ld/R3Ld, the retroflection eddy at 45°W 6°N.
- **v1 coastal jump gone.** In the 2°N transect, v1 had a jump in the first coastal U cell (−0.26 m/s vs ORIG −0.05). v3 does not (−0.07 m/s).

### 6. W boundaries (new)

![W 51 m shelf](analysis/boundaries/fig_w_shelf_50m.png)
![W 203 m shelf break](analysis/boundaries/fig_w_shelf_200m.png)
![W 20 m Amazon mouth](analysis/boundaries/fig_w_mouth_20m.png)
![W 51 m Amazon mouth](analysis/boundaries/fig_w_mouth_50m.png)
![W daily 51 m](analysis/boundaries/fig_w_shelf_daily_50m.png)
![W vs distance from coast](analysis/boundaries/fig_w_coastdist.png)

- **ORIG W is noisy at walls.** It carries grid-scale ±20–30 m/day alternating structure along the coast and the shelf break. This comes from GLORYS partial cells and topography, not from the filter.
  - 51 m shelf box: monthly-mean RMS W is 23.2 m/day in the first wet cell vs 16.8 at 3–10 cells from land.
  - 203 m: 19.4 vs 10.5 m/day.
- **Filtered W is smooth up to the wall, with no coastal banding.** First-wet-cell vs 3–10-cell RMS:
  - 51 m shelf: 1.2–3.4 vs 1.0–3.4 m/day
  - 203 m: 0.7–3.2 vs 0.7–2.9 m/day
  - Amazon mouth, 20 m: 0.3–1.3 vs 0.5–1.8 m/day
- **Coastal W variance.** Within 50 km of the coast, filtered W keeps only 1–6 % of the daily W variance, vs 10–29 % far offshore. The monthly-mean RMS ratio is 0.05–0.16 at the coast and 0.3–0.7 at 200–500 km.
  - W is dominated by grid-scale variability that ℓ ≫ Δx removes. The coastal noise is removed most strongly because it is the most grid-scale.
- **Lesser Antilles W at 51 m.** ORIG has ±10 m/day dipoles at every island tip (RMS 5.5 m/day in 62.5–59.5°W, 10.5–16.5°N). Filtered W is a smooth upwelling lobe in the lee of Grenada/Tobago (RMS 0.85–1.39 m/day), and nothing crosses the islands.
- **No land leakage.** The scalar W filter cannot smooth across land, and filtered W respects the level mask at the shelf break (fig. 203 m).

### 7. Shelf break at depth (task 4)

![Slope at depth](analysis/boundaries/fig_depth_slope.png)

Region 50–44°W, 0–8°N, v3 mean speed ratio:

| Depth | Config | Slope (H < 3000 m) | Deep (H > 3000 m) |
|---|---|---|---|
| 454 m | W2.0 | 0.80 | 0.75 |
| 454 m | R2Ld | 0.57 | 0.54 |
| 1062 m | W2.0 | **0.62** | 0.74 |
| 1062 m | R2Ld | **0.38** | 0.54 |

The narrow deep western boundary current on the steep slope at ≈49–50°W, 4–5.5°N is only 1–3 cells wide. It is diluted by the one-sided kernel, slightly more than in v1 (0.66 / 0.43). At 454 m the slope retains relatively more, because the fast zonal jets sit in open water.

### 8. Islands and passages (task 3)

![Passages map](analysis/boundaries/fig_passages_map.png)
![Passage transports](analysis/boundaries/fig_passages_transport.png)

Full-depth mean transport, Sv. Negative is westward into the Caribbean; surface wet width in brackets. The v1 range is over the 5 configs.

| Passage | ORIG | v3 range (5 configs) | v1 range |
|---|---|---|---|
| Grenada Passage (181 km) | −6.94 | −6.93 … −6.95 | −3.90 … −5.86 |
| St Vincent Passage (63 km) | −5.84 | −5.84 … −5.86 | −1.94 … −3.07 |
| St Lucia Channel (54 km) | −5.56 | −5.54 | −1.55 … −3.07 |
| Galleons (Trinidad–Tobago, 55 km) | −1.11 | −1.10 … −1.11 | −0.61 … −0.72 |
| Grenadines (136 km, shallow) | −1.18 | −0.99 … −1.03 (−13…−16 %) | −0.85 |
| Dominica Passage (72 km) | −0.16 | −0.16 … −0.17 | −0.09 … −0.47 |
| Guadeloupe Passage (54 km) | −0.040 | −0.040 … −0.048 | −0.19 … −0.52 |
| **Arc sum** | **−20.82** | **−20.63 … −20.65** | −9.8 … −13.9 |
| Dragon's Mouth (27 km, + north) | +0.123 | +0.142 … +0.154 (+16…+25 %) | +0.05 … +0.06 |
| Serpent's Mouth (27 km) | −0.119 | −0.103 … −0.105 (−12…−13 %) | −0.08 … −0.10 |

- **Transport is conserved in v3.** The passage jets are smoothed, but transport between islands is kept to < 1 % in the deep passages.
  - Why: the div–rot operator filters divergence and interior vorticity. The flow through a gap between two land masses is carried by the island circulation (a harmonic, divergence- and vorticity-free component). With free slip that component lies in the operator's null space, so filtering leaves it unchanged.
  - The remaining changes are in shallow passages and 1–3-cell mouths (Grenadines, Dragon's Mouth, Serpent's Mouth, ±12–25 %), where single-face pockets exist at some levels (section 1).
- **Gulf of Paria.** Mean surface speed rises by +19 % (R2Ld) to +44 % (W2.5) (v1 +2…+54 %). Momentum from the strong flow outside enters through the mouths: the same coastal speed-up as in section 3.
- **Unresolved islands.** Fernando de Noronha and St Peter & St Paul are not in the mask (no land cell within 3 cells), so the filter smooths across them.

### 9. Domain edges (task 5)

![Edge test](analysis/boundaries/fig_edge_test.png)

- **U,V wall error.** An artificial wall changes filtered U,V in the first row by 45–103 % of the RMS of the filtered field (v1 38–45 %). The larger wall error comes from the V (U) points of the last row (column): the vector operator treats them as unknowns with outward flux.
- **Decay.** The error decays as exp(−d/λ), λ = ℓ/√2, as in v1. The distance to 1 % is unchanged, e.g. north wall 76 km (R2Ld) to 203 km (W2.5).
- **W.** Scalar filter at 101 m: first-row error 0.29–0.66 (north) and 0.53–0.76 (south). 1 % is reached at 67–194 km (north) and 139–592 km (south, where ℓ is large).
- **Margin rule.** Over all v3 tests, d(1 %) = 3.0–4.9 λ and d(5 %) = 1.6–3.1 λ.
  - **Recommended margin: 5 λ ≈ 3.5 ℓ (one ℓ_box) for 1 % error, ≈ 2.2 ℓ for 5 %.**

  | Edge | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
  |---|---|---|---|---|---|
  | 30°N, 1 % | 144 km (1.3°) | 192 km (1.7°) | 240 km (2.2°) | 66 km (0.6°) | 99 km (0.9°) |
  | 10°S, 1 % | 164 km | 218 km | 273 km | ≥ 195 km | ≥ 293 km |

  - The analysis domain starts 555 km from the 10°S edge. That is safe for all W configs, and for R configs at the ~1–2 % level.
  - **At 30°N (no halo), exclude 28–30°N for W configs and 29–30°N for R configs.**
- **Empirical north-edge check.** Zonal sum over 60–40°W, surface EKE retained:
  - W2.0: 0.35 / 0.48 / 0.63 at 20–25 / 25–29 / 29–30°N
  - R2Ld: 0.53 / 0.68 / 0.77

  The uptick in the last degree is larger than in v1 (+0.15 vs +0.07 for W2.0), consistent with the larger wall error of the vector operator. At 454 m it is +0.12 (v1 +0.04).

![North edge](analysis/boundaries/fig_north_edge_lat.png)

**Continuity residual near the edges** (surface layer, points > 300 km from coast; threshold relative to the ORIG surface hdiv RMS of 2.3·10⁻⁶ s⁻¹):

![Residual vs distance from edges](analysis/boundaries/fig_residual_edges.png)

- **ORIG.** The first row itself is 10⁻⁶–10⁻⁵ s⁻¹ in ORIG too: the last row has no outer neighbour. Exclude it always.
- **30°N, filtered.** Updated after W was refiltered with the vector filter's T-cell operator. Beyond the first row, which is the ORIG artefact, R equals the ORIG interior level: 2.3×10⁻¹¹ s⁻¹ for W1.5/W2.0 against ORIG 2.4×10⁻¹¹. **The 30°N edge adds no continuity residual.**
- **10°S (W configs).** R > 1 % of hdiv within 16–19 cells (145–175 km) and > 0.1 % within 24–30 cells (220–275 km). The interior level is (2–4)×10⁻¹¹. This edge is 555 km south of the analysis domain.
- **95°W (Pacific).** R stays at ~2×10⁻⁷ over 550 km (ORIG 6×10⁻⁸). Here the regional cut crosses the Central-American coast and shelf, far outside the analysis domain.
- **Margin.** For continuity, only the western and southern cuts matter, and both lie outside 70–30°W, 5°S–30°N. For the velocity edge error (above), keep ≈ 3.5 ℓ from 30°N.

### Problems remaining in v3, with locations
1. **Coastal speed-up.** Monthly-mean speed rises by 12–29 % within ~45 km of all coasts, unchanged vs v1. Examples: inner Guiana/Amapá shelf (52–50°W, 3–7°N); Gulf of Paria (+19…+44 %).
2. **Coast-normal flow at the first wet faces** is still +18…+36 % RMS (v1 +44…+71 %). There is zero flux through land faces.
3. **Shelf depth-mean flow KE** rises by 22–29 % (NBC momentum moved across the shelf break), and the ~1000 m slope current at 50–48°W, 3–6°N is damped more than the deep ocean (W2.0 0.62 vs 0.74; R2Ld 0.38 vs 0.54).
4. **Small inlets and 1–3-cell mouths.**
   - Single-face U/V pockets are damped almost to zero (Serpent's Mouth 62°W 10°N at 16 m: −0.41 → −0.01 m/s).
   - Transport changes: Dragon's Mouth +16…+25 %, Serpent's Mouth −12 %, shallow Grenadines −13…−16 %.
   - One overshoot at Cozumel, 86.9°W 20.7°N at 78 m (outside the analysis domain).
5. **Domain edges.**
   - The U,V wall error at the first row is larger than in v1 (up to ~100 %). Keep ≈ 3.5 ℓ from the edges: 1.3–2.2° at 30°N for W configs, 0.6–0.9° for R configs. The EKE uptick over the last degree at 30°N is +0.09…+0.16.
   - The last column (9.9°E) and the last row are unusable.
6. **Fixed in v3:** Lesser Antilles and Guadeloupe/Dominica passage transports, and the v1 one-cell convergence line along walls.
7. **W:** no boundary artefacts found. The final W is filtered with the vector filter's T-cell operator, and the W maps and statistics here were regenerated for it.
