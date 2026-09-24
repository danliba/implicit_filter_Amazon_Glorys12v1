## Boundary behaviour of the filter (coasts, shelf, islands, passages, domain edges)

**Verdict.** The boundaries are *clean in the numerical sense*. Land faces stay NaN, so there is no flow through the coast, there is no ringing or overshoot of the velocity components, and the CPU and GPU operators agree. They are *not neutral in the physical sense*. The no-flux (Neumann) condition sets the normal gradient of each velocity component to zero at land, and the filter is much wider than narrow passages. This causes four boundary artefacts that you should know about before using the filtered fields near land:
(i) the monthly-mean speed rises in the first ~30 km from the coast (+6 to +31 %);
(ii) the flow normal to the coast through the first wet faces grows by 44–71 % RMS;
(iii) full-depth transport through the Lesser Antilles passages drops by 33–53 %, and in single passages by up to 72 %;
(iv) near a domain cut the filtered field is wrong by 13–45 % in the first row, and the error decays as exp(−d·√2/ℓ).
None of these are bugs in the implementation. They follow from filtering U and V independently with a Neumann condition.

Scripts: `accumulate.py` (monthly statistics, SLURM), `boundary_stats.py`, `maps.py`, `passages.py`, `edge_test.py`, `validate_cpu.py`, `mask_check.py`, `assemble_metrics.py`. All numbers are in `analysis/boundaries/metrics.json`.

### Method
- **Monthly statistics.** For ORIG and every config, `accumulate.py` streams the 31 days on all 50 levels and the whole domain. It stores the mean, the mean square and the mean squared daily (filtered − ORIG) difference at the U and V points. It also checks the NaN mask and the maximum principle: in every 4-connected wet region, each filtered component must lie within [min, max] of the original component, on 6 levels and 31 days.
- **Distance from the coast.** Euclidean distance transform of each grid's own wet mask at the level, times the local √(e1·e2). The model-domain edge does not count as coast. The analysis domain is 70–30°W, 5°S–30°N, without the last row and column of the file.
- **Regions.** Split by H_u = min(H) of the two adjacent T columns. Quantities are volume weighted (e1·e2·e3) over all levels, or taken at the surface.
- **Coast-normal faces.** A wet U point is a coast-normal face if U(i±1) is a land point; a wet V point if V(j±1) is a land point. Velocities are signed positive onshore.
- **Passage transports.** Staircase paths along cell edges between land T cells. Each step adds e2u·e3u·u or e1v·e3v·v over the full depth, daily.
- **Domain-edge test.** CPU re-filtering of 1 Jan, surface, with `cgrid_filter.build_operator`. First, `validate_cpu.py` checks that the CPU operator matches the production output on the full domain: max |CPU − GPU| = 6·10⁻⁸ m/s for W2.0 and R2Ld, U and V. The test itself filters 80–20°W twice: with an artificial no-flux wall, and with the real halo beyond it. North wall at 25°N (reference extends to 30°N); south wall at 0° (reference extends to 10°S).

### 1. Coast, mask and no-flux check (task 2c)
- **Input mask.** On all 50 levels, the NaN mask of the original U and V is exactly the C-grid mask (umask = tmask_i·tmask_i+1, vmask = tmask_j·tmask_j+1). There are 0 mismatches in the interior and no exact zeros at wet points. The only finite values outside that mask are in the last column (U) and last row (V) of the regional file, which have no outer neighbour.
  - So U and V points on land faces are NaN in the input and are excluded from the filter. *This contradicts the premise of variant A in `spectra_divergence/w_diagnose.py`.*
- **Filtered mask.** Identical to the input mask for all 5 configs: 0 mismatches over 31 days × 50 levels. The normal velocity through the coast is therefore exactly zero (it is undefined/NaN on land faces, as in NEMO).
- **Maximum principle.** No violation, largest exceedance 0.0 m/s. The system matrix A + γK is an M-matrix and preserves constants, so each filtered component is a positive weighted mean of wet values in the same connected region. **Per-component ringing or overshoot is impossible, and none was found.**
- **First wet faces normal to the coast.** Surface, 1166 faces in the analysis domain:

| | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|
| RMS monthly-mean onshore velocity (ORIG 12.4 cm/s) | 17.8 | 19.0 | 19.8 | 19.7 | 21.1 |
| Correlation with ORIG | 0.85 | 0.83 | 0.81 | 0.84 | 0.80 |
| RMS daily change (ORIG daily RMS 13.0 cm/s) | 10.4 | 11.8 | 12.9 | 12.2 | 14.2 |
| Σ\|onshore transport\| (ORIG 0.99 Sv) | 1.48 | 1.59 | 1.69 | 1.66 | 1.83 |

  The net onshore transport stays small: ORIG −0.04 Sv, filtered +0.003 to +0.06 Sv. The coastal normal velocity at the first wet face is **amplified** because the Neumann condition removes its decay towards the wall: ∂u/∂n = 0 instead of u → 0. The resulting convergence and divergence in the coastal T cell is passed to W through continuity.

![Coast-normal flow](analysis/boundaries/fig_coastal_normal.png)

**Vertical velocity from the filtered fields.** W computed by continuity (w = 0 at the bottom) is about 100–150 m/day RMS at the surface W level, compared with 0.05 m/day for ORIG. The value is almost flat from 10 km to 300 km from the coast and still ~40–60 m/day at >500 km. It is therefore **not a coastal artefact**. It is column-integrated divergence produced at topography and partial cells in the level-by-level filtering (see the spectra/divergence section). In the Lesser Antilles box it is also 101–113 m/day with grid-scale noise (fig. passages, bottom row). Near coasts it does not rise above the open-ocean level, so the coastal effect above does not dominate W.

### 2. Distance from the coast (task 2a)

![Distance from coast, surface](analysis/boundaries/fig_coastdist_surface.png)

Surface results by distance bin (0–13, 13–22, 22–32, 32–45, 45–60 km, …):

| cfg | speed ratio ⟨\|ū\|⟩ filt/ORIG, first 5 bins | far field (>750 km) | rel. RMS daily diff, 0–13 km → 135–175 km | EKE retained, 0–13 km / 45–80 km / >400 km |
|---|---|---|---|---|
| W1.5 | 1.06, 1.11, 1.02, 1.01, 0.99 | 0.87–0.92 | 0.52 → 0.20 | 0.49 / 0.44 / 0.62–0.66 |
| W2.0 | 1.10, 1.16, 1.05, 1.02, 0.99 | 0.82–0.88 | 0.59 → 0.25 | 0.44 / 0.37–0.39 / 0.52–0.57 |
| W2.5 | 1.13, 1.20, 1.07, 1.03, 1.00 | 0.78–0.84 | 0.63 → 0.29 | 0.40 / 0.33–0.35 / 0.44–0.50 |
| R2Ld | 1.16, 1.24, 1.11, 1.07, 1.04 | 0.84–0.87 | 0.55 → 0.28 | 0.62 / 0.46–0.50 / 0.42–0.48 |
| R3Ld | 1.22, 1.31, 1.13, 1.09, 1.04 | 0.76–0.79 | 0.64 → 0.35 | 0.53 / 0.35–0.40 / 0.27–0.35 |

(For the R configs, the far-field numbers mix latitude effects with distance effects, because ℓ varies strongly with latitude.)

- **Speed overshoot at the coast.** The mean speed ratio is **> 1 within ~45 km of the coast** for every config. It peaks in the 13–22 km bin and is not monotonic: the first bin is lower than the second. Of the coastal points with ORIG speed > 5 cm/s, 41–59 % have a filtered mean speed more than 1.2 times the original.
  - This is not ringing. The maximum principle holds per component. The cause is the one-sided kernel: fast offshore flow (NBC, Guiana Current) is averaged into the slow coastal boundary layer, and with ∂/∂n = 0 nothing pulls the value back toward zero at the wall.
  - The effect grows with ℓ: W1.5 < W2.0 < W2.5 < R2Ld < R3Ld near the equatorial coast.
- **Daily differences.** The relative RMS daily difference is largest at the coast, 52–64 %, and falls to 20–37 % at 100–200 km.
- **EKE retained.** The curve is non-monotonic. It is higher in the first 20 km than at 45–80 km, by +0.05 to +0.17. Part of the removed EKE is compensated by momentum spread into the quiet coastal band.
- **At depth.** At 110 m, 454 m and 1062 m the speed ratio stays < 1 at every distance (fig. below). The coastal overshoot is a feature of the surface and shelf levels, where a strong current flows next to a shallow wall. At 110 m the first 20 km reach 0.93–0.99 for the W configs, still slightly higher than at 30–100 km.

![Distance from coast, depth](analysis/boundaries/fig_coastdist_depth.png)

### 3. Shelf, slope and deep ocean (task 2b)
Values are volume weighted over the full depth, analysis domain.

| | KE ret. shelf / slope / deep | EKE ret. shelf / slope / deep | Depth-mean flow: rel. RMS change (shelf/slope/deep) | Depth-mean KE ret. (shelf/slope/deep) |
|---|---|---|---|---|
| W1.5 | 0.96 / 0.67 / 0.55 | 0.59 / 0.49 / 0.45 | 0.32 / 0.32 / 0.36 | **1.23** / 0.82 / 0.52 |
| W2.0 | 0.95 / 0.59 / 0.44 | 0.53 / 0.39 / 0.34 | 0.38 / 0.38 / 0.46 | **1.30** / 0.75 / 0.39 |
| W2.5 | 0.93 / 0.52 / 0.35 | 0.48 / 0.31 / 0.26 | 0.42 / 0.42 / 0.55 | **1.34** / 0.67 / 0.30 |
| R2Ld | 0.89 / 0.51 / 0.42 | 0.52 / 0.39 / 0.32 | 0.43 / 0.43 / 0.40 | **1.33** / 0.60 / 0.50 |
| R3Ld | 0.79 / 0.37 / 0.28 | 0.40 / 0.25 / 0.19 | 0.50 / 0.53 / 0.54 | **1.33** / 0.44 / 0.33 |

(shelf H < 200 m, slope 200–1000 m, deep H > 3000 m; the 1000–3000 m class is in `metrics.json`.)

- **Shelf KE and EKE.** The shelf keeps almost all of its KE (79–96 %) but only 40–59 % of its EKE.
- **Depth-averaged mean flow on the shelf gains 23–34 % KE.** At the upper levels the shelf is connected to the NBC over the slope. The filter moves NBC momentum onto the shelf and shelf momentum into the deep water: it *diffuses across the shelf break*, level by level, without regard to depth.
  - Along 5°N the shelf-mean surface speed hardly changes (ORIG 0.76, filtered 0.69–0.76 m/s). The shape changes, though: next to the coast the speed rises from ~0.1 to ~0.3–0.4 m/s, and the NBC core at the shelf break drops from 1.1 to 0.85–0.95 m/s (transects).
  - This is the expected behaviour of a horizontal filter, but it produces unrealistically strong velocities on the inner shelf, where the original flow is weak.

![Shelf transects](analysis/boundaries/fig_shelf_transects.png)

### 4. Amazon shelf and mouth maps (task 1)

![Shelf mean](analysis/boundaries/fig_shelf_mean.png)
![Shelf daily](analysis/boundaries/fig_shelf_daily.png)
![Amazon mouth](analysis/boundaries/fig_mouth_mean.png)

- **No artefacts at the coastline.** No sign alternation, checkerboard or spurious jets are visible along the coast or in the estuary channels.
- **Largest differences.** (a) Along the 200 m isobath (NBC core), |Δu| 0.3–0.5 m/s in the monthly mean and up to 0.8 m/s daily for R3Ld. (b) In the small-scale jets over the mid-shelf that exist in ORIG, which are removed.
- **Mouth and channels.** Inside the mouth and the Pará channels the filtered flow is smooth and weak. Each channel is filtered only along itself, because the filter cannot cross land. The differences there are small (< 0.1 m/s).
- **Inner shelf.** At 3–7°N, 52–50°W the inner shelf gets a stronger along-shore (north-westward) flow than ORIG, as described in section 3.
- **Discontinuities in ORIG itself.** ORIG shows straight zonal discontinuities near 0.7°N and 1.1°N, 49–47°W. They are present in GLORYS itself, not caused by the filter.

### 5. Shelf break at depth (task 4)

![Slope at depth](analysis/boundaries/fig_depth_slope.png)

Region 50–44°W, 0–8°N, monthly-mean speed ratio:

| Depth | Config | Slope (H < 3000 m) | Deep (H > 3000 m) |
|---|---|---|---|
| 454 m | W2.0 | 0.83 | 0.75 |
| 454 m | R2Ld | 0.66 | 0.54 |
| 1062 m | W2.0 | **0.66** | 0.74 |
| 1062 m | R2Ld | **0.43** | 0.54 |

- **At 1062 m the slope loses more than the flat deep ocean.** The narrow boundary current against the steep wall (≈49–50°W, 4–5.5°N) is only 1–3 cells wide. It can only be spread offshore, so the one-sided kernel dilutes it more strongly; |Δu| peaks directly along the wall.
- **At 454 m the pattern is reversed.** The slope keeps relatively more, because the level's mask is wide and the fast zonal jets (2.5–3°N, 5–6°N) sit in the open water.
- **Filter type.** R2Ld (ℓ ≈ 100–135 km at 0–5°N) removes much more than W2.0 (ℓ ≈ 63 km) at both depths.

### 6. Islands and narrow passages (task 3)

![Passages map](analysis/boundaries/fig_passages_map.png)
![Passage transports](analysis/boundaries/fig_passages_transport.png)

Full-depth mean transport in Sv (negative = westward into the Caribbean; surface wet width in brackets):

| Passage | ORIG | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|---|
| Grenada Passage (181 km) | −6.94 | −5.32 | −4.55 | −3.90 | −5.86 | −4.93 |
| St Vincent Passage (63 km) | −5.84 | −2.59 | −2.19 | −1.94 | −3.07 | −2.40 |
| St Lucia Channel (54 km) | −5.56 | −2.37 | −1.86 | −1.55 | −3.07 | −2.23 |
| Galleons (Trinidad–Tobago, 55 km) | −1.11 | −0.68 | −0.64 | −0.61 | −0.72 | −0.66 |
| Grenadines (136 km, shallow) | −1.18 | −0.85 | −0.85 | −0.85 | −0.86 | −0.85 |
| Dominica Passage (72 km) | −0.16 | −0.20 | −0.34 | −0.47 | −0.09 | −0.22 |
| Guadeloupe Passage (54 km) | −0.04 | −0.34 | −0.45 | −0.52 | −0.19 | −0.33 |
| **Arc sum, Galleons–Guadeloupe** | **−20.8** | −12.4 | −10.9 | −9.8 | −13.9 | −11.6 |
| Dragon's Mouth (27 km, + north) | +0.12 | +0.05 | +0.05 | +0.06 | +0.05 | +0.05 |
| Serpent's Mouth (27 km) | −0.12 | −0.09 | −0.08 | −0.08 | −0.10 | −0.08 |

- **Transport loss.** The filtered fields lose **33–53 % of the inflow through the southern Lesser Antilles**. The loss is concentrated in the narrow, deep passages: St Vincent and St Lucia keep only 28–55 %. Weak passages gain spurious westward transport; the Guadeloupe Passage goes from −0.04 to −0.5 Sv.
  - **Cause.** The passages are 54–63 km wide, while the filter width ℓ_box = 3.5ℓ is 140–270 km (at 13°N ℓ ≈ 46/62/77 km for W1.5/W2.0/W2.5 and ≈ 40/60 km for R2Ld/R3Ld). The filter cannot smooth across the islands. It smooths the passage jet *along* the only wet path, mixing it with the slow water in the island lee on both sides. Because U and V are filtered separately and the filter does not conserve transport, section transports are not conserved. The imbalance ends up in W by continuity.
- **Semi-enclosed basin.** In the Gulf of Paria the connection to the Caribbean (Dragon's Mouth) and to the Atlantic (Serpent's Mouth) is 3 cells each. The mean surface speed there *increases* by +2 % (R2Ld) to +54 % (W2.5), because momentum from the strong flow outside leaks in through the mouths. Dragon's Mouth transport drops by ~55 %.
- **Unresolved islands.** Fernando de Noronha (32.4°W 3.9°S) and the St Peter and St Paul rocks (29.3°W 0.9°N) are not in the mask: there are no land cells within 3 cells (minimum H 35 m and 1150 m). The filter smooths straight across them.
- **Disconnected regions.** Separate wet regions (lagoons, the Pacific side of Central America) are filtered independently. Their area-mean velocity is conserved and single-cell regions are unchanged.

### 7. Domain edges (task 5)

![Edge test](analysis/boundaries/fig_edge_test.png)

- **Error at the wall.** An artificial no-flux wall changes the filtered surface velocity in the first row by 13–45 % of the RMS of the filtered field. The error decays close to exp(−d/λ) with λ = ℓ/√2, as predicted by the 1-D Neumann Green's function (dotted lines).
- **Decay distances.** Across all tests the relative error drops below 5 % at d = 1.1–2.7 λ and below 1 % at d = 2.5–4.7 λ. For example: W2.5 north wall (U), 5 % at 126 km and 1 % at 211 km; R3Ld south wall at the equator (ℓ = 206 km), 5 % at 195 km and 1 % at 555 km.

**Recommended margin: about 4.7 λ ≈ 3.3 ℓ for a 1 % error, 2.7 λ ≈ 1.9 ℓ for 5 %.** Using ℓ at the real edges:

| Edge | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|
| 30°N, 1 % | 137 km (1.2°) | 182 km (1.6°) | 228 km (2.0°) | 62 km (0.6°) | 94 km (0.8°) |
| 10°S, 1 % | 156 km | 207 km | 259 km | ≥ 186 km | ≥ 280 km |

- **South edge.** The analysis domain starts 555 km from the 10°S edge, so it is safe for the W configs. For R3Ld, where ℓ grows to ~150–200 km at 5°S, the error at 5°S is estimated at ≤ 1–2 %.
- **North edge.** The analysis domain ends at the 30°N edge, where there is no halo. **Treat 28–30°N (W configs) and 29–30°N (R configs) as affected, or exclude them.**
- **Empirical check, zonal sum 60–40°W.** Surface EKE retained for W2.0 is 0.35 at 20–25°N, 0.48 at 25–29°N and 0.56 at 29–30°N. For R2Ld: 0.53 / 0.68 / 0.75.
  - The rise starts ~600 km from the edge, far outside the edge influence distance, so most of it is a real latitude effect: a larger share of large-scale EKE and a smaller ℓ.
  - The extra rise of +0.05 to +0.08 within the last degree fits the edge effect, but this check cannot separate the two cleanly.
  - At 454 m the last-degree rise is +0.04 to +0.06.

![North edge](analysis/boundaries/fig_north_edge_lat.png)

### Problems found, with locations
1. **Coastal speed overshoot.** Monthly-mean speed rises by 6–31 % within ~45 km of every coast, largest for R3Ld and W2.5. Examples: inner Guiana/Amapá shelf (52–50°W, 3–7°N), where speed at the first wet points goes from ~0.1 to ~0.3–0.4 m/s; and the Gulf of Paria (+54 % for W2.5).
2. **Amplified coast-normal velocity.** At the first wet faces the RMS rises by 44–71 %, with correlation to ORIG only 0.80–0.85. Zero flux through land faces is exact.
3. **Lesser Antilles transport.** The arc inflow falls from 20.8 Sv to 9.8–13.9 Sv. St Vincent Passage (61.2°W 13.5°N) and St Lucia Channel (61°W 14.1°N) keep ≤ 55 %. The Guadeloupe and Dominica passages gain spurious westward transport. Dragon's Mouth loses ~55 %.
4. **Mixing across the shelf break.** Depth-mean shelf KE rises by 23–34 % in all configs.
5. **Slope boundary current at ~1000 m.** Damped more than the deep ocean (W2.0 0.66 vs 0.74; R2Ld 0.43 vs 0.54). Location 50–48°W, 3–6°N.
6. **North edge, 30°N.** No halo. Errors > 5 % extend ~1° (W2.5: 1.2°) south of the edge, and > 1 % up to 2° for W2.5.
7. **Not a boundary effect.** Continuity W from the filtered fields is ~100 m/day RMS at the surface everywhere, not only near land; see the divergence section. Using W from the filtered U and V near coasts and in passages needs the same caution as elsewhere.
