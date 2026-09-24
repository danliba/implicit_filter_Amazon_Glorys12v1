## 6. Comparison of the five configurations

Sources: `analysis/nbc_energy/metrics.json`, `analysis/spectra_divergence/metrics.json`, `analysis/boundaries/metrics.json`.

**Definitions**
- Region: 70°W–30°W, 5°S–30°N.
- KE / EKE removed: 1 − filtered/original, volume-weighted. EKE is the KE of daily anomalies from the 31-day mean.
- NBC core: robust 3-point core of the 31-day mean section.
- NBC transport: 0–1000 m, westward at 44°W, northward at 5°N.
- R: RMS continuity residual ∂w̄/∂z + ∇h·ū. "Open" = >40 cells from grid edges and >30 cells from land and bottom steps; "coast" = ≤3 cells from land.
- For reference, the filtered RMS divergence at 110 m is 2.2–4.1×10⁻⁷ s⁻¹.

| | W1.5 | W2.0 | W2.5 | R2Ld | R3Ld |
|---|---|---|---|---|---|
| ℓ at 0° / 15°N / 30°N [km] | 48 / 46 / 41 | 64 / 61 / 55 | 79 / 77 / 69 | 136 / 41 / 19 | 203 / 62 / 28 |
| Total KE removed, 0–1000 m / full depth | 0.38 / 0.42 | 0.49 / 0.53 | 0.57 / 0.61 | 0.53 / 0.55 | 0.68 / 0.69 |
| EKE removed, 0–1000 m / full depth | 0.53 / 0.54 | 0.64 / 0.66 | 0.73 / 0.73 | 0.66 / 0.67 | 0.81 / 0.81 |
| Mean-flow KE removed, 0–1000 m | 0.32 | 0.43 | 0.51 | 0.48 | 0.63 |
| NBC core speed change, 44°W / 5°N | −19 % / −12 % | −23 % / −15 % | −28 % / −18 % | −34 % / −21 % | −40 % / −29 % |
| NBC transport change, 44°W / 5°N | −11 % / −11 % | −16 % / −15 % | −21 % / −19 % | −37 % / −25 % | −49 % / −35 % |
| Net transport change across the whole 44°W section | −3 % | −4 % | −5 % | −4 % | −7 % |
| Retroflection mean-pattern correlation (surface / 92 m) | 0.993 / 0.984 | 0.988 / 0.975 | 0.981 / 0.966 | 0.975 / 0.957 | 0.945 / 0.926 |
| >10° circulation, rel. RMS difference (surface / 92 m) | 1.0 % / 3.5 % | 1.5 % / 5.7 % | 2.2 % / 8.2 % | 3.8 % / 15 % | 7.0 % / 23 % |
| NBC-ring KE retained (3 cases) | 0.76–0.80 | 0.66–0.70 | 0.57–0.62 | 0.58–0.70 | 0.41–0.56 |
| Open-ocean RMS R at 110 m [s⁻¹] (rigid-lid original: 3.2e-11) | 2.9e-11 | 3.7e-11 | 9.1e-11 | 4.4e-10 | 1.5e-9 |
| Coast RMS R at 110 m [s⁻¹] (original: 9.5e-10) | 9.1e-7 | 8.1e-7 | 7.2e-7 | 9.0e-7 | 7.9e-7 |
| Mean surface speed within 45 km of coast (max ratio to original) | 1.13 | 1.18 | 1.22 | 1.24 | 1.29 |
| Lesser Antilles arc transport (original −20.8 Sv) | −20.6 | −20.6 | −20.6 | −20.6 | −20.6 |
| RMS W at 100 / 500 / 1000 m, ratio to original | 0.38 / 0.38 / 0.46 | 0.33 / 0.32 / 0.39 | 0.28 / 0.27 / 0.34 | 0.28 / 0.28 / 0.36 | 0.21 / 0.21 / 0.28 |
| Wall-clock on one A100 (U,V run + W run) | 21 + 13 min | 22 + 13 min | 24 + 13 min | 30 + 14 min | 39 + 16 min |

**Qualitative assessment**
- **W1.5: mild.** Half of the eddy variability remains, and rings keep ~80 % of their KE. The mean currents are essentially intact. It suits studies needing a light de-noising more than a mesoscale separation.
- **W2.0: balanced.** Two thirds of EKE is removed, and sub-100 km eddies and filaments are gone. The NBC keeps 84–85 % of its transport and the retroflection/NECC pattern is unchanged (r = 0.99). Rings are weakened but not removed.
- **W2.5: strong.** Nearly three quarters of EKE is removed and the mesoscale north of 15°N is largely gone. The narrow NBC loses ~20 % of its transport and 28 % of its core speed.
- **R2Ld: scale-aware but mismatched to this region.**
  - Box energetics are similar to W2.5, but the NBC loses 37 % of its westward transport, because ℓ ≈ 136 km at the equator.
  - North of 15°N it keeps the eddy field (ℓ < 40 km), i.e. the opposite of a mesoscale-removal filter there.
  - Its residual is 10× the window configs in open ocean, from the rapidly varying ℓ, though still < 1 % of the divergence.
- **R3Ld: too aggressive at the equator.** It halves NBC transport and ring KE and degrades the large-scale 92 m circulation by 23 %. It is not a clean mean/eddy separation.

**Common to all configurations**
- There is no flow through land.
- Passage transports through the Lesser Antilles are conserved to < 1 %.
- The coastal band of ~45 km speeds up by 13–29 %, and the depth-mean shelf KE rises by 22–29 %, because a free-slip horizontal smoother spreads slope-jet momentum onto the shelf.
- W keeps only 1–6 % of its daily variance near coasts, where the original W is grid-scale noise. There are no coastal artefacts in W.

## 7. Recommendation for the full-dataset processing
**Use W2.0**: ℓ = 2°·(π/180)·R·cos φ / 3.5, box-equivalent 222 km at the equator. The window configurations are equally cheap, so the choice is about physics.
1. **Mesoscale removal vs mean-flow preservation.** W2.0 is where the trade-off turns. From W1.5 to W2.0, EKE removal rises by 11 points (0.53 → 0.64) for a 5-point transport loss (−11 → −16 %). From W2.0 to W2.5 it rises by 9 points for another 5-point loss, and the >10° circulation difference at 92 m grows from 5.7 % to 8.2 %.
2. **Lagrangian tracking.**
   - The NBC–retroflection–NECC pathway, which carries Amazon water, is kept: r = 0.99, net section transport −4 %.
   - Filaments and sub-100 km eddies, which dominate dispersion noise, are removed.
   - Filtered U, V, W are mutually consistent in the open ocean to the level of the original data.
3. **A uniform, latitude-aware scale** (the grid spacing ∝ cos φ) avoids the Rossby filters' problem in this domain. Those filter hardest at the equator, where the NBC is, and least in the subtropics, where the eddies are.

**Caveats and practical rules for tracking**
- **No filter here removes NBC rings without damaging the NBC.** Rings (300–450 km diameter) have wavelengths comparable to the NBC's cross-shore scale, and the Lorentzian response G = 1/(1 + ½ℓ²K²) of the n = 1 implicit filter is gentle. If ring removal is essential, test the n = 2 filter (steeper response, not tested here), or a larger window combined with an explicit NBC-preserving treatment.
- **Seasonal cycle.** Only January 1993 was processed, so monsoon-driven NBC intensification and NECC formation could not be evaluated. The >10° circulation is preserved to 1–6 %, which suggests the seasonal signal will survive W2.0. Verify with a few months spanning boreal spring and autumn before the full run.
- **Domain edges.**
  - Keep particles ≥ 3.5ℓ (≈ 1.7° for W2.0) away from the 30°N cut, where filtered velocities are affected by the artificial wall.
  - Ignore the last row and column of the grid.
  - Near the 95°W and 10°S cuts, U,V/W consistency degrades within ~450 km; both are far outside the analysis domain. A larger halo in the next extraction removes this issue.
- **Coasts and shelf.**
  - Expect 13–29 % higher mean speeds within ~45 km of the coast, with more cross-shelf-break momentum.
  - Continuity residuals of ~10⁻⁶ s⁻¹ appear at bottom steps next to slopes, which is inherent to level-by-level horizontal filtering.
  - For beaching and shelf retention studies, compare against unfiltered runs.
- **W convention.** The filtered W is the rigid-lid product: zero at the surface, as in the existing Parcels setup. It does not contain ∂η/∂t.
