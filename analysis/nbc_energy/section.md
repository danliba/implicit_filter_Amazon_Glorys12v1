## North Brazil Current, kinetic energy and large-scale retention (filter v3)

**What was analysed**
- **Filter version:** v3 with the e3 fix. U and V are filtered jointly with the vector div–rot operator: zero normal flow at land, free slip, and ℓ_F² e3f/(e1f e2f) in the curl term. W is the rigid-lid product filtered as a scalar.
- **Data:** GLORYS12v1, January 1993, 31 daily fields. ORIG means unfiltered.
- **Reproducing:** `analysis/nbc_energy/run_all.sh` rebuilds everything in about 6 min on the login node. The scripts are `nbc_sections.py`, `energy.py`, `energy_analysis.py`, `large_scale.py`, `w_section.py`, `v1_v3_compare.py` and `make_metrics.py`. All numbers are in `analysis/nbc_energy/metrics.json`.
- **v1 results:** the earlier component-wise results are kept in `analysis/nbc_energy/v1/`.

**Excluded pre-fix v3 run.** A first v3 run had a depth-dependent bug: ℓ_F was effectively divided by √e3, so U and V were almost unfiltered below about 20 m. It was found in this analysis (rms(u_f−u_o)/rms(u_o) was 0.11 at 266 m, against 0.49 in v1) and fixed by the coordinator. **No numbers here come from that run.** After the fix, W2.5 in the box gives 0.28, 0.35, 0.49 and 0.54 at 0.5, 56, 266 and 763 m, against 0.29, 0.36, 0.49 and 0.55 for v1. In the open ocean v3 and v1 are therefore the same filter, as designed.

### Method

**Sections and signs**

| | Section A (primary) | Section B (secondary) |
|---|---|---|
| Line | U-grid column at glamu = −44.04°, from the first wet U point at the coast (2.42°S) to 6°N | V-grid row at 5.035°N, from the coast (52.5°W) to 46°W |
| Normal velocity | u, positive eastward | v, positive northward |
| NBC transport | **T_A = −Σ_{u<0} u·e2u·e3u** in Sv, **positive westward** | **T_B = +Σ_{v>0} v·e1v·e3v**, **positive northward** |

- **Why a zonal line at 5°N:** a row of V faces gives the exact C-grid volume flux. A coast-normal line would need interpolation. The line crosses the NW–SE slope at about 45°, so along-line distances are about 1.4 times the coast-normal distances and v is about 0.7 of the along-slope speed. Transport is not affected.
- **NBC region:** from the coast to a fixed offshore limit. The limit is the first sign change, offshore of the transport maximum, of the ORIG 31-day-mean, 0–1000 m integrated normal flow. That is 1.67°N (454 km) at 44°W and 47.92°W (508 km along the line) at 5°N. A fixed 300 km limit would cut through the 44°W jet, because the shelf there is about 230 km wide; that variant is still stored in the metrics.
- **Depth integrals:** 0–1000 m and 0–300 m, with partial cells clipped.
- **Core speed:** maximum NBC-signed velocity in the NBC region above 1000 m. It is given for the monthly mean, as the mean of the daily maxima, and as a robust version: the maximum of a 3-point along-section running mean (see the artefact note under section 1).
- **Jet width:** extent where the velocity exceeds 50 % of the maximum, measured at the fixed ORIG core depth.

**Energy**
- Region 70°W–30°W, 5°S–30°N. u and v are averaged to T points and weighted with e1t·e2t·e3t on wet cells.
- EKE is the KE of the daily anomalies from the 31-day mean.
- The residual ("filtered-out") KE is ½⟨|u_o − u_f|²⟩.
- Repeating the calculation for 5°S–25°N (away from the halo-free 30°N edge) changes the removed fractions by at most 0.02.

### 1. NBC sections

![NBC section at 44W](analysis/nbc_energy/fig_section_A_44W.png)
![NBC section at 5N](analysis/nbc_energy/fig_section_B_5N.png)
![NBC transport time series](analysis/nbc_energy/fig_nbc_transport_timeseries.png)

In both tables, T is the monthly mean of daily values over 0–1000 m (or 0–300 m). The net column is the net transport over the whole line, 0–1000 m, NBC-signed. Percentages are changes relative to ORIG.

**44°W (positive = westward)**

| Config | Core, monthly mean (m/s) | Core, robust 3-pt | Mean of daily max | Core depth | Width at ORIG core depth (km) | T 0–1000 m (Sv) | T 0–300 m (Sv) | Net, coast–6°N (Sv) |
|---|---|---|---|---|---|---|---|---|
| ORIG | 0.83 | 0.80 | 0.89 | 92 m | 121 | **36.0** | 26.9 | 23.5 |
| W1.5 | 0.71 (−15 %) | −19 % | 0.74 (−17 %) | 56 m | 153 | 32.0 (−11 %) | 25.0 (−7 %) | 22.9 (−3 %) |
| W2.0 | 0.67 (−20 %) | −23 % | 0.69 (−22 %) | 56 m | 157 | 30.3 (−16 %) | 24.0 (−11 %) | 22.7 (−4 %) |
| W2.5 | 0.63 (−25 %) | −28 % | 0.65 (−27 %) | 56 m | 161 | 28.5 (−21 %) | 22.9 (−15 %) | 22.5 (−5 %) |
| R2Ld | 0.53 (−36 %) | −34 % | 0.56 (−38 %) | 0.5 m | 180 | 22.9 (−37 %) | 19.1 (−29 %) | 22.5 (−4 %) |
| R3Ld | 0.49 (−42 %) | −40 % | 0.49 (−45 %) | 0.5 m | 210 | 18.3 (−49 %) | 15.7 (−42 %) | 22.0 (−7 %) |

**5°N (positive = northward)**

| Config | Core, monthly mean (m/s) | Core, robust 3-pt | Mean of daily max | Width at ORIG core depth (km) | T 0–1000 m (Sv) | T 0–300 m (Sv) | Net, coast–46°W (Sv) |
|---|---|---|---|---|---|---|---|
| ORIG | 0.81 | 0.80 | 0.97 | 224 | **34.7** | 24.7 | 14.9 |
| W1.5 | 0.72 (−10 %)\* | −12 % | 0.78 (−19 %) | 276 | 31.1 (−11 %) | 22.6 (−8 %) | 16.2 (+8 %) |
| W2.0 | 0.73 (−9 %)\* | −15 % | 0.76 (−22 %) | 296 | 29.5 (−15 %) | 21.6 (−12 %) | 16.8 (+12 %) |
| W2.5 | 0.72 (−11 %)\* | −18 % | 0.73 (−25 %) | 314 | 28.0 (−19 %) | 20.6 (−16 %) | 17.3 (+16 %) |
| R2Ld | 0.69 (−15 %)\* | −21 % | 0.70 (−28 %) | 315 | 26.0 (−25 %) | 19.0 (−23 %) | 18.8 (+26 %) |
| R3Ld | 0.57 (−29 %)\* | −29 % | 0.60 (−38 %) | 370 | 22.6 (−35 %) | 16.2 (−35 %) | 19.3 (+29 %) |

**Daily variability.** The standard deviation of daily T_A is 3.7 Sv in ORIG, 3.4, 3.2 and 3.0 Sv for W1.5–W2.5, and 2.4 and 1.9 Sv for R2Ld and R3Ld. For T_B it is 7.1 Sv in ORIG and between 6.9 and 4.6 Sv in the filters.

**Artefact at 5°N (marked \*).** In every v3 config, the single-cell maximum at 5°N sits at one V point (50.83°W, 66–78 m). That point is wedged against a one-cell bathymetric notch and has land on its western side. There v3 gives 0.73 m/s (W2.0), while ORIG has 0.26 m/s and the neighbouring points about 0.5 m/s. It is a local v3 wall artefact, plausibly because no curl constraint acts at a V point enclosed by land, so only the divergence term shapes it.
- The transport through the section is not affected: 66 m cells there are about 10 km × 10 m, so the anomaly is well below 0.1 Sv.
- The table therefore also gives the robust 3-point core, which is 0.68 m/s at 8 m for W2.0.

**Interpretation**
- **ORIG jet.** At 44°W the NBC is a narrow slope-trapped jet, 0.83 m/s at 92 m and about 120 km wide. Offshore of it lies an eastward subsurface current (2–3.5°N, 100–500 m) that feeds the retroflection and the equatorial undercurrents.
- **Where the "lost" transport goes.** The westward-only NBC transport in the fixed region drops by 11 % (W1.5) to 49 % (R3Ld). The net transport over the whole section changes by only −3 % to −7 %. The filter conserves area integrals level by level; the jet is widened across the section and cancels partly against the adjacent eastward flow.
- **Re-evaluated for v3.** This redistribution is a genuine property of any low-pass filter applied to a narrow jet next to an opposing current. It is **not** the v1 wall artefact: v1 and v3 give nearly the same NBC transport, see section 5. At 5°N the same mechanism *raises* the net transport (+8 % to +29 %), because the offshore southward recirculation is smoothed away.
- **Core depth.** For R2Ld and R3Ld the 44°W core moves to the surface: the subsurface slope jet is weakened more than the broad surface SEC/NBC flow. Widths are therefore given at the ORIG core depth.
- **Which filters are strongest here.** ℓ is 126–188 km at the equator for R2Ld/R3Ld, against 48–79 km for W1.5–W2.5, so the Rossby-radius filters are the most aggressive at NBC latitudes. R3Ld halves the westward transport at 44°W.

### 2. Kinetic energy

![Energy removed vs depth](analysis/nbc_energy/fig_energy_profiles.png)
![Surface mean speed maps](analysis/nbc_energy/fig_surface_speed_maps.png)
![Surface EKE maps](analysis/nbc_energy/fig_surface_eke_maps.png)

**ORIG box means**

| Depth range | KE (m²/s²) | EKE / KE |
|---|---|---|
| 0–200 m | 0.026 | 0.24 |
| 0–1000 m | 0.0115 | 0.30 |
| Full depth | 0.0039 | 0.28 |

The region is dominated by the mean flow in January 1993.

**Fractions removed by each filter.** KE removed = 1 − KE_f/KE_o; EKE removed = 1 − EKE_f/EKE_o. "Mean-flow KE" is the KE of the 31-day mean. "Residual KE" is ½⟨|u_o − u_f|²⟩ / KE_o.

| Config | KE removed 0–200 m | KE removed 0–1000 m | KE removed full | EKE removed 0–200 m | EKE removed 0–1000 m | EKE removed full | Mean-flow KE removed 0–1000 m | Residual KE 0–1000 m |
|---|---|---|---|---|---|---|---|---|
| W1.5 | 0.28 | 0.38 | 0.42 | 0.46 | 0.53 | 0.54 | 0.32 | 0.11 |
| W2.0 | 0.37 | 0.49 | 0.53 | 0.57 | 0.64 | 0.66 | 0.43 | 0.17 |
| W2.5 | 0.45 | 0.57 | 0.61 | 0.65 | 0.73 | 0.73 | 0.51 | 0.24 |
| R2Ld | 0.49 | 0.53 | 0.55 | 0.64 | 0.66 | 0.67 | 0.48 | 0.20 |
| R3Ld | 0.64 | 0.68 | 0.69 | 0.78 | 0.81 | 0.81 | 0.63 | 0.33 |

**Interpretation**
- **Two measures of "energy removed".** 1 − KE_f/KE_o is 2–4 times larger than the residual fraction. The reason is the cross term: KE_o = KE_f + residual + 2⟨u_f·(u_o − u_f)⟩. The Lorentzian transfer function G = 1/(1 + ½ℓ²K²) damps every scale partly; it does not project the flow onto large and small scales. Both measures are reported.
- **Depth dependence.** The removed fraction increases with depth. The weak deep flow is more dominated by narrow, topographically steered structures; this is an interpretation and was not checked with a spectrum.
- **Surface EKE.** Box surface EKE ratios are 0.62, 0.52, 0.45, 0.47 and 0.32 (W1.5 to R3Ld). All filters keep the NBC–retroflection EKE corridor. The W filters largely erase the surface mesoscale north of 15°N. The Rossby filters keep it there (Ld is 44–65 km) but damp the equatorial band strongly.

### 3. Large-scale and "seasonal" retention

**Only January 1993 is available, so no seasonal cycle can be computed.** The 31-day mean and 10°-smoothed fields serve as proxies.

![Mean zonal velocity and 10-degree part](analysis/nbc_energy/fig_largescale_maps.png)
![NBC rings](analysis/nbc_energy/fig_rings.png)

**(i) 31-day-mean circulation.** Centred vector pattern correlation with ORIG; the value in brackets is the relative RMS difference.

| Config | Retroflection 52–42°W, 3–10°N, 0.5 m | Retroflection, 92 m | NECC 40–20°W, 4–10°N, 0.5 m | NEC 60–30°W, 10–20°N, 0.5 m |
|---|---|---|---|---|
| W1.5 | 0.993 (0.13) | 0.984 (0.21) | 0.993 (0.17) | 0.955 (0.23) |
| W2.0 | 0.988 (0.19) | 0.975 (0.27) | 0.987 (0.24) | 0.924 (0.30) |
| W2.5 | 0.981 (0.24) | 0.966 (0.33) | 0.979 (0.31) | 0.888 (0.35) |
| R2Ld | 0.975 (0.29) | 0.957 (0.39) | 0.968 (0.37) | 0.960 (0.22) |
| R3Ld | 0.945 (0.42) | 0.926 (0.54) | 0.935 (0.53) | 0.913 (0.32) |

- The retroflection loop and the eastward jet at 6–8°N stay recognisable in all configs, with peak speeds reduced.
- In the NEC band the Rossby filters keep more structure than W2.0 and W2.5.

**(ii) Large scales (>~10°).** A NaN-aware 121×121-point box (about 10°) was applied to the full-domain monthly means of ORIG and each filter.

| Config | Relative RMS difference, 0.5 m | Relative RMS difference, 92 m |
|---|---|---|
| W1.5 | 0.010 | 0.035 |
| W2.0 | 0.015 | 0.057 |
| W2.5 | 0.022 | 0.082 |
| R2Ld | 0.038 | 0.15 |
| R3Ld | 0.070 | 0.23 |

- Pattern correlations are all ≥0.979. Only 0.1–1.5 % of the KE of the removed mean flow survives the 10° smoothing. The >10° circulation is essentially untouched; this is partly by construction, because the filter conserves area integrals.
- The larger 92 m difference for R2Ld/R3Ld reflects a weak large-scale flow at that depth (box-mean u = −0.011 m/s) plus the across-section redistribution next to the coast.

**(iii) NBC rings.** Rings were detected in ORIG from daily surface ζ and Okubo–Weiss (OW < −0.2 σ, ζ < 0, OW-core diameter > 100 km). Three cases were followed:
- 1 January: a shed ring at 54.9°W, 9.0°N (204 km core).
- 15 January: the same ring after drifting NW to 57.1°W, 9.5°N.
- 30 January: a ring forming at the retroflection, 49.8°W, 6.8°N (318 km).

| Config | Ring KE retained (1 Jan / 15 Jan / 30 Jan) | Detected anticyclones per day (ORIG: 3.3) |
|---|---|---|
| W1.5 | 0.76 / 0.77 / 0.80 | 2.2 |
| W2.0 | 0.66 / 0.68 / 0.70 | 2.0 |
| W2.5 | 0.57 / 0.60 / 0.62 | 1.7 |
| R2Ld | 0.67 / 0.70 / 0.58 | 2.1 |
| R3Ld | 0.51 / 0.56 / 0.41 | 1.2 |

- The ring is still detected on all dates, except W2.5 and R3Ld on 15 January. There the OW core remains visible but covers less than 30 % of the ORIG core.
- **No filter removes the rings; all of them weaken them.** Sub-100 km eddies and filaments disappear in every config.

### 4. Vertical velocity at 44°W

![W section 44W](analysis/nbc_energy/fig_w_section_44W.png)

| Field | rms of 31-day-mean w (m/day) | rms of daily w (m/day) | Correlation of mean pattern with ORIG |
|---|---|---|---|
| ORIG (rigid-lid) | 6.4 | 11.7 | 1 |
| W2.0 | 1.7 | 2.9 | 0.48 |
| R2Ld | 1.0 | 1.6 | 0.40 |

- The ORIG W is dominated by vertically alternating grid-scale columns over the continental slope (±20–70 m/day between −0.5°N and 1°N), i.e. slope-following up- and downwelling at the model's step topography. It also shows banded upwelling at 2–3°N, 200–500 m, below the eastward undercurrent.
- The scalar filter removes the slope noise and keeps the smooth large-scale pattern: weak upwelling of up to 9 m/day at 200–500 m, weak downwelling below 600 m offshore. Amplitude drops by a factor of 4 (W2.0) to 7 (R2Ld).
- W is filtered as a scalar rather than derived from the filtered U,V. Any residual inconsistency with filtered continuity is diagnosed in the coordinator's continuity residual R, not here.

### 5. v1 (component-wise) vs v3 (vector div–rot)

![v1 vs v3 profiles](analysis/nbc_energy/fig_v1_v3_profiles.png)

Near-coast speed is the 31-day-mean surface speed, averaged over wet T points within 50 km of land (70°W–30°W, 5°S–30°N); the ORIG value is 0.258 m/s.

| Metric | W2.0 v1 | W2.0 v3 | R2Ld v1 | R2Ld v3 |
|---|---|---|---|---|
| 44°W core, single cell | −28 % | −20 % | −36 % | −36 % |
| 44°W core, robust 3-pt | −25 % | −23 % | −33 % | −34 % |
| 5°N core, robust 3-pt | −16 % | −15 % | −23 % | −21 % |
| 44°W T 0–1000 m | 29.9 Sv (−17 %) | 30.3 Sv (−16 %) | 21.9 Sv (−39 %) | 22.9 Sv (−37 %) |
| 44°W net, whole section | −6 % | −4 % | −7 % | −4 % |
| 5°N T 0–1000 m | −15 % | −15 % | −25 % | −25 % |
| KE removed, 0–200 m / 0–1000 m / full | 0.39 / 0.50 / 0.54 | 0.37 / 0.49 / 0.53 | 0.51 / 0.55 / 0.57 | 0.49 / 0.53 / 0.55 |
| EKE removed, 0–200 m / 0–1000 m / full | 0.57 / 0.64 / 0.66 | 0.57 / 0.64 / 0.66 | 0.63 / 0.66 / 0.68 | 0.64 / 0.66 / 0.67 |
| Near-coast surface speed | 0.275 (×1.07) | 0.282 (×1.09) | 0.291 (×1.13) | 0.295 (×1.14) |

**What the switch to v3 changed**
- **Open ocean and energetics: practically identical.** EKE removed differs by ≤0.01. KE removed is 0.01–0.02 lower in v3, because less mean-flow KE is lost along coasts.
- **Along coasts, v3 keeps slightly more of the slope jets.**
  - The single-cell 44°W NBC core loses 20 % in v3 instead of 28 % for W2.0. Most of that gain sits in the wall-adjacent cell (free slip, no wall-induced convergence); the robust 3-point core differs by only 1–2 points.
  - NBC transports are 1–3 % of ORIG higher in v3, and the net section loss shrinks from −6/−7 % to −4 %.
- **Near-coast surface speed.** Both versions raise the 0–50 km mean speed above ORIG, because smoothing spreads the fast slope jets onto slow shelves. v3 raises it 2–3 points more.
- **Shelf structure.** v3 keeps closed shelf pockets separate from the slope flow; v1 diffused across them. For example, at 5°N, 51.5°W, 66 m: ORIG −0.03 m/s, v1 +0.18, v3 +0.01.
- **Transport redistribution caveat, revisited.** With v3 the across-section redistribution of NBC transport remains: 16 % (W2.0) to 49 % (R3Ld) of the westward NBC transport is moved outside the fixed NBC region. It is therefore a property of the scale separation, not of the v1 wall treatment.
- **New v3 artefact:** isolated single-cell velocity maxima at V/U points trapped in one-cell bathymetric notches (5°N, 50.83°W, 66–78 m: 0.73 against 0.26 m/s in ORIG). They are harmless for transports and energy but bias single-cell "maximum speed" metrics.

### 6. Summary and assessment

The NBC core column uses the robust 3-point core. Transport is westward-only in the fixed NBC region at 44°W and northward-only at 5°N, both 0–1000 m. Retroflection correlation is the surface pattern correlation.

| Config | Total KE removed (0–1000 m / full) | EKE removed (0–1000 m / full) | NBC core change, 44°W / 5°N | NBC transport change, 44°W / 5°N | Retroflection correlation | Ring KE retained |
|---|---|---|---|---|---|---|
| W1.5 | 0.38 / 0.42 | 0.53 / 0.54 | −19 % / −12 % | −11 % / −11 % | 0.993 | 0.78 |
| W2.0 | 0.49 / 0.53 | 0.64 / 0.66 | −23 % / −15 % | −16 % / −15 % | 0.988 | 0.68 |
| W2.5 | 0.57 / 0.61 | 0.73 / 0.73 | −28 % / −18 % | −21 % / −19 % | 0.981 | 0.60 |
| R2Ld | 0.53 / 0.55 | 0.66 / 0.67 | −34 % / −21 % | −37 % / −25 % | 0.975 | 0.65 |
| R3Ld | 0.68 / 0.69 | 0.81 / 0.81 | −40 % / −29 % | −49 % / −35 % | 0.945 | 0.50 |

**Assessment by configuration**
- **W1.5, mildest.** Half of the EKE is removed. NBC transport stays within 11 %, the mean circulation is almost unchanged (r ≥ 0.98), and rings keep about 78 % of their KE. The narrow subsurface NBC core still loses about 19 %.
- **W2.0, balanced.** Two-thirds of the EKE is removed. NBC transport drops by 15–16 %; rings are clearly weakened but intact.
- **W2.5, strong eddy removal (73 % of EKE).** NBC transport drops by about 20 %, and the mesoscale north of 15°N is nearly erased.
- **R2Ld, scale-aware but aggressive at the equator.** Box energetics are close to W2.5, but the NBC loses 37 % of its westward transport at 44°W. It preserves the NEC-band mesoscale best (r = 0.96).
- **R3Ld, most aggressive.** 81 % of EKE and 68 % of KE are removed. NBC transport at 44°W and ring KE are roughly halved, the retroflection correlation falls to 0.945, and the 92 m large-scale difference reaches 23 %. At these latitudes this is no longer a clean eddy/mean separation.

**Caveats**
- EKE here means intra-monthly variability only.
- Near 30°N the filter has no halo; the effect on the box metrics is ≤0.02.
- Single-cell core metrics are affected by the v3 notch artefact; use the robust 3-point core.
