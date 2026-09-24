## North Brazil Current, kinetic energy and large-scale retention

Scripts: `analysis/nbc_energy/run_all.sh` runs `nbc_sections.py`, `energy.py`, `energy_analysis.py`, `large_scale.py` and `make_metrics.py` (about 5 min on the login node). All numbers are in `analysis/nbc_energy/metrics.json`.
Data: GLORYS12v1, January 1993, 31 daily fields. ORIG is the unfiltered field. The filters are W1.5, W2.0, W2.5, R2Ld and R3Ld.

### Method

**Sections and sign conventions**

- **Section A (primary): 44°W.**
  - Uses the U-grid column with glamu = −44.04°. It runs from the first wet U point at the Brazilian coast (2.42°S) to 6°N.
  - Normal velocity is u, positive eastward.
  - NBC transport is **T_A = −Σ_{u<0} u·e2u·e3u** over the NBC region, in Sv. **Positive means westward.**
- **Section B (secondary): 5.035°N.**
  - Uses a V-grid row from the coast (52.5°W) to 46°W.
  - Normal velocity is v, positive northward.
  - NBC transport is **T_B = +Σ_{v>0} v·e1v·e3v**. **Positive means northward.**
  - Why a zonal line of V faces: it gives the exact C-grid volume flux. A coast-normal (NE–SW) line would need u and v to be interpolated.
  - Side effect: the line crosses the NW–SE Guiana slope at about 45°. Distances along the line are therefore about 1.4 times the coast-normal distances, and v is only about 0.7 of the along-slope speed. The transport is not affected by this.
- **Offshore limit of the NBC region.** It is fixed for all configurations and all days. It is the first point offshore of the transport maximum where the ORIG 31-day-mean, 0–1000 m integrated normal flow changes sign:
  - Section A: 1.67°N, 454 km from the coast. A fixed 300 km limit would cut through the jet, because the shelf is about 230 km wide. The 300 km variant is still stored in the metrics file.
  - Section B: 47.92°W, 508 km along the line.
- **Vertical integration.** 0–1000 m and 0–300 m, with partial cells clipped at the limit.
- **Core.** Maximum of the NBC-signed velocity within the region above 1000 m. It is given for the monthly mean and as the mean of the daily maxima.
- **Jet width.** Horizontal extent at the core depth where the velocity exceeds 50 % of the core value (edges interpolated). It is also given at the ORIG core depth.

**Energy**

- Region: 70°W–30°W, 5°S–30°N.
- u and v are averaged to T points and weighted with e1t·e2t·e3t on wet points.
- KE_tot = ⟨½|u|²⟩_t. The KE of the mean is KE_mean = ½|⟨u⟩_t|². EKE = KE_tot − KE_mean, i.e. the KE of anomalies from the 31-day mean.
- The residual ("filtered-out") KE is ½⟨|u_o − u_f|²⟩.
- Depth ranges are 0–200 m, 0–1000 m and the full depth.
- As a check, repeating everything for 5°S–25N, which avoids the halo-free northern edge, changes the removed fractions by ≤0.02.

### 1. NBC sections

![NBC section at 44W](analysis/nbc_energy/fig_section_A_44W.png)
![NBC section at 5N](analysis/nbc_energy/fig_section_B_5N.png)
![NBC transport time series](analysis/nbc_energy/fig_nbc_transport_timeseries.png)

In the tables, "Core" is the monthly-mean core and "daily max" is the mean of the daily maxima. Transports are the mean of the daily values over 0–1000 m and 0–300 m. "Net full section" is the net transport from the coast to 6°N (Section A) or to 46°W (Section B), 0–1000 m, NBC-signed. Values in brackets are the change relative to ORIG.

**Section A, 44°W (positive = westward)**

| Config | Core (m/s) | Daily max (m/s) | Core depth | Width at ORIG core depth (km) | T 0–1000 m (Sv) | T 0–300 m (Sv) | Net full section (Sv) |
|---|---|---|---|---|---|---|---|
| ORIG | 0.83 | 0.89 | 92 m | 121 | **36.0** | 26.9 | 23.5 |
| W1.5 | 0.62 (−25 %) | 0.66 (−26 %) | 0.5 m | 151 (+24 %) | 31.9 (−11 %) | 25.0 (−7 %) | 22.7 (−4 %) |
| W2.0 | 0.60 (−28 %) | 0.63 (−30 %) | 0.5 m | 155 (+28 %) | 29.9 (−17 %) | 23.8 (−12 %) | 22.1 (−6 %) |
| W2.5 | 0.58 (−30 %) | 0.60 (−33 %) | 0.5 m | 160 (+32 %) | 27.9 (−23 %) | 22.6 (−16 %) | 21.6 (−8 %) |
| R2Ld | 0.53 (−36 %) | 0.54 (−39 %) | 0.5 m | 179 (+47 %) | 21.9 (−39 %) | 18.6 (−31 %) | 21.8 (−7 %) |
| R3Ld | 0.49 (−41 %) | 0.50 (−45 %) | 0.5 m | 208 (+71 %) | 17.5 (−51 %) | 15.3 (−43 %) | 21.3 (−10 %) |

**Section B, 5°N (positive = northward)**

| Config | Core (m/s) | Daily max (m/s) | Core depth | Width at ORIG core depth (km) | T 0–1000 m (Sv) | T 0–300 m (Sv) | Net full section (Sv) |
|---|---|---|---|---|---|---|---|
| ORIG | 0.81 | 0.97 | 22 m | 224 | **34.7** | 24.7 | 14.9 |
| W1.5 | 0.70 (−13 %) | 0.75 (−22 %) | 8 m | 286 (+28 %) | 31.0 (−11 %) | 22.8 (−8 %) | 16.3 (+9 %) |
| W2.0 | 0.67 (−17 %) | 0.71 (−27 %) | 8 m | 322 (+44 %) | 29.5 (−15 %) | 21.9 (−11 %) | 16.9 (+13 %) |
| W2.5 | 0.63 (−21 %) | 0.67 (−31 %) | 8 m | 360 (+61 %) | 28.0 (−19 %) | 21.0 (−15 %) | 17.3 (+16 %) |
| R2Ld | 0.61 (−24 %) | 0.64 (−34 %) | 8 m | 365 (+63 %) | 26.0 (−25 %) | 19.8 (−20 %) | 18.6 (+25 %) |
| R3Ld | 0.54 (−33 %) | 0.57 (−41 %) | 8 m | 392 (+75 %) | 22.3 (−36 %) | 17.4 (−29 %) | 19.0 (+27 %) |

**Daily variability of T_A.** The standard deviation of daily T_A is 3.7 Sv in ORIG, 3.4–3.7 Sv for W1.5–W2.5 and 2.1 Sv for R3Ld. The 31-day-mean T_B is 30.5 Sv in ORIG. The ORIG transports, 36 Sv at 44°W and 35 Sv at 5°N, lie in the upper range of observed winter NBC transports.

**Interpretation**

- **44°W, jet structure.** In ORIG the NBC at 44°W is a narrow slope-trapped jet: 0.83 m/s at 92 m, about 120 km wide, pressed against the continental slope at about 0.2°S. Just offshore (2–3.5°N, 100–500 m) there is an eastward subsurface current that feeds the retroflection and the equatorial undercurrents.
- **Why the NBC transport drops so much.** The filter works level by level with a no-flux wall at the slope. It therefore spreads the jet offshore, where it partly cancels against that eastward current.
  - The westward-only transport inside the fixed NBC region drops by 11 % (W1.5) to 51 % (R3Ld).
  - The net transport over the whole section changes by only −4 to −10 %.
  - This matches a filter that conserves the area integral per level (checked: the domain integral of u·e1u·e2u on level 10 is unchanged to 10⁻⁹). **Most of the "lost" NBC transport is redistributed across the section, not destroyed.**
  - The same effect at 5°N raises the net full-section transport (+9 to +27 %): smoothing weakens the southward recirculation offshore.
- **Core depth.** In all filtered fields the core moves to the surface. The subsurface slope jet is weakened more than the broad surface westward flow (South Equatorial Current plus NBC), whose maximum then becomes the section maximum. The width at the core depth jumps to 580–700 km, so that width is not meaningful. The table therefore reports the width at the fixed ORIG core depth.
- **Ranking.** At the NBC latitudes Ld is large (220 km at the equator, 164 km at 5°N). The Rossby-radius filters are therefore the most aggressive here: ℓ is 126–188 km at the equator for R2Ld/R3Ld versus 48–79 km for W1.5–W2.5. Accordingly, R3Ld removes half of the westward transport at 44°W.

### 2. Kinetic energy

![Energy removed vs depth](analysis/nbc_energy/fig_energy_profiles.png)
![Surface mean speed maps](analysis/nbc_energy/fig_surface_speed_maps.png)
![Surface EKE maps](analysis/nbc_energy/fig_surface_eke_maps.png)

**ORIG reference values (box mean)**

| Depth range | KE (m²/s²) | EKE / KE |
|---|---|---|
| 0–200 m | 0.026 | 0.24 |
| 0–1000 m | 0.0115 | 0.30 |
| Full depth | 0.0039 | 0.28 |

In January 1993 the region is dominated by the mean flow (NBC, SEC, retroflection). Most eddy energy sits along the NBC–retroflection corridor.

**Removed fractions.** "KE removed" is 1 − KE_f/KE_o. "EKE removed" is 1 − EKE_f/EKE_o. "Mean-flow KE removed" refers to the KE of the 31-day mean. "Residual KE" is ½⟨|u_o − u_f|²⟩ / KE_o.

| Config | KE removed 0–200 m | KE removed 0–1000 m | KE removed full | EKE removed 0–200 m | EKE removed 0–1000 m | EKE removed full | Mean-flow KE removed 0–1000 m | Residual KE 0–1000 m |
|---|---|---|---|---|---|---|---|---|
| W1.5 | 0.29 | 0.39 | 0.43 | 0.46 | 0.53 | 0.55 | 0.34 | 0.11 |
| W2.0 | 0.39 | 0.50 | 0.54 | 0.57 | 0.64 | 0.66 | 0.44 | 0.18 |
| W2.5 | 0.47 | 0.59 | 0.63 | 0.65 | 0.73 | 0.74 | 0.53 | 0.24 |
| R2Ld | 0.51 | 0.55 | 0.57 | 0.63 | 0.66 | 0.68 | 0.50 | 0.21 |
| R3Ld | 0.66 | 0.70 | 0.71 | 0.78 | 0.81 | 0.81 | 0.65 | 0.34 |

**Interpretation**

- **Removed KE is much larger than the residual KE.** For example, W1.5 removes 39 % of KE over 0–1000 m but the residual KE is only 11 %. The reason:
  - KE_o = KE_f + residual + 2⟨u_f·(u_o − u_f)⟩.
  - For the Lorentzian transfer function G = 1/(1 + ½ℓ²K²), the cross term 2G(1−G) is large at intermediate scales. The filter does not project the flow onto large and small scales; it damps every scale partially.
  - "Energy removed" therefore depends on the definition; both are reported.
- **EKE versus mean flow.** EKE is removed more strongly than mean-flow KE, as expected. The mean flow still loses 34–65 % of its KE, mostly in the narrow NBC and on the shelf.
- **Depth dependence.** The removed fraction increases with depth. The deep flow is weak and appears to be dominated by narrow, topographically steered structures, so a larger share of its KE is at small scales. This is an interpretation; it was not checked with a spectrum.
- **Surface EKE maps.** All filters keep the NBC–retroflection EKE corridor and the ring pathway. The W filters largely erase the surface mesoscale EKE north of 15°N. The Rossby filters keep it (Ld is 44–65 km there) but strongly damp the equatorial band. Box-integrated surface EKE ratios are 0.62, 0.52, 0.44, 0.47 and 0.32 for W1.5, W2.0, W2.5, R2Ld and R3Ld.

### 3. Large-scale and "seasonal" signal retention

**Only January 1993 is available, so no seasonal cycle can be computed.** The 31-day mean and 10°-smoothed fields are used as proxies.

![Mean zonal velocity and 10-degree part](analysis/nbc_energy/fig_largescale_maps.png)
![NBC rings](analysis/nbc_energy/fig_rings.png)

**(i) 31-day-mean circulation.** The table gives the centred vector pattern correlation with ORIG. The value in brackets is the relative RMS vector difference.

| Config | Retroflection 52–42°W, 3–10°N, 0.5 m | Retroflection, 92 m | NECC 40–20°W, 4–10°N, 0.5 m | NEC 60–30°W, 10–20°N, 0.5 m |
|---|---|---|---|---|
| W1.5 | 0.993 (0.13) | 0.984 (0.21) | 0.993 (0.17) | 0.955 (0.23) |
| W2.0 | 0.988 (0.19) | 0.975 (0.28) | 0.987 (0.24) | 0.925 (0.30) |
| W2.5 | 0.982 (0.24) | 0.966 (0.34) | 0.979 (0.31) | 0.890 (0.35) |
| R2Ld | 0.976 (0.29) | 0.955 (0.40) | 0.967 (0.37) | 0.960 (0.22) |
| R3Ld | 0.950 (0.43) | 0.922 (0.55) | 0.933 (0.54) | 0.913 (0.32) |

- The retroflection loop and the eastward jet at 6–8°N, 45–38°W are recognisable in every configuration. Their peak speeds are reduced; R3Ld shows the largest loss (see the maps).
- The NECC band east of 40°W is weak in boreal winter and stays well correlated.
- In the NEC band, where Ld is small, the Rossby filters keep more structure than W2.0 and W2.5.

**(ii) Large scales (>~10°).** Both fields were smoothed with a NaN-aware 121×121-point (about 10°) running box on the full domain.

| Config | Relative RMS difference, 0.5 m | Relative RMS difference, 92 m | Pattern correlation |
|---|---|---|---|
| W1.5 | 0.009 | 0.035 | ≥0.979 |
| W2.0 | 0.015 | 0.056 | ≥0.979 |
| W2.5 | 0.022 | 0.080 | ≥0.979 |
| R2Ld | 0.035 | 0.15 | ≥0.979 |
| R3Ld | 0.063 | 0.23 | ≥0.979 |

- Only 0.1–1.2 % of the KE of the removed mean flow survives the 10° smoothing. **The filters leave the >10° circulation essentially untouched.**
- This is partly by construction: the filter conserves area integrals, and box averages are close to area integrals.
- The larger relative difference at 92 m for R3Ld comes from a weak large-scale flow at that depth (box-mean u of −0.011 m/s) and from the redistribution next to the coast described in section 1.

**(iii) NBC rings.**

- **Detection in ORIG.** Rings are detected from daily surface ζ and Okubo–Weiss: OW < −0.2 σ_OW, ζ < 0, equivalent diameter of the OW core > 100 km. The OW core is smaller than the ring's velocity-maximum diameter of about 300–400 km.
- **Rings analysed:**
  - 1 January: a shed ring at 54.9°W, 9.0°N (OW-core diameter 204 km).
  - 15 January: the same ring drifted NW to 57.1°W, 9.5°N.
  - 30 January: a new ring forming at the retroflection, 49.8°W, 6.8°N (318 km).
- **What the filters do.** Mean over the 3 dates, inside the ORIG ring:

| Config | KE ratio | Mean ζ/f ratio | Ring still detected | Detected anticyclones per day (ORIG 3.3) |
|---|---|---|---|---|
| W1.5 | 0.78 | 0.76 | 3 of 3 dates | 2.2 |
| W2.0 | 0.68 | 0.67 | 3 of 3 dates | 2.0 |
| W2.5 | 0.60 | 0.60 | 2 of 3 dates | 1.7 |
| R2Ld | 0.66 | 0.66 | 3 of 3 dates | 2.1 |
| R3Ld | 0.50 | 0.51 | 2 of 3 dates | 1.2 |

- For W2.5 and R3Ld the OW core of the ring is still visible on 15 January but covers less than 30 % of the ORIG core, so it counts as not detected.
- **No filter removes the rings. All weaken them.**
  - W1.5 keeps about ¾ of ring KE.
  - R3Ld halves it, and more so for the retroflection ring at 7°N (0.41), where Ld is 150 km.
  - The drifted ring at 9–10°N is treated similarly by R2Ld and W2.0.
- Small submesoscale eddies and filaments (<100 km) disappear in all filters.

### 4. Summary and assessment

The NBC transport change is the westward-only transport inside the fixed NBC region at 44°W, 0–1000 m. The retroflection correlation is the surface pattern correlation.

| Config | Total KE removed (0–1000 m / full) | EKE removed (0–1000 m / full) | NBC core velocity change, 44°W / 5°N | NBC transport change, 44°W / 5°N | Retroflection pattern correlation |
|---|---|---|---|---|---|
| W1.5 | 0.39 / 0.43 | 0.53 / 0.55 | −25 % / −13 % | −11 % / −11 % | 0.993 |
| W2.0 | 0.50 / 0.54 | 0.64 / 0.66 | −28 % / −17 % | −17 % / −15 % | 0.988 |
| W2.5 | 0.59 / 0.63 | 0.73 / 0.74 | −30 % / −21 % | −23 % / −19 % | 0.982 |
| R2Ld | 0.55 / 0.57 | 0.66 / 0.68 | −36 % / −24 % | −39 % / −25 % | 0.976 |
| R3Ld | 0.70 / 0.71 | 0.81 / 0.81 | −41 % / −33 % | −51 % / −36 % | 0.950 |

**Assessment by configuration**

- **W1.5: mildest.**
  - Removes about half of the EKE but keeps the NBC transport within about 11 % and the rings at about 75 % KE.
  - The mean circulation is almost unchanged (r ≥ 0.98).
  - Still, the narrow subsurface NBC core loses 25 % of its speed.
- **W2.0: intermediate.**
  - Two-thirds of the EKE is removed.
  - NBC transport drops 15–17 %; rings are clearly weakened but intact.
- **W2.5: strong eddy removal (73 % of EKE).**
  - NBC transport drops 19–23 %.
  - Mesoscale variability north of 15°N is nearly erased.
- **R2Ld: scale-aware but aggressive near the equator.**
  - Close to W2.5 in box-integrated KE, but concentrated where it matters here: the NBC loses 39 % of its westward transport at 44°W.
  - It preserves the NEC-band mesoscale field (NEC correlation 0.96).
- **R3Ld: most aggressive.**
  - 80 % of EKE and 70 % of KE are removed.
  - The NBC transport at 44°W is halved and ring KE is halved.
  - The retroflection correlation drops to 0.95 and the 92 m large-scale difference to 23 %.
  - At these latitudes this is no longer a clean eddy/mean separation.

**Caveats**

- The NBC "transport loss" is mostly cross-section redistribution by a per-level filter with no-flux walls at the slope. Net section transports change by only −10 % to +27 %.
- Core-depth and width metrics jump when the core moves to the surface. Use the width at the ORIG core depth.
- EKE here means intra-monthly variability only.
- Near 30°N the filter has no halo. This does not affect the box metrics (≤0.02).
