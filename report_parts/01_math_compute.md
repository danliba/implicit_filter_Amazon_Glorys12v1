## 1. Mathematical summary

### 1.1 Filter equation, transfer function and box-filter equivalence
The coarse-grained field solves the low-pass implicit filter of Danilov et al. (2023) and Nowak et al. (2025) with γ = ½ and n = 1:

$$\left(1 + \gamma\,\mathcal{L}_\ell\right)\overline{\phi} = \phi, \qquad G(K) = \frac{1}{1 + \tfrac12 \ell^2 K^2}, \qquad \lambda = 2\pi\ell,$$

where $\mathcal{L}_\ell$ is a (negative) Laplacian scaled by ℓ², and $k_\ell = 1/\ell$ is the filter wavenumber.

**Box-filter factor 3.5.** A box filter of width $L$ has transfer $\operatorname{sinc}(KL/2)\approx 1-K^2L^2/24$. Matching the second moment to $G\approx1-\tfrac12\ell^2K^2$ gives $\ell = L/\sqrt{12} = L/3.46$. This is Nowak et al.'s $\ell_{box}/\ell\approx 3.5$ for γ = ½.

**Package parameter convention (checked in the source).** `implicit_filter` solves $(I + 2(-L/k^2))\phi$, i.e. γ = 2 with $k$. To obtain the γ = ½ filter at scale ℓ, the argument must be **k = 2/ℓ**. The task sheet's suggestion `k = 1/ℓ` would silently double ℓ. All code here uses γ and ℓ explicitly.

### 1.2 Velocity: vector (div–rot) form on the C-grid
**Why U,V are coupled.** Following the vector addendum, the spherical metric cross-terms of the vector Laplacian scale as $\ell^2\sin\varphi/(R^2\cos^2\varphi)$. That is 0 at the equator, ~0.01 % at 10°N, ~0.04 % at 20°N and ~0.07 % at 30°N, so they are negligible and none are included. Nevertheless, U and V are **not** filtered as independent scalars in the final product, for a different, discrete reason:
- A first production run (v1) filtered U on the U-grid and V on the V-grid, each with its own no-flux (Neumann) walls.
- That smooths each component toward offshore values right at the coast, while the flow through the land face stays zero.
- The result is a one-cell convergence line along **every** coastline and bottom step at **every** level: RMS divergence within 3 cells of land is 1.4–1.7× the original, while offshore it falls to 0.14–0.25×.
- Integrated vertically this gave $|w|\sim10^{-3}$ m/s (section 6).

The final filter (v3) therefore uses the C-grid vector Laplacian in div–rot form. For constant ℓ in open water it is identical to the component-wise Laplacian (metric terms dropped, as the addendum justifies):

$$\mathcal{L}_\ell\mathbf u = -\nabla\!\left(\ell_T^2\,\nabla\!\cdot\mathbf u\right) + \nabla\times\!\left(\ell_F^2\,\zeta\right),\qquad
\nabla\!\cdot\mathbf u = \frac{\delta_i(e_{2u}e_{3u}u)+\delta_j(e_{1v}e_{3v}v)}{e_{1t}e_{2t}e_{3t}},\quad
\zeta = \frac{\delta_i(e_{2v}v)-\delta_j(e_{1u}u)}{e_{1f}e_{2f}} .$$

This is NEMO's discrete divergence (T points, partial steps) and vorticity (F points). The discrete system is symmetric positive definite:

$$\Big(W + \gamma\big[F^{T}\Lambda_T F + C^{T}\Lambda_F C\big]\Big)\overline{\mathbf u} = W\mathbf u,$$

- $W=\mathrm{diag}(e_{1u}e_{2u}e_{3u},\,e_{1v}e_{2v}e_{3v})$
- $F$ is the T-cell flux-difference matrix, with $\Lambda_T=\ell_T^2/(e_{1t}e_{2t}e_{3t})$
- $C$ is the F-point circulation matrix, with $\Lambda_F=\ell_F^2 e_{3f}/(e_{1f}e_{2f})$ and $e_{3f}$ the minimum of the surrounding $e_{3u},e_{3v}$

**Boundary conditions.** Zero normal flow through land faces (they are not unknowns). Free slip: ζ = 0 at F points touching land. The open edges of the regional grid behave as walls.

**Exact discrete property.** $F W^{-1} C^T = 0$ (div curl = 0), therefore at every level

$$\nabla\!\cdot\overline{\mathbf u} = \left(1+\gamma(-\Delta_T\,\ell_T^2)\right)^{-1}\nabla\!\cdot\mathbf u ,$$

i.e. the filtered divergence is the no-flux scalar filter of the original divergence. There is no wall artefact. This was verified to a relative residual of 5×10⁻⁵.

**Scales.** With variable ℓ, $\ell_T$ is used at T points and $\ell_F$ at F points. The operator conserves the volume-weighted integral of each component.

### 1.3 W: scalar filter at the same scale (correction to Section 1.4 of the task sheet)
Following the W correction, W is **not** recomputed from the divergence. It is the rigid-lid W of `data/Fix_W.ipynb` (`data/variables_c/UVW/W_1993-01fc.nc`), filtered level by level as a scalar on the W/T points with the same $\ell_T$. There is no metric coupling between W and (U,V), so this is the exact vector-Laplacian treatment of W.

To make $\partial_z\overline w$ and $\nabla_h\cdot\overline{\mathbf u}$ pass through *the same* discrete operator, W at the top of T-cell k is filtered with exactly the T-cell operator implied by the level-k vector filter (`cgrid_filter.filter_level_w`):

$$\mathrm{vol}\,\overline w + \gamma\,F W^{-1}F^{T}\left(\ell_T^2\,\overline w\right) = \mathrm{vol}\,w, \qquad \mathrm{vol}=e_{1t}e_{2t}e_{3t},$$

- no-flux at land, with the same $e_3$ weights, the same placement of ℓ² and the same regional-edge treatment as the U,V filter
- solved in the symmetric variable $y=\ell_T^2\overline w$
- only cells above the sea floor (tmask) are filtered; the surface value 0 (rigid lid) and the sea-floor value 0 are kept
- no barotropic correction

**Why this operator.** A first W run used the plain scalar Laplacian $\nabla\cdot(\ell^2\nabla)$. It left an avoidable open-ocean residual where ℓ varies, 1–2×10⁻⁸ s⁻¹ in the Rossby configs. With the consistent operator, the open-ocean residual equals the original, a few ×10⁻¹¹ s⁻¹.

**Validation quantity.** The continuity residual on T cells:

$$R = \partial_z\overline w + \nabla_h\cdot\overline{\mathbf u}_h = \frac{\overline w_k - \overline w_{k+1}}{e_{3t,k}} + \nabla_h\cdot\overline{\mathbf u}_{h,k}.$$

Where levels k and k+1 share the same wet mask, R is the filter applied to the original residual, which is small. There are two exceptions:
1. **Bottom steps**, where the mask of level k+1 differs from level k. There $\overline w_{k+1}$ is filtered on a different domain; this is inherent to level-by-level filtering.
2. **Near the open edges of the regional cut.** There the discrete "divergence" of the edge cells misses the flow through the cut, and the filter spreads that deficit inward over ~2 box widths.

Near the Amazon mouth R also carries the filtered surface source.

### 1.4 Scales used
| Config | Definition | ℓ range in domain | ℓ at 0° / 10°N / 20°N / 30°N | box scale $3.5\ell$ at 0° |
|---|---|---|---|---|
| W1.5 | $\ell = 1.5°\cdot\frac{\pi}{180}R\cos\varphi/3.5$ | 41–48 km | 47.6 / 46.9 / 44.8 / 41.3 km | 167 km |
| W2.0 | same, 2.0° | 55–64 km | 63.5 / 62.5 / 59.7 / 55.0 km | 222 km |
| W2.5 | same, 2.5° | 69–79 km | 79.4 / 78.2 / 74.6 / 68.8 km | 278 km |
| R2Ld | $\ell = 2L_d/3.5$, Gaussian-smoothed (σ = 6 cells) | see diag | ≈135 / 61 / 30 / 19 km | ≈474 km |
| R3Ld | $\ell = 3L_d/3.5$ | see diag | ≈203 / 91 / 45 / 28 km | ≈711 km |

R = 6371 km and φ is the local latitude; the Rossby values are Atlantic zonal medians.

**Rossby radius.** $L_d$ is the WKB first-baroclinic radius from the GLORYS January-1993 mean T/S: $c_1=\frac1\pi\int N\,dz$, $L_d=\min(c_1/|f|,\sqrt{c_1/2\beta})$ (Chelton et al. 1998).
- On the shelf (H < 1000 m) and over land, $c_1$ is extrapolated from deep water.
- Atlantic zonal medians: 237 km (0°), 196 km (5°N), 106 km (10°N), 52 km (20°N), 33 km (30°N).
- The discrete eigenproblem gives values ~4–5 % below WKB.
- The Chelton file could not be downloaded, so no comparison was made.
- At 10°N, $L_d$ is larger than the task sheet's 50–80 km because $c_1\approx2.7$ m/s and $f=2.5\times10^{-5}$ s⁻¹. This matches Chelton et al. for the deep tropical Atlantic.

## 2. Computational summary
**Environment.** `/work/bk1450/b383184/conda/envs/implicit_filter`, built by `build_env.sh`:
- Python 3.11, CuPy 14.2 with the CUDA 12.4 runtime and headers from conda-forge (NVIDIA driver 580 / CUDA 13)
- SciPy, xarray, netCDF4, JAX, papermill
- `implicit_filter` 1.3.1 installed editable from `../implicit_filter`
- Jupyter kernel `implicit_filter`

**Hardware.** DKRZ Levante GPU partition, one NVIDIA A100-80GB per configuration (account `bk1450_gpu`). The five configurations run as parallel jobs.

**Solver.** Batched Jacobi-preconditioned CG in CuPy, solving all 31 days of a level simultaneously, relative tolerance 10⁻⁸, float64.
- U,V: 847,626 coupled unknowns at the surface.
- W: 417,000 unknowns at the surface.
- The operators are rebuilt per level from the NaN mask of the data, in vectorised NumPy on the CPU, in about 1 s.

| Config | CG its U,V | CG its W | GPU time U,V per level (31 days) | per field per day | W per level | wall-clock (U,V+W run / final W rerun) |
|---|---|---|---|---|---|---|
| W1.5 | 110–114 | 94–99 | 5.1 s | 82 ms (U+V) | 3.1 s | 20.8 + 13.3 min |
| W2.0 | 150–154 | 126–133 | 6.6 s | 106 ms | 3.7 s | 22.4 + 13.3 min |
| W2.5 | 188–195 | 157–167 | 8.1 s | 131 ms | 4.3 s | 24.0 + 13.3 min |
| R2Ld | 288–329 | 249–294 | 13.3 s | 215 ms | 6.7 s | 30.4 + 13.7 min |
| R3Ld | 443–504 | 373–433 | 19.7 s | 318 ms | 9.3 s | 38.8 + 15.6 min |

**Where the wall-clock time goes.** GPU solves take 6–23 min per job. Compressed NetCDF writing takes ~9 min. The continuity-residual diagnostics, computed for original and filtered fields on CPU, take ~5 min. Output is ~5.6 GB per configuration.

**Issues encountered and how they were resolved**
1. **CUDA libraries.** CuPy from conda-forge initially lacked the CUDA runtime and the headers for JIT kernels. Fixed by adding `cuda-cudart(-dev)`, `libcusparse(-dev)`, `libcublas(-dev)`, `cuda-nvrtc(-dev)` and `cuda-cccl`, and setting `CUDA_PATH=$ENV/targets/x86_64-linux`.
2. **`implicit_filter` package bugs.** The GPU backend fails with CuPy ≥ 14 (`tol` was renamed `rtol`). `k` must be a Python `float`. `NemoFilter` builds only a T-grid operator, with index errors in its metric arrays. With variable scales its column scaling is not symmetric. The package was used only for a cross-check: its solver reproduces this operator to 2×10⁻⁸ for constant ℓ.
3. **Component-wise U,V filtering (v1)** created wall-divergence artefacts (section 1.2). This was replaced by the vector div–rot operator.
4. **W from continuity (v1, v2)** contained no river or free-surface source and was replaced by the scalar W filter (W correction). The intermediate barotropic correction (v2) was dropped.
5. **Missing e3 weight in the first v3 build.** The vorticity term lacked $e_{3f}$, so below ~20 m U,V were barely filtered. An analysis agent caught this, and it was fixed and rerun. The regression test `tests/test_e3_independence.py` checks shear and divergent waves at e3 = 1, 50 and 200 m.
6. **Empty depth axis in W files.** `netCDF4.renameVariable` on the depth coordinate silently wrote fill values; the v1/v2 W files have an empty `depthw`. Fixed by creating the coordinate directly.
7. **No umask/vmask in the mesh files.** Wet masks come from the data NaNs. They equal $t_{mask}(i)\,t_{mask}(i{+}1)$ except in the last column (U) and last row (V) of the regional cut.
8. **W operator inconsistent with the U,V filter (first W run).** It was flagged by the divergence analysis. The plain scalar Laplacian ∇·(ℓ²∇) differs from the T-cell operator implied by the vector filter, ∇·∇(ℓ²·) with e3 weights. Where ℓ varies, this left an open-ocean continuity residual of 1–2×10⁻⁸ s⁻¹. W was refiltered with the consistent operator (`filter_level_w`, `run_filter.py --w-only`), bringing the open-ocean residual down to the original level.
