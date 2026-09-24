# Why this repository implements its own C-grid filter

This note answers a question that comes up whenever someone sees that
`cgrid_filter.py` exists alongside the
[`implicit_filter`](https://github.com/FESOM/implicit_filter) package:

> Is it using the filter with the existing option to work directly on the
> velocity points, or did it code something new itself (which might be tricky)?

Short answer: **the package has no option to filter on the NEMO velocity
points, so this part is new code.** It is not, however, a new *method* — only a
new discretisation, and it is the direct analogue of something the package
already does for triangular meshes.

## 1. What the package offers on a NEMO grid

`NemoFilter` subclasses `LatLonFilter` and inherits its `compute_velocity`.
That method does this:

```python
return (
    np.reshape(self._compute(n, k, np.reshape(ux, self._e2d), x0=ux0_flat), (self._nx, self._ny)),
    np.reshape(self._compute(n, k, np.reshape(vy, self._e2d), x0=vy0_flat), (self._nx, self._ny)),
)
```

Two independent scalar solves with the same T-cell operator. The contract in
the abstract base class (`filter.py`) is explicit about where the data has to
live:

> Compute the filtered velocity data using a specified filter size.
> **Data must be placed on mesh nodes**
> […]
> Tuple containing NumPy arrays with filtered data ux and uy velocities **on mesh nodes**.

So using it on a C-grid means interpolating `U` and `V` off their native points
onto T points, filtering the two components as unrelated scalars, and
interpolating back.

## 2. That route was tried first, and it failed

It is the **v1** run, kept in `output_v1_componentwise/` for the comparison
figures in [REPORT.md](REPORT.md). Filtering the components separately gives
each one its own land mask, so the no-flow wall sits in a different place for
`U` than for `V`. The result was a one-cell convergence line along every
coastline and vertical velocities of order 1e-3 m/s — unusable for Lagrangian
tracking, which is the entire purpose of these fields.

This is not a tuning problem. Two separately filtered components are not the
filtered vector field, and on a staggered grid the discrete divergence of the
pair is not the filtered divergence.

## 3. What is implemented instead

Same filter, different discretisation. Unchanged from Danilov et al. (2023) and
Nowak et al. (2025): the implicit filter itself, γ = ½, the length scale ℓ, and
the transfer function `G(K) = 1/(1 + ½ℓ²K²)`.

What is new is that `U` and `V` are solved **jointly, on their own C-grid
points**, in div–rot form:

$$M\mathbf{u} = -\nabla(\ell_T^2\,\nabla\cdot\mathbf{u}) + \nabla\times(\ell_F^2\,\zeta)$$

assembled from **NEMO's own discrete divergence** (carrying `e3u`, `e3v`) and
vorticity, with zero normal flow through land and free slip. Nothing about the
stencil is invented: it is NEMO's discretisation, used as the filter operator.

This is also what the package already does for triangular meshes.
`TriangularFilter` with `full=True` solves the coupled system on the
concatenated components via `_compute_full`. The package simply does not offer
that for structured grids — and even on a triangular mesh it refuses to combine
coupled filtering with element (i.e. velocity) points:

```python
if is_elem and self._full:
    raise ValueError("Coupled full metric filtering not supported for elements. "
                     "Please use full=False or switch to nodal filtering.")
```

So "coupled" and "on the velocity points" are not jointly available anywhere in
the package, for any mesh type. On a C-grid, neither is.

`W` is handled separately: the rigid-lid `W` product is filtered as a scalar at
the same ℓ with the T-cell operator implied by the level-*k* vector filter
(`filter_level_w`). It is deliberately **not** recomputed from divergence,
because NEMO's continuity also carries runoff, E−P and free-surface sources —
which matters in an Amazon plume configuration.

## 4. Evidence that it behaves

| property | value | how |
|---|---|---|
| operator symmetry | rel. asymmetry **9.81e-17** | direct measurement of `S - Sᵀ` |
| positive definiteness | λ_min = **2.58e8** > 0 | `eigsh`, smallest algebraic |
| agreement with the package | **2e-8** | scalar path, constant ℓ, `tests/validate_filter.py` |
| `div(ū)` = scalar no-flux filter of `div(u)` | exact, every level | `tests/validate_vector.py` |
| independence of layer thickness | regression test | `tests/test_e3_independence.py` |

The last row exists because an early v3 build omitted the `e3` weight in the
vorticity term, which left `U`, `V` almost unfiltered below ~20 m. The test
covers shear and divergent waves at e3 = 1, 50 and 200 m.

The commutation property in row four is the reason the extra complexity is
worth it: it is what keeps the filtered field discretely consistent with NEMO
continuity, and it is precisely what the component-wise route cannot provide.

## 5. A related consequence: the package's GPU preconditioner

The `elements` branch of the package adds a V-cycle (AMG) preconditioner that
substantially reduces iteration counts. It **cannot** be reached through
`NemoFilter`, for reasons internal to the package rather than anything about
this grid. Its V-cycle setup requires `D·A` to be symmetric under a diagonal
weight, and refuses anything above a relative asymmetry of 1e-6. Measured on
this domain (499 × 1260, level 22, W2.0 scale):

| configuration | rel. asymmetry | gate |
|---|---|---|
| package `NemoFilter`, as released | 2.479e-01 | rejected |
| + correcting the `hc[2]`/`hc[3]` index fill | 2.473e-01 | rejected |
| + forcing all `e3` = 1 (index fill as released) | 5.802e-04 | rejected |
| + both corrections together | **8.952e-17** | passes |

The dominant term is that the stencil takes the layer-thickness ratio at the
*neighbour* cell, `(hh·h3u[j])/(hc·h3t[j])/area[i]`, which is not edge-symmetric
and cannot be repaired by any diagonal weight. The geometry of the grid is not
the problem — the last row shows the curvilinear, stretched metrics symmetrise
to machine precision once those two issues are removed.

The operator in this repository is symmetric to 9.81e-17 as released (§4), so
AMG preconditioning is directly applicable to it. That is a performance
opportunity, not a correctness concern: a preconditioner changes the path to
the solution, never the solution.

## References

- Danilov, S., et al. (2023). *JAMES*. [doi:10.1029/2023MS003946](https://doi.org/10.1029/2023MS003946)
- Nowak, K., et al. (2025). *GMD* **18**, 6541. [doi:10.5194/gmd-18-6541-2025](https://doi.org/10.5194/gmd-18-6541-2025)
