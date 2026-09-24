"""Builds implicit_data/v0.Implicit_filtering_ARP.ipynb (keeps the user's original cells at the top)."""
import nbformat as nbf

NB = "/work/bk1450/b383184/Amazon/Mercator/implicit_data/v0.Implicit_filtering_ARP.ipynb"
old = nbf.read("/work/bk1450/b383184/Amazon/Mercator/implicit_run/scratch/v0.Implicit_filtering_ARP.orig_backup.ipynb", as_version=4)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""## Trying the implicit filter
Implicit filtering of **GLORYS12v1 (NEMO ORCA12 C-grid)** U, V and W for **January 1993**, for Lagrangian tracking.

This notebook is the interactive version of the workflow. The heavy production (5 filter configurations × 31 days × 50 levels) runs as GPU batch jobs
from `../implicit_run/`; the report is `../implicit_run/REPORT.md`.

**Kernel:** select `implicit_filter` (env `/work/bk1450/b383184/conda/envs/implicit_filter`, with CuPy for the GPU).
On a CPU-only Jupyter node everything below still runs; the solver automatically uses SciPy on the CPU (slower, ~1 min per day and level).

Contents
1. Paths and data
2. Mesh, masks and a look at the data
3. The filter: equation, scale convention (k vs ℓ), vector (div–rot) form
4. Synthetic check of the transfer function
5. Filter one level and one day (four window/Rossby scales)
6. Why U,V must be filtered as a vector on the C-grid (divergence at walls)
7. W: filtered as a scalar (same scale) and the continuity residual
8. Production on the GPU (sbatch) and loading the filtered output""")

md("### 1. Paths and data")
cells.append(old.cells[1])  # user's original path cell
code("""path0 = './'   # = /work/bk1450/b383184/Amazon/Mercator/implicit_data/
U_file = path0 + 'U_1993-01c.nc'   # vozocrtx (time_counter, deptht, y, x)
V_file = path0 + 'V_1993-01c.nc'   # vomecrty
T_file = path0 + 'T_1993-01c.nc'   # votemper (used for the Rossby radius)
S_file = path0 + 'S_1993-01c.nc'   # vosaline
RUN = '/work/bk1450/b383184/Amazon/Mercator/implicit_run'   # code, batch scripts, outputs, report""")

code("""import sys, time, math
import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
sys.path.insert(0, RUN)
import cgrid_filter as cf

try:
    import cupy as cp
    BACKEND = 'gpu' if cp.cuda.runtime.getDeviceCount() > 0 else 'cpu'
except Exception:
    BACKEND = 'cpu'
print('solver backend:', BACKEND)""")

md("### 2. Mesh, masks and a look at the data")
code("""mesh = cf.CMesh(hgr=Hgr, zgr=Zgr)     # scale factors e1/e2 (t,u,v,f), partial-step e3t/e3u/e3v, tmask from mbathy
print('grid (ny, nx, nz):', mesh.ny, mesh.nx, mesh.nz)
print('lon range: %.2f .. %.2f   lat range: %.2f .. %.2f' % (mesh.glamt.min(), mesh.glamt.max(), mesh.gphit.min(), mesh.gphit.max()))
print('e1t %.0f-%.0f m, e2t %.0f-%.0f m' % (mesh.e1t.min(), mesh.e1t.max(), mesh.e2t.min(), mesh.e2t.max()))
print('U(j,i) sits east of T(j,i): glamu-glamt =', float(mesh.glamu[200, 600] - mesh.glamt[200, 600]))

dsU = netCDF4.Dataset(U_file); dsV = netCDF4.Dataset(V_file)
print(dsU['vozocrtx'])""")

code("""t, k = 10, 0            # 1993-01-11, surface
u0 = dsU['vozocrtx'][t, k].filled(np.nan).astype(float)
v0 = dsV['vomecrty'][t, k].filled(np.nan).astype(float)
# the wet masks of U and V equal tmask(i)*tmask(i+1), tmask(j)*tmask(j+1) (except the last column/row of the cut)
umask = np.zeros_like(mesh.tmask[k]); umask[:, :-1] = mesh.tmask[k][:, :-1] & mesh.tmask[k][:, 1:]
print('U NaN pattern vs mask mismatches (interior):', int(((~np.isnan(u0)) != umask)[:, :-1].sum()))

def to_t(u, v):
    ut = np.zeros_like(u); vt = np.zeros_like(v)
    ut[:, 1:] = 0.5*(np.nan_to_num(u[:, 1:]) + np.nan_to_num(u[:, :-1]))
    vt[1:, :] = 0.5*(np.nan_to_num(v[1:, :]) + np.nan_to_num(v[:-1, :]))
    s = np.hypot(ut, vt); s[~mesh.tmask[k]] = np.nan
    return s

box = dict(x=slice(np.argmin(abs(mesh.glamt[200]+70)), np.argmin(abs(mesh.glamt[200]+30))),
           y=slice(np.argmin(abs(mesh.gphit[:, 600]+5)), np.argmin(abs(mesh.gphit[:, 600]-30))))
sl = (box['y'], box['x'])
plt.figure(figsize=(9, 6))
plt.pcolormesh(mesh.glamt[sl], mesh.gphit[sl], to_t(u0, v0)[sl], vmin=0, vmax=1.2, cmap='viridis')
plt.colorbar(label='surface speed [m/s]'); plt.title('GLORYS12 1993-01-11 surface speed (Amazon / NBC)')""")

md(r"""### 3. The filter

**Equation** (Danilov et al. 2023 JAMES; Nowak et al. 2025 GMD), low-pass form with $\gamma=1/2$, $n=1$:

$$\left(1 + \gamma\,\mathcal{M}_\ell\right)\overline{\mathbf u} = \mathbf u, \qquad G(K) = \frac{1}{1+\tfrac12\ell^2K^2}, \qquad \ell_{box}\approx 3.5\,\ell .$$

The factor 3.5 comes from matching second moments: a box of width $L$ has $\operatorname{sinc}(KL/2)\approx 1-K^2L^2/24$, and $G\approx1-\tfrac12\ell^2K^2$ gives $\ell = L/\sqrt{12}\approx L/3.5$.

**Parameter convention (important).** The `implicit_filter` package solves $(I + 2(-L/k^2))\phi$, i.e. $\gamma=2$ with $k$.
To get the γ = ½ filter with scale ℓ you must pass **`k = 2/ℓ`**. Passing `k = 1/ℓ` would silently double the effective ℓ.
The code here works directly with γ and ℓ [m].

**Scales.**
- Windows W1.5/W2.0/W2.5: $\ell = \Delta\theta\,\frac{\pi}{180} R\cos\varphi / 3.5$.
- Rossby configurations: $\ell = m\,L_d/3.5$ with $m=2,3$ and $L_d$ from the GLORYS T/S (WKB), in `../implicit_run/rossby/`.

**Vector form on the C-grid.** Instead of a scalar Laplacian per component, $\mathcal{M}_\ell\mathbf u = -\nabla(\ell^2\,\nabla\!\cdot\mathbf u) + \nabla\times(\ell^2\zeta)$.
- It uses NEMO's discrete divergence (T points, with $e_{3u},e_{3v}$) and vorticity (F points).
- For constant ℓ in the open ocean it equals the component-wise Laplacian, so the spherical metric terms are < 0.1 % (addendum). The transfer function is unchanged.
- The boundary conditions are zero normal flow and free slip.
- The filtered divergence is exactly the scalar-filtered divergence (section 6), so filtered (U, V) stay consistent with a W filtered at the same scale.
- W is filtered as a scalar on the W/T points with the same ℓ. The vertical component has no metric coupling to U, V. W is **not** recomputed from divergence, because NEMO's continuity includes runoff, E−P and free-surface sources.

The addendum's metric-term argument is correct, but it is not what decides the choice: section 6 shows that per-component no-flux walls break continuity.""")

code("""ell_W20_T, ell_W20_F = cf.ell_T_F(mesh, 'window', window_deg=2.0)
ld = netCDF4.Dataset(RUN + '/rossby/rossby_radius_T.nc')['Ld'][:].astype(float)
ell_R2_T, ell_R2_F = cf.ell_T_F(mesh, 'rossby', ld_T=ld, mult=2.0)
lat = mesh.gphit[:, 600]
plt.figure(figsize=(7, 4))
for w in (1.5, 2.0, 2.5):
    plt.plot(lat, cf.ell_T_F(mesh, 'window', window_deg=w)[0][:, 600]/1e3, label=f'W{w}')
atl = slice(np.argmin(abs(mesh.glamt[200]+60)), np.argmin(abs(mesh.glamt[200]+20)))
plt.plot(lat, np.median(ell_R2_T[:, atl], 1)/1e3, label='R2Ld (Atlantic median)')
plt.plot(lat, np.median(cf.ell_T_F(mesh, 'rossby', ld_T=ld, mult=3.0)[0][:, atl], 1)/1e3, label='R3Ld')
plt.xlabel('latitude'); plt.ylabel('ℓ [km]  (box scale = 3.5 ℓ)'); plt.legend(); plt.grid(alpha=.3)""")

md("### 4. Synthetic check of the transfer function\nSinusoids of 50–800 km on the real ORCA12 metrics (no land, constant ℓ for W2.0 at the equator). Filtered amplitude vs $G(K)$.")
code("""import copy
jj, ii = slice(150, 350), slice(300, 700)          # sub-box for speed
sub = copy.copy(mesh)
for name in ['e1t','e2t','e1u','e2u','e1v','e2v','e1f','e2f']:
    setattr(sub, name, getattr(mesh, name)[jj, ii])
ny_, nx_ = sub.e1t.shape
sub.ny, sub.nx = ny_, nx_
sub.tmask = np.ones((1, ny_, nx_), bool); sub.e3t = np.ones((1, ny_, nx_)); sub.e3u = sub.e3t.copy(); sub.e3v = sub.e3t.copy()
sub.e3t_0 = np.ones(1)
ell = 222e3/3.5
lT = np.full((ny_, nx_), ell)
x = np.cumsum(sub.e1u, 1); y = np.cumsum(sub.e2v, 0)
lams = np.array([50, 100, 200, 400, 800])*1e3
Us = np.stack([np.cos(2*np.pi*x/l) for l in lams]); Vs = np.stack([np.cos(2*np.pi*y/l) for l in lams])
uo_, vo_, its, n = cf.filter_level_vector(sub, 0, Us, Vs, lT, lT, backend=BACKEND, tol=1e-10)
c = (slice(60, -60), slice(80, -80))
gain_u = [(uo_[i][c]*Us[i][c]).sum()/(Us[i][c]**2).sum() for i in range(5)]
gain_v = [(vo_[i][c]*Vs[i][c]).sum()/(Vs[i][c]**2).sum() for i in range(5)]
theory = 1/(1 + 0.5*ell**2*(2*np.pi/lams)**2)
print('CG iterations', its)
for l, gu, gv, th in zip(lams, gain_u, gain_v, theory):
    print(f'lambda = {l/1e3:4.0f} km   gain u {gu:.3f}  v {gv:.3f}   theory {th:.3f}')""")

md("### 5. Filter one level and one day\nThis is the same routine as production (`cf.filter_level_vector`). U and V are solved together on their native points.")
code("""results = {}
for name, (lT, lF) in {'W1.5': cf.ell_T_F(mesh, 'window', window_deg=1.5),
                       'W2.0': (ell_W20_T, ell_W20_F),
                       'W2.5': cf.ell_T_F(mesh, 'window', window_deg=2.5),
                       'R2Ld': (ell_R2_T, ell_R2_F)}.items():
    t0 = time.time()
    uf_, vf_, its, n = cf.filter_level_vector(mesh, k, u0[None], v0[None], lT, lF, backend=BACKEND, tol=1e-8)
    results[name] = (uf_[0], vf_[0])
    print(f'{name}: {n} unknowns, {its} CG iterations, {time.time()-t0:.1f} s ({BACKEND})')

fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
panels = [('GLORYS12', to_t(u0, v0))] + [(nm, to_t(*results[nm])) for nm in results]
for a, (nm, s) in zip(ax.flat, panels):
    pc = a.pcolormesh(mesh.glamt[sl], mesh.gphit[sl], s[sl], vmin=0, vmax=1.2, cmap='viridis'); a.set_title(nm)
d = to_t(*results['W2.0']) - to_t(u0, v0)
pc2 = ax.flat[5].pcolormesh(mesh.glamt[sl], mesh.gphit[sl], d[sl], vmin=-.5, vmax=.5, cmap='RdBu_r'); ax.flat[5].set_title('W2.0 − GLORYS (speed)')
fig.colorbar(pc, ax=ax[:, :2], shrink=.6, label='m/s'); fig.colorbar(pc2, ax=ax.flat[5], label='m/s')""")

md("""### 6. Why U and V must be filtered as a vector on the C-grid
The first production run (v1, kept in `../implicit_run/output_v1_componentwise/`) followed the task sheet: U on the U-grid and V on the V-grid, each a scalar with its own no-flux walls.
The spherical metric terms are negligible (< 0.1 %, as the addendum says). The problem is elsewhere.
- Each component is smoothed toward its offshore values right at the wall, while the normal velocity through the wall stays zero.
- This leaves a **one-cell convergence line along every coast, at every level**.
- Integrated over the water column it gives $|w|\\sim10^{-3}$ m/s (≈ 80 m/day), against $5\\times10^{-5}$ m/s in GLORYS.

The div–rot form fixes this exactly: $\\nabla\\cdot\\overline{\\mathbf u} = (1+\\gamma(-\\Delta_T\\ell^2))^{-1}\\,\\nabla\\cdot\\mathbf u$ at every level.""")
code("""# v1: component-wise scalar filter (same scale), same level/day
uv1 = cf.filter_level(mesh, 'U', u0[None].astype(np.float32), cf.ell_from_window_deg(mesh, 'U', 2.0), backend=BACKEND)[0][0]
vv1 = cf.filter_level(mesh, 'V', v0[None].astype(np.float32), cf.ell_from_window_deg(mesh, 'V', 2.0), backend=BACKEND)[0][0]

def hdiv2d(u, v):
    return cf.horizontal_divergence_e3(mesh, u[None], v[None])[0] / np.where(mesh.e3t[0] > 0, mesh.e3t[0], 1)

zoom = (slice(np.argmin(abs(mesh.gphit[:, 600]+3)), np.argmin(abs(mesh.gphit[:, 600]-12))),
        slice(np.argmin(abs(mesh.glamt[200]+62)), np.argmin(abs(mesh.glamt[200]+42))))
fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
for a, (nm, dd) in zip(ax, [('GLORYS12', hdiv2d(u0, v0)), ('v1 component-wise W2.0', hdiv2d(uv1, vv1)),
                            ('v2 vector div-rot W2.0', hdiv2d(*results['W2.0']))]):
    dd = np.where(mesh.tmask[0], dd, np.nan)
    pc = a.pcolormesh(mesh.glamt[zoom], mesh.gphit[zoom], dd[zoom]*1e6, vmin=-5, vmax=5, cmap='RdBu_r'); a.set_title(nm)
fig.colorbar(pc, ax=ax, label='horizontal divergence [1e-6 1/s]', shrink=.8)
from scipy.ndimage import distance_transform_edt
dist = distance_transform_edt(mesh.tmask[0])
for nm, (uu, vv) in [('GLORYS12', (u0, v0)), ('v1', (uv1, vv1)), ('v2', results['W2.0'])]:
    dd = hdiv2d(uu, vv)
    print(f'{nm:9s} rms div  coast (<=3 cells): {np.sqrt(np.mean(dd[(dist>0)&(dist<=3)]**2)):.2e}   open (>15 cells): {np.sqrt(np.mean(dd[dist>15]**2)):.2e}')""")

md("""### 7. W: filtered as a scalar, and the continuity residual
W comes from the rigid-lid product of `data/Fix_W.ipynb` (`data/variables_c/UVW/W_1993-01fc.nc`). It is filtered level by level on the W/T points with the same ℓ as U, V.
The operator is exactly the T-cell operator implied by the vector filter at that level, $\\mathrm{vol}\\,\\overline w + \\gamma F W^{-1}F^T(\\ell^2\\overline w) = \\mathrm{vol}\\,w$, so $\\partial_z\\overline w$ and $\\nabla\\cdot\\overline{\\mathbf u}$ are filtered identically wherever neighbouring levels share the same wet mask.
- Only points above the sea floor (tmask) are filtered, with no-flux at land.
- The sea-floor W points keep their value (0), and the surface stays 0 (rigid lid).

Validation: $R = \\partial_z \\overline w + \\nabla_h\\cdot\\overline{\\mathbf u}_h$ should be small and smooth. It is not zero near the Amazon mouth, where it is the filtered river source.""")
code("""W_file = '/work/bk1450/b383184/Amazon/Mercator/data/variables_c/UVW/W_1993-01fc.nc'
dsW = netCDF4.Dataset(W_file)
RUN_3D = (BACKEND == 'gpu')      # one full 3D day; on CPU it takes a few minutes, set True to force
if RUN_3D:
    u3 = dsU['vozocrtx'][t].filled(np.nan).astype(float); v3 = dsV['vomecrty'][t].filled(np.nan).astype(float)
    w3 = dsW['vovecrtz'][t].filled(np.nan).astype(float)
    uf3 = np.full_like(u3, np.nan); vf3 = np.full_like(v3, np.nan); wf3 = w3.copy()
    t0 = time.time()
    for kk in range(mesh.nz):
        a_, b_, its, n = cf.filter_level_vector(mesh, kk, u3[kk][None], v3[kk][None], ell_W20_T, ell_W20_F, backend=BACKEND)
        uf3[kk], vf3[kk] = a_[0], b_[0]
        wet = mesh.tmask[kk] & ~np.isnan(w3[kk])
        if kk > 0 and wet.any():
            # same T-cell operator as implied by the level-kk vector filter -> consistent with div(u_bar)
            wo_ = cf.filter_level_w(mesh, kk, np.nan_to_num(w3[kk])[None], ~np.isnan(u3[kk]), ~np.isnan(v3[kk]), ell_W20_T, backend=BACKEND)[0][0]
            wf3[kk] = np.where(mesh.tmask[kk], wo_, w3[kk])
    print(f'3D filtering of one day: {time.time()-t0:.0f} s')
    R_o = cf.continuity_residual(mesh, u3, v3, w3)
    R_f = cf.continuity_residual(mesh, uf3, vf3, wf3)
    hd_o = cf.horizontal_divergence_e3(mesh, u3, v3) / np.where(mesh.e3t > 0, mesh.e3t, 1)
    hd_f = cf.horizontal_divergence_e3(mesh, uf3, vf3) / np.where(mesh.e3t > 0, mesh.e3t, 1)
    for kk in (0, 10, 22, 33):
        m_ = mesh.tmask[kk][sl]
        print(f'z = {mesh.gdept_0[kk]:6.0f} m  rms R  orig {np.sqrt(np.nanmean(R_o[kk][sl][m_]**2)):.2e}  filt {np.sqrt(np.nanmean(R_f[kk][sl][m_]**2)):.2e}'
              f'   | rms hdiv orig {np.sqrt(np.mean(hd_o[kk][sl][m_]**2)):.2e} filt {np.sqrt(np.mean(hd_f[kk][sl][m_]**2)):.2e}  [1/s]')
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for a, (nm, rr) in zip(ax, [('R original (surface layer)', R_o[0]), ('R filtered W2.0 (surface layer)', R_f[0]),
                                ('hdiv filtered W2.0 (surface layer)', hd_f[0])]):
        pc = a.pcolormesh(mesh.glamt[sl], mesh.gphit[sl], np.where(mesh.tmask[0], rr, np.nan)[sl]*1e6, vmin=-3, vmax=3, cmap='RdBu_r'); a.set_title(nm)
    fig.colorbar(pc, ax=ax, label='[1e-6 1/s]', shrink=.8)
else:
    print('skipped (set RUN_3D = True)')""")

md("""### 8. Production on the GPU and the filtered output
One A100 per configuration, ~20–30 min each (`../implicit_run/run_filter.sh`).
- **U, V:** `output/<CFG>/U_1993-01c_<CFG>.nc` (vozocrtx) and `V_...` (vomecrty), with the same dims and coordinates as the input.
- **W:** `W_...`, with `vovecrtz` = filtered rigid-lid W (same convention as `data/variables_c/UVW/W_1993-01fc.nc`).
- **Diagnostics:** `diag_<CFG>.npz` holds the continuity residual statistics and maps.
- **Parcels:** use `FieldSet.from_nemo` with `Hgr_cmesh.nc`, exactly as for the unfiltered files.""")
code("""# submit (uncomment); the env build script is ../implicit_run/build_env.sh
# import subprocess
# for c in ['W1.5', 'W2.0', 'W2.5', 'R2Ld', 'R3Ld']:
#     print(subprocess.run(['sbatch', '-J', f'v2_{c}', f'{RUN}/run_filter.sh', c], capture_output=True, text=True).stdout)
import os, json
for c in ['W1.5', 'W2.0', 'W2.5', 'R2Ld', 'R3Ld']:
    f = f'{RUN}/output/{c}/diag_{c}.json'
    if os.path.exists(f):
        d = json.load(open(f)); print(f"{c}: done, {d['total_time']/60:.1f} min, l_T = {d['l_T_min_m']/1e3:.0f}-{d['l_T_max_m']/1e3:.0f} km")
    else:
        print(c, 'not (yet) available')""")
code("""cfg = 'W2.0'
if os.path.exists(f'{RUN}/output/{cfg}/diag_{cfg}.json'):
    Uf = xr.open_dataset(f'{RUN}/output/{cfg}/U_1993-01c_{cfg}.nc')
    Vf = xr.open_dataset(f'{RUN}/output/{cfg}/V_1993-01c_{cfg}.nc')
    Wf = xr.open_dataset(f'{RUN}/output/{cfg}/W_1993-01c_{cfg}.nc')
    print(Uf.attrs['filter']); print(Wf)
    uf_ = Uf.vozocrtx.isel(time_counter=t, deptht=0).values; vf_ = Vf.vomecrty.isel(time_counter=t, deptht=0).values
    w100 = Wf.vovecrtz.isel(time_counter=t, depthw=10).values
    dg = np.load(f'{RUN}/output/{cfg}/diag_{cfg}.npz')
    print('rms residual R (surface layer, whole domain, 31-day mean): filt %.2e  orig %.2e' % (dg['R_rms_filt'][:, 0].mean(), dg['R_rms_orig'][:, 0].mean()))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    pc = ax[0].pcolormesh(mesh.glamt[sl], mesh.gphit[sl], to_t(uf_, vf_)[sl], vmin=0, vmax=1.2); ax[0].set_title(f'{cfg} surface speed, 1993-01-11'); fig.colorbar(pc, ax=ax[0])
    pc = ax[1].pcolormesh(mesh.glamt[sl], mesh.gphit[sl], w100[sl]*86400, vmin=-10, vmax=10, cmap='RdBu_r'); ax[1].set_title(f'{cfg} w at 100 m [m/day]'); fig.colorbar(pc, ax=ax[1])""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "implicit_filter", "language": "python", "name": "implicit_filter"},
               "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, NB)
print("written", NB, len(cells), "cells")
