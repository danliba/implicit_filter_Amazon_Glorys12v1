"""Transfer function of the vector div-rot filter for rotational (shear) and divergent waves at
several layer thicknesses e3 (regression test for the missing e3f weight in the vorticity term)."""
import sys, copy
sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run")
import numpy as np, cgrid_filter as cf
m = cf.CMesh()
jj, ii = slice(150, 350), slice(300, 700)
for E3 in [1.0, 50.0, 200.0]:
    sub = copy.copy(m)
    for n in ['e1t', 'e2t', 'e1u', 'e2u', 'e1v', 'e2v', 'e1f', 'e2f']:
        setattr(sub, n, getattr(m, n)[jj, ii])
    ny, nx = sub.e1t.shape; sub.ny, sub.nx = ny, nx
    sub.tmask = np.ones((1, ny, nx), bool); sub.e3t = np.full((1, ny, nx), E3)
    sub.e3u = sub.e3t.copy(); sub.e3v = sub.e3t.copy(); sub.e3t_0 = np.array([E3])
    ell = 222e3 / 3.5; lT = np.full((ny, nx), ell)
    x = np.cumsum(sub.e1u, 1); y = np.cumsum(sub.e2v, 0)
    lams = np.array([100, 200, 400]) * 1e3
    Us = np.stack([np.cos(2*np.pi*y/l) for l in lams] + [np.cos(2*np.pi*x/l) for l in lams]); Vs = np.zeros_like(Us)
    uo, vo, its, n = cf.filter_level_vector(sub, 0, Us, Vs, lT, lT, backend='cpu', tol=1e-10)
    c = (slice(60, -60), slice(80, -80))
    g = [(uo[i][c]*Us[i][c]).sum()/(Us[i][c]**2).sum() for i in range(6)]
    th = 1/(1 + 0.5*ell**2*(2*np.pi/lams)**2)
    print(f'e3={E3:5.0f}: shear gains {np.round(g[:3], 3)}  divergent gains {np.round(g[3:], 3)}  theory {np.round(th, 3)}')
    assert np.allclose(g[:3], th, atol=0.01) and np.allclose(g[3:], th, atol=0.01)
print("OK")
