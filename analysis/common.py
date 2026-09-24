"""Shared helpers for the analysis of the filtered GLORYS12v1 fields."""
import os, sys
import numpy as np
import netCDF4

RUN = "/work/bk1450/b383184/Amazon/Mercator/implicit_run"
IN = "/work/bk1450/b383184/Amazon/Mercator/implicit_data"
sys.path.insert(0, RUN)
import cgrid_filter as cf  # noqa: E402

MONTH = "1993-01"
CONFIGS = ["W1.5", "W2.0", "W2.5", "R2Ld", "R3Ld"]
LABELS = {"ORIG": "GLORYS12 (unfiltered)", "W1.5": "W1.5 (1.5°)", "W2.0": "W2.0 (2.0°)",
          "W2.5": "W2.5 (2.5°)", "R2Ld": "Rossby 2·Ld", "R3Ld": "Rossby 3·Ld"}
COLORS = {"ORIG": "k", "W1.5": "#1b9e77", "W2.0": "#d95f02", "W2.5": "#7570b3",
          "R2Ld": "#e7298a", "R3Ld": "#66a61e"}

_mesh = None


def mesh():
    global _mesh
    if _mesh is None:
        _mesh = cf.CMesh()
    return _mesh


def available_configs():
    return [c for c in CONFIGS if os.path.exists(f"{RUN}/output/{c}/diag_{c}.json")]


def path(var, cfg):
    """var in U,V,W; cfg 'ORIG' or a config name."""
    if cfg == "ORIG":
        if var == "W":
            return "/work/bk1450/b383184/Amazon/Mercator/data/variables_c/UVW/W_1993-01fc.nc"
        return f"{IN}/{var}_{MONTH}c.nc"
    return f"{RUN}/output/{cfg}/{var}_{MONTH}c_{cfg}.nc"


VNAME = {"U": "vozocrtx", "V": "vomecrty", "W": "vovecrtz"}


def read(var, cfg, t=slice(None), k=slice(None), j=slice(None), i=slice(None)):
    """Read a hyperslab (time, depth, y, x) as float64 with NaN on land."""
    with netCDF4.Dataset(path(var, cfg)) as ds:
        return ds[VNAME[var]][t, k, j, i].filled(np.nan).astype(np.float64)


def depths():
    return mesh().gdept_0


def uv_to_t(u, v):
    """Average C-grid u (…,y,x) and v (…,y,x) to T points (NaN treated as 0 = land faces)."""
    u0 = np.nan_to_num(u); v0 = np.nan_to_num(v)
    ut = np.zeros_like(u0); vt = np.zeros_like(v0)
    ut[..., 1:] = 0.5 * (u0[..., 1:] + u0[..., :-1])
    vt[..., 1:, :] = 0.5 * (v0[..., 1:, :] + v0[..., :-1, :])
    return ut, vt


def ij_nearest(lon, lat, grid="T"):
    m = mesh()
    d = (m.lon(grid) - lon) ** 2 + (m.lat(grid) - lat) ** 2
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return int(j), int(i)


def box_slices(lon0, lon1, lat0, lat1, grid="T"):
    """Index slices bounding a lon/lat box (grid is nearly lat-lon aligned in this domain)."""
    m = mesh()
    lon = m.lon(grid); lat = m.lat(grid)
    sel = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    jj, ii = np.where(sel)
    return slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1)
