"""
Kinetic-energy budget of the filters in the analysis box 70W-30W, 5S-30N (T points).

For every level k and every configuration c (ORIG, W1.5, ..., R3Ld), with C-grid u,v averaged to
T points (common.uv_to_t; land faces count as 0) and the weight w = e1t*e2t*e3t on wet T points:

  KE_tot  = < 0.5 (u^2 + v^2) >_t                       (time mean of daily KE)
  KE_mean = 0.5 (<u>_t^2 + <v>_t^2)                      (KE of the 31-day mean)
  EKE     = KE_tot - KE_mean = < 0.5 (u'^2 + v'^2) >_t,  u' = u - <u>_t
  SS      = < 0.5 ((u_o-u_f)^2 + (v_o-v_f)^2) >_t         (KE of the filtered-out small scales)
  SS_mean = 0.5 ((<u_o>-<u_f>)^2 + ...)                   (small-scale part of the mean flow)

All quantities are stored as sums over i and t (/nt) per latitude row j, i.e. arrays
(ncfg, nz, ny_box) in m^5/s^2 (KE per unit mass x volume), so any latitude band and any depth
range (with partial-cell clipping) can be formed later.
Also stored: 31-day-mean T-point u,v on the full model domain at levels 0 and K100 (92 m),
EKE maps at those levels in the box, and daily surface native-grid U,V in the box (+1 halo) for
vorticity / Okubo-Weiss.

Run as SLURM job (see energy.sh); uses a process pool over levels.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/work/bk1450/b383184/Amazon/Mercator/implicit_run/analysis")
import common as C  # noqa: E402

OUT = os.environ.get("OUTDIR", os.path.dirname(os.path.abspath(__file__)) + "/data")
CFGS = os.environ["CFGS"].split(",") if "CFGS" in os.environ else ["ORIG"] + C.CONFIGS
K100 = 21                      # gdept_0[21] = 92.3 m
m = C.mesh()
JS, IS = C.box_slices(-70, -30, -5, 30, "T")
J0, J1, I0, I1 = JS.start, JS.stop, IS.start, IS.stop
# read with a 1-point halo on the west/south side so that uv_to_t is correct on the box
RJ = slice(J0 - 1, J1); RI = slice(I0 - 1, I1)
NZ = int(os.environ.get("NZ", m.nz))


def to_t(u, v):
    ut, vt = C.uv_to_t(u, v)
    return ut[..., 1:, 1:], vt[..., 1:, 1:]


def level(k):
    t0 = time.time()
    wet = m.tmask[k, J0:J1, I0:I1]
    w = np.where(wet, m.e1t[J0:J1, I0:I1] * m.e2t[J0:J1, I0:I1] * m.e3t[k, J0:J1, I0:I1], 0.0)
    ny = J1 - J0
    res = {q: np.zeros((len(CFGS), ny)) for q in ["KEtot", "KEmean", "SS", "SSmean"]}
    extra = {}
    uo = vo = None
    for ic, c in enumerate(CFGS):
        u = C.read("U", c, k=k, j=RJ, i=RI)
        v = C.read("V", c, k=k, j=RJ, i=RI)
        if k == 0:
            extra[f"U0_{c}"] = u.astype(np.float32); extra[f"V0_{c}"] = v.astype(np.float32)
        ut, vt = to_t(u, v); del u, v
        um, vm = ut.mean(0), vt.mean(0)
        res["KEtot"][ic] = np.sum(0.5 * (ut ** 2 + vt ** 2).mean(0) * w, 1)
        res["KEmean"][ic] = np.sum(0.5 * (um ** 2 + vm ** 2) * w, 1)
        if k in (0, K100):
            extra[f"EKEmap_{c}"] = np.where(wet, 0.5 * ((ut - um) ** 2 + (vt - vm) ** 2).mean(0), np.nan).astype(np.float32)
            uf = C.read("U", c, k=k); vf = C.read("V", c, k=k)
            uft, vft = C.uv_to_t(uf.mean(0), vf.mean(0))
            extra[f"UMfull_{c}"] = np.where(m.tmask[k], uft, np.nan).astype(np.float32)
            extra[f"VMfull_{c}"] = np.where(m.tmask[k], vft, np.nan).astype(np.float32)
        if c == "ORIG":
            uo, vo, umo, vmo = ut, vt, um, vm
        else:
            res["SS"][ic] = np.sum(0.5 * ((uo - ut) ** 2 + (vo - vt) ** 2).mean(0) * w, 1)
            res["SSmean"][ic] = np.sum(0.5 * ((umo - um) ** 2 + (vmo - vm) ** 2) * w, 1)
        del ut, vt
    res["vol"] = w.sum(1)
    print(f"level {k} done in {time.time() - t0:.0f} s", flush=True)
    return k, res, extra


def main():
    os.makedirs(OUT, exist_ok=True)
    nproc = int(os.environ.get("NPROC", 16))
    ny = J1 - J0
    prof = {q: np.zeros((len(CFGS), NZ, ny)) for q in ["KEtot", "KEmean", "SS", "SSmean"]}
    vol = np.zeros((NZ, ny))
    maps = {}
    with Pool(nproc) as p:
        for k, res, extra in p.imap_unordered(level, range(NZ)):
            for q in prof:
                prof[q][:, k] = res[q]
            vol[k] = res["vol"]
            for key, val in extra.items():
                maps[f"{key}_k{k}"] = val
    np.savez(f"{OUT}/energy_profiles.npz", cfgs=np.array(CFGS), box=np.array([J0, J1, I0, I1]),
             lat=m.gphit[J0:J1, I0:I1].mean(1), vol=vol, **prof)
    np.savez(f"{OUT}/energy_maps.npz", cfgs=np.array(CFGS), box=np.array([J0, J1, I0, I1]), K100=K100,
             **{k: v for k, v in maps.items() if not k.startswith(("U0_", "V0_"))})
    np.savez(f"{OUT}/surface_daily_uv.npz", cfgs=np.array(CFGS), box=np.array([J0, J1, I0, I1]),
             **{k: v for k, v in maps.items() if k.startswith(("U0_", "V0_"))})
    print("saved")


if __name__ == "__main__":
    main()
