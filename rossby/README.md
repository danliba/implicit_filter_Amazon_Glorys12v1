# First-baroclinic Rossby radius, GLORYS12v1 regional T-grid (Jan 1993)

Script: `rossby_radius.py` (run with `/work/bk1450/b383184/conda/envs/parcels_4/bin/python`, ~1-3 min on login node).
Output: `rossby_radius_T.nc` (Ld, c1, Ld_raw, H, glamt, gphit on y=499, x=1260), `rossby_radius_map.png`,
`rossby_radius_zonal_median.png`, `sanity_check.txt`, `run.log`.

Method
1. Monthly mean of daily votemper (potential T) and vosaline (practical S), 31 days.
2. Density: Jackett & McDougall (1995) EOS in NEMO form (theta, SP, depth in m as pressure);
   gsw not available in the env. N^2 at W level k from the two adjacent T parcels both evaluated
   at the W-level depth (locally referenced), divided by the T-point separation (partial bottom
   cell T depth from Zgr `deptht`). Wet points only (k < mbathy); N^2 clipped to >= 1e-8 s^-2.
3. WKB: c1 = (1/pi) * int N dz over 0..H, H = sum of wet e3t (e3t_ps for the bottom level);
   top half cell and bottom partial cell use the nearest W-level N.
   Check: discrete Sturm-Liouville problem w'' + N^2/c^2 w = 0, w=0 at surface and bottom,
   solved on 3000 random deep columns (lowest eigenvalue of a symmetric tridiagonal problem).
4. Ld = min(c1/|f|, sqrt(c1/(2 beta))) (Chelton et al. 1998), f and beta from gphit, R = 6371 km.
5. Columns with H < 1000 m and land are masked (Ld_raw = NaN). c1 is extrapolated into them by
   a Laplacian (harmonic) fill from deep water, Gaussian-smoothed (sigma = 5 cells), and blended
   into the raw field over 10 cells next to the mask. Ld is then recomputed from the filled c1
   with the local f/beta, so the latitude structure stays physical over land and shelves.
   Ld is finite and positive everywhere.
6. Sanity check: zonal medians over 60W-20W, see `sanity_check.txt`.
   Chelton's rossrad.dat could not be found at the web page (no data link, 404), so no comparison.
