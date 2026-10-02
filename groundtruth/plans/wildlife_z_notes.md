# wildlife z: long-hold levels and hunting effort (`gtlab/ode/wildlife_z*.py`)

Date: 2026-09-27/28. Data: the same 4 runs as x13 (`p1.hold_rec` 120, `p2.pulse200_200` 400, `p3.compose` 291,
`p4.corridor_habitat` 300; 1,111 ticks). No credits spent, nothing uploaded.
Judge: `scripts/ode_lab.py --sigma-cal 1.0 --budget 300 --starts 8 --nfev 60` (Cauchy least squares and
scoring at the calibrated organizer sigma $\sigma = (9.28, 0.187, 7.66, 0.253)$), no polishing, pair AB.
Leave-one-run-out (LOO) folds get 150 s each; fold noise ≈ 0.02. Rule (MATH_LOG §32): a candidate must not
lose the pulse/recovery-shaped fold `p2.pulse200_200` against x13 (0.714).

## 1. The question

Public wildlife (x13, u013/u014): 0.724 = sustained 0.655, sequence 0.748. Sustained holds last thousands of
ticks; our runs are at most 400. The long-hold audit (`plans/longhold_audit.md`) says: if our model were the
truth, persistence would score $\hat s_{pers} = \text{mean}\,1/(1+|y_0 - P_t|/\sigma) \approx 0.11$ on our
sustained-style schedules, but persistence actually scored **0.191**. Reading: the predictor moves further from
$y_0$ on long holds than the truth. The brief hint: make long-hold levels less extreme (refuge that grows as
prey fall, predator floor, hunting effort that falls with scarcity).

Tools (scratch, not in the repo): `lh.py` = the audit's persistence check on the same 20 schedules (seeds
7000-7019, $y_0$ cycled over our four runs) plus 11 constant 4,000-tick holds from the typical $y_0 = (82.7, 11.4,
70.0, 10.5)$; `tc.py` = the audit's truth-consistency check (candidate as truth, implied sustained score of
every scored upload vs its public band; statistics over the six independent predictors u001-u010b);
`seg.py` = level at the end of every constant-control segment, data vs model.

## 2. What the data say about long-hold depth

Segment ends (mean of the last 10 ticks), x13 vs data:

| segment (hunt, hab, cor) | ticks | data prey N / S | x13 prey N / S | x13 too |
|---|---|---|---|---|
| pulse (7, .1, 1) | 200 | 7.0 / 7.6 (flat from t = 60) | 9.4 / 7.9 | mild |
| habitat .24 (p3) | 45 | 77.0 / 73.2 | 88.3 / 77.0 | mild |
| habitat .5 (p4) | 100 | 90.9 / 78.8 | 98.2 / 80.9 | mild |
| corridor 1 (p4) | 60 | 111.4 / 77.6 | 113.3 / 84.6 | mild |
| hunting 5.95 at hab 1 (p3) | 45 | 67.1 / 28.2 (falling) | 63.1 / 40.2 | mild (south) |
| joint (5.95, .24, .85) (p3) | 45 | 21.5 / 15.5 | 17.5 / 14.5 | deep |
| recovery (0, 1, 0) | 120+ | 121.2 / 96.9 | 120.4 / 96.0 | right |

On every regime we have observed for 45+ ticks except one, x13's prey level is **less** extreme than the data,
not more. So the data do not support a shallower long-hold map in the regimes we can see.

Per level kind of our sustained schedules, the persistence-if-truth score (mean of the four observables) under
the recommended model z9: recovery 0.208, uniform 0.113, mid 0.108, corner 0.083, pulse 0.068. Predators fall
from 9-12 to 1.4-2.9 under every control we have seen (observed in all four runs), so their component is
≈ 0.025 whatever the model. Even a truth that sat at the recovery level all the time would give only 0.208.
To reach 0.191 the organizer's sustained holds would have to sit at the recovery action ~85 % of the time, or
start nearer the held level than our resets do. **The 0.191 band is a statement about the organizer's schedule
mix and initial states, not about prey depth**; no data-consistent structure we tried moves the check by more
than ±0.01 (table §3).

## 3. Candidates (all start from x13 = `wildlife_z` with no switch; one or two switches each)

All modules come from one template; switches (module constant `FLAGS`):

| switch | change (per region $i$) | new parameter |
|---|---|---|
| c0 | base refuge: $K_i = s_i(c_0 + c_h u_p)$ (shelter without protection) | $c_0$ |
| n | refuge sharpness: $E = P\,z/(1+z)$, $z = (P/K)^{n}$ ($n = 1$ in x13) | $n_R$ |
| g | quota power: take $\propto 7\,(q/7)^{g}$ | $g_q$ |
| eff | hunting effort with memory: $\dot\phi = (P/(P+P_e) - \phi)/t_e$ | $t_e, P_e$ |
| pt | predators follow prey: predator journeys $\propto e_m P/(P+P_t)$ | $P_t$ |
| cg / hg | corridor / habitat act through $u_c^{c_g}$ / $u_p^{h_g}$ | $c_g$ / $h_g$ |
| **hs** | **hunting effort falls with scarcity: take $\times\,\phi$, $\phi = P/(P+P_e)$** | $P_e$ |
| hsE | the same on exposed prey, $\phi = E/(E+P_e)$ | $P_e$ |
| bh2 | south renewal loss $b_{h,S} = x_b\,b_h$ | $x_b$ |

Results (LOO folds p1 p2 p3 p4 at 1.0σ; long holds are 4,000-tick levels from the typical $y_0$, prey N / S;
pers = persistence-if-truth on the 20 schedules, actual band 0.191; TC = RMSE of implied vs public sustained bands):

| family | switches | LOO folds | LOO | in-sample | pulse N / pred | mid (4,.5,.5) | (6,1,0) | (0,0,0) | (8,0,1) N | pers | TC rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x13 = z | – | .683 .714 .614 .602 | 0.653 | 0.734 | 9.4 / 1.41 | 46 / 32 | 48 / 37 | 63 / 58 | 0.5 | 0.110 | 0.064 |
| z2 | c0 | .683 **.634** .614 .602 | 0.633 | 0.737 | 10.5 / 1.41 | 30 / 22 | 32 / 26 | 66 / 57 | 7.5 | 0.103 | 0.079 |
| z3 | n | .692 **.698** .611 .597 | 0.650 | 0.731 | 9.3 / 1.41 | 42 / 29 | 34 / 27 | 53 / 53 | 0.4 | 0.103 | 0.067 |
| z4 | g ($g_q$ → 0.3 bound) | .682 .706 .614 .605 | 0.652 | 0.734 | 9.4 / 1.41 | 31 / 25 | 47 / 37 | 59 / 57 | 0.4 | 0.101 | 0.088 |
| z5 | eff ($t_e$ → 2 bound) | .687 .720 .602 .605 | 0.653 | 0.740 | 8.6 / 1.39 | 57 / 42 | 42 / 30 | 52 / 52 | 0.6 | 0.113 | 0.075 |
| z6 | pt ($P_t$ → 0.85) | .685 .727 .601 .599 | 0.653 | 0.734 | 9.4 / 1.42 | 46 / 32 | 48 / 37 | 64 / 59 | 0.5 | 0.110 | |
| z7 | cg ($c_g$ = 1.26) | .677 **.697** .617 .619 | 0.652 | 0.734 | 9.4 / 1.40 | 47 / 33 | 48 / 37 | 64 / 58 | 0.5 | 0.110 | |
| z8 | hg | .698 **.646** .614 .600 | 0.640 | 0.735 | 10.2 / 1.41 | 37 / 25 | 34 / 27 | 0 / 17 | 0.0 | 0.094 | |
| **z9** | **hs** | **.704 .749 .643 .616** | **0.678** | **0.741** | 8.9 / 1.39 | 54 / 38 | 38 / 30 | 50 / 51 | 0.8 | 0.108 | 0.073 |
| z9 cold | hs, started from x13's theta | .704 .749 .643 .616 | 0.678 | 0.741 | same | | | | | | |
| z10 | hs + c0 ($c_0$ → 0) | .704 **.635** .666 .616 | 0.655 | 0.741 | 8.9 / 1.39 | 54 / 38 | 38 / 30 | 50 / 51 | 0.8 | 0.108 | |
| z11 | hs + cg | .698 .731 .643 .640 | 0.678 | 0.741 | 8.8 / 1.37 | 55 / 39 | 38 / 29 | 51 / 51 | 0.8 | 0.110 | 0.073 |
| z12 | hs + pt ($P_t$ → 0.5 bound) | .705 .749 .623 .615 | 0.673 | 0.741 | 8.9 / 1.40 | 53 / 37 | 38 / 30 | 51 / 52 | 0.9 | 0.108 | |
| z13 | hsE ($P_e$ → 0.5 bound = x13) | .683 .703 .615 .601 | 0.651 | 0.733 | 9.4 / 1.41 | 46 / 32 | 48 / 37 | 64 / 58 | 0.7 | 0.109 | |
| z14 | hs + n | .706 .705 .636 .611 | 0.664 | 0.741 | | | | | | | |
| z15 | hs + g ($g_q$ = 2.3) | .697 .673 .643 .622 | 0.659 | 0.742 | | | | | | | |
| z16 | hs + cg + pt | .698 .719 .623 .643 | 0.671 | 0.742 | | | | | | | |
| z17 | hs + bh2 ($x_b$ = 0.81) | .704 .747 .651 .617 | 0.680 | 0.740 | | | | | | | |

Bold fold = below x13's pulse fold by more than the noise (rejected by the rule).

Readings:
1. Every "shallower refuge" idea (c0, n, hg) makes the fitter trade the pulse floor for the partial-habitat
   levels and loses the held-out pulse run by 0.02-0.08. The base refuge is not identified: $c_0$ fits to 0 in
   z10's full fit and to 192 in z2's (with $c_h$ at its 1,000 bound), and in the fold without the pulse run it
   raises the floor and that fold loses 0.08 in both. The data give no support for it.
2. The quota power went to its lower bound (take nearly independent of the quota): the opposite of a
   "less extreme" map. Settlement-style and corridor variants (pt, cg) change nothing beyond noise.
3. **One change moves every fold: hunting effort falls with scarcity (hs).** LOO 0.653 → 0.678, all four folds up
   (p1 +0.021, p2 +0.035, p3 +0.029, p4 +0.014), in-sample 0.734 → 0.741. The dynamic version (z5) fitted
   $t_e$ to its 2-tick lower bound, i.e. the effort adjusts at once, which is z9's static form with one
   parameter fewer. The same switch on exposed prey (hsE) collapses back to x13: the effort follows the whole
   population, not the exposed part.
4. Stacking anything on hs (z11-z17) stays within ±0.005 of 0.678 or loses the pulse fold. z17 (+0.002, one more
   parameter) is noise.

## 4. The recommended model: z9 (`gtlab/ode/wildlife_z9.py`, 16 states, 23 parameters)

x13 (`plans/wildlife_x_notes.md` §5) with one change in the harvest. Per region $i$ ($s_N = 1$, $s_S = k_s$):

$$K_i = s_i\,c_h\,u_p,\qquad E_i = \frac{P_i^2}{K_i + P_i},\qquad
\text{harvest}_i = H\,u_h\,\underbrace{\frac{P_i}{P_i + P_e}}_{\text{effort}}\,\frac{E_i^2}{E_i^2 + P_h^2}$$

Everything else as x13: nursery $\dot J = rP - m_J J$, recruitment $m_J J/(1 + J/(c_J s_i R))$; exposure death
$d_{h,i}(1-u_p)E$; food $\dot R = w(1 - b_h(1-u_p))(r_l + (1-r_l)R)(1-R) - k_R (P/s_i) R$; predators
$\dot Q = c_q(q_0 + q_1 G - Q)\,Q/(Q+4) - e_m u_c Q + s_v V_{in}/\tau$ with $G$ lagging $E/(E + 100 s_i)$ by $t_z$;
transit pools with survival $s_v$. (The module carries two unused effort states, constant at 1.)

Why it helps: with a fixed quota the take stays near $H u_h$ until the prey are almost gone, so the model
either over-harvests medium stocks or under-harvests large ones. With the catch per trip falling as
$P/(P+P_e)$, $P_e = 56$, the take at a full stock (P ≈ 85) is 60 % of the request and falls in proportion
below ~50: the pulse decline, the hunting-at-full-habitat decline and the joint-stress floor are fitted by one
law. The shelter capacity grows to $c_h = 424$ (x13 253) and the Hill floor $P_h$ drops to 1.1 (x13 5.6):
effort now does the flattening that the floor did.

Fitted theta (`plans/wildlife_wildlife_z9_a.json`; $m_J$ at its upper bound 5):

| $r$ | $d$ | $k_s$ | $H$ | $P_h$ | $e_m$ | $e_{mS}$ | $\tau$ | $s_v$ | $c_J$ | $m_J$ | $P_e$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.334 | 0.151 | 0.798 | 2.238 | 1.13 | 0.0303 | 0.0451 | 27.1 | 0.793 | 17.97 | 5.0 | 55.7 |

| $w$ | $k_R$ | $b_h$ | $r_l$ | $d_{hN}$ | $d_{hS}$ | $c_h$ | $q_0$ | $q_1$ | $c_q$ | $t_z$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0231 | 2.26e-4 | 0.440 | 0.498 | 0.0857 | 0.0550 | 424 | 1.711 | 3.007 | 0.0929 | 37.5 |

$m_J = 5$ means the nursery passes young through within a tick (competition stays, the delay goes). At
$N_{SUB} = 2$, $m_J\Delta t = 2.5$ is inside RK4's stability limit (2.78); rolling at 2 / 4 / 8 substeps gives
identical scores to the fourth decimal (0.7321, 0.8185, 0.7094, 0.7033 per run), so the relocation trap of
x5/x7 does not apply.

LOO per fold and observable (prey_N, pred_N, prey_S, pred_S), x13 → z9:

| held out | x13 | z9 |
|---|---|---|
| p1.hold_rec | .704 .702 .743 .583 = 0.683 | .696 .733 .738 .651 = **0.704** |
| p2.pulse200_200 | .669 .711 .717 .758 = 0.714 | .709 .717 .799 .771 = **0.749** |
| p3.compose | .568 .619 .507 .763 = 0.614 | .596 .671 .525 .780 = **0.643** |
| p4.corridor_habitat | .685 .573 .611 .537 = 0.602 | .662 .596 .637 .570 = **0.616** |
| mean | 0.653 | **0.678** |

Per observable over folds: prey_N 0.656 → 0.666, pred_N 0.651 → 0.679, prey_S 0.644 → 0.675, pred_S 0.660 → 0.693.

Checks:
- Cold start: refitting z9 from x13's theta (H scaled to 2.74, $P_e$ = 85) instead of the z5 fit gives the same
  four folds and the same theta to three digits: the gain is the structure, not the warm start.
- 4,000-tick lab rollouts on all four categories: finite, 0.5-0.6 s, 0 % of ticks outside the observed range
  ±5 %. Packaged through `gtlab.flatpack.write_folder` into a scratch folder: smoke test passes, `predict()` equals
  the lab doc to 0.0 on a 4,000-tick mixed schedule, 0.57 s per episode.
- Long holds (4,000 ticks, typical $y_0$), prey N / pred N / prey S / pred S:

| hold (hunt, hab, cor) | x13 | z9 | observed |
|---|---|---|---|
| recovery (0, 1, 0) | 120.4 / 2.36 / 96.0 / 2.36 | 120.3 / 2.34 / 95.9 / 2.34 | 121.2 / 2.35 / 96.9 / 2.34 |
| pulse (7, .1, 1) | 9.4 / 1.41 / 7.9 / 1.41 | 8.9 / 1.39 / 7.4 / 1.39 | 7.0 / 1.62 / 7.6 / 1.65 |
| habitat .5 | 98.5 / 2.41 / 81.1 / 2.44 | 98.4 / 2.43 / 81.0 / 2.46 | 91.0 / 2.44 / 78.7 / 2.34 (100 ticks) |
| corridor 1 | 116.9 / 1.92 / 87.9 / 1.90 | 117.3 / 1.91 / 87.7 / 1.89 | 111.7 / 1.72 / 77.5 / 1.67 (60 ticks) |
| hunting 4 at hab 1 | 72.0 / 2.04 / 50.6 / 1.98 | 72.2 / 2.00 / 51.9 / 1.95 | – |
| hunting 6 at hab 1 | 48.4 / 1.89 / 37.4 / 1.88 | 38.1 / 1.80 / 29.6 / 1.80 | 66 / – / 28 after 45 ticks, falling |
| mid (4, .5, .5) | 46.2 / 1.77 / 32.3 / 1.74 | 53.7 / 1.80 / 38.0 / 1.77 | – |
| (4, .25, 0) | 27.0 / 1.90 / 21.1 / 1.89 | 42.8 / 2.04 / 33.3 / 2.03 | – |
| (0, 0, 0) | 63.2 / 2.61 / 58.2 / 2.69 | 50.2 / 2.72 / 51.1 / 2.88 | 77 / 73 at habitat .24 after 45 ticks |
| (8, 1, 0) | 39.1 / 1.84 / 31.8 / 1.84 | 26.9 / 1.76 / 22.9 / 1.76 | – |
| (8, 0, 1) | 0.45 / 1.37 / 0.45 / 1.37 | 0.78 / 1.37 / 0.80 / 1.38 | never observed |

  z9 is less extreme than x13 where hunting meets partial habitat (mid +8 / +6, (4,.25,0) +16 / +12) and more
  extreme at the corners (full hunting at full habitat, no protection), where the data point the same way
  (hunting-6 south already at 28 and falling; habitat .24 already below both models). Persistence-if-truth 0.108
  (x13 0.110); truth-consistency RMSE 0.073 vs 0.064, rank correlation 0.94 for both: no evidence either way.

## 5. Risks

1. **Habitat 0 with heavy hunting** still drives prey to ≈ 0.8 (x13 0.45; v8h 8). We have no run below habitat
   0.1; every refuge that would stop it ($c_0$) is rejected by the pulse run. Same exposure as the shipped x13.
2. Sustained corners are deeper than x13 ((8,1,0) 27 vs 39, (0,0,0) 50 vs 63). If the 0.191 band did mean
   shallow holds, z9 would lose a little sustained there; the observed segments say the opposite (§2).
3. The p3 final rebound is still too high (141 vs 124) and the p2 overshoot too low (166 vs 206): the size of the
   post-depletion overshoot does not scale with the depletion length the way the data do. Unchanged from x13.
4. Pulse predators 1.39 vs 1.63 observed: unchanged. The predators-follow-prey switch fitted to zero.

## 6. Verdict and forecast

Recommended doc: **`plans/wildlife_wildlife_z9_a_doc.json`** (family `wildlife_z9`, pair AB).
Held-out gain +0.025 (0.653 → 0.678), every fold up, pulse fold +0.035. Forecast by 0.6 × held-out gain:
**+0.015 public, ≈ 0.739** (from 0.724). We expect it mostly in sequence (pulse, hunting and joint-stress
transients); sustained should move little (its levels agree with x13 on the regimes the tests visit most:
recovery, pulse, habitat, corridor).

The long-hold persistence argument does not hold up for wildlife: no data-consistent structure moves the
implied persistence off ≈ 0.11, and the observed segments say x13's levels are too mild, not too deep. The gap
to 0.191 is explained by the schedule mix or the initial states of the organizer's sustained episodes, which
we cannot see. The data that would settle it: a 300-tick hold at hunting 4, habitat .5 (the mid level, where
z9 and x13 differ by 8 / 6 and v8h by 20), and a hold at habitat 0-0.1 with and without hunting (risk 1).

Files: `gtlab/ode/wildlife_z.py` (x13 control), `wildlife_z2.py` ... `wildlife_z17.py` (families
`wildlife_z`, `wildlife_z2` ... `wildlife_z17`); lab reports `plans/wildlife_wildlife_z*_a.json` and
`plans/wildlife_wildlife_z9_cold.json`, docs `..._doc.json`.
