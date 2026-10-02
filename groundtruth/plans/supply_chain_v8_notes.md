# supply_chain v8: structural pass at the organizer's per-observable scale (2026-09-26)

Target scale: the per-observable calibrated sigma from `plans/sigma_calibrated.json`
(`sigma_per_obs_fit` = [13.7, 185, 49.0] for shipments, inventory_supplier, inventory_retail).
We also report the single-k sigma [9.0, 98.1, 235]. At the per-observable scale retail is
5x tighter than the single-k says, so retail is the lever. No credits; same 3 runs
(p1.hold_rec 120, p2.pulse200_200 400, p3.hold_mid 450).

## Method

- Fit: Nelder-Mead (adaptive, small initial simplex 0.04-0.08 in the unit-box z coordinates,
  2,500 evaluations) directly on $1 - \overline{\text{score}}$ at the per-observable sigma,
  averaged over runs and observables. Scipy Powell with bounds jumps along whole coordinate
  lines and landed in far basins (kc at 5, supplier filled on the interior hold); the
  Cauchy least-squares fitter from `gtlab/ode/fit.py` did the same. Both rejected.
- LOO: for each run, refit on the other two starting from the full-fit theta (1,500 evals),
  score the held-out run. The start leaks information from the held-out run, equally for
  every version, so we use LOO only to compare versions, not as an absolute number.
- Keep rule: calibrated in-sample up AND LOO not down.

## Error budget of the current best (v7 doc, u011) at the per-observable sigma

| run | shipments | supplier | retail | mean |
|---|---|---|---|---|
| hold_rec | 0.986 | 0.982 | 0.993 | 0.987 |
| pulse200_200 | 0.877 | 0.923 | 0.806 | 0.869 |
| hold_mid | 0.795 | 0.926 | **0.597** | 0.773 |

Mean 0.876 (single-k 0.920). Top leaks: (1) hold_mid retail, (2) pulse retail (hump at
ticks 3-66 and the post-pulse burst 200-260), (3) hold_mid shipments after tick 205.
Refitting v7b unchanged at this scale gives 0.894 (retail on hold_mid 0.697): about 0.02 of
the gap is the fitting scale alone.

## What the data say (sales = arrivals - dR)

- Pulse: sales 16.3 -> 17.0/tick for ticks 5-42 (retail rises 3/tick), then a step to
  ~24/tick at tick 43 with arrivals unchanged at 19.2 (retail 134 -> 0 by tick 66), then
  sales = arrivals once empty. The step is not stock-driven (sales stay 24 at R = 20).
  It comes as the supplier first reads ~350 (ticks 38-43).
- Pulse supplier: 0 until tick 30, then refills at 40-50/tick. Production of
  $e(p_0 + k_c o)$ with a short commitment fits the interior hold ramp (32 shipped by tick 7)
  only if the pulse's 30-tick delay comes from somewhere else: dispatch. Before tick 31 all
  production (~65/tick) is dispatched while the terminal passes 19.2/tick; the conveyor
  fills (~1,500 units); once it is congested, dispatch throttles to the terminal rate and
  the supplier refills. The ~1,100 units that arrive after the pulse are that backlog.
- Interior hold: sales 26.7 -> 36.5/tick while R goes 120 -> 1,100 (≈ 0.01 per unit stock);
  arrivals 34.6 until tick 205, a dip to ~31 (ticks 215-235), then a 29/45 alternation
  (mean 37.4) to the end; supplier leaves 0 at tick 385 (+2.3/tick).
- Brief: congested transport and rework is one of the three listed memory mechanisms.

## Versions (per-observable sigma; in-sample mean over 3 runs; LOO mean)

| version | change | params | in-sample | LOO (rec / pulse / mid) | LOO mean |
|---|---|---|---|---|---|
| v7b refit | v7b equations at this scale | 12 | 0.894 | 0.986 / 0.727 / 0.642 | 0.785 |
| v8 | class-resolved conveyors, class-B dispatch share $mix(1 - w_s S/S_{cap})$ | 13 | 0.897 | - | - |
| **v8b** | v8 + congested transport replaces the receiving-scaled dispatch cap | 13 | **0.903** | 0.981 / 0.869 / 0.641 | **0.830** |
| v8c | v8b + affine terminal cap $g(a_0 + a_1 r)$ + slow commitment $C_3$; tau_c, kcool fixed | 14 | 0.908 | 0.988 / 0.896 / 0.552 | 0.812 |
| v8d | v8c with the class split at the terminal (one conveyor) | 14 | 0.908 | 0.988 / 0.872 / 0.624 | 0.828 |
| v8e | v8d without $C_3$; kcool free | 13 | 0.907 | 0.988 / 0.788 / 0.641 | 0.805 |
| v8f | v8e, switch $\propto (S/S_{cap})^8$, initial retail split in demand shares | 13 | 0.908 | 0.988 / 0.814 / 0.659 | 0.820 |
| v8g | v8f with the proportional cap $g a_0 r$ | 12 | 0.906 | 0.988 / 0.876 / 0.617 | 0.827 |

Kept: **v8b** (`gtlab/ode/supply_chain_v8b.py`, doc `plans/supply_chain_supply_chain_v8b_a_doc.json`).
It is the only change that raises in-sample (+0.009 over the refit, +0.027 over the u011 doc)
and LOO (+0.045). Every later addition gains <= 0.005 in-sample and loses 0.002-0.025 LOO:
the slow commitment stage (v8c) and the affine cap (v8e) are identified by the interior hold
alone and extrapolate badly when it is held out (v8c supplier on the hold_mid fold 0.75;
v8e shipments on the pulse fold 0.72). v8d is the runner-up: in-sample 0.908 (single-k
0.927) with LOO 0.828, 0.002 below v8b, within the noise of the local LOO refits.

## v8b equations (10 states, 13 parameters, RK4, N_SUB = 2)

States $S, C_1, C_2, Q_{1A}, Q_{1B}, Q_{2A}, Q_{2B}, R_A, R_B, W$; controls $o, \ell, mix, e, r, m$.

$$P = e(p_0 + C_2),\quad Q = Q_{1A}+Q_{1B}+Q_{2A}+Q_{2B},\quad
D = \min_\epsilon\big(\min_\epsilon(o,\ d_q (q_m - Q)^+),\ P + 1.5 S\big)$$
$$sh_B = mix\,(1 - w_s S/362),\quad k_q = \frac{k_{q0}}{1 + k_l \ell},\quad
g = 1 - \frac{0.5}{1 + e^{-(W-1)/0.05}},\quad A = \min_\epsilon\big(k_q (Q_{2A}+Q_{2B}),\ g\,a_0 r\big)$$
$$\dot S = P - D\ (S \le 362),\quad \dot C_1 = (k_c o - C_1)/\tau_c,\quad \dot C_2 = (C_1 - C_2)/\tau_c$$
$$\dot Q_{1A} = (1-sh_B) D - k_q Q_{1A},\ \dot Q_{1B} = sh_B D - k_q Q_{1B},\
\dot Q_{2c} = k_q Q_{1c} - A_c,\ A_c = A\,Q_{2c}/(Q_{2A}+Q_{2B})$$
$$\dot R_A = A_A - \min_\epsilon((1-f_b)dem + k_d R_A,\ 1.5 R_A),\quad
\dot R_B = A_B - \min_\epsilon(f_b\,dem + k_d R_B,\ 1.5 R_B)$$
$$\dot W = (1-m)/100 - k_{cool}\, m\, W\ (W \le 1.5)$$

Observed: shipments $= A$, inventory_supplier $= S$, inventory_retail $= R_A + R_B$.
$x_0$: $S = y_1$ (clipped to 362), $R_A = R_B = y_2/2$, $Q_{1A} = Q_{1B} = y_0/2$, rest 0.

Theta: p0 12.07, kc 0.388, tau_c 1.00 (on its lower bound 1.0 to 3 decimals; production
commitment is effectively immediate), qm 1,887, dq 0.0831, kq0 1.165, kl 7.01, a0 53.3,
dem 41.1, fb 0.314, kd 0.00998, kcool 0.0131, ws 0.598. No other bound hit.
Production on the pulse: 1.5 (12.1 + 0.388 x 80) = 64.7; on the interior hold
1.25 (12.1 + 15.5) = 34.5, which is the observed 34.6.

## Scores of v8b vs the u011 doc (v7), in-sample

| run | scale | v7 (u011) ship / sup / ret | v8b ship / sup / ret |
|---|---|---|---|
| hold_rec | per-obs | 0.986 / 0.982 / 0.993 | 0.985 / 0.966 / 0.991 |
| pulse | per-obs | 0.877 / 0.923 / 0.806 | 0.899 / 0.971 / 0.843 |
| hold_mid | per-obs | 0.795 / 0.926 / 0.597 | 0.813 / 0.964 / 0.697 |
| mean | per-obs | 0.876 | **0.903** |
| hold_rec | single-k | 0.979 / 0.968 / 0.998 | 0.979 / 0.942 / 0.998 |
| pulse | single-k | 0.840 / 0.892 / 0.935 | 0.874 / 0.948 / 0.954 |
| hold_mid | single-k | 0.735 / 0.873 / 0.856 | 0.759 / 0.946 / 0.878 |
| mean | single-k | 0.920 | 0.920 |

At the single-k scale v8b only ties the u011 doc (v8d: 0.927); the gain is on retail, which
only the per-observable scale weights. LOO at single-k: v7b refit 0.819, v8b 0.852, v8d 0.851.

## Eval sanity (4,000-tick rollouts, all finite, 0.3-0.8 s each)

Lab eval-shaped schedules, max [shipments, supplier, retail]: sustained [36, 362, 417],
order [67, 362, 715], recovery [80, 362, 509], composition [60, 362, 801]
(v7b: order-category retail 5,670). Constant holds: recovery, pulse, zero-action and low
production end with retail 0-6; the interior-hold action levels at 964 (v7: 1,039); the
extreme corner order 80 / receiving 1.5 / maintenance 1 climbs to 5,194 unclipped (v7:
4,165), which the doc's clip caps at 2,245.

## What still limits it

- hold_mid retail (0.697) after tick 205: arrivals dip to ~31 and then alternate 29/45
  (mean 37.4) while the supplier refills from tick 385. Production must creep up (~34.6 ->
  ~40) against a terminal/transport limit near 37.4; the slow stage and the affine cap that
  model it (v8c) are identified by this run alone and failed LOO. The alternation phase is
  not predictable; at this scale predicting the low level of an alternation scores higher
  than its mean, which pulls the fit away from the mean inflow retail needs.
- pulse retail (0.84): the class switch (sales 16.7 -> 24 at tick 43) is abrupt; our
  class split passes through a ~1,600-unit mixed conveyor and arrives smeared over ~80 ticks.
  Moving the split to the terminal (v8d/v8f) did not lift LOO.
- The post-pulse burst: data release ~12/tick for 30 ticks then stop (a rate-limited
  queue), the model releases exponentially.
- The interior-hold fold (0.64) stays below persistence: without that run the model has not
  seen sales at high stock nor receiving 0.925.
