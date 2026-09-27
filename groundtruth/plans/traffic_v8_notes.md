# traffic: v8 speed-law pass (`gtlab/ode/traffic_v8*.py`)

Sat Sep 26, evening. No credits. Everything is scored at the organizer's scale
($\sigma$ = `plans/sigma_calibrated.json["traffic"]` = 4.01, 3.53, 5.21, 4.60 for flow_a, flow_b,
speed_a, speed_b), score per tick $1/(1+|e|/\sigma)$, mean over observables, then mean over the 5
runs (hold_rec 120, pulse40 80, mid40 40, multilevel200 200, hold_mid 400).
Starting point: `plans/traffic_traffic_min_p3_doc.json`, calibrated in-sample **0.777**.

## Error budget of the starting model

Loss $=\sum_t (1-s_t)$ per observable over each constant-control segment. Top leaks:

| run / segment | controls | loss (fa, fb, sa, sb) | what is wrong |
|---|---|---|---|
| pulse40 [0,40) | sig 0.15, lane 0.65, ramp 1 | 18, 23, 19, 16 | speeds fall too slowly (data 45 -> 24 in 10 ticks, model -> 33) |
| pulse40 [40,80) | recovery | 26, 19, 20, 11 | speed_a sits at 11 for 22 ticks in the data, model recovers at once |
| multilevel [76,109) | sig 0.61, ramp 0.9 | 8, 17, 15, 11 | speeds 4.6 / 2.4 too low |
| multilevel [48,70), [151,161) | sig 0.1 / 0.9, ramp 0 | speed of the starved route +7 / +8 | with no arrivals the starved route still slows in the data |
| hold_mid | interior hold | 16, 49, 12, 47 | small biases over 400 ticks; start overshoot 44 missed |

Flows are bursty exit counts (0 / 29 / 38 under the pulse), so their loss is mostly irreducible.
The fixable part is the speed law: it reacts too slowly on the way down, has no memory on the way
up, and ignores which route the signal is starving.

Refitting the unchanged traffic_min directly on the calibrated score (Powell on the score itself,
bounded, from the p3 theta) already gives 0.791: the p3 fit used least squares with a Cauchy loss,
a close but not identical objective. All numbers below use the same direct fit so the comparison
is structural.

## Structures tried (all keep the traffic_min flow pipeline and its 8 flow/speed states)

Let $n_{e,i}$ be the occupancy seen by the speed law, target speed $v^*_i = v_f / (1 + (n_{e,i}/n_{\mathrm{ref}})^\gamma)$,
$\dot s_i = \rho (v^*_i - s_i)$.

| version | change | params | in-sample | LOO | kept |
|---|---|---:|---:|---:|---|
| min (p3 doc) | $n_e = p_1+p_2+p_3$, $\gamma = 1$ | 12 | 0.777 | - | reference |
| min refit | same, fitted on the calibrated score | 12 | 0.791 | 0.746 | baseline |
| v8 (5 extras) | + $\gamma$, approach weight $w_1$, speed floor $v_{\min}$, falling-rate factor, route-B capacity factor | 17 | 0.803 | - | no: $v_{\min} \to 0$, falling factor 0.96, B factor 1.04 (unused) |
| v8 | $n_e = w_1 p_1 + w_2 p_2 + p_3$, $\gamma$ | 15 | 0.804 | 0.785 | no: $w_2 = 0.95$ (unused) |
| v8 (lab least squares) | same 15 parameters, `ode_lab.py` Cauchy least squares at sigma-cal 1.0, 10 starts | 15 | 0.791 | - | no: $w_2 \to 0$, lanes at bounds, sustained speeds down to 7 |
| v8b | lagged occupancy $\dot m_i = r_m (w_1 p_{1i} + p_{2i} + p_{3i} - m_i)$, $v^*$ uses $m_i$ | 15 | 0.811 | 0.802 | superseded |
| v8c | approach vehicles counted by queueing delay: $w_1 p_{1i}/(2\phi_i)$ | 15 | 0.814 | 0.810 | yes (15-parameter fallback) |
| v8d | softer delay weight $w_1 p_{1i} (2\phi_i)^{-\kappa}$, $\kappa = 0.77$ | 16 | **0.817** | **0.812** | **yes: best** |
| v8e | v8c with lane sensitivities up to 1.5 (exit shut at $u_{\mathrm{lane}} \ge 1/\ell$, floor 2%) and $w_{\mathrm{sig}}$ up to 2 | 15 | 0.816 | 0.809 | no: LOO drops vs v8c ($\ell_b$ = 1.12) |
| v8f | v8d + v8e bounds | 16 | 0.816 | 0.783* | no: in-sample below v8d, LOO drops |

\* v8f folds were refit from the start theta with 1,200 evaluations instead of from the full-fit theta,
so its LOO is less optimistic than the others; its in-sample is below v8d anyway.

LOO = leave-one-run-out, refit on 4 runs from the full-fit theta (600 evaluations, same for every
row), held-out run scored at the calibrated $\sigma$. Warm starts make every LOO optimistic by the
same mechanism, so they rank structures but are not absolute.

Why v8c: the signal gives route $i$ junction service $J\phi_i$, so a vehicle in the approach
stage waits about $p_{1i}/(J\phi_i)$ ticks. Counting approach vehicles by that delay makes a route
slow down the moment the signal turns against it, even with ramp 0 (multilevel [151,161): speed_b
43 -> 30 in 3 ticks with nothing admitted). The lag state $m$ gives the second time constant the
p3 notes asked for: speeds keep falling for ~6 ticks after the ramp closes (multilevel [48,58))
and stay low while the approach queue drains after the pulse.

## Equations (v8c, 10 states, 15 parameters)

Flow part unchanged from traffic_min (admission $\tfrac12 d_0 u_{\mathrm{ramp}} \max(0, 1-n_i/n_{\max})$,
three stages at rate $r$, junction $\operatorname{smin}(r p_{1i}, J\phi_i)$, exits
$\operatorname{smin}(r p_{3i}, C(1-\ell_i u_{\mathrm{lane}})(1+\kappa_c u_{\mathrm{clr}}))$). Speed part:

$$\phi_a = \operatorname{clip}(0.5 + w_{\mathrm{sig}}(u_{\mathrm{sig}} - 0.5), 0.02, 0.98),\quad \phi_b = 1-\phi_a$$
$$\dot m_i = r_m\left(\frac{w_1 p_{1i}}{2\phi_i} + p_{2i} + p_{3i} - m_i\right),\qquad
  \dot s_i = \rho\left(\frac{v_f}{1 + (m_i/n_{\mathrm{ref}})^{\gamma}} - s_i\right)$$
$$x_0 = (0,\dots,0,\ \operatorname{clip}(s_{a,0},1,80),\ \operatorname{clip}(s_{b,0},1,80),\ 0, 0)$$

Fitted (v8c): $d_0$ 50.67, $w_{\mathrm{sig}}$ 0.982, $r$ 0.283, $J$ 36.18, $C$ 13.44, $\ell_a$ 0.214,
$\ell_b$ 1.000 (bound), $\kappa_c$ 0.686, $v_f$ 48.88, $n_{\mathrm{ref}}$ 210.4, $\rho$ 0.301,
$n_{\max}$ 785.6, $\gamma$ 1.046, $w_1$ 0.523, $r_m$ 0.352.
Doc: `plans/traffic_traffic_v8c_v8c_doc.json`.

v8d (best) replaces $1/(2\phi_i)$ by $(2\phi_i)^{-\kappa}$. Fitted: $d_0$ 51.64, $w_{\mathrm{sig}}$ 1.001,
$r$ 0.290, $J$ 34.86, $C$ 13.21, $\ell_a$ 0.117, $\ell_b$ 0.989, $\kappa_c$ 0.708, $v_f$ 48.88,
$n_{\mathrm{ref}}$ 194.4, $
ho$ 0.295, $n_{\max}$ 715.2, $\gamma$ 1.166, $w_1$ 0.532, $r_m$ 0.401,
$\kappa$ 0.768. No parameter at a bound. Doc: `plans/traffic_traffic_v8d_v8d_doc.json`.

## Leave-one-run-out per held-out run (mean over observables)

| held out | min refit | v8b | v8c | v8d |
|---|---:|---:|---:|---:|
| hold_rec | 0.971 | 0.986 | 0.991 | 0.992 |
| pulse40 | 0.531 | 0.586 | 0.566 | 0.567 |
| mid40 | 0.819 | 0.864 | 0.865 | 0.868 |
| multilevel200 | 0.650 | 0.643 | 0.681 | 0.681 |
| hold_mid | 0.758 | 0.932 | 0.947 | 0.951 |
| **mean** | **0.746** | **0.802** | **0.810** | **0.812** |

The biggest out-of-sample gain is on hold_mid (0.758 -> 0.951): with the old speed law, a fit
without the interior hold put its speed_a at 0.42; the lagged, delay-weighted occupancy carries
the interior level from the other runs.

## Per run, calibrated in-sample (flow_a, flow_b, speed_a, speed_b)

| run | min p3 doc | v8c | v8d |
|---|---|---|---|
| hold_rec | 1.000, 1.000, 0.933, 0.926 (0.965) | 1.000, 1.000, 0.983, 0.983 (0.991) | 1.000, 1.000, 0.984, 0.983 (0.992) |
| pulse40 | 0.451, 0.479, 0.514, 0.663 (0.527) | 0.487, 0.523, 0.669, 0.630 (0.577) | 0.483, 0.526, 0.673, 0.653 (0.584) |
| mid40 | 0.825, 0.753, 0.828, 0.832 (0.810) | 0.828, 0.792, 0.944, 0.913 (0.869) | 0.820, 0.793, 0.952, 0.918 (0.871) |
| multilevel200 | 0.662, 0.580, 0.699, 0.701 (0.660) | 0.666, 0.600, 0.751, 0.713 (0.682) | 0.664, 0.601, 0.754, 0.709 (0.682) |
| hold_mid | 0.961, 0.879, 0.970, 0.882 (0.923) | 0.889, 0.958, 0.978, 0.979 (0.951) | 0.896, 0.967, 0.978, 0.980 (0.955) |
| **mean** | **0.777** | **0.814** | **0.817** |

## Eval-shaped sanity (4,000 ticks, `scripts/ode_lab.py` schedules)

All finite, < 1.1 s per rollout. Fraction of ticks outside the observed range $\pm 5\%$
(sustained / order / recovery / composition): p3 doc 0.30 / 0.22 / 0.00 / 0.02 (speed min 8.1);
v8c 0.31 / 0.10 / 0.00 / 0.02 (speed min 7.3 / 12.3); v8d 0.31 / 0.11 / 0.00 / 0.02 (7.6 / 11.3).
v8b is worse here (speed min 4.3: $\gamma = 1.26$ with a large buffer), a second reason to prefer v8c.
The sustained 0.31 is the same as the shipped model's: long harsh holds drive both routes to their
buffers; nothing in our data says what speed they settle at below 11.

## What still leaks

- Pulse flows (loss ~40 per observable per 40 ticks): bursty exits, irreducible for a smooth model.
- After the pulse speed_a is predicted to recover ~10 ticks early (+7 mean residual on [40,80)).
- Route B stuck at signal 0.9 with lane 0.75 (multilevel [121,161)): data speed_b holds at 34 with
  zero exits; the model lets B drain through its 10% junction share (+8 to +10 on speed_b).
- hold_mid route split: data flow_a 10.2 vs flow_b 10.8; the model splits admission equally.
- Toll and freight priority are still unused.
