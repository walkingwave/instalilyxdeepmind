# traffic: minimal grey-box ODE (`gtlab/ode/traffic_min.py`)

Fri Sep 25. Fitted on the 4 real runs (440 ticks: hold_rec 120, pulse40 80, mid40 40,
multilevel200 200) with `scripts/ode_lab.py --system traffic --family traffic_min`
(least squares, cauchy loss, 10 Latin-hypercube starts, 60 nfev, leave-one-run-out).

## What the data says

- Flows are bursty exit counts (0 / 7 / 22 / 41 in consecutive ticks); speeds are smooth.
  `sigma_proxy` = (9.1, 7.5, 12.1, 10.8), so speeds carry most of the score.
- Flows are exactly 0 at $t=0$ in every run whatever the initial value (35-42): roads start
  empty. The initial flow values are noise; the initial speeds are real and relax to the free
  speed 48.9 with $\tau \approx 3$-4 ticks under recovery.
- Exit flow lags admission by ~12 ticks, on and off: mid40 (ramp 0.5) shows 0 at $t=8$,
  6.8 at $t=12$, 10.5 from $t=16$; in multilevel the switch to ramp 0 at $t=48$ keeps flow at
  14.5 until $t=56$ and drops it at $t=60$.
- Under the pulse (signal 0.15, lane 0.65, ramp 1) route A exits ~12/tick and B ~30/tick,
  speed_a falls to 11 and stays there 20 ticks into the recovery while A drains in bursts up
  to 41; B recovers first. With signal 0.9, lane 0.75, ramp 0 ($t=121$-140) both flows are 0
  but speed_b sits at 34 while speed_a is back at 48.9: vehicles left on B cannot cross.
  A single lane-closure effect cannot give both; signal timing as the share of junction
  service to route A does (A starved under the pulse, B starved at 0.9).
- Toll had no measurable effect on totals (a demand term $e^{-k\,\mathrm{toll}}$ fitted
  $k \to 0$): the brief says toll changes the mix, not the volume. Ramp 0 gives zero admission.

## Equations (8 states, 12 parameters)

Per route $i \in \{a, b\}$ three first-order stages $p_{1i}, p_{2i}, p_{3i}$ (approach,
crossing, exit) and a reported speed $s_i$. Smooth minimum
$\operatorname{smin}(x, c) = x / (1 + (x/c)^4)^{1/4}$, $n_i = p_{1i} + p_{2i} + p_{3i}$.

$$a_i = \tfrac12\, d_0\, u_{\mathrm{ramp}}\, \max(0, 1 - n_i / n_{\max})$$
$$\phi_a = \operatorname{clip}(0.5 + w_{\mathrm{sig}} (u_{\mathrm{sig}} - 0.5), 0.02, 0.98),\quad \phi_b = 1 - \phi_a$$
$$j_i = \operatorname{smin}(r\, p_{1i},\; J \phi_i)$$
$$c_i = C\,(1 - \ell_i\, u_{\mathrm{lane}})(1 + \kappa\, u_{\mathrm{clr}}),\quad o_i = \operatorname{smin}(r\, p_{3i},\; c_i)$$
$$\dot p_{1i} = a_i - j_i,\quad \dot p_{2i} = j_i - r\, p_{2i},\quad \dot p_{3i} = r\, p_{2i} - o_i$$
$$\dot s_i = \rho \left( \frac{v_f}{1 + n_i / n_{\mathrm{ref}}} - s_i \right)$$
$$y = (o_a, o_b, s_a, s_b),\qquad x_0 = (0, 0, 0, 0, 0, 0, \operatorname{clip}(s_{a,0}, 1, 80), \operatorname{clip}(s_{b,0}, 1, 80))$$

RK4 with 2 substeps per tick. Freight priority and toll are read but unused.

## Parameters (full fit on all 4 runs, v2)

| name | meaning | init | bounds | fitted |
|---|---|---:|---|---:|
| dem0 | admitted demand at ramp 1 | 40 | 5-300 (log) | 43.35 |
| w_sig | junction share slope on signal timing | 1.0 | 0-1.5 | 0.997 |
| r_pipe | stage rate (mean transit 3/r) | 0.25 | 0.05-1.5 (log) | 0.252 |
| junc | junction service, total | 40 | 3-600 (log) | 32.02 |
| cap | exit capacity per route | 15 | 3-300 (log) | 11.51 |
| lc_a | lane-closure sensitivity, exit A | 0.3 | 0-1 | 1.178e-08 |
| lc_b | lane-closure sensitivity, exit B | 0.7 | 0-1 | 1 |
| clr | clearance boost of exit capacity | 0.8 | 0-3 | 0.8575 |
| v_free | free speed | 48.9 | 30-70 | 49.39 |
| n_ref | occupancy that halves speed | 200 | 10-5000 (log) | 206 |
| rho | reported-speed relaxation rate | 0.25 | 0.02-1.5 (log) | 0.1813 |
| n_max | approach buffer (rejects arrivals) | 800 | 50-20000 (log) | 1101 |

At bound: lc_b. Cost 265.5.

## Leave-one-run-out and in-sample (v2)

| held out | ode (flow_a, flow_b, speed_a, speed_b) | mean | l0b_lin mean | persistence mean |
|---|---|---:|---:|---:|
| p1.hold_rec | 1.000, 1.000, 0.881, 0.870 | 0.938 | 0.624 | 0.375 |
| p2.pulse40 | 0.618, 0.601, 0.660, 0.741 | 0.655 | 0.607 | 0.328 |
| p2.mid40 | 0.869, 0.817, 0.778, 0.786 | 0.812 | 0.651 | 0.497 |
| p2.multilevel200 | 0.753, 0.712, 0.777, 0.746 | 0.747 | 0.624 | 0.428 |
| **mean** | | **0.788** | **0.626** | |


In-sample: p1.hold_rec 0.978 (1.00, 1.00, 0.96, 0.96); p2.pulse40 0.699 (0.63, 0.64, 0.72, 0.80); p2.mid40 0.833 (0.87, 0.82, 0.81, 0.83); p2.multilevel200 0.782 (0.78, 0.72, 0.82, 0.82); mean 0.823

Eval-shaped 4,000-tick rollouts (finite, seconds, fraction of ticks outside the observed range
$\pm 5\%$): sustained: finite=True, 0.4 s, outside 0.07, speeds min 9.5/9.6; order: finite=True, 0.4 s, outside 0.13, speeds min 10.2/9.8; recovery: finite=True, 0.5 s, outside 0.00, speeds min 17.6/17.4; composition: finite=True, 0.5 s, outside 0.01, speeds min 9.6/9.3

Verdict: **ship** (LOO ode 0.788 vs l0b_lin 0.626, in-sample 0.823)

## Structures tried

| version | change | LOO ode / l0b_lin | in-sample | eval outside (sust/order/rec/comp) | kept |
|---|---|---|---|---|---|
| v1 | 3-stage pipeline per route, admission split by signal, exit cap with tanh limit, toll factor, speed exponent | 0.778 / 0.626 (hold 0.967, pulse 0.602, mid 0.787, multi 0.754) | 0.824 | 0.30 / 0.38 / 0.00 / 0.04 | no |
| v2 | signal = junction service share, equal admission, smooth-min limits, finite approach buffer, toll and speed exponent removed | 0.788 / 0.626 (hold_rec 0.938, pulse40 0.655, mid40 0.812, multilevel200 0.747) | 0.823 | 0.07 / 0.13 / 0.00 / 0.01 | yes |

v1 rejected: with no arrival rejection, any sustained hold with admission above exit capacity
grows the queue without bound and speeds fall to ~0.3 (30-38% of eval ticks outside the observed
range); the real speeds never go below ~11. Its toll factor fitted to 0.04 and the speed exponent
to 1.07, so both were dropped. Its signal-as-admission-split came out negative ($w = -0.35$),
i.e. it was using the signal to put more vehicles on the starved route, a proxy for the junction
share that v2 models directly.

| v3 | lane closure moves from exit capacity to the speed law ($n_{\mathrm{ref}}(1-\ell_i u_{\mathrm{lane}})$), rejected arrivals divert to the other route | quick fit only (90 s, 4 starts): cost 272.9 vs 265.5 for v2, no parameter at bound, pulse speeds 0.79/0.79 vs 0.72/0.80 | - | - | no (full LOO not run) |

## What limits it

- The pulse run is the weak fold (LOO 0.655): route B's exits under the pulse (30-38/tick) are
  under-predicted because the same exit capacity $C$ serves both routes; per-route capacities
  would cost a parameter we do not have data to identify (lc_b sits at its bound 1.0).
- Flows are bursty event counts, so their score is capped near 0.6-0.8 whatever the model;
  the gain over l0b_lin comes from speeds (0.66-0.88 vs 0.40-0.62).
- Speed has one relaxation rate ($\rho = 0.18$); the data shows a fast drop when the junction
  loads and a slow, queue-driven decline and recovery. A second speed time constant or a
  journey-time memory would need more states than the 8 allowed.
- Freight priority and toll are unused: no run varied them independently of the rest.
- Sustained holds above capacity fill the approach buffer ($n_{\max} = 1100$); speeds bottom
  at ~9.5, below the 11 seen in the data, so the sustained category is the least certain
  (7-13% of eval ticks outside the observed range).
