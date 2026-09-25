# power_grid: minimal grey-box ODE (`gtlab/ode/power_grid_min.py`)

Lab notebook, Fri Sep 25. Data: the three real runs (500 ticks): `p1.hold_rec` (120, recovery
hold), `p2.pulse60_120` (60 pulse + 120 recovery), `p2.multilevel200` (4 levels x ~50-60).
`sigma_proxy` = [22.4, 0.71, 0.15] for (load, frequency, renewable_share). Fit = `scripts/ode_lab.py`
(least squares, cauchy loss, 10 LHS starts, 60 nfev, 240 s), leave-one-run-out against `l0b_lin`
fitted on the same folds and against persistence.

## What the data says

- Load under a price step is second order, not first order. Recovery (price 0.8 -> 1.5): load
  106 -> 93 (t=0) -> 71.5 (t=13) -> 105 (t=45) -> 90 (t=105). Pulse (price 0, reserve 150):
  118 -> 140 -> 154 (t=12) -> 110 (t=40) -> 120 (t=55). Period about 54 ticks in both regimes,
  so $\omega \approx 2\pi/54 = 0.116$. Amplitude of the swing is nearly the same (21, 22, 30) for
  demand steps of 18, 21 and 39 power units: the synchronised group saturates.
- The three resets give load $y_0$ = 106, 118, 93 but the pulse trajectories from 118 and 93
  coincide within 5 ticks (154.3 vs 157.7 at t=15). The population is the same every reset; the
  observed $y_0$ is not the state. We therefore use $y_0$ only to seed fast states that forget it
  in 2-3 ticks.
- Segment means give the demand curve: $D(0) \approx 132$, $D(0.18) \approx 120$, $D(0.8) \approx 108$,
  $D(1.5) \approx 93$: roughly $D(p) = d_0 - d_1 p$ with $d_1 \approx 22$-26.
- Frequency is a droop: it moves with the load within one tick, and stays offset for 60 ticks in
  every level of the multilevel run (49.3 at reserve 15.5, 51.2 at reserve 126/ic 0.58, 49.6 at
  reserve 126/ic 0.27). Reserve at 150 lifted the balance by about +85 with ic = 1 and by +27
  with ic = 0.2: the reserve is delivered through the interconnector. The interconnector alone
  (0.2 -> 1 at reserve 0) did nothing visible. No depletion of the reserve was visible over 60
  ticks with charging 0, so no stock state and no charging effect.
- Frequency drops more for a deficit than it rises for an equal surplus (load 64 -> +1.0 Hz, load
  118 -> -0.7 Hz, both at reserve 0): governor output limits. One asymmetry parameter fixes this.
- Renewable share: 0.35-0.44 at reserve 0 / ic 1, 0.31-0.39 at reserve 15.5, 0.05-0.08 at reserve
  126-150 whatever the interconnector (0.05 at ic 0.2, 0.08 at ic 1). It is a function of the reserve
  request (sharp, roughly $1/(1+(R/70)^2)$), weakly of ic, and not of the load (0.44 at load 64
  and 0.36 at load 118 is far from a $1/L$ law).

## Equations (final version)

States $x = (D_s, x_1, x_2, R_s, \delta f, s)$, controls $(p, r, c, i)$ = (price_signal,
reserve_dispatch, charging_allowance, interconnector). Fixed constants: $a_s = 1$, $a_r = 0.6$,
$a_f = 0.5$, $a_{sh} = 1.5$ per tick, $i_0 = 0.15$, $\iota_0 = 0.5$. RK4, 2 substeps per tick.

$$D(p) = d_0 - d_1 p, \qquad g = D(p) - D_s, \qquad L = D_s + x_1$$
$$\dot D_s = a_s g$$
$$\dot x_1 = x_2, \qquad \dot x_2 = -2\zeta\omega x_2 - \omega^2 x_1 + k\, a_s D_{sat} \tanh(g / D_{sat})$$
$$\dot R_s = a_r (r - R_s), \qquad R_{eff} = c_r R_s (i_0 + (1-i_0) i)$$
$$G_0 = D(0.8), \qquad b = G_0 + R_{eff} - L, \qquad \dot{\delta f} = a_f\big(k_f (b - k_{asym} \max(-b, 0)) - \delta f\big)$$
$$s^\star = s_0 (\iota_0 + (1-\iota_0) i) \big/ \big(1 + (R_s/k_c)^2\big), \qquad \dot s = a_{sh}(s^\star - s)$$
$$y = (L,\ 50 + \delta f,\ s)$$

Initial state from $y_0$: $D_s = \text{load}_0$, $\delta f = f_0 - 50$, $s = \text{share}_0$,
$x_1 = x_2 = R_s = 0$. The kick term integrates to $k D_{sat} \int_0^{|\Delta D|/D_{sat}} \tanh(z)/z\, dz$,
which grows only logarithmically once $|\Delta D| > D_{sat}$: that is the saturation the data asks for.
Rebound amplitude $\approx$ impulse$/\omega$.

## Fitted parameters (full fit on all three runs)

| name | value | bounds | role |
|---|---:|---|---|
| d0 | 126.13 | 60-250 | demand at price 0 |
| d1 | 21.15 | 0-80 | demand slope per unit price |
| w | 0.1016 (period 62 ticks) | 0.03-0.4 (log) | thermostat cycle, $2\pi/\omega$ ticks |
| zeta | 0.281 | 0.02-1 (log) | dispersion damping |
| kick | 0.459 | 0-1 | synchronisation impulse gain |
| Dsat | 4.45 | 0.5-100 (log) | saturation scale of the kick |
| kf | 0.0226 | 0.005-0.3 (log) | droop, Hz per power unit |
| k_asym | 2.60 | 0-4 | extra droop in deficit |
| c_r | 0.583 | 0-2 | reserve delivered per unit request at ic=1 |
| s0 | 0.374 | 0.05-0.9 | renewable share at reserve 0, ic 1 |
| k_c | 71.6 | 10-500 (log) | reserve request that halves the share |

At bound: none. State clips (not fitted): $x_1 \in [-32, 32]$ (largest observed rebound 29.4), $\delta f \in [-2.5, 2.5]$ Hz (observed 49.2-51.9), $D_s \in [0, 260]$.

## Leave-one-run-out and in-sample (score per observable: load, frequency, share)

| fold / run | ode (load, freq, share) | mean | l0b_lin | mean | persistence |
|---|---|---:|---|---:|---:|
| LOO hold_rec | 0.797 0.843 0.898 | 0.846 | 0.776 0.745 0.876 | 0.799 | 0.633 |
| LOO pulse60_120 | 0.784 0.730 0.866 | 0.793 | 0.670 0.654 0.874 | 0.733 | 0.564 |
| LOO multilevel200 | 0.816 0.717 0.859 | 0.797 | 0.596 0.425 0.803 | 0.608 | 0.485 |
| **LOO mean** | | **0.812** | | **0.713** | 0.561 |
| in-sample hold_rec | 0.904 0.841 0.899 | 0.882 | | | |
| in-sample pulse60_120 | 0.808 0.780 0.881 | 0.823 | | | |
| in-sample multilevel200 | 0.899 0.739 0.913 | 0.850 | | | |
| **in-sample mean** | | **0.852** | | | |

The ode beats l0b_lin on every fold and on 8 of 9 fold x observable cells (share on the pulse fold
is a tie, 0.866 vs 0.874). Before the state clips the hold_rec fold scored 0.894 on load (LOO mean
0.817); the clip costs that fold 0.10 on load because the fold fit (pulse + multilevel only) leans
on a larger rebound, while the full fit is unchanged (theta moved < 2%).

Eval-shaped 4,000-tick rollouts (finite, 0.3 s each): load stays in [51.8, 158.1] against observed
[63.6, 157.7]; frequency in [47.5, 52.5] against observed [49.2, 51.9]; share in [0.0, 0.4]. The
fraction of ticks outside the observed range +-5% is 0.24 sustained (long deficit holds sit at the
frequency clip), 0.08 order, 0.01 recovery, 0.07 composition. Before the clips the sustained
schedule reached 46.2 Hz and the recovery schedule 172 load.

## Structures tried

| version | load | frequency | share | states / params | LOO mean | in-sample | notes |
|---|---|---|---|---|---:|---:|---|
| v1 | one oscillator, kick saturating | droop with governor state $G$ relaxing to $G_0 + a_L(L - G_0)$ at $\tau_g$, lagged $\delta f$ | filtered | 7 / 12 | 0.817 | 0.845 | fit killed the governor: $a_L = 0$, $\tau_g$ at lower bound, Dsat at lower bound (3) |
| v2 | two oscillators (fast $\omega$, slow $\omega/\rho$), shared kick | algebraic droop, governor state, $G(0)$ from $f_0$ | algebraic | 7 / 13 | 0.812 | 0.851 | second group did not buy anything (0.906 / 0.777 / 0.793 on load vs 0.888 / 0.790 / 0.840); noisy $f_0$ in $G(0)$ hurt the pulse fold |
| v3 | two oscillators | lagged asymmetric droop, no governor | algebraic | 7 / 12 | 0.810 | 0.854 | nothing at bound, but LOO load worse than v1 on two folds |
| v4 (kept) | one oscillator, Dsat bound lowered | lagged asymmetric droop, no governor | filtered | 6 / 11 | 0.817 | 0.851 | same LOO as v1 with two fewer parameters, nothing at bound, best in-sample frequency on every run |

Mini-lab numbers (v2-v4) come from the same protocol as `ode_lab.py` (240 s budget, 10 starts, 60 nfev)
run on scratch copies; only v4 was promoted to `power_grid_min.py` and re-run through the lab
(table above).

Rejected on the data, without a fit: a reserve energy stock refilled at `charging_allowance`
(no depletion visible in 60 ticks at reserve 150 / charging 0); an interconnector supply term
(the 0.2 -> 1 switch at reserve 0 moved nothing); a share law $R_n / L$ (share barely depends on load);
a per-episode demand offset from $y_0$ (the three resets are the same population).

## Verdict

`ode_lab` verdict: **ship** (LOO 0.812 vs l0b_lin 0.713, margin 0.10 > 0.03; in-sample 0.852; load in-sample 0.81-0.90). Fitted document: `plans/power_grid_power_grid_min_doc.json` (theta in `plans/power_grid_power_grid_min.json`; the v1 governor version is kept as `_v1`). Use it as a candidate for the power_grid slot, behind the usual local screen and the public-score check; the sustained category is the one to watch since long deficit holds sit at the frequency clip.

What limits it: (1) dispersion is not exponential; the real rebound loses amplitude fast between
the second and third swing (25 -> 7) and the period shortens (54 -> 45), a linear damped oscillator
cannot do both; (2) the demand curve near price 0 is steeper than linear ($D(0.18) \approx 120$ vs
127 predicted), 3 runs are too few to fit a curvature; (3) frequency at high load with reserve on
is still under-predicted by 0.3-0.5 Hz (the pulse recovery peak at t=110), so the droop is
likely state dependent (governor limits + reserve thermal duration) in a way one asymmetry
parameter only approximates; (4) share at reserve 126 / ic 0.58 is 0.055 observed vs 0.07 modelled.
