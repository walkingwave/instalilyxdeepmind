# market: minimal grey-box ODE (`gtlab/ode/market_min.py`)

Data: 4 real runs, 440 ticks (hold_rec 120, pulse40 80, mid40 40, multilevel200 200).
Observables price, volume, depth; controls interest_rate $r \in [0, 0.1]$, transaction_tax $x \in [0, 0.05]$.
Fitted with `scripts/ode_lab.py --system market --family market_min --budget 240 --starts 10 --nfev 60`
(least squares, Cauchy loss, 10 LHS starts, RK4 with 2 substeps). $\sigma_{proxy}$ = (9.54, 8.71, 26.5).

## What the data says

- Volume. $y_0$ is 90-110 in every run, then 63-78 at $t=0$, 2-3 by $t=20$, ~2 forever. Subtracting a floor of 2,
  the ratio between consecutive ticks is 0.69 in all four runs, whatever the controls: a reset backlog of orders
  draining with $\tau_b = 1/\ln(1/0.69) = 2.7$ ticks. The floor wanders 1.4-3.3 with no clean control dependence.
- Depth. At recovery it settles at ~91 (hold_rec: 117 -> 91 with an S-shape, half way at $t \approx 13$).
  Under the pulse (0.1, 0.05) it falls 91 -> 37 in 40 ticks, fast at first (-5/tick) then slow (-0.5/tick);
  after release it refills to 85 in 20 ticks ($\tau \approx 6.5$) then creeps to ~91. Under tax 0.05 the
  plateau is ~22, under (0.02, 0.03) ~45, under (0.05, 0.025) ~50: tax dominates, roughly $91 e^{-28 x}$.
  A single first-order pool cannot give fast-then-slow on the way down and clean-fast on the way up; a sum of a
  fast pool and a slow pool with an asymmetric slow rate does (checked by hand on pulse40 before fitting:
  frac 0.42, $\tau_f = 5$, $\tau_{s,down} = 40$, $\tau_{s,up} \approx 13$ reproduce every printed tick within 2).
- Price. Flat at recovery (hold_rec 93-95, mid40 at (0.05, 0.025) 91.6 -> 90). Under (0.1, 0.05) it moves only
  -2 in 40 ticks, but under (0.09, 0.02) it falls 103 -> 85 with an accelerating slope (-0.25 -> -0.45/tick),
  and after the pulse40 release (controls back to 0) it falls 106 -> 94.7, also accelerating. It stops at ~79 in
  multilevel and stays there for 90 ticks under any control, including 16 ticks of recovery. So: a transaction
  tax freezes the price (no trades, no price moves), a rate pushes it down, the fall feeds on itself (momentum),
  and there is a floor near 79 (producer reservation value).

## Model (5 states, 12 fitted parameters, 3 fixed constants)

States $B$ (backlog), $P$ (price), $m$ (momentum), $D_f$, $D_s$ (fast and slow dealer-capacity pools).

$$\dot B = -B/\tau_b, \qquad \tau_b = 2.7 \text{ (fixed)}$$
$$g(x) = e^{-a_x x}, \qquad s(P) = \mathrm{clip}\big((P - p_{lo})/10,\, 0,\, 1\big)$$
$$\dot P = \big(m - \text{drive}(r)\big)\, g(x)\, s(P), \qquad \dot m = (k_m \dot P - m)/\tau_m, \quad \tau_m = 10 \text{ (fixed)}$$
$$D^* = d_0\, e^{-a_t x - a_r r}, \qquad \dot D_f = (D^* - D_f)/\tau_f$$
$$e = D^* - D_s, \quad w = \tfrac12\big(1 + e/\sqrt{e^2 + 4}\big), \quad \dot D_s = \big(k_{sd} + (k_{su} - k_{sd}) w\big)\, e$$
$$\text{price} = P, \quad \text{volume} = v_0 + B, \quad \text{depth} = \text{frac}\, D_f + (1 - \text{frac}) D_s$$

Reset: $B_0 = \max(\text{volume}_0 - v_0, 0)$, $P_0 = \text{price}_0$, $m_0 = 0$, $D_{f0} = D_{s0} = \text{depth}_0$
(every entry of $y_0$ is used).

Momentum with $k_m > 1$ is locally unstable on purpose (a started fall accelerates, as observed); it is bounded
by the floor $s(P)$ and by the tax factor $g(x)$, and the state clip keeps $|m| \le 50$. All rates are within
0.02-1.5 per tick except the slow pool ($k_{sd} \ge 0.005$).

Two versions of the rate drive were tried:

- v1: $\text{drive}(r) = k_r r$ (linear).
- v2: $\text{drive}(r) = 10\, k_r r^2$ (convex: rate 0.05 barely moves the price, 0.09 moves it fast).

## Results (score per observable = mean of $1/(1+|err|/\sigma_{proxy})$)

Leave-one-run-out (fit on 3 runs, score the 4th), v1 = kept version:

| held out | ode v1 (price, vol, depth) | mean | l0b_lin | mean | persistence | ode v2 mean |
|---|---|---:|---|---:|---:|---:|
| hold_rec | 0.902 0.933 0.944 | 0.926 | 0.835 0.901 0.827 | 0.854 | 0.515 | 0.926 |
| pulse40 | 0.659 0.970 0.909 | 0.846 | 0.715 0.918 0.711 | 0.782 | 0.468 | 0.804 |
| mid40 | 0.831 0.962 0.844 | 0.879 | 0.930 0.893 0.894 | 0.906 | 0.533 | 0.903 |
| multilevel200 | 0.653 0.931 0.779 | 0.788 | 0.530 0.918 0.694 | 0.714 | 0.294 | 0.772 |
| mean | | 0.860 | | 0.814 | 0.453 | 0.851 |

In-sample (v1, full fit on all 4 runs, cost 12.7):

| run | price | volume | depth | mean |
|---|---:|---:|---:|---:|
| hold_rec | 0.902 | 0.950 | 0.947 | 0.933 |
| pulse40 | 0.930 | 0.972 | 0.955 | 0.952 |
| mid40 | 0.838 | 0.962 | 0.865 | 0.888 |
| multilevel200 | 0.932 | 0.933 | 0.922 | 0.929 |
| mean | | | | 0.926 |

Fitted parameters (v1, `plans/market_market_min.json`; none at a bound):

| name | value | bounds | meaning |
|---|---:|---|---|
| $v_0$ | 2.353 | 0.5-6 (log) | base volume flow |
| $d_0$ | 89.48 | 60-130 | depth target at recovery |
| $a_t$ | 23.08 | 2-100 (log) | tax elasticity of depth target ($e^{-a_t x}$: 0.32 at tax 0.05) |
| $a_r$ | 3.65 | 0-30 | rate elasticity of depth target ($e^{-a_r r}$: 0.69 at rate 0.1) |
| $\tau_f$ | 4.59 | 1.5-20 (log) | fast pool time constant |
| $k_{sd}$ | 0.0356 | 0.005-0.3 (log) | slow pool rate when shrinking ($\tau = 28$) |
| $k_{su}$ | 0.0945 | 0.005-0.5 (log) | slow pool rate when refilling ($\tau = 10.6$) |
| frac | 0.247 | 0.05-0.95 | fast-pool share of reported depth |
| $k_r$ | 4.45 | 0.2-60 (log) | rate drive on price (rate 0.1, no tax: -0.44/tick before momentum) |
| $a_x$ | 37.2 | 0-120 | tax freeze ($e^{-a_x x}$: 0.16 at tax 0.05) |
| $k_m$ | 1.393 | 0-4 | momentum gain (> 1: a started fall accelerates, e-folding $\tau_m/(k_m-1) = 25$ ticks) |
| $p_{lo}$ | 73.4 | 20-90 | price floor |

Eval sanity (4,000-tick rollouts on eval-shaped schedules, every category): finite, 0.4-0.5 s each,
27-33% of ticks outside the observed range +-5% (the price sits at the floor 73.4 for most of a long
high-rate hold and the depth sits at ~20 under a long tax; both are below what 440 ticks of data
covered, so "outside" here is extrapolation, not blow-up). Ranges: price 73.4-93.4, volume 2.4-64,
depth 19.6-115.

## Structures tried and rejected

- v2, convex rate drive $10 k_r r^2$: in-sample 0.932 (better on mid40 price, 0.905 vs 0.838) but LOO 0.851
  vs 0.860, pulse40 price fold 0.53 vs 0.66. With one run held out the momentum/floor parameters lean on
  a single run and the steeper drive over-shoots. Rejected on LOO. Files: `plans/market_market_min_v2*.json`.
- Price as an integrator of $-k_r r + k_t x$ with a slow anchor pull (hand check before coding): predicts a
  rise of +0.3/tick under (0.02, 0.03) and +0.5/tick under (0.03, 0.05); the data shows -0.05 and 0. Tax
  does not push the price up, it freezes it. Not coded.
- Momentum without the tax factor (hand check): under the pulse the momentum would self-excite and the
  price would fall during the 40-tick pulse instead of after it. Not coded.
- Sell-pressure pipeline (commitments made under a high rate, executed once the tax is lifted): explains the
  post-release fall in pulse40 but predicts a 20-point fall at the multilevel release ($t = 184$) where the
  price rose +1.9. Not coded.
- Single first-order depth pool: cannot give -5/tick then -0.5/tick on the way down and $\tau = 6.5$ on the
  way up (hand check on pulse40). Two cascaded pools ($\tau = 8, 8$) reproduce the hold_rec S-shape exactly
  but miss the pulse40 decline by 8 at $t = 4$. The fast+asymmetric-slow mixture was kept; it misses hold_rec
  depth by up to 4.5 at $t = 8$ and mid40 depth by 5 (target at (0.05, 0.025) too low).
- $a_r$ on depth, $\tau_b$ free: to stay at 12 parameters $\tau_b$ was fixed at the hand-measured 2.7
  (ratio 0.69 per tick in every run) and $a_r$ kept, because the (0.09, 0.02) segment of multilevel needs a
  rate effect on depth (target 41 with the fitted $a_r$, 56 without; observed 39-48).

## Verdict

Lab verdict: ship (LOO 0.860 vs l0b_lin 0.814, +0.046; above l0b_lin on 3 of 4 folds, below on mid40 by 0.027;
in-sample 0.926; no parameter at a bound; all eval rollouts finite and fast).

What limits it: (1) the price floor and momentum are identified from multilevel alone, so the two price folds
that lean on them (pulse40 0.66, multilevel 0.65) are the weak spots; (2) the volume floor wanders 1.4-3.3 with
no control dependence we could pin down, modelled as a constant 2.35; (3) the depth target at intermediate
controls is off by ~5 (mid40) and the hold_rec S-shaped approach is not a sum of exponentials; (4) the eval
schedules spend a third of their ticks below the data range, so the floor value 73.4 (fit from 90 ticks near 79)
carries a lot of the sustained-category risk; (5) 4 runs, 440 ticks: with p3 composition/order data the
momentum and floor should be re-fit before this replaces l0b_lin in a submission.
