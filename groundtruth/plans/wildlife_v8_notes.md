# wildlife v8: structural fixes at the organizer's scale (`gtlab/ode/wildlife_v8h.py`)

Date: 2026-09-26. Data: 4 runs (`p1.hold_rec` 120, `p2.pulse200_200` 400, `p3.compose` 291,
`p4.corridor_habitat` 300; 1,111 ticks). No credits spent. Fitter: `scripts/ode_lab.py --sigma-cal 1.5`
(Cauchy least squares at 1.5x the calibrated sigma), budget 240-400 s, 8-10 starts, 60-80 evaluations.
All "calibrated" scores below are $\frac1{T p}\sum 1/(1+|e|/\sigma)$ at exactly
$\sigma = (9.28, 0.187, 7.66, 0.253)$ from `plans/sigma_calibrated.json`, averaged over the 4 runs, with
the doc rolled through `gtlab.runtime.infer.rollout_from_blob`.

## 1. Error budget of the starting point (s1 cal4ABJ, calibrated in-sample 0.660)

Loss share = $\sum_{t,k}(1 - \text{score}_{tk})/(4 \cdot 1111)$ per constant-control segment.
Total loss 0.340.

| segment | loss share | prey_N | pred_N | prey_S | pred_S | what is wrong |
|---|---|---|---|---|---|---|
| p2 recovery after pulse [200,400) | 0.051 | .009 | **.018** | .010 | **.014** | predators 2.39 vs 2.26 mean; rise after release too early |
| p2 pulse [0,200) | 0.050 | .012 | **.016** | .007 | **.015** | prey decline too slow (14.4 vs 10.7 mean), predators flat-exp |
| p1 recovery from reset | 0.037 | .007 | .011 | .006 | **.013** | predator decay from 8-12 exponential, data two-speed |
| p4 habitat 0.5 [180,280) | 0.034 | .008 | .012 | .007 | .007 | prey 101 vs 91 at end; predators do not rise (data 2.5 -> 2.95) |
| p4 corridor 0.5 from reset | 0.027 | | | | | predators 2.30 vs 2.73 at close (no travellers return) |
| p3 recovery [246,291) | 0.022 | | | | | overshoot mistimed: 154 vs 124 at end |

By observable the predators carried 0.55 of the loss. Three sources: (1) predator equation decoupled
from prey (relaxes to $q_0 - q_c u_{cor}$), (2) prey decline and floor under hunting, (3) prey level under
partial habitat and the corridor.

## 2. What the data say (read off the four runs)

- **Harvest is a quota, not a rate.** Under the pulse both regions lose about 5 prey/tick in absolute
  terms from 84 to 40 (north 11.6, 10.4, 9.4, 8.6, 8.8 per 2 ticks; south 11.4, 9.8, 9.1, 8.7), then
  the loss stops sharply at 7.0 / 7.6. A constant take with a Holling type-III shortfall at low density,
  $H u\,P^2/(P^2+P_h^2)$, gives both the linear decline and a stable floor; a proportional take gives
  neither.
- **Predators: two-speed decay from reset** (excess rate 0.065/tick in the first 10 ticks at both 8.7
  and 12.6, then 0.03/tick over ticks 28-60), a slow rise after the pulse (2.04 -> 2.35 over 50 ticks,
  *after* the prey peak), a rise under habitat 0.5 while prey fall. A predator response to prey that
  lags by ~50 ticks explains the first two.
- **Corridor**: predators drop to 1.73 within ~20 ticks when it opens and return to 2.4 within ~20 ticks
  after it closes (travellers arriving); prey drop 8 % north and 20 % south. Visible predators stay low
  for the whole 60-tick opening, so travel must cost animals (lossless transit would restore the level
  through density dependence).
- **Habitat** acts fast: under 0.235 the prey fall 143 -> 76 in 40 ticks and rebound 76 -> 116 in 20
  ticks after protection returns; a slow food pool alone (the s1 mechanism) cannot do the fast
  rebound. The north reacts more (121 -> 91 at 0.5 vs 97 -> 79 south), as the brief says.

## 3. Model v8h (12 states, 18 fitted parameters, RK4 with 2 substeps)

Per region $i \in \{N, S\}$, scale $s_N = 1$, $s_S = k_s$; controls $u_h$ (hunting), $u_p$ (habitat),
$u_c$ (corridor):

$$\dot P_i = \frac{r R_i P_i}{1 + P_i/(c_J s_i)} - \big(D_0 + d_{h,i}(1-u_p)\big) P_i
 - H u_h \frac{P_i^2}{P_i^2 + P_h^2} - e_i u_c P_i + \frac{s_v}{\tau} T_{\to i}$$

$$\dot R_i = w\big(1 - b_h(1-u_p)\big)(1 - R_i) - k_R \frac{P_i}{s_i} R_i$$

$$\dot Q_i = c_q\,(q_0 + q_1 G_i - Q_i)\,\frac{Q_i}{Q_i + 4} - e_N u_c Q_i + \frac{s_v}{\tau} V_{\to i},\qquad
\dot G_i = \frac{1}{t_z}\Big(\frac{P_i}{P_i + 100 s_i} - G_i\Big)$$

$$\dot T_{i\to j} = e_i u_c P_i - T_{i\to j}/\tau,\qquad \dot V_{i\to j} = e_N u_c Q_i - V_{i\to j}/\tau,\qquad
e_N = e_m,\ e_S = e_{mS}$$

Observed $y = (P_N, Q_N, P_S, Q_S)$. Reset: $P, Q$ from $y_0$, $R = 1$, pools empty,
$G = P/(P + 100 s)$. Mechanisms: A = nursery crowding ($c_J$), B = finite food renewal ($w, k_R, b_h$).
Fixed constants: $D_0 = 0.7$ (only $r - D_0$ is identified: fits landed on $(r, d_0)$ = (0.87, 0.69),
(1.49, 1.32), (1.98, 1.82) with the same cost) and the predator half-saturation 4.

Fitted theta (`plans/wildlife_wildlife_v8h_v8h_loo.json`):

| $r$ | $k_s$ | $H$ | $P_h$ | $e_m$ | $e_{mS}$ | $\tau$ | $s_v$ | $c_J$ | $w$ | $k_R$ | $b_h$ | $d_{hN}$ | $d_{hS}$ | $q_0$ | $q_1$ | $c_q$ | $t_z$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.876 | 0.799 | 1.387 | 26.7 | 0.0281 | 0.0408 | 33.4 | 0.747 | 966 | 0.0258 | 2.0e-5 | 0.337 | 0.0401 | 0.0265 | 1.620 | 1.510 | 0.0966 | 63.4 |

No parameter at a bound. The fit starts from theta
`[0.8748, 0.8010, 1.3753, 26.708, 0.024521, 0.038547, 32.118, 0.75975, 899.51, 0.025576, 2.6182e-05, 0.32847, 0.038904, 0.025869, 1.6053, 1.5338, 0.096203, 60.605]`
(PARAMS order; the v8f "no xq" fit with $r$ shifted by $0.7 - d_0$).

## 4. Path (calibrated in-sample, 4 runs, full fit only)

| version | change | params | calibrated in-sample |
|---|---|---|---|
| s1 cal4ABJ | start | 16 | 0.660 |
| v8 (v8a) | predators logistic to $q_0 + q_1 Z$, predator transit pools, transit survival, saturating harvest | 18 | 0.682 |
| v8b | type-III harvest; predator relaxation $\propto Q/(Q+q_h)$ | 19 | 0.686 |
| v8c | + predation on prey $a_p E Q P$ | 20 | 0.686 (no gain, dropped) |
| v8d | lagged predator food $G$ ($t_z$); renewal habitat weight per region | 21 | 0.701 |
| v8e | $P_h$ not scaled by region; south emigration $e_{mS}$ | 22 | 0.706 |
| v8e mech B only | no crowding | 21 | 0.698 (fast-food basin: better p3, worse p1) |
| **v8f** | habitat exposure death $d_{h,i}(1-u_p)$, one $b_h$, $q_h = 4$ fixed | 22 | **0.714** |
| v8g | v8f without $b_h$ | 21 | 0.699 |
| v8f, harvest shelter $e_h = 0$ | | 21 | 0.713 |
| v8f, $e_h = 0$, mech B | | 20 | 0.692 |
| v8f, $e_h = x_q = 0$ | | 20 | 0.711 |
| v8f, $e_h = 0$, $s_v = 1$ | | 20 | 0.697 |
| v8f, $e_h = x_q = 0$, no lag ($t_z = 0.5$) | | 19 | 0.702 |
| v8t: $e_{mS} = e_m$, $d_{hS} = d_{hN}$ | | 18 | 0.697 |
| **v8h**: $e_h = x_q = 0$, $d_0 = 0.7$ fixed, predators use $e_m$ | | **18** | **0.709** |
| v8i: as v8h but $t_z = 50$ fixed, own $e_{mq}$ | | 18 | 0.711 |

Pruning kept what each removal test said matters: the lag $t_z$ (-0.009), transit loss $s_v$ (-0.016),
separate south emigration and exposure death (-0.014), $b_h$ (-0.015), crowding $c_J$ (-0.021).
Harvest shelter $e_h$ (-0.001) and predator exposure $x_q$ (-0.002) went.

## 5. v8h scores

Calibrated in-sample per observable (prey_N, pred_N, prey_S, pred_S):

| run | s1 (0.660) | v8h |
|---|---|---|
| p1.hold_rec | .748 .589 .776 .519 = 0.658 | .688 .702 .715 .626 = 0.683 |
| p2.pulse200_200 | .768 .619 .815 .681 = 0.721 | .760 .768 .801 .801 = 0.782 |
| p3.compose | .534 .699 .571 .769 = 0.643 | .604 .683 .594 .843 = 0.681 |
| p4.corridor_habitat | .661 .545 .611 .651 = 0.617 | .673 .643 .756 .694 = 0.691 |
| **mean** | **0.660** | **0.709** |

Predators gain the most (+0.086 north, 0.613 -> 0.699; +0.086 south, 0.655 -> 0.741, mean over runs); prey_N on p1 loses 0.06
(the overshoot peak is now slightly too high/late).

Leave-one-run-out (lab scale $1.5\sigma$; each fold refits from the theta above, which has seen the
held-out run, so this is mildly optimistic for both models alike):

| held out | s1 cal4ABJ | v8h | v8i | l0b_lin |
|---|---|---|---|---|
| p1.hold_rec | 0.712 | 0.732 | 0.735 | 0.478 |
| p2.pulse200_200 | 0.595 | 0.678 | 0.697 | 0.563 |
| p3.compose | 0.646 | 0.672 | 0.650 | 0.601 |
| p4.corridor_habitat | 0.628 | 0.693 | 0.706 | 0.598 |
| **mean** | **0.645** | **0.694** | **0.697** | 0.560 |

In-sample at $1.5\sigma$: 0.774 (s1: 0.728). Eval-shaped 4,000-tick rollouts (lab): finite, 0.5-0.6 s,
0 % of ticks outside the observed range $\pm 5\%$ on all four categories.

Long-hold levels from $y_0 = (85, 9, 80, 10)$, 1,500 ticks:

| hold | prey_N | pred_N | prey_S | pred_S | observed |
|---|---|---|---|---|---|
| recovery (0, 1, 0) | 120.6 | 2.45 | 96.3 | 2.45 | 121.6 / 2.35 / 97.2 / 2.34 (p2 end) |
| pulse (7, .1, 1) | 10.0 | 1.37 | 9.1 | 1.37 | 7.0 / 1.62 / 7.6 / 1.65 (still falling) |
| habitat 0.5 | 95.5 | 2.36 | 80.1 | 2.38 | 91.0 / 2.44 / 78.7 / 2.34 (100 ticks) |
| corridor 1 | 116.5 | 1.99 | 88.0 | 1.98 | 111.7 / 1.72 / 77.5 / 1.67 (60 ticks) |

## 6. Error budget after (v8h, total loss 0.291 vs 0.340)

| segment | loss share | before |
|---|---|---|
| p2 pulse [0,200) | 0.043 | 0.050 |
| p2 recovery [200,400) | 0.035 | 0.051 |
| p1 recovery from reset | 0.034 | 0.037 |
| p4 habitat 0.5 | 0.024 | 0.034 |
| p4 corridor 0.5 from reset | 0.022 | 0.027 |
| p3 recovery after hunting at habitat 0.235 [246,291) | 0.022 | 0.022 |

## 7. What is left

1. **Pulse prey floor**: model 10.0 / 9.1 vs observed 7.0 / 7.6, and the approach to the floor is too
   slow in the north (mean 13.9 vs 10.7). Type-III harvest fixes the shape but not the level; a refuge
   or a growth term that collapses at low habitat is the next thing to try.
2. **Recovery predator level**: 2.45 at 1,500 ticks vs 2.34 observed on the only long recovery; the
   lagged numerical response keeps pulling it up after the p1 decay is fitted. Recovery is half of the
   test ticks, so 0.1 predators there is worth about 0.01-0.02 on the system.
3. **Rebound after low-habitat hunting** (p3 [246,291)): model rises late and overshoots to 147 vs 124.
4. Corridor-open prey south: 88 vs 77.5.

Files: `gtlab/ode/wildlife_v8h.py` (best, 18 params), `wildlife_v8i.py` (18, $t_z$ fixed),
`wildlife_v8f.py` (22, best in-sample), `wildlife_v8`, `_v8b`-`_v8g`, `_v8t` (steps above);
lab outputs `plans/wildlife_wildlife_v8*_*.json` and `_doc.json`. Best doc:
`plans/wildlife_wildlife_v8h_v8h_loo_doc.json`.
