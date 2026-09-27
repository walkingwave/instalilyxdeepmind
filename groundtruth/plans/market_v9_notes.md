# market v9: depth rebuilt (Sat Sep 26, no credits)

Goal: move market from 0.645 (calibrated in-sample, `market_v8t2`) toward 0.70 with structural changes.
Scale: $\sigma$ = (0.816, 0.643, 2.077) for (price, volume, depth), `plans/sigma_calibrated.json`. Score per tick
$1/(1+|e|/\sigma)$. Same 6 runs, 940 ticks. "Calibrated in-sample" = run mean at exactly $\sigma$ from the doc
via `rollout_from_blob` (clipped, as packaged).

**Result: `market_v9g`, doc `plans/market_market_v9g_v9cal_doc.json`. Calibrated in-sample 0.687 (from 0.645,
+0.042); depth 0.725 (from 0.603); price and volume unchanged (0.568 / 0.768). Lab LOO (1.5σ) 0.661 (from 0.650).
17 fitted parameters (from 16), none at a bound.** Price and volume equations are v8t2's; only depth changed.

## 1. What the depth data says (per-segment exponential fits)

We fitted $y = a + b\,e^{-t/\tau}$ to depth on every constant-control segment (skipping the first 3-6 ticks):

| segment | controls (r, x) | start | asymptote $a$ | $\tau$ |
|---|---|---:|---:|---:|
| compose [90,150) | (0, 0.0425) | 83.6 | 43.7 | 7.3 |
| compose [180,240) | (0.085, 0.0425) | 84.5 | 43.6 | 8.0 |
| 5 recoveries to zero controls (pulse40, compose x3, multilevel) | (0, 0) | 42-80 | 87.6-91 | 7.1-8.4 |
| hold_rec [0,120) (post reset) | (0, 0) | 116 | 90.6 | 13.2 |
| mid40 [0,40) (post reset) | (0.05, 0.025) | 82.6 | 47.0 | 22.3 |
| multilevel [0,37) (post reset) | (0.023, 0.034) | 94.4 | 44.1 | 14.5 |
| pulse40 [0,40) (post reset, frozen) | (0.1, 0.05) | 86.0 | 29.9 | 18.1 |
| multilevel [95,116) (frozen) | (0.1, 0.05) | 36.5 | 18.6 | 13.1 |

What follows:
1. **Every mid-run step is a single exponential with $\tau \approx 7.5$, up or down.** No slow tail: the compose
   tax plateau is flat at 43.7. v8t2's slow pool ($k_{sd}$, $\tau \approx 34$) was the leak.
2. **Right after a reset the approach is 2-3x slower** ($\tau$ 13-22), and slower the higher the tax. The reset
   is the only thing that differs, so the slow component must be a reset-only state.
3. **The capacity target is flat in the tax.** Plateaus sit at 44-47 for every interior tax 0.025-0.0425, not
   the $d_0 e^{-a x}$ of v8t2 (which gave 28 at 0.0425 on long holds). A freeze roughly halves it again (22).
4. The risk-capacity dip (depth falls while the price falls) shows up under a rate with no or low tax
   (rate_hold 90 -> 72; compose [0,60) 90 -> 78; multilevel [37,95) 48 -> 39), but **not** in compose [180,240)
   (rate 0.085, tax 0.0425: price falls 0.3/tick, depth flat at 43.7), and not in pulse40 [40,80) (the price falls
   at zero controls and depth recovers to 87.5).

## 2. Model (market_v9g)

Price and volume: exactly `market_v8t2` (anchor $A$ with the Hill rate threshold, speed-limited follower, reset
imbalance $W$). $c_W$, $\tau_W$, $r_c$ are now fixed at their v8t2 values (1.139, 70.71, 0.0795).

Depth:
$$\text{depth} = D\,(1-R) + H$$
$$\dot D = (D^* - D)/8, \qquad D^* = d_0\Big(1 - m_1 \frac{x}{x + 0.005}\Big)\big(1 - m_2 (1-g)\big)$$
$$\dot G = c_H B - G/3, \qquad \dot H = G/3 - H\, e^{-a_H x}/\tau_{H}$$
$$\dot R = \big(k_R \max(-\dot P, 0)\,[1 - w_r + w_r\,\phi(r)]\,s_R(x) - R\big)/20, \qquad s_R = \frac{1}{1 + e^{(x - x_R)/0.004}}$$

- $D$: quoted capacity. One pool, $\tau$ = 8 fixed from the segment fits. $D^*$ drops by $m_1$ as soon as a tax
  is on and by $m_2$ more when trading freezes ($g$ = v8t2's gate at 0.0452).
- $H$: inventory the reset order burst $B$ puts on dealer books, through one settlement stage $G$ (3 ticks). It
  exists only after a reset, so it makes post-reset approaches slow and leaves mid-run steps untouched. It
  clears with $\tau_H \approx 9.6$ at zero tax; a tax slows settlement by $e^{a_H x}$: 3.6x at 0.034, 6.7x at
  the freeze level 0.05.
- $R$: adverse moves cost capacity mainly when funding is dear ($\phi(r)$ = the anchor's Hill rate threshold;
  $w_r$ = 0.77 of the effect is rate-gated), and not when a tax above $x_R$ = 0.034 has already thinned dealer
  inventory.
- Reset: $D_0$ = depth$_0$, $G_0 = H_0 = R_0 = 0$ (the rest as v8t2). Every observed initial is used.

Fitted (17): $c_p$ 0.960, $d_0$ 91.47, $m_1$ 0.582, $m_2$ 0.470, $c_H$ 0.0759, $\tau_H$ 9.57, $a_H$ 38.2,
$k_R$ 0.490, $x_R$ 0.0337, $w_r$ 0.766, $p_{lo}$ 69.79, $k_x$ 131.8, $\tau_A$ 7.63, $\tau_{Au}$ 62.5, $\kappa$ 0.0878,
$s_v$ 0.624, $\tau_v$ 14.0.

Fitting: we polished directly on the calibrated score (Powell in the unit-cube parameter space; depth
parameters first with price/volume held at v8t2, then all 17), and wrote the doc with
`ode_lab.py --family market_v9g --theta0 <polished> --starts 1 --nfev 1 --sigma-cal 1.0 --no-loo --tag v9cal`.
The lab's own Cauchy fit at 1.5σ (`plans/market_market_v9g_v9_doc.json`) scores 0.675 calibrated, so the
Powell polish is worth +0.012; the structure is worth +0.030.

## 3. The ladder (calibrated in-sample, run mean)

| version | change vs previous | params | cal. in-sample | depth | kept |
|---|---|---:|---:|---:|---|
| v8t2 | reference | 16 | 0.6446 | 0.603 | |
| v9 | v8t2 + $J$: depth used by the reset adjustment flow, $\dot J = c_J W(10r+20x) - J/\tau_J$ | 18 | 0.6496 | 0.619 | no: multilevel depth 0.561 -> 0.430 |
| v9b | one fast pool, flat tax target $m_1$, freeze step $m_2$; $H$ clears slower when frozen | 16 | 0.6468 | 0.610 | step |
| v9c | + settlement stage $G$ for $H$; a tax slows $H$ clearing ($a_H$) | 17 | 0.6535 | 0.630 | step |
| v9d | + $R$ partly rate-gated ($w_r$), cut by $e^{-a_R x}$ | 19 | 0.6768 | 0.700 | step (multilevel depth 0.41) |
| v9e | $R$ fully rate-gated, logistic tax cut-off $x_R$ | 18 | 0.6744 | 0.693 | step (pulse40 recovery worse) |
| v9f | v9e + $w_r$, $\tau_D$ fixed at 8; full polish | 18 | 0.6868 | 0.725 | no: LOO 0.639 |
| **v9g** | v9f with one clearing law $e^{-a_H x}/\tau_H$ (no separate frozen $\tau_{H0}$) | 17 | **0.6868** | **0.725** | **best** |

v9f's separate frozen clearing time $\tau_{H0}$ is identified by pulse40 alone: with pulse40 held out it fell
from 62 to 5 and the held-out depth collapsed. The fitted $\tau_{H0}$ = 62.5 was almost exactly
$\tau_H e^{a_H \cdot 0.05}$ = 64.6, so v9g ties the two: same in-sample, one parameter fewer, better LOO.

## 4. Best version in detail (market_v9g)

Calibrated in-sample at exactly $\sigma$ (v8t2 in brackets):

| run | price | volume | depth | mean |
|---|---:|---:|---:|---:|
| hold_rec | 0.511 | 0.871 | 0.802 (0.675) | 0.728 (0.686) |
| pulse40 | 0.429 | 0.731 | 0.776 (0.540) | 0.645 (0.565) |
| mid40 | 0.681 | 0.678 | 0.746 (0.598) | 0.702 (0.649) |
| multilevel200 | 0.511 | 0.670 | 0.567 (0.561) | 0.583 (0.580) |
| compose | 0.626 | 0.789 | 0.739 (0.542) | 0.718 (0.652) |
| rate_hold | 0.648 | 0.868 | 0.722 (0.704) | 0.746 (0.736) |
| **mean** | 0.568 | 0.768 | 0.725 (0.603) | **0.687** (0.645) |

Tick-weighted 0.690 (0.577, 0.781, 0.710). Every run improves.

Leave-one-run-out, two protocols:

| held out | lab v8t2 | lab v9f | **lab v9g** | cal. v8t2 | cal. v9f | **cal. v9g** |
|---|---:|---:|---:|---:|---:|---:|
| hold_rec | 0.731 | 0.781 | 0.778 | 0.683 | 0.727 | 0.727 |
| pulse40 | 0.636 | 0.514 | 0.554 | 0.558 | 0.492 | 0.566 |
| mid40 | 0.719 | 0.681 | 0.716 | 0.649 | 0.600 | 0.655 |
| multilevel | 0.542 | 0.572 | 0.594 | 0.528 | 0.545 | 0.560 |
| compose | 0.600 | 0.582 | 0.615 | 0.566 | 0.590 | 0.588 |
| rate_hold | 0.669 | 0.703 | 0.707 | 0.697 | 0.682 | 0.689 |
| **mean** | 0.650 | 0.639 | **0.661** | 0.614 | 0.606 | **0.631** |

- Lab: `ode_lab.py`, Cauchy least squares at $1.5\sigma$, scored at $1.5\sigma$ (v8t2's 0.650 used the same
  scale; v9f/v9g budget 200 s, 4 starts).
- Calibrated: per fold, Powell on the calibrated score over the 5 training runs from the full-data fit
  (1,000 evaluations), held-out run scored at exactly $\sigma$. Same protocol for all three models. It starts
  from the full-data fit, so it is optimistic in level, but fair as a comparison.
- The one fold that drops is pulse40 in the lab (depth 0.35 vs 0.61): without pulse40 no run shows a freeze
  right after a reset, so the post-reset freeze depth is extrapolated. Under the calibrated protocol pulse40 is
  level (0.566 vs 0.558).

Eval sanity (lab eval-shaped 4,000-tick schedules, and constant holds from the mean $y_0$):
- All finite; 0.7 s per 4,000 ticks.
- Long-hold price: unchanged from v8t2 (r 0.1 -> 74.2; r 0.07 -> 91.2; (0.085, 0.0425) -> 77.5). Minimum on the
  sustained schedules 67.4 (v8t2 67.1), 9% of ticks outside the observed range, same cause as before
  ($p_{lo} - k_x x$ with a rate of 0.1 and an interior tax; $k_x$ fell 141 -> 132, so slightly less).
- Long-hold depth: 43.7 at tax 0.0425 (observed plateau 43.7; v8t2 went to 28), 23 frozen, 91 at zero controls.
  Depth 22.8-115.1 on the eval schedules. At bound: none.

## 5. What did not work, and why

- $J$ (depth consumed by the reset adjustment flow $W$, v9): the drain is too slow for mid40 and too strong on
  multilevel's long low-tax stretch. The slow post-reset component is not a drain but a slowly clearing
  positive stock ($H$) whose clearing slows with the tax.
- Exponential tax target $d_0 e^{-a x}$: forces either compose's plateau or the post-reset asymptotes to be
  wrong. The flat target plus a freeze step fits both.
- $R$ cut smoothly by $e^{-a_R x}$ (v9d): to remove $R$ at 0.0425 it also removes it at 0.022, and multilevel
  [37,95) needs it there. A logistic cut-off between the two fits both.
- $R$ fully rate-gated (v9e): pulse40's recovery ends at 87.5, not 90.5; a quarter of $R$ at zero rate
  ($w_r$ = 0.77) gives that.
- A separate clearing time when frozen (v9f): identified by one run only (see section 3).

## 6. Remaining biggest losses (v9g)

1. **Price** is now the weakest observable (0.568), untouched in this pass: the hold_rec hump at zero controls
   (93.4 -> 96.4 -> 94.3), pulse40 (0.43: the anchor after a frozen pulse), multilevel after the freeze.
2. **multilevel depth (0.567):** post-reset [5,40) the model sits ~5 above the data (tax 0.034 slows $H$ clearing
   too much there, while mid40 at 0.025 needs it slow); frozen floor [115,135) 23-24 vs 21.5; the release at
   t 181 recovers too fast (58.6 vs 49.1 at t 185).
3. **rate_hold depth recovery [65,95):** the model recovers from the dip ~10 ticks early (73.8 vs 71.8 at t 65,
   80.3 vs 74.5 at t 80). In the data the dip stays flat until the price stops, then recovers.

Files:
- modules `gtlab/ode/market_v9.py`, `market_v9b.py` ... `market_v9g.py`
- lab reports `plans/market_market_v9f_v9.json`, `plans/market_market_v9g_v9.json` (with LOO),
  `plans/market_market_v9g_v9cal.json` (+ `_doc.json` each)
- ship candidate `plans/market_market_v9g_v9cal_doc.json`
