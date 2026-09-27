# market v8: structural rebuild at the organizer's scale (Sat Sep 26, no credits)

Goal: move market from 0.524 (in-sample at the calibrated sigma, `market_gate` cal4AB) toward 0.70.
Scale: $\sigma$ = (0.816, 0.643, 2.077) for (price, volume, depth), `plans/sigma_calibrated.json`.
Score per tick $1/(1+|e|/\sigma)$. Data: 6 runs, 940 ticks (hold_rec, pulse40, mid40, multilevel200, compose, rate_hold).
Fits: least squares, Cauchy loss on residuals in units of $1.5\sigma$ (`--sigma-cal 1.5`), RK4 with 2 substeps.
"Calibrated in-sample" = mean over the 6 runs of the score at exactly $\sigma$, from the doc via
`rollout_from_blob` (clipped, as packaged). LOO numbers are the lab's, at $1.5\sigma$.

**Result: `market_v8t2`, doc `plans/market_market_v8t2_v8_doc.json`. Calibrated in-sample 0.645 (from 0.524,
+0.121); tick-weighted 0.651 (from 0.522). LOO 0.650 (from 0.499). 16 fitted parameters, none at a bound.**
Short of 0.70. What still costs the most is depth (see the end).

## 1. Error budget before (market_gate cal4AB)

Loss = $\sum_t (1 - \text{score})$ at $\sigma$, summed over the 940 ticks (out of 2,820 obs-ticks):

| | price | volume | depth | total |
|---|---:|---:|---:|---:|
| loss | 518 | 340 | 492 | 1,350 |
| score (run mean) | 0.444 | 0.640 | 0.489 | 0.524 |

Largest segments (constant-control stretches):

| run [ticks) | controls | loss P / V / D | mean error P / V / D |
|---|---|---|---|
| rate_hold [0,200) | (0.1, 0) | 86 / 77 / 117 | +0.5 / -0.1 / +3.2 (depth dip 91 -> 72 -> 91 missed) |
| hold_rec [0,120) | (0, 0) | 59 / 42 / 47 | -1.0 / +0.3 / -0.3 |
| compose [90,150) | tax 0.0425 | 56 / 13 / 37 | **-11.1** / 0 / -2.4 |
| multilevel [37,95) | (0.09, 0.022) | 28 / 35 / 38 | -0.6 / -1.0 / +5.0 |
| compose [180,240) | (0.085, 0.0425) | 48 / 17 / 35 | -6.2 / +0.3 / -3.1 |
| compose [150,180) | (0, 0) | 29 / 13 / 12 | **-15.1** / +0.5 / -1.9 |

The three biggest sources:
1. **Price never recovers.** The gate model has no anchor: a rate pushes the price down to a hard floor, and at
   zero controls nothing brings it back. Compose rises 76.5 -> 92 (t 60-150) and 73.5 -> 85 (t 240-300); the model
   stays at the floor (errors -11 to -15). The tax gate at $a_x$ = 0.0345 also froze compose at tax 0.0425, where
   the data trades freely.
2. **Volume floor wander.** $\sigma_V$ = 0.64 and the floor moves 1.35-3.8 with a constant $v_0$: every run
   scores 0.53-0.70 on the most predictable observable (noise 0.017).
3. **Depth under a rate.** No rate effect on depth ($a_r$ fitted to 0.13), but rate_hold's depth falls 97 -> 72
   while the price falls and comes back to 91 once it stops.

## 2. Model (market_v8t2)

States $B, P, v, A, D_f, D_s, R, H, Q, W$. Controls $r$ (interest), $x$ (tax).

Trading gate: $g = 1/(1+e^{(x - X_C)/W_X})$, with $X_C$ = 0.0452 and $W_X$ = 0.0005. Trading is free at 0.0425
(compose) and frozen at 0.0459 or above (multilevel).

**Price.** A moving anchor, and a follower with a speed limit:
$$\phi(r) = \frac{(r/r_c)^N}{1+(r/r_c)^N},\ N = 7.6, \qquad A_{eq} = Q + (p_{lo} - k_x\, x\, g - Q)\,\phi(r)$$
$$\dot A = (A_{eq} - A)\,[\,(1-s)/\tau_A + s/\tau_{Au}\,],\quad s = \tfrac12(1 + e_A/\sqrt{e_A^2+1}),\ e_A = A_{eq}-A$$
$$\dot P = g\,v, \qquad \dot v = g\,\big(s_v \tanh(\kappa (A-P)/s_v) - v\big)/\tau_v$$
$Q$ = price at reset: the price sits where the hidden stocks put it (multilevel stays at 106 for 37 ticks at a
low rate). Above $r_c \approx 0.08$, interest drains working cash and the anchor moves to the producers'
reservation level $p_{lo}$. It falls fast ($\tau_A \approx 10$) and recovers slowly ($\tau_{Au} \approx 58$):
cash drains quickly under the rate and refills slowly through revenue. The follower has inertia ($\tau_v$), which
gives the S-shaped start, and a top speed $s_v$ = 0.63/tick. That speed limit is what the data shows: rate_hold
falls at a nearly constant -0.6/tick, then stops hard at 72. With a trading tax the clearing floor drops by
$k_x x$. This only applies while trading ($g$), so a freeze does not drag the anchor further down.

**Volume.**
$$V = V_0 + B + C_G|A-P| + c_p\,g\,|v| + c_W\,W\,g\,(10 r + 20 x), \quad \dot B = -B/2.7,\ \dot W = -W/\tau_W,\ W_0 = 1$$
- Traded volume follows the price velocity: $|v|$ alone explains 34-37% of the floor variance.
- It also follows the anchor-price gap.
- It follows a reset imbalance $W$: warehouses start half full. Their adjustment flow runs while controls are on,
  decays with $\tau_W \approx 71$, and stops when trading freezes.
- $V_0$ = 1.77 is the quiet floor seen late in hold_rec, rate_hold and compose.

**Depth.**
$$\text{depth} = (f D_f + (1-f) D_s)(1-R) + H, \quad D^* = d_0 e^{-a_t x}$$
$$\dot D_f = (D^* - D_f)/4, \quad \dot D_s = k(e)\,e,\ e = D^* - D_s,\ k = k_{sd} \text{ shrinking}, K_{SU} = 0.133 \text{ refilling}$$
$$\dot R = (k_R \max(-\dot P, 0) - R)/20, \qquad \dot H = 0.05\,B - H/11.1$$
- $R$: adverse moves cut risk capacity, which is mechanism B in the brief.
- $H$: inventory the reset burst puts on dealer books. It gives the +7 to +14 depth bump right after a reset
  (rate_hold 82.6 -> 97, compose 95 -> 102).

**Reset.** $B_0 = y_{V} - V_0$, $P_0 = A_0 = Q = y_P$, $v_0 = 0$, $D_{f0} = D_{s0} = y_D$, $R_0 = H_0 = 0$, $W_0 = 1$.
Every observed initial is used.

**Constants and parameters.**
- Fixed from earlier fits (all stable to ±2% across the fits below): $X_C$, $N$, $K_{SU}$, $C_G$ = 0.0184,
  $V_0$, $\tau_H$ = 11.1, $C_B$ = 0.05, $\tau_f$ = 4, $\tau_R$ = 20, $\tau_B$ = 2.7.
- Fitted (16): $c_p$ 1.076, $c_W$ 1.139, $\tau_W$ 70.7, $d_0$ 92.05, $a_t$ 27.5, $k_{sd}$ 0.0292, $f$ 0.449,
  $k_R$ 0.455, $p_{lo}$ 69.85, $r_c$ 0.0795, $k_x$ 140.9, $\tau_A$ 9.55, $\tau_{Au}$ 58.1, $\kappa$ 0.0882,
  $s_v$ 0.632, $\tau_v$ 14.4.

## 3. The ladder (each step: calibrated in-sample, run mean; LOO at 1.5σ)

| version | change | params | cal. in-sample | LOO | kept |
|---|---|---:|---:|---:|---|
| market_gate cal4AB | reference | 12 | 0.524 | 0.499 | |
| v8 | anchor $A$ relaxing to $p_a - dl\,(r/0.1)^\gamma$; 2nd-order follower; volume +gap; depth $\times(1-R)$ + $H$ | 17 | 0.580 | | step |
| v8b | slow anchor forced ($\tau_A \ge 15$), sharp gate | 17 | 0.568 | | no |
| v8c | anchor centre $wQ + (1-w)p_a$ | 18 | 0.577 | | no |
| v8d | $A_{eq} = Q + (p_{lo}-Q)\phi(r)$, Hill threshold in rate | 17 | 0.598 | | step |
| v8e/f | fix $\tau_f$, $C_B$, $N$; asymmetric anchor ($\tau_{Au}$) | 15-16 | 0.598-0.602 | | step |
| v8g | volume from $|A_{eq}-A|$ instead of $|A-P|$ | 15 | 0.585 | | no |
| v8h | separate tax elasticity for the slow pool | 16 | 0.598 | | no |
| **v8i** | volume $+ c_p g |v|$ (price velocity) | 16 | 0.6125 | 0.568 | yes |
| v8j | $R$ scaled by $e^{-a_t x}$; $\tau_R$ free | 16 | 0.606 | | no |
| v8k | $\tau_R$ free | 16 | 0.612 | | no |
| v8l | tax lowers the floor, $-k_x x$ | 16 | 0.6163 | 0.567 | yes (LOO -0.001, noise) |
| v8m/n | $R$ from price below its EMA ($\tau_E$ 20 / 40) | 16 | 0.616 / 0.607 | | no |
| **v8o** | $C_B$, $\tau_H$ free; $K_{SU}$, $C_G$ fixed | 16 | 0.6237 | 0.572 | yes |
| v8p | speed-limited follower $s_v\tanh(\cdot)$ | 16 | 0.623 | | tie (cost 1236 -> 1015) |
| **v8q** | + Hill exponent free, $X_C$ fixed | 16 | 0.6246 | 0.612 | yes |
| **v8r** | $k_x$ acts only while trading ($g$) | 16 | 0.6259 | 0.612 | yes |
| v8s | $R$ only while the rate is on | 16 | 0.617 | | no |
| **v8t** | reset imbalance $W$ in volume | 16 | 0.632 | 0.631 | yes |
| v8u | slow-pool shrink scaled by $W$ | 16 | 0.6215 | | no (cost 907, but pulse40/mid40 depth worse) |
| v8v | reset excess put in the slow pool only | 16 | 0.614 | | no |
| v8w/x | anchor centre $wQ + (1-w)p_a$ (w 0.84, $p_a$ 100) | 16 | 0.631 / 0.6365 | x: 0.624 | no: LOO dropped (hold_rec fold price 0.59 -> 0.50) |
| v8y | leak through the freeze ($g_{min}$) | 16 | 0.6365 | | no ($g_{min} \to 0$) |
| v8z / v8z2 | v8x + $W$ gated by $g$ (+ $V_0$ 1.77) | 16 | 0.6497 / 0.6503 | 0.641 / 0.642 | no: carries v8x's anchor centre, LOO below v8t2 |
| **v8t2** | v8t + $W$ gated by $g$, $V_0$ = 1.77 | 16 | **0.6446** | **0.650** | **best** |

The saturating follower (v8p) is the structural change that moved the Cauchy cost most (1236 -> 1015). The
calibrated run mean did not move, because mid40 and pulse40 got worse while rate_hold improved. With the
Hill exponent free (v8q), LOO jumped 0.572 -> 0.612.

## 4. Best version in detail (market_v8t2)

Calibrated in-sample at exactly $\sigma$:

| run | price | volume | depth | mean | gate (before) |
|---|---:|---:|---:|---:|---:|
| hold_rec | 0.511 | 0.871 | 0.675 | 0.686 | 0.589 |
| pulse40 | 0.430 | 0.725 | 0.540 | 0.565 | 0.539 |
| mid40 | 0.670 | 0.679 | 0.598 | 0.649 | 0.471 |
| multilevel200 | 0.504 | 0.675 | 0.561 | 0.580 | 0.529 |
| compose | 0.622 | 0.792 | 0.542 | 0.652 | 0.483 |
| rate_hold | 0.640 | 0.862 | 0.704 | 0.736 | 0.534 |
| **mean** | 0.563 | 0.767 | 0.603 | **0.645** | 0.524 |

Tick-weighted: 0.651, per observable (0.572, 0.782, 0.600).

LOO (1.5σ), fold means:

| held out | score | l0b_lin |
|---|---:|---:|
| hold_rec | 0.731 | 0.624 |
| pulse40 | 0.636 | 0.386 |
| mid40 | 0.719 | 0.641 |
| multilevel | 0.542 | 0.357 |
| compose | 0.600 | 0.478 |
| rate_hold | 0.669 | 0.343 |
| **mean** | **0.650** | 0.471 |

The gate scored 0.499 on the same folds.

Error budget after (loss at σ): price 402 (from 518), volume 205 (from 340), depth 376 (from 492); total 983
(from 1,350). For scale, smoothing the data with a 5-tick moving average and scoring it against itself gives
0.91 (price 0.82: noise 0.35 against $\sigma$ 0.82). So in-sample has room up to about 0.9.

Eval sanity (lab eval-shaped 4,000-tick schedules, and constant holds from the mean $y_0$):
- All finite; 0.3 s per 4,000 ticks.
- Long-hold price: r 0.1 -> 74.2 (observed 74.4); r 0.07 -> 91.2; (0.085, 0.0425) -> 77.2; a freeze tax
  holds the price where it was.
- Minimum price on the sustained schedules: 67.1, 9% of ticks outside the observed range. This happens when
  a rate of 0.1 meets an interior tax: the floor becomes $69.85 - 140.9x$, about 64 at x = 0.04. That pair was
  never observed, so it is an extrapolation of $k_x$. $k_x$ comes from compose [180,240) alone, and part of its
  size may stand in for the dynamics.
- Depth 22.7-117.8, volume 1.8-63.8. At bound: none.

## 5. What did not work, and why

- A common zero-rate price level instead of the reset price (v8c, v8w, v8x): better in-sample (hold_rec drifts
  up to 95.9), worse out of sample. Multilevel sits at 106 for 37 ticks, so the anchor must start at the reset price.
- A slow anchor instead of the rate threshold (v8b): rate_hold's fall is too steep for it.
- Every depth variant aimed at the compose tax plateaus (v8h separate slow elasticity, v8u $W$-scaled drain,
  v8v reset excess in the slow pool, v8j $R$ reduced under tax, v8s $R$ only with the rate on) lost on
  pulse40/mid40. The data conflict: after a reset the depth keeps draining slowly under any tax (pulse40 0.05,
  mid40 0.025, multilevel 0.034). Mid-run (compose t=90 and t=180) the same kind of tax gives a flat plateau at
  44 within 25 ticks. No state we tried separates these.
- $R$ from a price EMA (v8m/n): same score, worse rate_hold dip.
- A leak through the freeze (v8y): the fit sets it to zero.

## 6. Remaining biggest losses (v8t2)

1. **Depth under tax mid-run:** compose [180,240) mean error -7.2 and [90,150) -3.2, loss 72. The slow pool keeps
   draining where the data sits flat at 43.5.
2. **rate_hold overall (159):** the depth dip timing (the model dips earlier and shallower) and the price rebound
   75.9 -> 74.4.
3. **hold_rec price (59):** a hump 93.4 -> 95.9 -> 94.3 at zero controls that nothing in the model produces.
4. **Freeze transitions:** pulse40 price 0.43, and multilevel [181,200) price -2.9 after release (the anchor
   sits at the floor while the frozen price is above it).

Files:
- modules `gtlab/ode/market_v8*.py`
- lab reports `plans/market_market_v8{i,l,o,q,r,t,x,z,z2,t2}_v8.json` (+ `_doc.json`)
- ship candidate `plans/market_market_v8t2_v8_doc.json`
