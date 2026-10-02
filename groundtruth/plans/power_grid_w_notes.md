# power_grid w: the long-hold leak (p8.longhold)

Lab notebook, Mon Sep 28. No credits spent here. Runs: `p1.hold_rec` 120, `p2.pulse60_120` 180,
`p2.multilevel200` 200, `p8.longhold` 700 (fitting folds); `p6.exam` 400 is excluded from every fit
(wrapper that drops it from `load_runs`, same pattern as the p5.testlike wrapper) and scored separately.
Scale: calibrated sigma [load 9.65, frequency 0.308, share 0.066]. Every fit is a plain lab fit:
`scripts/ode_lab.py --mech AB --budget 300 --starts 8 --nfev 60 --sigma-cal 1.0`.

## 1. What p8 shows

Schedule (ledger): t0-206 price 1.0, reserve 75, charging 0.5, ic 0.5; t207-384 price 0, reserve 0,
charging 1, ic 1; t385-699 price 0, reserve 0, charging 0, ic 0.

| tick | truth (load, f, share) | public perobs | v9c refit on 4 runs | **w5** |
|---|---|---|---|---|
| 100 | 104.8, 51.18, 0.18 | 105.6, 50.71, 0.14 | 106.0, 51.05, 0.15 | 108.0, 51.11, 0.14 |
| 350 | 120.8, 49.44, 0.34 | 127.0, 49.32, 0.35 | 123.5, 48.94, 0.34 | 123.0, 49.34, 0.36 |
| 699 | 121.4, 48.89, 0.15 | 126.4, 49.35, 0.18 | 123.1, 48.96, 0.17 | 122.6, 48.91, 0.15 |

Reading the three segments against the v9c balance ($\delta f_{ss}=k_f b/(1+k_fk_g)$, about
0.029 Hz per power unit):

1. **Segment 1 is flat.** Frequency holds 51.14-51.19 Hz from t42 to t206. No sign of reserve energy
   running out, thermal derating or a governor hitting its limit within 165 ticks. So the leak is
   *not* a slow decay. v9c sits 0.45 Hz low there: at ic 0.5 its delivery is
   $c_r(1-\beta)R\,(0.15+0.85\,ic)=26$ units, the data need about 50.
2. **Delivery saturates.** Multilevel t116-178 (request 126, ic 0.58) and p8 segment 1 (request 75,
   ic 0.5) settle at the same 51.15-51.2 Hz with loads 8 units apart: 51 extra units of request buy
   about 8. At ic 0.2 (pulse, request 150) the frequency only reaches 50.6-50.75. That is a capacity
   that grows with the interconnector, not a gain on the request.
3. **The interconnector delivers without a reserve request.** At t385 (reserve 0, ic 1 -> 0) the
   frequency falls 49.44 -> 48.9 and stays: about 17-20 power units of remote delivery at ic 1.
   Closing the charging allowance at the same tick can only raise the frequency, so the drop is the
   interconnector.
4. **Price-0 load settles at 121**, not at the $D(0)=126$ of v9c. At price 1 (51.2 Hz) it sits at 106
   and at price 1.5 at 91-93. A straight line through all three does not exist.

## 2. Structures (all copy v9c; always-on, `--mech AB`)

| family | change vs v9c | params |
|---|---|---:|
| `power_grid_w` | remote delivery $g_i(ic-1)$ in the balance; load $+k_d\,\delta f$ (frequency-sensitive load); fitted floors $i_0$ (reserve/refill) and $s_{i0}$ (share) | 20 |
| `power_grid_w2` | w + capacity: $R_{eff}=\text{smin}\big(c_r(1-\beta)R_s+\text{draw},\;p_l+p_r\,ic\big)$, request no longer scaled by ic; refill still $P_{ch}(i_0+(1-i_0)ic)$ | 22 |
| `power_grid_w3` | w2 + reset load as a decaying offset: $D_s(0)=D(0.8)$, $z(0)=\text{load}_0-D(0.8)$, $\dot z=-a_zz$ | 23 |
| `power_grid_w4` | w2 but one fleet capacity: $\text{smin}\big((\dots)\,icf,\;p_l\big)$ | 21 |
| `power_grid_w5` | w2 with the load term replaced by a curved price response $D(p)=d_0-d_1p+d_2(p-0.8)^2$ | 22 |
| `power_grid_w6` | w5 + the w3 reset offset | 23 |

smin$(a,b)=\tfrac12\big(a+b-\sqrt{(a-b)^2+3^2}\big)$ (smooth minimum, width 3 units).
The balance of w5:
$$b = D(0.8)+g_b+g_i(ic-1)+G+\text{smin}\big(c_r(1-\beta)R_s+c_r\beta R_sE,\;p_l+p_r\,ic\big)-p_{ch}\,c\,(1-E)\,(i_0+(1-i_0)ic)-L$$
with frequency, governor, ceiling, storage and share exactly as v9c apart from the share floor
$s^\star\propto s_{i0}+(1-s_{i0})ic$.

Why w5 over w2: in w2 the load damping $k_d\delta f$ and the governor gain $k_g$ both set the
steady-state droop. With the pulse run held out the fit sent $k_g\to0$ and $p_l\to0$, and the pulse
fold lost 0.004 to v9c. The curved demand explains the same price-0 load (121 at 48.9 Hz) without
touching the frequency loop, and the pulse fold then gains 0.015.

## 3. Results (lab; LOO = held-out run fitted on the other three)

| candidate | LOO hold_rec | LOO pulse | LOO multilevel | LOO p8 | **LOO mean** | in-sample | **exam** | at bound |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| public perobs (fit on 3 runs, no refit) | - | - | - | (0.630) | - | - | 0.657 | |
| v9c refit on 4 runs (`v9c_wref`) | 0.738 | 0.673 | 0.734 | 0.639 | 0.696 | 0.763 | 0.634 | a_g |
| w | 0.759 | 0.644 | 0.690 | 0.651 | 0.686 | 0.777 | 0.647 | |
| w2 | 0.760 | 0.669 | 0.720 | 0.674 | 0.706 | 0.789 | 0.673 | |
| w3 | 0.740 | 0.678 | 0.690 | 0.659 | 0.692 | 0.793 | 0.671 | |
| w4 (screen only, 1 start) | - | 0.652 | - | - | - | 0.779 | 0.644 | |
| **w5** | **0.770** | **0.688** | **0.733** | 0.650 | **0.710** | 0.789 | **0.685** | none |
| w6 | 0.751 | 0.681 | 0.688 | 0.656 | 0.694 | 0.792 | 0.683 | a_g, p_ch, i0 |

The public model never saw p8, so its 0.630 on p8 is a held-out number comparable to the LOO p8 column.
Exam per observable: public [0.632, 0.587, 0.751], w5 [0.669, 0.624, 0.761].
w5 in-sample per run: hold_rec 0.823, pulse 0.708, multilevel 0.800, p8 0.826.

w5 theta: d0 127.5, d1 19.15, w 0.0993, zeta 0.320, kick 0.626, kf 0.0384, gb -9.21, a_g 0.0554,
kg 4.59, c_r 1.090, cap 304, p_ch 86.1, s0 0.365, k_th 0.250, f_hi 1.89, beta 0.503, d2 -7.71,
g_i 13.7, i0 0.0002, sh_i0 0.432, p_l 21.9, p_r 82.2. Nothing at a bound.
Reading: the fleet can deliver 22 units with the interconnector closed and 104 fully open; at ic 1
the remote link also brings 13.7 units on its own; $D(0)=122.6$, $D(1)=107.1$, $D(1.5)=94.9$,
$D(2)=78.1$; refill passes the connection in proportion to ic ($i_0\approx0$).

## 4. Long holds (4,000 ticks, 36 constant schedules: price 0/0.8/2 x reserve 0/75/150 x charging x ic)

| model | all finite | load range | frequency range | final frequency range |
|---|---|---|---|---|
| v9c (shipped member) | yes | 54.0-155.9 | 47.97-51.73 | 49.35-51.70 |
| w5 | yes | 46.1-153.5 | 47.71-51.88 | 48.91-51.82 |

Steady frequency at reserve 150 minus reserve 0 on the recovery base (u008 +0.30 / +1.14 / +1.98):
w5 +0.71 / +1.17 / +1.35 at ic 0 / 0.5 / 1 (v9c +0.38 / +1.15 / +1.41). Lab eval-shaped 4,000-tick
rollouts: finite, 0.3 s per episode, 0.00 outside the observed range in all four categories.
Caveat: the price-2 load (78) is an extrapolation of the curvature; no run goes above price 1.5.

## 5. Decision

- **Recommend w5 as a single family for all three observables**: `plans/power_grid_power_grid_w5_w5_doc.json`
  (lab doc, no polishing). LOO 0.710 vs 0.696 for v9c refit the same way; every fold at least ties
  (multilevel -0.001), the pulse fold gains +0.015; exam +0.028 over the public model.
- Renewable share: keep w5's own. Swapping the public share members (v8k/min mean) back in via the
  perobs map lowers exam 0.685 -> 0.681 and p8 0.826 -> 0.790; they win only in-sample on hold_rec.
- Rejected: w (kd/kg trade-off, pulse fold -0.029), w2 (pulse fold -0.004), w3 and w6 (the reset
  offset lifts pulse in-sample but costs the multilevel fold 0.04; again, as in v8b), w4 (single
  capacity, pulse fold 0.652). Reserve energy exhaustion, interconnector thermal derating and a
  governor output limit were not fitted: p8 shows no drift at all over 165 ticks of segment 1 and
  300 ticks of segment 3, so there is nothing for a slow state to explain.
- Forecast: held-out gain = mean of exam (+0.028) and the p8 fold vs the public model's p8
  (+0.020) = +0.024; x 0.6 -> **about +0.015 public (0.775 -> ~0.79)**.
- Open: the p8 fold is still the weakest (0.650): without p8 no run shows ic 0 at reserve 0, so
  $g_i$ is set by p8 alone. The exam's remaining leak is load dynamics in fast switching and the
  second rebound (t240-290), not long holds.
