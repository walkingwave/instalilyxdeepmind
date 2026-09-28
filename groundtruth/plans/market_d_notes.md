# market d: the p7.longhold depth leak (Mon Sep 28, no credits)

Baseline: `market_y3` AB refit on all 9 runs (`plans/market_market_y3_r9.json`): LOO 0.564, p7 fold 0.293
(P/V/D 0.149/0.637/0.091), in-sample p7 0.324. Lab: `scripts/ode_lab.py --system market --family <fam> --mech AB
--budget 300 --starts 8 --nfev 60 --sigma-cal 1.0`.

## 1. What differs between p7 and compose 180-240

Reading every run block by block, p7 at (0.0949, 0.0403) is not a trading block with extra drain: it is a
**freeze from reset**. Its first 40 ticks match pulse40 at (0.1, 0.05) (frozen) almost tick for tick:

| run, block | price t19 → t39 | volume | depth t19 → t39 |
|---|---|---|---|
| pulse40 (0.1, 0.05) | 108.4 → 105.8 | 2.6–2.9 | 46.9 → 35.0 |
| p7 (0.0949, 0.0403) | 104.7 → 102.7 | 3.0–3.2 | 52.3 → 38.1 |
| compose 180–240 (0.085, 0.0425) | 87.1 → 73.7 (fast) | 1.8–2.1 | 47 → 43.7 flat |

Slow slide at the frozen order speed ($g_l$), high volume, depth draining below the tax level: all freeze
signatures. Compose at a *higher* tax trades. So no tax-only gate fits both; the gate threshold must fall with the
rate. Every other block agrees with $x_c(r)$ dropping by ≈ 0.006–0.010 between r = 0.085 and 0.095:
(0.097, 0.036) testlike trades (fast-ish fall), (0.095, 0.044) testlike frozen (price rises), (0.09, 0.022) and
(0.079, 0.037) trade. Second finding: the frozen drain is slow and almost linear (52 → 11 over 150 ticks), and the
rebuild depends on the controls: τ ≈ 8 at (0, 0) (pulse40, testlike 200), τ > 100 at (0.05, 0.025) (p7 tail).

## 2. Structures (all on the y3 base, AB)

- `market_d`: rate-dependent gate $x_c(r) = x_c - \Delta x_c\,\sigma((r - r_c)/0.002)$, $g = \sigma((x_c(r) - x)/W_X)$.
- `market_d2`: d + slow funding tie-up by stranded inventory (mechanism A):
  $\dot F = (1-g)(r/0.1)^{n_F}(1-F)/\tau_{Fi} - F e^{-a_F x/0.05 - b_F r/0.1}/\tau_{Fo}$,
  $D^* = d_0(1-m_1 s(x))(1-m_2 S)(1-m_3 F)$; volume $+ c_F(1-g)(r/0.1)^{n_F}$.
- `market_d3`: d2 without the volume term.

## 3. Lab results (9 runs; per-fold LOO mean)

| fold | y3 r9 | d | d2 | d3 |
|---|---:|---:|---:|---:|
| hold_rec | 0.725 | 0.681 | 0.707 | 0.716 |
| pulse40 | 0.589 | **0.631** | 0.502 | 0.496 |
| mid40 | 0.579 | 0.602 | 0.636 | 0.682 |
| multilevel | **0.589** | 0.490 | 0.488 | 0.445 |
| compose | 0.593 | 0.599 | 0.629 | 0.641 |
| rate_hold | 0.603 | 0.590 | 0.636 | 0.592 |
| testlike (exam) | **0.581** | 0.533 | 0.523 | 0.552 |
| voi | 0.522 | 0.512 | 0.518 | 0.520 |
| p7 (P/V/D) | 0.293 (.149/.637/.091) | 0.295 (.145/.650/.091) | 0.267 (.091/.619/.091) | 0.267 (.091/.619/.091) |
| **LOO** | **0.564** | 0.548 | 0.545 | 0.546 |
| in-sample p7 | 0.324 | 0.405 | **0.680** (.676/.706/.659) | 0.664 |
| in-sample mean | 0.624 | 0.614 | 0.652 | 0.652 |

Readings:
1. **The structure explains p7**: in-sample p7 0.32 → 0.68 (depth 0.09 → 0.66) while the other 8 runs lose only
   0.012 in-sample (0.661 → 0.649). The full fits put $x_c = 0.0458$ (upper bound), $\Delta x_c ≈ 0.010$,
   $r_c ≈ 0.0926$, so $x_c(0.0949) ≈ 0.038$; the fast freeze thinning $m_2$ drops 0.50 → 0.18 and the slow $F$
   takes over ($m_3$ 0.95, $\tau_{Fi}$ ≈ 65–72, release τ 1 at (0, 0) but ≈ 245 at (0.05, 0.025)).
2. **The p7 fold cannot show it**: no other run identifies a freeze at tax 0.040, so without p7 the fit has no
   reason to move $\Delta x_c$; the p7 fold stays at 0.27–0.30 for every family. This is a one-run structure.
3. **The other folds lose ≈ 0.018** (mean of folds 1–8: y3 0.598, d 0.580, d2 0.580, d3 0.581), and all three lose
   multilevel (−0.10 to −0.14) and exam; d2/d3 also lose pulse40. Per §1 of the z notes, ±0.02 on the mean is the
   lab's one-polish noise, but the multilevel loss is consistent across all three.
4. 4,000-tick holds (price at 0/0, 0.05/0.025, 0.1/0, 0.095/0.04; depth at 0.095/0.04): y3 86.0 / 80.2 / 76.7,
   depth 44; d2 86.2 / 79.8 / 75.0, depth **2.1** (the tie-up saturates). All start-independent.

## 4. Pinned-structure check (running at the time of writing)

`--free` = y3's 26 parameters, the new ones pinned at the d3 / d2 full-fit values ($m_2$ init 0.2, $x_c$ init
0.0455): reports `plans/market_market_d{,2,3}_ABpin.json`. This asks whether the p7 structure, held fixed, costs
the other 8 folds anything; its p7 fold is not a hold-out (the pins came from fits that saw p7).

## 5. Recommendation

- **Keep `plans/market_market_y3_AB_doc.json`.** No d candidate beats y3 on LOO (0.545–0.548 vs 0.564), and all
  lose the multilevel fold. Forecast public gain = 0.6 × (< 0) ≈ 0.
- The finding worth keeping: p7 is a freeze at (0.095, 0.040), i.e. the trading gate depends on the rate. If a
  test episode holds r ≳ 0.093 with tax 0.037–0.043, y3 crashes the price to the floor and holds depth at 44,
  while the truth slides slowly and drains depth. `plans/market_market_d2_AB_doc.json` is the hedge for that case
  (in-sample 0.652 vs 0.624), not a replacement, until a second run confirms the gate.
- One cheap confirming experiment if credits remain: 60 ticks at (0.095, 0.037) and 60 at (0.088, 0.040) from a
  settled state; it would identify $r_c$ and $\Delta x_c$ outside p7.

Files: `gtlab/ode/market_d.py`, `market_d2.py`, `market_d3.py`; `plans/market_market_d*_AB{,pin}.json` + `_doc.json`.
