# market x: agent-flow rebuild (Sun Sep 27, no credits)

Goal: +0.08 public on market (0.588: sustained 0.491, sequence 0.620) with a rebuild from the brief, judged only
by leave-one-run-out at the organizer's scale ($\sigma$ = 0.816, 0.643, 2.077) and by 4,000-tick holds.

**Result: short of the goal.** Best candidate `market_x4` (doc `plans/market_market_x4_x7_doc.json`): lab LOO
over 7 runs at 1.0σ **0.591** vs v9g **0.582** under the same protocol; held-out exam on the new test-shaped run
(`p5.testlike`, fitted on the other 6) **0.525** vs v9g 0.467; calibrated in-sample (7 runs) 0.664 vs 0.646. Its
long holds settle to a level set by the controls, independent of the initial price (v9g keeps the initial price
on 8 of 13 holds). LOO 0.70 was not reached.

Protocol note: the lab's LOO at 1.0σ is stricter than the Powell "calibrated" LOO used for v9g before (0.631):
the same v9g scores 0.591 (6 runs) and 0.582 (7 runs) in the lab. Every number below is the lab at 1.0σ, 200 s,
4 starts, 7 runs (the 7th fold = the exam).

## 1. What the new test-shaped run says (p5.testlike, 400 ticks)

| segment | controls (r, x) | price | reading |
|---|---|---|---|
| [0,77) | (0.05, 0.025) | 99.9 → 88.8, accelerating (−0.1 → −0.3/tick) | a mid rate drains to the floor on long holds; v9g's Hill threshold ($\phi(0.05)=0.03$) says 99 forever |
| [100,120) | (0, 0) | 82.6 → 86.3 | quick rebound |
| [120,140) | (0.095, 0.0436) | +2 then flat; volume 1.9 | **0.0436 is frozen**: the gate sits between 0.0425 (compose, trades) and 0.0436 |
| [140,180) | (0.039, 0.0196) | 87.8 → 80.2, then flat | the release after a freeze moves the price to where cash put it |
| [200,285) | (0, 0) | 82 → 91.8 (logistic: slow, fast, saturating) | zero-control level ≈ 92 again, not the reset price 99.75 |
| [300,324) | pulses (0.1, 0.05) | keeps falling ≈ −0.3/tick inside the freeze | committed orders keep executing under a freeze |
| [348,360) | (0.1, 0) | flat 83.2 | inertia: orders take ~10 ticks to build |
| [360,400) | (0, 0) | 82.2 → 73.6 | delayed fall to the floor after the pulse block, at zero controls |

Depth: (0.039, 0.0196) holds ≈ 60 while 0.025 goes to ≈ 45: the tax effect on dealer quoting is a steep step near
$x \approx 0.019$, not $x/(x+0.005)$. A pure-tax freeze (0, 0.05) takes depth only to ≈ 47, a freeze with the rate
at 0.1 to ≈ 22: the freeze loss scales with the rate.

## 2. Candidates

All: numpy + math, batched f/h/x0, N_SUB = 2, every observed initial used, B/W volume terms and G/H reset-burst
inventory as v9g (physically: reset orders in execution; warehouses' half-full imbalance; burst inventory on dealer
books settling slower under tax).

**market_x (18 params)**: producers' working cash $C$ replaces v9g's anchor:
$\dot C = \rho g (C + c_0)(1 - C) - \delta\,\phi(r)\,C$ (revenue regrows funded production logistically; interest
past the margin drains), $A = p_f + (Q - p_f) C$, $Q$ = reset price (fixed). Follower as v9g; depth with a freeze
loss $\propto r$.

**market_x2 (22)**: a structurally different price law.
- $Q$ relaxes from the reset price to a common fundamental: $\dot Q = (p_{full} - Q)/\tau_Q$.
- Self-reinforcing financing drain on borrowed cash: $\dot C = \rho g (C + c_0)(1-C) - a_r (r/0.1)^{n_r}(1 - C + b_0)$.
  With small $b_0$ any rate above ≈ 0.04 collapses $C$ (accelerating fall, then the floor); a transcritical
  balance gives interior levels only for small rates.
- $A = p_{lo} + (Q - p_{lo})C$; committed orders: $\dot P = v$, $\dot v = (g\,s_v \tanh(\kappa (A-P)/s_v) - v)/\tau_v$
  (the price keeps moving while committed orders execute under a freeze; new orders stop).
- Gate centre $x_c$ fitted; depth $D^* = d_0 (1 - m_1 s((x - x_d)/w_d))(1 - m_2 (1-g) r/0.1)$.

**market_x3 (24)**: x2 + supported-price lag (goods dumped by cash-starved producers must be absorbed before the price
recovers): $\dot A = e_A[(1-s)/\tau_A + s/\tau_{Au}]$ toward $p_{lo} + (Q - p_{lo})C$; stranded-inventory state
$S \to (1-g) r/0.1$ ($\tau_s$) in the freeze depth loss; tax drain dropped (fitted 0); $p_{full} \in [89, 99]$.

**market_x4 (24)**: x3 + a leak through the freeze ($g_l$ of new orders still trade: $\dot v$ uses
$g_l + (1-g_l) g$), $w_r$ fixed at 0.7.

## 3. Results (lab, 1.0σ, 7 runs)

LOO per fold (mean of price, volume, depth):

| held out | v9g | x | x2 | x3 | **x4** |
|---|---:|---:|---:|---:|---:|
| hold_rec | 0.708 | 0.712 | 0.689 | 0.688 | 0.694 |
| pulse40 | 0.552 | 0.540 | 0.536 | 0.617 | 0.589 |
| mid40 | 0.623 | 0.640 | 0.679 | 0.674 | 0.674 |
| multilevel200 | 0.530 | 0.515 | 0.479 | 0.475 | 0.471 |
| compose | 0.591 | 0.572 | 0.530 | 0.560 | 0.567 |
| rate_hold | 0.604 | 0.556 | 0.525 | 0.588 | 0.616 |
| **testlike (exam)** | 0.467 | 0.465 | 0.480 | 0.519 | **0.525** |
| **mean** | 0.582 | 0.571 | 0.559 | 0.589 | **0.591** |
| calibrated in-sample (7 runs) | 0.646 | 0.639 | 0.659 | 0.668 | 0.664 |
| params / at bound | 17 / – | 18 / k_x, c0 | 22 / – | 24 / p_full, a_r, tau_s | 24 / p_full, tau_s |

Exam per observable (x4 fold): price 0.310, volume 0.788, depth 0.477 (v9g 0.258 / 0.713 / 0.429). In-sample on the
exam the x-family reaches 0.62-0.63 (v9g 0.50): the structure can describe the test-shaped run, but only once it
has seen it.

## 4. Long holds (4,000 ticks, initial price 92 / 100 / 108)

| hold (r, x) | v9g t4000 | x4 t450 | x4 t4000 | x4 spread over y0 |
|---|---|---|---|---:|
| (0, 0) | 92 / 100 / 108 | 89.4 / 90.4 / 91.4 | 89.0 | 0 |
| (0.05, 0) | 91.4 / 99.1 / 106.9 | 74.5 | 74.5 | 0 |
| (0.07, 0) | 85.8 / 91.6 / 97.4 | 74.5 | 74.5 | 0 |
| (0.1, 0) | 72.9 / 74.1 / 75.3 | 74.5 | 74.5 | 0 |
| (0, 0.025) | 92 / 100 / 108 | 89.4-91.4 | 89.0 | 0 |
| (0, 0.05) frozen | 92 / 100 / 108 | 90.6-102.7 | 89.0 | 0 (settles by ~2,000) |
| (0.1, 0.05) frozen | 92 / 100 / 108 | 85.9-101.8 | 74.5 | 0 (settles by ~3,000) |
| (0.085, 0.0425) | 74.7-80.7 | 74.5 | 74.5 | 0 |

Volume 1.78-1.79 at t4000 everywhere (2.2 frozen in v9g), depth 22-91, max volume after t10 ≤ 5.7; all finite.
Price settles within 450 ticks on every trading hold and depends only on the controls. Two gaps against the
long-hold audit: the zero-control level sits at the $p_{full}$ lower bound 89 (the audit wants 95-100; our runs
return to 91.5-94.5), and frozen holds take 2,000-3,000 ticks to settle (the audit wants 450-1,000). The audit's
"price rises 10-40 with tax at 0.05" has no support in our runs (every freeze holds or lowers the price), so we did
not build it in.

## 5. Why LOO did not reach 0.70

1. **multilevel fold** (0.47 in the x-family): it is the only run that starts at 106 under a low rate and a tax and
   then freezes for 80 ticks. Without it, the fit puts the gate, the fundamental and the drain curvature elsewhere
   (price 0.30 held out vs 0.67 in-sample).
2. **hold_rec price** (0.40): $p_{full}$ is pulled to 89 by the falls after freezes (pulse40, the exam's end), so a
   run starting at 93.4 drifts down while the data rise to 94.5 (the reset hump at t 40-70 is not modelled).
3. The price is second order with ≈ 10-tick order build-up and a floor at 74.5; each run shows a different
   combination of drain history and freeze, so every fold extrapolates one regime.

## 6. Recommendation and what data would help most

- If market is changed at all, `plans/market_market_x4_x7_doc.json` (x4, fitted on all 7 runs by the lab: no
  score-polishing). It gains +0.009 LOO, +0.058 on the held-out test-shaped run, and fixes the frozen-at-initial-price
  failure that costs v9g's sustained score. Expected public gain is modest (the LOO gain is small); the sustained
  band is where it should show.
- Data (by value): (1) a 300-400-tick zero-control hold started from a high reset price (pins $p_{full}$ vs the
  reset-price premium and the hold_rec hump); (2) a long frozen hold at zero rate (0, 0.05) for 300+ ticks (does a
  frozen price drift, and to where: this is the audit's "rises with tax" question); (3) a 200-tick hold at
  (0.03, 0) or (0.04, 0) with no tax (is there an interior level between 89 and 74.5, i.e. where the drain
  threshold is); (4) a tax step between 0.043 and 0.045 at rate 0 to pin the gate.

Files: `gtlab/ode/market_x.py`, `market_x2.py`, `market_x3.py`, `market_x4.py`; lab reports
`plans/market_market_{v9g_x7ref,x_x7,x2_x7,x3_x7,x4_x7}.json` (+ `_doc.json`), `plans/market_market_v9g_xref10.json`
(v9g, 6 runs).
