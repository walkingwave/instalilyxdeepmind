# market canon: "what would the simulator code literally look like" (Mon Sep 28, no credits)

Protocol: leave-one-run-out on all 9 runs at 1.0 sigma (0.816 / 0.643 / 2.077), ode_lab settings (150 s per fold,
8 starts, 60 nfev, module-default init), `scripts/lab_canon_market_loo.py` (= lab_mkt9_loo with its own output
`plans/market_canon_<tag>_loo.json`). Reference = y3 AB folds in `plans/market_mkt9_y3_loo.json` (same protocol).

**Result: no canon family passes the rule. Keep y3 AB (`plans/market_market_y3_AB_doc.json`).** The literal hard
switch does explain the boundary: **hysteresis** (the book closes above $x_{hi} pprox 0.044$ and reopens only below
$x_{lo} pprox 0.039$; it starts closed at reset) fits every trade/freeze block with no rate term, and a slow funding
pool $F$ built only while frozen at a rate explains the p7 depth drain and its slow recovery (in-sample p7 0.325 ->
0.66). Out of sample, canon4 (hard hysteresis gate only) is the first structure that gains on the p7 fold (0.287 ->
0.377) and on the mean (0.568 -> 0.573), but it loses pulse40 (-0.007), exam (-0.008) and multilevel (-0.093).

## 1. Brief wording (kit/briefs.md, market)

Observables price, volume, depth; controls interest_rate [0, 0.1], transaction_tax [0, 0.05] ("fractions per trading
period"). Base: producer/consumer groups with reservation values, storage capacities, placement speeds; production
consumes working cash, consumption returns revenue; orders go through preparation and execution and "a new policy
does not cancel commitments already made"; trades move goods and cash between customers and finite dealer books.
Mechanisms (two of three): A inventory ties up funding until settlement, B adverse price moves reduce risk capacity,
C investors shift exposure toward recent winners. Reset: warehouses half full, full cash, books neutral, orders and
settlement commitments empty.

## 2. The code we would expect from that spec

The brief is not a textbook spec (it reads like an agent-flow sim), so the literal guess is a tick loop with hard
branches. Three sketches, Euler, one tick = one period:

**V1 dealer-quote switch (the one we tested).**
```
quoting = False                                   # books neutral, commitments empty at reset
for t:
    if quoting and tax > X_HI: quoting = False     # margin after tax gone: dealers pull quotes
    elif not quoting and tax < X_LO: quoting = True  # restart needs a clearly positive margin
    target = P_FULL - K_R * rate * cash_gap ...    # rate = carrying cost of working cash / storage
    flow   = clip(k * (target - price), -F_MAX, F_MAX) if quoting else LEAK * (target - price)
    orders.append(flow); price += orders.pop(0)    # preparation/execution pipeline, not cancelled
    depth  = D0 / (1 + tax / X_D) * (1 - funding_tied) * (1 - risk_loss)
```
**V2 margin rule without memory.** `quoting = spread(state) > tax + rate * HOLD`. Linear in (tax, rate) unless the
spread is state dependent; see §3 for why it fails.

**V3 funding pool.** `funding -= rate * inventory; funding += settled`; inventory builds only when books cannot
unload (frozen), settlement slower at high rate; `depth = D0 * funding / F0`. This is the literal form of mechanism A
and the candidate for the p7 drain (F in canon2/3).

## 3. The trade/freeze boundary: hysteresis fits every block, a function of (rate, tax) does not

Every block with its entry state (frozen = price stops tracking the rate target, volume at the floor):

| run / tick | (rate, tax) | entered from | observed |
|---|---|---|---|
| multilevel 0 | (0.0227, 0.0342) | reset | trades |
| p7 0 | (0.0949, 0.0403) | reset | **frozen** (slide −0.09/tick with a 30-point gap) |
| pulse40 0 | (0.1, 0.05) | reset | frozen |
| testlike 0, voi, mid40 | (0.05, 0.025) | reset | trade |
| compose 90 | (0, 0.0425) | trading | trades |
| compose 180 | (0.085, 0.0425) | trading | trades (−0.4/tick) |
| testlike 77 | (0.0965, 0.0362) | trading | trades |
| testlike 120, 180 | (0.0951, 0.0436) | trading | frozen |
| multilevel 95 | (0.1, 0.05) | trading | frozen |
| multilevel 116, 132 | (0.0888, 0.0476), (0.0323, 0.0459) | frozen | stay frozen |
| testlike 140 | (0.0389, 0.0196) | frozen | reopens |

A static rule "freeze iff $x > \theta(r)$" needs $\theta(0.085) > 0.0425$, $\theta(0.0949) < 0.0403$,
$\theta(0.0965) > 0.0362$, $\theta(0.0323) < 0.0459$: only a cliff at $r \approx 0.09$ passes (mkt9 g9b), and nothing
but p7 asks for it. A state-dependent threshold on producers' cash also fails (y3's own states: compose 180 has
$C = 0.99$ and trades, p7 $C = 1.00$ and freezes, both at similar rates).

Hysteresis with the book **closed at reset** fits all blocks with no rate term:
$$\text{open} \to \text{frozen iff } x > x_{hi},\ x_{hi} \in (0.0425, 0.0436); \qquad
\text{frozen (or reset)} \to \text{open iff } x < x_{lo},\ x_{lo} \in (0.0342, 0.0403).$$
p7 is simply "tax 0.0403 at reset is above the opening threshold"; compose 180 trades at 0.0425 because the book was
already open. Without p7 the data give only $x_{lo} \in (0.0342, x_{hi})$; our p7-blind init was the midpoint of
(0.0342, 0.043) = 0.0386, which happens to sit below 0.0403 (a flat prior on that interval does so with p ≈ 0.73).

Smooth version used in the fits (state $g$, reset $g = 0$, $\tau_g = 1$):
$$\dot g = \big(s((x_{lo} + (x_{hi} - x_{lo}) g - x)/w) - g\big)/\tau_g .$$
**Width matters.** With y3's $w = 5\times10^{-4}$ a tax 0.0005 above $x_{lo}$ starts $g$ at $s(-1) = 0.27$, above the
unstable point $(x - x_{lo})/(x_{hi} - x_{lo}) \approx 0.1$, and the book tips open: every canon1/canon2 p7 fold fit
had $x_{lo}$ = 0.0395–0.0398 < 0.0403 and still traded p7 (price crash to 74, fold 0.27). canon3/canon4 use
$w = 10^{-4}$ (a literal hard switch).

## 4. Families

| family | = |
|---|---|
| `market_canon1` | y3 + hysteretic gate state g ($x_c$ = $x_{hi}$, new $x_{lo}$), $w = 5\times10^{-4}$ |
| `market_canon2` | canon1 + slow funding tie-up F (mech A) + frozen linear leak $\lambda_f (A - P)$ (nests y3) |
| `market_canon3` | canon2 with $w = 10^{-4}$ and frozen price flow = linear leak only (no $g_l$ push) |
| `market_canon4` | canon1 with $w = 10^{-4}$ |

$\dot F = (1-g)(r/0.1)^{n_F}(1-F)/\tau_{Fi} - F\,e^{-b_F r/0.1 - a_F x/0.05}/\tau_{Fo}$, $D^* \leftarrow D^*(1 - m_3 F)$.

## 5. Leave-one-run-out (mean of P/V/D per held-out run)

| model | hold_rec | pulse40 | mid40 | multilevel | compose | rate_hold | exam | voi | p7 long | **LOO** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y3 AB | 0.717 | 0.596 | 0.597 | 0.589 | 0.624 | 0.599 | 0.576 | 0.523 | 0.287 | 0.568 |
| canon1 | 0.725 | 0.607 | 0.662 | 0.484 | 0.614 | 0.496 | 0.576 | 0.518 | 0.273 | 0.551 |
| canon2 | 0.683 | 0.627 | 0.599 | 0.455 | 0.635 | 0.620 | 0.555 | 0.510 | 0.279 | 0.551 |
| canon3 | 0.669 | 0.592 | 0.616 | 0.451 | 0.619 | 0.612 | 0.572 | 0.563 | 0.285 | 0.553 |
| canon4 | 0.712 | 0.589 | 0.658 | 0.496 | 0.635 | 0.555 | 0.568 | 0.565 | 0.377 | 0.573 |

Rule: LOO mean up, pulse40 and exam (testlike) not below y3 AB. canon1 ties exam and gains pulse40 but loses the
mean; canon4 gains the mean and p7 but loses pulse40 and exam by < 0.01 and multilevel by 0.09. **None passes.**

Fold gate thresholds: $x_{lo}$ drifts up from 0.0386 to 0.0391-0.0403 in the fits (flat objective; nothing but p7
has a tax in (0.0386, 0.0403)). In the p7 fold: canon1/2 ($w$ = 5e-4) tip open, canon3 lands at 0.0403 = p7's tax
and trades, canon4 (0.0391, $w$ = 1e-4) freezes: price 105 -> 100 by 168 (truth 90; frozen slide too slow, $g_l$
0.06), depth 24 (truth 11), then reopens at (0.05, 0.025) and falls to 75 (truth 83). The gain is the gate, not the
drain.

## 6. In-sample (all 9 runs): which literal form explains p7

| full fit, 9 runs, in-sample (300 s) | hold_rec | pulse40 | mid40 | multilevel200 | compose | rate_hold | testlike | voi | longhold | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y3 AB | 0.752 | 0.660 | 0.641 | 0.639 | 0.678 | 0.639 | 0.625 | 0.652 | 0.325 | 0.623 |
| canon1 | 0.775 | 0.665 | 0.626 | 0.613 | 0.686 | 0.643 | 0.626 | 0.642 | 0.533 | 0.645 |
| canon2 | 0.760 | 0.649 | 0.638 | 0.635 | 0.687 | 0.619 | 0.640 | 0.630 | 0.432 | 0.632 |
| canon3 | 0.778 | 0.646 | 0.641 | 0.594 | 0.661 | 0.573 | 0.626 | 0.672 | 0.662 | 0.650 |
| canon4 | 0.775 | 0.663 | 0.626 | 0.612 | 0.685 | 0.643 | 0.625 | 0.641 | 0.530 | 0.644 |
| canon3 (p7-informed init) | 0.768 | 0.620 | 0.637 | 0.594 | 0.675 | 0.606 | 0.616 | 0.640 | 0.592 | 0.639 |

- **Boundary + frozen slide: the hysteresis gate.** y3 cannot hold p7 frozen and multilevel at the same time
  (in-sample p7 0.325). canon1/canon4 hold both: p7 0.53, multilevel 0.61 (-0.03), everything else level.
  The old "frozen slide speed" conflict (mkt9 §3.3) was a symptom: y3 had to trade p7 or slide multilevel.
- **Depth drain: slow funding tie-up $F$ (mechanism A, literal V3).** canon3 full fit: $m_3$ 0.92, $n_F$ 0.96,
  $	au_{Fi}$ 116, $	au_{Fo}$ 1.5, $b_F$ 6.7, $a_F$ 3.6. So $F$ fills at $(1-g)(r/0.1)/116$ per tick while frozen,
  empties in 1.5 ticks at zero controls but in ~260 ticks at (0.05, 0.025) ($e^{-3.34-1.81}$). p7 depth 110 -> 12.1 at
  168 (truth 11.1) and 15 -> 24 over 120 ticks after (truth 13.5 -> 25.6). Pulse40's quick recovery at (0, 0) and
  multilevel's rise at (0.032, 0.046) follow from the rate term. canon3 in-sample 0.650 (p7 0.662) vs y3 0.623.
  Out of sample canon3 never sees F build in a long freeze except in p7, so its folds fall back to y3-like depth.
- Seeded from the p7-derived guess ($b_F$ 7, $m_3$ 0.8) the fit lands lower (0.639): the landscape is rough; the
  default-init result above is the better optimum.

## 7. Decision and what it would take

- Keep y3 AB for the Final. The canon evidence is in-sample only for the drain, and the gate's LOO gain is bought
  with multilevel/pulse/exam losses.
- If a p7-regime hedge is wanted (tax 0.040-0.044 applied right after reset or after a freeze), canon3/canon4 full
  fits are the structures; the hysteresis claim is testable for ~30 credits: from reset hold (0.0, 0.041) 15 ticks
  (hysteresis: frozen, price flat; static cliff at r ~ 0.09: trades), then (0, 0) 10, then (0, 0.041) 15 from the
  open book (hysteresis: trades). Not bought; needs Dylan's go.

Files: `gtlab/ode/market_canon{1,2,3,4}.py`, `scripts/lab_canon_market_loo.py`,
`plans/market_canon_{c1,c2,c3,c4}_loo.json` (folds + full fit), `plans/market_canon_y3ref_full.json` (y3 AB full
fit on 9 runs, same budget), `plans/market_canon_c3p7init_full.json`.


## 8. Follow-up: 140-tick discovery schedule and the canon4 multilevel loss

**Schedule `plans/p9_market_alt.json`** (one run from reset, 140 steps, controls [interest_rate, transaction_tax],
built by `scripts/lab_canon_market_p9design.py --write`; leaves 20 of market's 160):

| ticks | (rate, tax) | tests | predicted end state c3 / c4 / y3 (P, D) | separation, mean over block, sigma units |
|---|---|---|---|---|
| 0-40 | (0.06, 0.041) | hysteresis from reset; F at a 2nd rate | frozen 92/42, frozen 92/41, trades 86/53 | canon vs y3: P 2.6-3.4, D 3.4-4.8; c3 vs c4 D 1.4 |
| 40-60 | (0.06, 0) | F release at rate 0.06 | depth 72 / 83 / 84 | c3 vs c4 D 3.2 (the F test) |
| 60-75 | (0.06, 0.041) | same tax from an open book | trades / trades / trades | hysteresis vs static threshold 0.0395: D 2.3 |
| 75-83 | (0.1, 0.05) | close the book | frozen | vs static: D 4.7 |
| 83-95 | (0.1, 0.040) | x_lo step | reopens if x_lo > 0.040 | x_lo 0.039 vs 0.0402: D 3.0 |
| 95-107 | (0.1, 0.038) | x_lo step | | 0.037 vs 0.039: D 3.0; 0.039 vs 0.0402: D 5.1 |
| 107-119 | (0.1, 0.036) | x_lo step | | 0.0355 vs 0.037: D 3.0; 0.037 vs 0.039: D 5.1 |
| 119-140 | (0, 0) | recovery, F release at zero | depth 86-87 all | c3 vs y3 P 4.5, D 2.2 (weakest block, kept as the recovery tail) |

Numbers are for y0 = hold_rec's reset; the other two resets tried (mid, high price) give the same or larger separations.
No block fell below ~2 sigma mean separation on its target pair, so none was dropped. x_hi gets no block: the data
already pin it to (0.0425, 0.0436) (compose 0.0425 trades from open, testlike 0.0436 freezes from open).
Reopening is read mainly from depth (the quote gate moves depth ~5 sigma), price second (rate 0.1 makes a reopened
book fall fast).

**Why canon4 loses multilevel (-0.093).** The loss is all price (P 0.571 -> 0.308; V, D level). In the multilevel
fold canon4 falls too fast in the trading stretch at (0.090, 0.022) (88 at t80 vs truth 90) and then slides on to 75
while frozen (g_l 0.154 in that fold; y3's fold 0.036) where truth holds 79.

Quick variant: canon4 with g_l pinned at y3's default 0.031, everything else refit (tags c4gla/b/c, 3 fold
chunks): LOO 0.557 (hold .748, pulse40 .656, mid40 .627, multilevel .473, compose .593, rate_hold .429,
exam .565, voi .521, p7 .401). Pinning g_l helps pulse40 (+0.06) and p7 (0.401), but multilevel gets worse: that fold
now reaches 77 at t100 (truth 82) during trading and stays there. **So g_l is not the cause.** The multilevel
fold overshoots in the trading phase (flow speed and rate target) and loses rate_hold (0.429). Not fixed. Files:
`plans/market_canon_c4gl{a,b,c}_loo.json`.
