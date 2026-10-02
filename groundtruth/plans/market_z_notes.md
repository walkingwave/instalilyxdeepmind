# market z: keep y3's long-hold gains, recover sequence (Sun Sep 27 night – Mon Sep 28, no credits)

Starting point: u014 ships `market_y3` pair AB (public 0.634: sustained 0.582, sequence 0.651; lab LOO 0.615 on
8 runs, exam 0.591). Goal: LOO ≥ 0.635 without losing the pulse40 or multilevel fold. Judged only by the lab
(`scripts/ode_lab.py --system market --family <fam> --mech <pair> --budget 300 --starts 8 --nfev 60
--sigma-cal 1.0`, σ = 0.816 / 0.643 / 2.077), exam = p5.testlike fold, 4,000-tick holds from price 92 / 100 / 108.
No polishing.

**Result: short. No candidate beats y3 AB (LOO 0.615) on the 8 runs; the structural variants land at
0.568–0.611. Keep `plans/market_market_y3_AB_doc.json`. A new run (p7.longhold) arrived mid-round and shows a
leak none of our models has; the structure aimed at it (`market_z12`) is built but its lab run did not finish.**

## 1. What we learned about the lab itself

- **The lab is deterministic.** A re-run of y3 AB (`plans/market_market_y3_zref.json`) reproduces every fold to
  the third decimal (0.615). The per-fold spread between structures is not timing; it is how the
  Levenberg–Marquardt path from the PARAMS init changes with the structure.
- **Each LOO fold is essentially one polish from the init** (≈ 60 residual + Jacobian evaluations in 150 s; one
  Jacobian on all runs costs 1.7–1.9 s). Later starts rarely complete. So adding parameters that end up unused
  still moves the result:
  - z8 BC = y3 BC plus two parameters BC never reads: 0.603 → 0.597, folds moving by up to 0.10 (compose
    0.643 → 0.545, multilevel 0.466 → 0.524).
  - z5 = y3 plus a freeze-debt state the fit switched off ($k_L$ → 0.0005): 0.615 → 0.568.
  - z7 = y3 with the six depth parameters pinned at the values every fit returns to within 1 %: 0.615 → 0.595.
- Consequence: a structural change must gain well over ≈ 0.02 on the mean before the lab can see it, and "does
  not lose the pulse40 / multilevel fold" is close to a coin flip at the ±0.05–0.10 per-fold level. y3 AB's 0.615
  sits at the top of 16 near-equivalent fits (mean 0.600, sd ≈ 0.011).

## 2. Structures tried

All numpy + math, batched, N_SUB 2; y3's inits for shared parameters, neutral inits for new ones.

| family | pair | change vs y3 | hold_rec | pulse40 | mid40 | multilevel | compose | rate_hold | **exam** | voi | **LOO** | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| y3 (ref) | AB | – | 0.712 | 0.627 | 0.662 | 0.560 | 0.644 | 0.605 | 0.591 | 0.518 | **0.615** | holds 86 / 79.4 / 75.1 |
| y3 | BC | – | 0.673 | 0.629 | 0.662 | 0.466 | 0.643 | 0.576 | 0.586 | 0.591 | 0.603 | |
| z | AB | supported price $A$ moves only through trades ($\dot A \times g_{eff}$) | 0.723 | 0.554 | 0.658 | 0.456 | 0.620 | 0.659 | 0.575 | 0.566 | 0.601 | holds as y3 |
| z | BC | same | 0.681 | **0.655** | 0.666 | 0.398 | 0.633 | 0.624 | 0.570 | 0.585 | 0.602 | |
| z2 | AB | z + y5's two-stage order pipeline | 0.731 | 0.571 | 0.676 | 0.455 | 0.625 | 0.684 | 0.584 | 0.560 | 0.611 | holds as y3 |
| z3 | AB | y3 + unsold output $K$ filling in a freeze, premium $/(1+k_K K)$ | 0.718 | 0.561 | 0.589 | 0.502 | 0.608 | 0.695 | 0.574 | 0.600 | 0.606 | $\tau_f$ 334: barely used |
| z4 | AB | z + unsold output | 0.656 | 0.556 | 0.690 | 0.496 | 0.641 | 0.584 | 0.597 | 0.538 | 0.595 | 7 params at bounds |
| z5 | AB | y3 + freeze debt blocking cash regrowth | 0.728 | 0.549 | 0.582 | 0.488 | 0.604 | 0.520 | 0.553 | 0.517 | 0.568 | $k_L$ → 0 |
| z7 | AB | y3, depth block pinned | 0.716 | 0.646 | 0.655 | 0.479 | 0.586 | 0.560 | 0.582 | 0.535 | 0.595 | |
| z8 | AB | y3 + settlement time for A: $\dot S = (1-g)(r' - S)/\tau_{Si} - gS/\tau_{So}$ | 0.705 | 0.554 | 0.688 | 0.455 | 0.604 | 0.684 | 0.576 | 0.518 | 0.598 | $\tau_{Si}$ 5, $\tau_{So}$ 0.4 |
| z8 | BC | (A unused: noise control) | 0.729 | 0.643 | 0.667 | 0.524 | 0.545 | 0.558 | 0.586 | 0.526 | 0.597 | |
| z9 | AB | cash regrowth on the producers' margin, $\propto \mathrm{softplus}(P - p_{lo})$ | 0.697 | 0.566 | 0.638 | 0.493 | 0.681 | 0.599 | 0.541 | 0.516 | 0.591 | |
| z9 | BC | same | 0.660 | 0.592 | 0.640 | 0.517 | 0.622 | 0.553 | 0.548 | 0.538 | 0.584 | |
| z10 | AB | reset-price memory $\tau_Q$ 80 (y3: 204), $p_{full} \in [89, 99]$ | 0.685 | 0.624 | 0.683 | 0.515 | 0.618 | 0.556 | 0.565 | 0.522 | 0.596 | **holds 89 / 81.8 / 75.0** |
| z11 | AB | $\tau_Q$ fitted (→ 259) | 0.721 | 0.617 | 0.687 | 0.480 | 0.616 | 0.556 | 0.590 | 0.519 | 0.598 | (0.05) → 74.7; freezes do not settle |

Holds = 4,000-tick price at (0, 0) / (0.05, 0.025) / (0.1, 0). Every family settles to a level independent of the
start on trading holds (spread 0 by t 1,000); z11's frozen holds keep a spread of 1.6–2.4.

Readings:
1. **(a) The post-freeze fall.** pulse40 after release and testlike 360–400 keep falling for 40 ticks at zero
   controls (pulse40: flat 10 ticks, then accelerating to −0.5/tick; testlike down to the floor 74), while y3
   recovers because its cash regrows within ≈ 20 ticks ($c_0$ at its upper bound 2 in every fit). A frozen
   anchor (z), a freeze glut (z3, z4), a freeze debt (z5) and margin-driven revenue (z9) each describe the shape,
   but the polish either switches the new term off or trades it against multilevel. The pipeline (z2) again helps
   rate_hold, not pulse40.
2. **(b) The zero-control level.** $p_{full}$ sits at its lower bound in every fit, but it is weakly identified: an
   in-sample profile with $p_{full} \in [91, 95]$ costs 0.003 (0.664 vs 0.667). z10 (level 89; (0.05, 0.025) →
   81.8, closer to voi's 83) costs 0.019 LOO, inside the spread of §1, and its long holds match the observed returns
   (91.5–94.5) better than y3's 86. Not shipped on LOO evidence; it is the hedge for the sustained band.
3. **(c) A vs C.** AB − BC: y3 +0.012, z −0.001, z9 +0.007 (z8 BC is a noise control, not a C test); mean +0.006,
   inside the noise. B stays supported (AC last in y3 by 0.040). p7 (below) is the first run that points at A.

## 3. New data: p7.longhold (300 ticks, arrived during the round)

Controls: (0.0949, 0.0403) for 175 ticks from a reset at 104.7, then (0.05, 0.025) for 125 ticks.

| ticks | price | volume | depth |
|---|---|---|---|
| 0–170 | 104.7 → 90.7, slow slide (−0.08/tick) | 3.0–3.2 throughout | 111 → 43 (t30) → **10.9** (t170), still falling |
| 175–300 | 89.7 → 83.0, settled | 2.9 → 1.8 | 11 → 26.7, slow rebuild (τ ≳ 100) |

y3 AB (fitted on the other 8) scores 0.29 on it (price 0.15, volume 0.64, depth 0.09): it drops the price to the
floor 74 by t 90 and holds depth at the tax level 45. What our structures cannot produce: (i) a sustained sell
flow at a high rate and tax that moves the price slowly while volume stays at 3 and dealer depth drains far below
the tax level; (ii) compose 180–240 at (0.085, 0.0425) shows the opposite (fast fall, depth flat at 44), so the
effect needs a steep rate threshold (0.085 → none, 0.095 → strong). B cannot do it (the price falls too slowly to
build capacity loss at our $k_R$, and y3 cuts B off above tax 0.037). It reads as **A beyond the freeze**: dealers
take on inventory through trades and fund it at the rate.

Every 8-run full fit of this round, scored on p7 (a true hold-out: fitted before it existed), P / V / D, mean:
y3 AB .149/.637/.091 0.293; y3 BC 0.291; y5 AB 0.289; z AB 0.288; z BC 0.286; z2 AB 0.287; z3 AB 0.291;
z4 AB 0.295; z5 AB 0.291; z7 AB 0.293; z8 AB 0.287; z8 BC 0.290; z9 AB 0.272; z9 BC 0.273; z10 AB 0.292;
z11 AB 0.268. Depth is 0.09 for all of them: none of the structures can drain depth below the tax level while
trading.

`market_z12` (built; lab run on 9 runs not finished): y3 +
$\dot F = k_F\,g\,(r/0.1)^{n_F}\,s_d(x)\,(1-F) - F/\tau_F$ under A, depth $\times (1-F)$, price support
$A_1 + k_{FP}F$, volume $+\,c_F \times$ build. No build in a freeze (multilevel holds depth 22 through 80 frozen
ticks at rate 0.1). Only its first fold ran: hold_rec 0.673 vs y3's 0.722 under the same 9-run protocol.

## 4. Recommendation

- **Keep `plans/market_market_y3_AB_doc.json`** (u014). Nothing here beats it on held-out folds; forecast public gain
  from any z candidate = 0.6 × (≤ 0) ≈ 0.
- Sustained hedge if a slot is spare: `plans/market_market_z10_AB_doc.json` (zero-control level 89 instead of 86;
  LOO −0.019, pulse40 −0.003, multilevel −0.045).
- Next: refit on all 9 runs including p7 (y3 AB baseline, then z12 AB against it). p7's depth (≈ 0.09 for every
  shipped model) is the largest single leak we can see, and the sustained band is where it would show.

Files: `gtlab/ode/market_z.py`, `market_z2.py` … `market_z12.py`; lab reports `plans/market_market_z*_{AB,BC}.json`
(+ `_doc.json`), `plans/market_market_y3_zref.json` (y3 AB re-run).
