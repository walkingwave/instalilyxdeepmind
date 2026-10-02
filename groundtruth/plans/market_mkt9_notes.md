# market mkt9: rate-dependent freeze, frozen-flow funding tie-up, committees (Mon Sep 28, no credits)

Protocol: leave-one-run-out on all 9 runs at 1.0 sigma (0.816 / 0.643 / 2.077), ode_lab settings (150 s per fold,
8 starts, 60 nfev), held-out predictions saved per fold (`scripts/lab_mkt9_loo.py`, reports
`plans/market_mkt9_<tag>_loo.json`), committees scored from the saved folds (`scripts/lab_mkt9_ens.py`).
All runs in this round shared the same machine load, so they compare with each other; y3 reproduces the lab
(0.568 here vs 0.564 in `plans/market_market_y3_r9.json`).

**Result: nothing beats y3 AB under the rules. Keep `plans/market_market_y3_AB_doc.json`.**

## 1. Structures

- `market_mkt9b` (tag g9b): y3 + rate-dependent gate $x_c(r) = x_c - \Delta x_c\,s((r - r_c)/0.002)$ with $x_c$ held in
  [0.0440, 0.0452] (so the low-rate freeze (0.032, 0.046) in multilevel stays frozen), $\Delta x_c$ init 0.0055, $r_c$ 0.090.
- g9c: same, gate pinned at (0.0055, 0.090), y3's 26 parameters free (p7 fold NOT a hold-out: pins came from p7).
- `market_mkt9a` (g9a): g9b + funding tie-up built only by flow executed during a freeze,
  $\dot F = (1-g)(r/0.1)^2 (|v|/0.1)(1-F)/\tau_{Fi} - F e^{-1.5 x/0.05 - b_F r/0.1}/\tau_{Fo}$, $D^* \times (1 - m_3 F)$.
  Idea: a freeze at a settled price (multilevel 116-181) builds no F, a freeze with a sliding price (p7) does.

## 2. Per-fold LOO (mean of P/V/D)

| model | hold_rec | pulse40 | mid40 | multilevel | compose | rate_hold | exam | voi | p7 long | **LOO** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y3 AB (pick) | 0.717 | 0.596 | 0.597 | **0.589** | 0.624 | 0.599 | **0.576** | 0.523 | **0.287** | 0.568 |
| g9b gate | 0.768 | 0.598 | 0.673 | 0.480 | 0.645 | 0.687 | 0.566 | 0.512 | 0.268 | 0.577 |
| g9c gate pinned | 0.747 | 0.545 | 0.644 | 0.466 | 0.610 | 0.527 | 0.563 | 0.510 | (0.319) | 0.548 |
| g9a gate + F | 0.737 | 0.530 | 0.656 | 0.465 | 0.632 | 0.589 | 0.527 | 0.575 | 0.267 | 0.553 |
| d3 | 0.747 | 0.492 | 0.671 | 0.454 | 0.596 | 0.644 | 0.559 | 0.519 | 0.280 | 0.551 |
| z12 | 0.673 | 0.584 | 0.636 | 0.480 | 0.595 | 0.585 | 0.412 | 0.497 | 0.273 | 0.526 |
| y3 BC | 0.675 | 0.638 | 0.607 | 0.477 | 0.653 | 0.596 | 0.542 | 0.515 | 0.269 | 0.552 |
| y3 + 0.25 d3 | 0.730 | 0.627 | 0.614 | 0.555 | 0.617 | 0.604 | 0.572 | 0.591 | 0.281 | 0.577 |
| med(y3, g9b, z12) | 0.749 | 0.599 | 0.675 | 0.587 | 0.641 | 0.620 | 0.564 | 0.502 | 0.270 | 0.579 |
| med(y3, g9b, g9a, d3, y3BC) | 0.748 | 0.654 | 0.663 | 0.503 | 0.644 | 0.653 | 0.563 | 0.564 | 0.268 | 0.584 |

Every single structure and every committee (≈ 60 mean/median/weighted combinations of 7 members) that gains on the
mean loses the exam fold (−0.010 to −0.05) and the p7 long hold (−0.006 to −0.02); all but med(y3, g9b, z12) also
lose multilevel by ≥ 0.03. Best-of-60 selection alone is worth ≈ +0.01, so +0.011 to +0.016 is not evidence.

## 3. What we learned

1. **The p7 regime cannot be validated by leave-one-out.** With p7 held out, the fit shrinks the gate drop: g9b's
   p7 fold fit takes $\Delta x_c$ 0.0055 → 0.0025 (g9a: → 0.0014), $x_c(0.0949)$ = 0.0427 / 0.0438 > 0.0403, so p7
   trades and the price crashes (p7 price 0.09 in every held-out fit). With p7 in, every fold fit keeps
   $x_c(0.0949)$ ≈ 0.039. The other eight runs do not ask for the rate-dependent freeze; testlike's (0.0951, 0.0436)
   blocks are satisfied by a drop of 0.001.
2. **A linear trade/freeze boundary is impossible**: (0.085, 0.0425) trades, (0.0949, 0.0403) freezes needs
   $\beta > 0.22$ in "freeze iff $x + \beta r > \theta$", while (0.0323, 0.0459) frozen and (0.0965, 0.0362) trading
   then need $\theta$ < 0.054 and > 0.064. The boundary is flat in r up to ≈ 0.085 and drops above ≈ 0.09, or it
   depends on state.
3. **multilevel is decided by the frozen slide speed $g_l$**: truth holds 79 through 80 frozen ticks at rate 0.1;
   y3's fold fit has $g_l$ 0.036 (holds 79), every other structure's fold fit $g_l$ ≈ 0.12 (slides to 74–76, fold −0.1).
   p7 needs the opposite: a steady −0.08/tick frozen slide from 104. That tension, not the depth, is what loses
   multilevel for every p7-aware structure (d, d3, z12, g9a/b/c).
4. The frozen-flow tie-up (g9a) fits the idea but its depth loss in pulse40's freeze (0.575 → 0.425) outweighs it.

## 4. Recommendation

Keep y3 AB. Forecast public gain of any mkt9 candidate: 0.6 × (≤ 0 on the ruled folds) ≈ 0. If a sustained/pulse
hedge is wanted for rate ≥ 0.093 with tax 0.036–0.044 (≈ 14 % of recovery pulses at alpha ~ U(0.7, 1)), the
full-fit gate is the structure, but it rests on p7 alone.

Files: `gtlab/ode/market_mkt9a.py`, `gtlab/ode/market_mkt9b.py`, `scripts/lab_mkt9_loo.py`, `scripts/lab_mkt9_ens.py`,
`plans/market_mkt9_{y3,y3BC,d3,z12,g9a,g9b,g9c}_loo.json`.
