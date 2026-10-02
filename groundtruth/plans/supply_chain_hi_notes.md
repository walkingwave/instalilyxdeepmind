# supply_chain hi: discrete-time literal family + medians (Tue Sep 29, no credits)

Question: does a dt = 1 literal-code reading (integer 3-tick delay line, hard min() caps, orders
withdraw available stock, conveyor capacity) beat the pick (final1 v8b, public 0.8217) with gains
spread across folds? Protocol: leave-one-run-out on the 4 runs at the calibrated sigma
[9.0, 98.1, 234.7] (1.0x). v8b / v8c held-out predictions from the ens9 cache (cold refits);
hi members refit per fold with `scripts/lab_hi_sc_loo.py` (4-6 starts, 60 evals, 300 s).

## Families

| family | reading | free params |
|---|---|---:|
| hi1 (`gtlab/ode/supply_chain_hi1.py`) | per tick: $P = e_p(p_0 + C_2 + C_3)$; $D = \min(o,\ d_q(q_m - Q_{transit}),\ S + P)$; $S' = \min(S+P-D, 362)$; dispatch rides a 3-slot delay line per class into the terminal queue; $A = \min(k_q Q,\ v_0/(1+k_l \ell),\ g(a_0 + a_1 r)(1 - k_m m))$; retail class $c$ sells $\min(d_c + k_d R_c, R_c + A_c)$; commitment $C_1, C_2$ (rate $a_c$), slow $C_3 \to k_s o$ ($\tau_s$); wear $W' = W + (1-m)/100 - k_{cool} m W$, $g = 1 - w_l[W>1]$ | 19 |
| hi2 (`gtlab/ode/supply_chain_hi2.py`) | hi1 with the literal constants pinned: terminal cap $8 + 32r$ (19.2 at r 0.35, 37.6 at 0.925, 56 at 1.5: matches all four runs), $d_q = 1$ (fill conveyor to capacity), $a_c = 1$, wear as v8c; transport cap $v_0 (1 - k_l \ell)^+$ (11.7 at lead 1 and >= 39 at 0.5 cannot both hold with $v_0/(1+k_l\ell)$) | 13 |

Runtime: the shared RK4 rollout runs with $f = 0$ and the map is applied inside $h$ (writes the new
state into $x$ in place). Parity vs a plain loop: 0.0 on all runs, batched fit path included.

## Per-fold held-out table (mean over the 3 observables)

| model | hold_rec | pulse200_200 | hold_mid | longhold | mean | folds > v8b |
|---|---:|---:|---:|---:|---:|---|
| v8b (pick) | 0.974 | 0.609 | **0.792** | 0.981 | 0.839 | - |
| v8c | 0.965 | 0.922 | 0.723 | 0.981 | 0.898 | 1/4 |
| med(v8b, v8c) | 0.969 | 0.670 | 0.792 | 0.981 | 0.853 | 1/4 (+0.061 pulse, rest ties) |
| hi1 warm | 0.980 | 0.442 | 0.353 | 0.983 | 0.690 | 2/4 |
| hi2 warm (theta0 = v8c full fit) | 0.972 | 0.899 | 0.651 | 0.982 | 0.876 | 2/4 |
| hi2 cold (family defaults) | 0.970 | 0.914 | 0.646 | 0.983 | 0.878 | 2/4 |
| med(v8b, v8c, hi2 cold) | 0.966 | 0.922 | 0.726 | 0.982 | **0.899** | 2/4 |
| med(v8c, hi2 cold) | 0.967 | 0.918 | 0.676 | 0.982 | 0.886 | 2/4 |
| med(v8b, hi2 cold) | 0.971 | 0.667 | 0.714 | 0.983 | 0.833 | 2/4 |

Full table: `plans/supply_chain_hi_folds.json` (`scripts/lab_hi_sc_combine.py`).
In-sample (full fit): hi2 0.985 / 0.933 / 0.852 / 0.983, v8b 0.973 / 0.925 / 0.861 / 0.981.
hi2 segments held out: pulse t200-300 0.81-0.84 (v8b warm 0.814), hold_mid t300-450 0.57.

## Reading

- hi1 (19 free) is not identified by 3 runs: the pulse fold puts the terminal intercept at 81
  (unpinned at r = 0.35), the hold_mid fold sets the transport cap to 19.7. Pinning the literal
  constants (hi2) fixes the pulse fold (0.44 -> 0.91) but not hold_mid: retail held out 0.35
  (class demand at mix 0.65 is only seen in p3).
- Every candidate trades the same two folds: +0.3 on pulse200, -0.07 to -0.15 on hold_mid, ties on
  the two recovery holds. That is the same profile as v8c. Gains are not spread (rule from the
  power_grid lesson), so nothing here passes.
- The pulse fold gap of v8b is protocol-sensitive: warm refit 0.866, cold refit 0.609. The pick is
  trained on p2, so its public pulse/recovery episodes sit between those numbers.
- med(v8b, v8c): only the pulse fold moves (+0.061), others tie. Cheapest hedge if one is wanted,
  still one fold.

## Alternation (p3 t237-450, shipments, sigma 9)

Even ticks 29.35, odd ticks 44.74: phase is locked to the tick count (even low from t238 to the end).
Score on those ticks: mean 37.4 -> 0.544, low mode 29.35 -> 0.675, high mode -> 0.619, exact parity
-> 0.924. On the p3 run mean that is +0.021 (low mode) or +0.06 (parity). Not used: seen in one run
only, onset (t236) not predicted by any fitted model, no alternation on p2 or p7 at their caps.
If a rule is ever added: predict the LOW mode, not the mean (concave score).

## Doc and forecast

- `plans/supply_chain_hi_med3_doc.json` = median(v8b final1, v8c alt1, hi2 full fit). Checks:
  runtime == manual median (0.0), finite on 4,000-tick sustained / order / recovery / composition,
  in-sample 0.976 / 0.933 / 0.861 / 0.982. Runtime 3.7-6.1 s per 4,000-tick episode on the loaded
  machine (hi2 alone 1.3-3.0 s): 3-4x v8b; check the 1,200 s budget before any use.
- `plans/supply_chain_hi_supply_chain_hi2_a_doc.json` = hi2 alone.
- Forecast (public = 0.6 x held-out gain vs v8b): med3 +0.060 -> 0.858, hi2 +0.039 -> 0.845,
  med(v8b,v8c) +0.014 -> 0.830. All three rest on the pulse fold alone and lose hold_mid, so the
  realistic range is 0.80-0.86.

Decision: keep v8b. If the alt1b probe (v8c) beats v8b on public, v8c stays the better swap
(med3 adds +0.001 held-out at 3-4x runtime).
