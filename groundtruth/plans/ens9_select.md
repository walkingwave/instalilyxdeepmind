# ens9: ensembles across fitted models, all ten systems (Mon Sep 28, no credits)

## Protocol

- Members per system: the current pick plus 3-4 structurally different fitted families, plus l0b_lin.
- Held-out predictions: every member refit cold (family default init, 6 LHS starts, spread 0.5, 50
  evaluations, 60 s per fold, no early phase) on all runs but one, at the calibrated sigma (1.0x);
  every run is a fold (exams, pulse runs and the p7/p8 long holds included). `scripts/lab_ens9_loo.py`.
  Seed check (seed 1 vs 0): supply v8b 0.839/0.839, v8c 0.898/0.897, social z20 0.598/0.594, z21 0.628/0.625.
- Finalize with the Final-slot-1 clip vectors; post rules applied after combining.
- Option pool per observable: single, median of 2 (= mean) or 3, median of all ODEs, median of all;
  ext (needs a new runtime kind): horizon switch A for H ticks after each control change then B
  (H 5-80), fixed mixtures w A + (1-w) B.
- Strategies scored by nested leave-one-run-out (choice for fold k made on the other folds only):
  sys_best, obs_best, obs_best_ext, and "median of the top-k members by inner LOO" (k = 2, 3; per
  system or per observable). `scripts/lab_ens9_select.py` -> `plans/ens9_select.json`.
- Acceptance: nested gain > 0; all-fold gain > 0.005; does not lose the pulse / recovery-shaped
  fold or the exam fold by more than 0.002; fold wins reported. Forecast public = 0.6 x held-out gain.

## Result per system (held-out mean over folds)

| system | pick (LOO) | best nested strategy | candidate | held-out | fold wins | guard folds | ship? |
|---|---:|---|---|---:|---|---|---|
| ad_auction | 0.859 | none > 0 | keep | 0.859 | - | - | keep |
| epidemic | 0.684 | none > 0 (obs_best -0.005) | keep | 0.684 | - | - | keep |
| hospital_queue | 0.675 | obs_best +0.033, top2 +0.027 | med(p3,y5) all obs | 0.693 (+0.018) | 8/8 | pulse40 +0.019, long recovery +0.035, exam +0.017 | yes, with the public-transfer caveat |
| market | 0.564 | top3 +0.010 | price = med(y3,d3,z8), volume/depth = y3 | 0.575 (+0.011) | 7/9 | pulse40 +0.001, testlike -0.004 | marginal |
| power_grid | 0.697 | top3_sys +0.005, top2_obs +0.004 | med(v9c,w5) all obs | 0.713 (+0.016) | 4/5 | pulse +0.009, exam +0.047, p8 +0.062; hold_rec -0.047 | yes |
| reservoir | 0.841 | none > 0 (3 folds) | keep | 0.841 | - | - | keep |
| social_contagion | 0.598 | top2_sys +0.022, sys_best (z21) +0.022 | med(z20,z21) all obs | 0.622 (+0.024) | 5/6 | pulse +0.012, testlike +0.056 | yes |
| supply_chain | 0.839 | top3_sys +0.048 (unstable across strategies) | v8c alone (family swap, not an ensemble) | 0.898 (+0.059) | 1/4 | pulse200 +0.313; hold_mid -0.069 | only with a public probe |
| traffic | 0.745 | none > 0 | keep | 0.745 | - | testlike lost by every blend | keep |
| wildlife | 0.684 | none > 0 (obs_best -0.031) | keep | 0.684 | - | - | keep |

Horizon switches and fixed mixtures never won a nested comparison (they top the all-fold tables,
then lose when chosen out of sample): no new runtime kind is needed.

### Fold tables (held-out, mean over observables)

social_contagion

| fold | z20 | z21 | med(z20,z21) | med(z21,z15) |
|---|---:|---:|---:|---:|
| p1.hold_rec | 0.774 | 0.888 | 0.830 | 0.837 |
| p2.pulse200_200 | 0.560 | 0.592 | 0.573 | 0.575 |
| p3.compose | 0.599 | 0.575 | 0.620 | 0.626 |
| p4.interior_holds | 0.576 | 0.568 | 0.565 | 0.565 |
| p5.testlike | 0.589 | 0.674 | 0.645 | 0.661 |
| p6.voi | 0.486 | 0.471 | 0.498 | 0.477 |
| mean | 0.598 | 0.628 | 0.622 | 0.623 |

power_grid (pick = v9c; v9c; med(v8k,min))

| fold | pick | w5 | med(v9c,w5) | med(v9c,v8k,w5) |
|---|---:|---:|---:|---:|
| p1.hold_rec | 0.819 | 0.721 | 0.771 | 0.795 |
| p2.pulse60_120 | 0.701 | 0.694 | 0.710 | 0.715 |
| p2.multilevel200 | 0.754 | 0.751 | 0.762 | 0.772 |
| p6.exam | 0.612 | 0.685 | 0.660 | 0.638 |
| p8.longhold | 0.601 | 0.638 | 0.662 | 0.626 |
| mean | 0.697 | 0.698 | 0.713 | 0.709 |

If w5 alone becomes the pick: med(v9c,w5) beats it on 4/5 folds (+0.015) but loses the exam (-0.025).

hospital_queue

| fold | p3 | y5 | med(p3,y5) |
|---|---:|---:|---:|
| p1.hold_rec | 0.665 | 0.677 | 0.670 |
| p2.pulse40 | 0.716 | 0.730 | 0.735 |
| p2.mid40 | 0.662 | 0.669 | 0.683 |
| p2.multilevel200 | 0.686 | 0.673 | 0.689 |
| p3.compose | 0.536 | 0.586 | 0.552 |
| p4.pulse_long_recovery | 0.743 | 0.808 | 0.778 |
| p6.voi | 0.712 | 0.777 | 0.744 |
| p6.exam | 0.678 | 0.681 | 0.695 |
| mean | 0.675 | 0.700 | 0.693 |

Caveat: y-family models have beaten p3 on held-out folds before and lost on public (u014 y3 0.666 vs
p3 0.690). The median keeps p3 in every prediction, so it halves that risk.

market (price only changes)

| fold | y3 | price med(y3,d3,z8) |
|---|---:|---:|
| p1.hold_rec | 0.725 | 0.725 |
| p2.pulse40 | 0.589 | 0.590 |
| p2.mid40 | 0.599 | 0.652 |
| p2.multilevel200 | 0.589 | 0.589 |
| p3.compose | 0.591 | 0.608 |
| p4.rate_hold | 0.603 | 0.633 |
| p5.testlike | 0.587 | 0.583 |
| p6.voi | 0.524 | 0.524 |
| p7.longhold | 0.268 | 0.268 |
| mean | 0.564 | 0.575 |

supply_chain

| fold | v8b (pick) | min | min2 | v8c |
|---|---:|---:|---:|---:|
| p1.hold_rec | 0.974 | 0.958 | 0.962 | 0.965 |
| p2.pulse200_200 | 0.609 | 0.604 | 0.755 | 0.922 |
| p3.hold_mid | 0.792 | 0.321 | 0.321 | 0.723 |
| p7.longhold | 0.981 | 0.981 | 0.981 | 0.981 |
| mean | 0.839 | 0.716 | 0.755 | 0.898 |

## Ready-to-ship docs (checked: runtime == manual median, 4,000-tick finite on all four categories)

| doc | model | s/episode (loaded machine) | forecast public |
|---|---|---:|---|
| plans/ens9_social_contagion_doc.json | median(z20 shipped, z21 full refit) | 0.93 | 0.645 -> 0.660 |
| plans/ens9_power_grid_doc.json | median(v9c shipped, w5 full refit) | 0.79 | 0.774 -> 0.783 |
| plans/ens9_market_doc.json | price median(y3 shipped, d3, z8); volume, depth y3 | 1.62 | 0.634 -> 0.640 |
| plans/ens9_hospital_queue_doc.json | median(p3 shipped, y5 full refit) | 0.87 | 0.690 -> 0.701 (risk) |
| plans/ens9_supply_chain_doc.json | v8c full refit | 0.38 | 0.822 -> 0.857 (risk: 1 of 4 folds) |

Full refits (non-pick members): cold, 10 starts, 60 evaluations, 240 s, all current runs.
Build configs: `plans/ens9_cfg_safe.json` (social, power_grid, market changed; rest = final1 docs),
`plans/ens9_cfg_full.json` (also hospital and supply_chain):
`python scripts/build_cfg.py --tag <tag> --cfg plans/ens9_cfg_safe.json`.
Forecast mean: safe +0.003 (0.7555 -> 0.759); full +0.008 (-> 0.763).
