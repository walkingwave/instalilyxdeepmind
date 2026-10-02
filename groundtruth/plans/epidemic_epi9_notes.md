# Epidemic epi9: structure round on the six-run set (with p8)

Mon Sep 28. No credits. Runs: p1 120, p2 400, p3 291, p4 300, p6 300, p8 400 (180 ticks at
(0.5, 0.5, 0.0015), then 220 at zero). Every LOO at $1.0\sigma$ = (12.95, 5.23), 150 s per fold,
8 starts, nfev 50 (ode_lab settings). Repeat fits of the same family move the mean by up to 0.004
and single folds by up to 0.02 (time-budgeted fits on a shared machine).

## Where y2 loses (fold fits, held out)
- p6 (0.55-0.57, worst): trained without p6, mask 1 alone is too weak (first wave 272 vs 253) and
  vaccination alone too strong (second wave 121 vs 151, tail 68 vs 87).
- p3 (0.65): trained with p6, mask 0.85 alone is too strong (21 -> 15) and the vaccination-alone
  segment is again too low (bias -1.0 sigma); closure-only first wave peaks early and falls fast (0.37).
- The shipped 5-run y2 on p8 (held out): 0.768 (cases 0.81, beds 0.73). No leak; beds lead truth
  by ~4 ticks after the release.
- v8g_Bonly plan (untracked): old 4-run fit, in-sample only 0.694, no LOO. Not a candidate.

## Structures tested (all nest y2; pair AC)
| family | change | p1 | p2 | p3 | p4 | p6 | p8 | LOO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| y2 (baseline, 2 runs) | - | .742 | .734 | .652 | .729/.732 | .551/.567 | .760/.768 | .695/.699 |
| epi9a | contacts / (1 + k B), B = memory of reported cases | .725 | .721 | .653 | .748 | .563 | .728 | .690 |
| epi9c | age-specific mask protection (children, elders) | .694 | .662 | .643 | .745 | .449 | .778 | .662 |
| epi9d | curved vaccination (vac/0.003)^q | .726 | .725 | .653 | .745 | .552 | .744 | .691 |
| epi9e | vaccine immunity wanes on its own clock | .735 | .723 | .653 | .721 | .541 | .736 | .685 |
| epi9g | closure blunts masks: mask x (1 - k closure) | .693 | .716 | .700 | .717 | .546 | .778 | .692 |
| epi9h | vaccination x (1 + k restriction) | .741 | .734 | .652 | .736 | .565 | .767 | .699 |
| epi9gh | g + h | .694 | .687 | .704 | .721 | .565 | .780 | .692 |
| epi9gi | g + curved masks | .760 | .654 | .652 | .719 | .589 | .692 | .678 |
| y7 | curved masks (q) | .778 | .737 | .654 | .643 | .561 | .641 | .669 |
| c (control-audit lane) | fuller closure term | .704 | .734 | .673 | .708 | .590 | .781 | .698 |

Nothing beats y2 alone. y7's curved masks, left open by the y notes, are now settled by p8: they
lose p4 and p8 (0.64 each). Hospital-driven behaviour (BEH=2) and overflow (OVF) were written but
not run (time box).

## Ensemble (saved fold predictions, scripts/lab_epi9_loo.py + lab_epi9_ens.py)
| combo (mean) | p1 | p2 | p3 | p4 | p6 | p8 | LOO |
|---|---:|---:|---:|---:|---:|---:|---:|
| y2 | .742 | .734 | .652 | .732 | .567 | .768 | .699 |
| **y2 + c** | .725 | .741 | .660 | .752 | .578 | .774 | **.705** |
| y2 + g | .726 | .732 | .679 | .741 | .562 | .789 | .705 |
| c + g + h + y2 | .725 | .742 | .671 | .746 | .569 | .783 | .706 |

y2 + c: +0.006, up on 5 of 6 folds incl. the pulse/recovery fold p2 (+0.007) and the long hold p8
(+0.006); loses p1 (-0.017, the 120-tick free wave). y2 + g loses p2. Best of ~30 combos, so the
+0.006 carries selection optimism. Docs at n_sub 2: `plans/epidemic_epi9_y2n2_doc.json` (y2 refit
on 6 runs) + `plans/epidemic_epi9_cn2_doc.json` (c_ca fit on 6 runs); cfg `plans/epidemic_epi9_ens_cfg.json`.
Pair: finite on all four 4,000-tick eval shapes, 1.8 s per episode on a loaded machine, in-sample 0.7626
(y2 alone 0.7618). Forecast: 0.6 x 0.006 ≈ +0.004 public: noise level.
