# social_contagion soc9: ensembles, bagging and the disappointed-pool delay (Mon Sep 28, no credits)

Same 6 runs, calibrated sigma, LOO at 1.0 sigma, fit settings of ode_lab (8 starts, 60 nfev, 300 s full / 150 s
per fold). Scripts: `scripts/lab_soc9_loo.py` (LOO that saves fold thetas and predictions of every run, optional
two-run-out subsets for nested bagging), `scripts/lab_soc9_combine.py` (fold ensembles), `scripts/lab_soc9_hold.py`
(long holds), `scripts/lab_soc9_mk.py` (generates the families). Fold outputs live in the scratch folder `soc9/`.

Fits are wall-clock limited, so a rerun of z20 under tonight's load gave p5 0.614 (stored 0.647): the exam fold
moves by +-0.03 with fit luck alone. Differences below ~0.01 in LOO mean are noise.

| candidate | LOO | p1 | p2 pulse | p3 | p4 | p5 exam | p6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| z20 (rerun, same settings) | 0.598 | 0.770 | 0.560 | 0.594 | 0.569 | 0.614 | 0.480 |
| z20 stored / deep (4x) | 0.603 / 0.607 | 0.770 | 0.560 / 0.585 | 0.594 | 0.569 | 0.647 / 0.645 | 0.480 |
| y10 AB | 0.581 | 0.735 | 0.571 | 0.635 | 0.557 | 0.564 | 0.424 |
| y10 BC | 0.579 | 0.682 | 0.551 | 0.560 | 0.570 | 0.631 | 0.482 |
| mean(z20, y10 AB) | 0.594 | 0.762 | 0.563 | 0.636 | 0.567 | 0.591 | 0.447 |
| mean(z20, y10 BC) | 0.599 | 0.732 | 0.564 | 0.576 | 0.571 | 0.648 | 0.501 |
| mean / median of z20, y10 AB, y10 BC | 0.596 / 0.593 | | | | | | |
| z20 bag mean (6 members, nested) | 0.601 | 0.723 | 0.570 | 0.637 | 0.582 | 0.604 | 0.489 |
| z20 bag median (6) | 0.603 | 0.751 | 0.571 | 0.604 | 0.564 | 0.625 | 0.501 |
| z20 bag mean of 2 / of 4 | 0.589 / 0.585 | | | | | | |
| soc9a: two-stage disappointed pool | 0.604 | 0.770 | 0.566 | 0.599 | 0.568 | 0.635 | 0.487 |
| soc9a deep (4x budget, 24 starts) | 0.607 | 0.770 | 0.574 | 0.599 | 0.568 | 0.644 | 0.486 |
| soc9b: three-stage pool | 0.592 | 0.769 | 0.570 | 0.600 | 0.568 | 0.552 | 0.490 |
| soc9c: soc9a, m0 up to 1 | 0.604 | 0.766 | 0.564 | 0.595 | 0.568 | 0.640 | 0.491 |
| soc9d: soc9c + m0 per community | 0.596 | 0.720 | 0.564 | 0.595 | 0.568 | 0.640 | 0.487 |
| soc9e: soc9a + former members return through the offer | 0.605 | 0.762 | 0.567 | 0.599 | 0.571 | 0.644 | 0.484 |

Findings:
1. Averaging structures does not pay: the families make the same errors (p6 b 0.24 everywhere, p2 b ~0.5).
2. Bagging z20 gains +0.003 (mean) / +0.005 (median) against the same-run P0, not the +0.014..0.024 the fitproc study
   measured on y10 BC; z20's fits are already stable across subsets. Six rollouts per episode also break the 2 s rule.
3. A waiting time before former members reconsider (Erlang-2 disappointed pool) is a tie: deep LOO 0.607 = z20 deep
   0.607, pulse fold 0.574 vs 0.585. The full fits are nearly the same model (zero-control level 95/77 vs 97/79,
   seeding and bridge maps within 2 adopters). Erlang-3, m0 bounds and incentive-led return (k_ret -> 0) add nothing.
4. Verdict: keep z20. No soc9 candidate clears fold noise.
