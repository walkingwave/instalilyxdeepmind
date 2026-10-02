# ad_auction str9 (Mon, no credits): no change

Shipped perobs (final1) on the runs it never saw: p6.exam 0.830, p8.longhold 0.890.
Error concentrates in **conversions** (every run 0.75-0.85; win_rate 0.88-0.95, spend 0.87-0.96),
10-50 ticks after a change (0.67-0.73), with a negative level bias (-0.2 to -0.33 sigma):
- steady recovery conversions 3.08 (p8 t150-290) vs model 2.90;
- conversion peaks after a heavy phase too low: exam t18-21 6.97 vs 5.29, t200 5.77 vs 3.74,
  p8 t610 5.62 vs 3.60; the post-pulse decay is slower than the fulfilment cap allows;
- spend at narrow targeting with high bid after a zero phase: exam t180-190 52-55 vs 27-30.

Tried (lab LOO at 1.0 sigma, 5 folds, same start = shipped v8b theta):

| fold | v8b refit on all 5 runs | str9 = v8b + price (breadth/0.55)^-kb |
|---|---|---|
| p1.hold_rec | 0.907 | 0.907 |
| p2.pulse60_120 | 0.834 | 0.833 |
| p2.multilevel200 | 0.848 | 0.848 |
| p6.exam | 0.823 | 0.821 |
| p8.longhold | 0.899 | 0.897 |
| mean | 0.862 | 0.861 |

Single-member refit with p8 does not beat the shipped perobs ensemble on the exam (0.823 vs 0.830);
breadth-priced impressions tie. Keep final1's ad_auction. Next structural target, if any: the
fulfilment backlog (conversions above F for ~30 ticks after heavy phases).
Files: `gtlab/ode/ad_auction_str9.py`, `plans/ad_auction_str9_ad_auction_*`.
