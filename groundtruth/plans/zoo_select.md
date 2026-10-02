# Per-observable zoo selection (held-out evidence only)

We took every model family we already have per system, refitted a shortlist leave-one-run-out on all current runs, and chose per observable among each member alone and the mean of each pair (the runtime's two-member median). Everything below is held-out, scored at the calibrated organizer sigma (plans/sigma_calibrated.json, 1.0x). No credits were used.

## Protocol

- Folds: leave one run out over all current runs (including the round-2 runs bought Sep 27 19:09).
- ODE members: `fit_ode` robust cauchy, one start at the member's full-data theta, no early phase, `max_nfev` 10, 12 s per fold (run: `loo --workers 1 --budget 12 --nfev 10`). No direct score polishing.
- l0b_lin: `make_model('l0b', cfg={'sq': False, 'pairs': None})`, clip margin 1, refitted per fold.
- Every member prediction is finalized with the current public doc's clip vectors; public post rules (ad_auction spend cap, hospital_queue queue cap) are applied after combining.
- Score per observable: $\frac{1}{T}\sum_t 1/(1+|\hat y_t-y_t|/\sigma)$, mean over folds.
- Accept a change on an observable only if the best option beats the public option by more than 0.01 (mean over folds) AND wins on a strict majority of folds.
- Exam runs `p6.exam` are never used in any fit; they are scored with members fitted on all other runs. Clean-run veto: a change is dropped if, averaged over the exam runs and the `p6.*` folds (runs no member doc had seen), it loses more than 0.01 to the public option.
- Cold-start check: the warm-start folds start from a theta fitted on all runs, so they leak the held-out run. Where time allowed we refitted the competing members from the family's default init (4 starts, spread 0.5, 40 evaluations, 45 s) and reverted every changed observable whose cold-start gain is not positive.
- Forecast public gain = 0.6 x held-out gain of the assembled combination over the public family.
- Validation: each shortlisted doc reproduces its lab in-sample scores exactly (max diff 0.0); each shipped u013 predict.py reproduces the dev rollout of its doc exactly.

## Summary

| system | public (u013) | held-out public | held-out pick | warm gain | cold gain | forecast (0.6 x min) | map | doc |
|---|---|---|---|---|---|---|---|---|
| ad_auction | 0.8565 | 0.823 | 0.832 | +0.0088 | +0.0106 | +0.0053 | pub_v8b, mean(pub_v8b,s1), mean(pub_v8b,min) | plans/zoo_ad_auction_doc.json |
| epidemic | 0.6867 | 0.617 | 0.644 | +0.0273 | +0.0065 | +0.0039 | mean(x7AC,x8AC), pub_x11AC | plans/zoo_epidemic_doc.json |
| hospital_queue | 0.6721 | 0.705 | 0.720 | +0.0151 | +0.0090 | +0.0054 | mean(v9w,min), pub_v9g, v9w | plans/zoo_hospital_queue_doc.json |
| market | 0.6272 | 0.569 | 0.569 | +0.0000 | +0.0000 | +0.0000 | mean(pub_x4,l0b), pub_x4, pub_x4 | keep current |
| power_grid | 0.7690 | 0.734 | 0.746 | +0.0122 | +0.0123 | +0.0073 | pub_v9c, pub_v9c, mean(v8k,min) | plans/zoo_power_grid_doc.json |
| reservoir | 0.8311 | 0.745 | 0.745 | +0.0000 | n/a | +0.0000 | pub_v8, pub_v8, pub_v8, pub_v8 | keep current |
| social_contagion | 0.6261 | 0.584 | 0.584 | +0.0000 | n/a | +0.0000 | pub_x7, pub_x7 | keep current |
| supply_chain | 0.8217 | 0.728 | 0.728 | +0.0000 | -0.1094 | +0.0000 | pub_v8b, pub_v8b, pub_v8b | keep current |
| traffic | 0.7757 | 0.729 | 0.729 | +0.0000 | n/a | +0.0000 | pub_v8d, pub_v8d, pub_v8d, pub_v8d | keep current |
| wildlife | 0.7244 | 0.647 | 0.647 | +0.0000 | -0.0161 | +0.0000 | pub_x13, pub_x13, pub_x13, pub_x13 | keep current |

Forecast mean public gain over the 10 systems: **+0.0022** (0.6 x the smaller of the warm and cold-start held-out gains of the final map; systems without a change count 0).

## ad_auction

Runs (4): p1.hold_rec, p2.pulse60_120, p2.multilevel200, p6.exam.

Shortlist: `pub_v8b` (u013 shipped), `v8c` (ad_auction_ad_auction_v8c_loo_doc.json), `min` (ad_auction_ad_auction_min_v8ref_doc.json), `s1` (ad_auction_ad_auction_s1_cal4_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | win_rate | spend | conversions | mean |
|---|---|---|---|---|
| pub_v8b | 0.893 | 0.891 | 0.685 | 0.823 |
| v8c | 0.888 | 0.884 | 0.690 | 0.821 |
| min | 0.810 | 0.892 | 0.663 | 0.788 |
| s1 | 0.810 | 0.891 | 0.663 | 0.788 |
| l0b | 0.468 | 0.598 | 0.436 | 0.501 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| win_rate | pub_v8b | 0.893 | pub_v8b | 0.893 | 0/3 | **pub_v8b** (clean-run delta +0.000) |
| spend | pub_v8b | 0.891 | mean(pub_v8b,s1) | 0.903 | 2/3 | **mean(pub_v8b,s1)** (clean-run delta +0.030) |
| conversions | pub_v8b | 0.685 | mean(pub_v8b,min) | 0.700 | 2/3 | **mean(pub_v8b,min)** (clean-run delta +0.017) |

Held-out (warm-start folds): public 0.823 -> pick 0.832 (gain +0.0088, forecast public +0.0053).
Per fold (public -> pick): p1.hold_rec 0.842->0.850, p2.pulse60_120 0.783->0.802, p2.multilevel200 0.844->0.843.
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.820, selection before reverts 0.831 (+0.0106), per observable +0.000, +0.012, +0.020; fold wins 3/3; nothing reverted.
p6.exam (exam (never fitted; members fitted on all fit runs)): public 0.814, selection before reverts 0.830; members pub_v8b 0.814, v8c 0.812, min 0.803, s1 0.801, l0b 0.556.
Assembled `plans/zoo_ad_auction_doc.json`: runtime vs manual combination max err 0.0e+00 sigma; 4,000-tick eval-like rollouts: sustained finite=True 3.13s outside=0.00, order finite=True 3.02s outside=0.00, recovery finite=True 2.94s outside=0.00, composition finite=True 3.05s outside=0.00.

## epidemic

Runs (5): p1.hold_rec, p2.pulse120_280, p3.compose, p4.joint_hold, p6.voi.

Shortlist: `pub_x11AC` (u013 shipped), `x7AC` (epidemic_epidemic_x7_AC_doc.json), `x10AC` (epidemic_epidemic_x10_AC_doc.json), `x8AC` (epidemic_epidemic_x8_AC_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | daily_cases | hospital_load | mean |
|---|---|---|---|
| pub_x11AC | 0.637 | 0.596 | 0.617 |
| x7AC | 0.661 | 0.616 | 0.639 |
| x10AC | 0.652 | 0.610 | 0.631 |
| x8AC | 0.663 | 0.632 | 0.647 |
| l0b | 0.234 | 0.198 | 0.216 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| daily_cases | pub_x11AC | 0.637 | mean(x7AC,x8AC) | 0.692 | 4/5 | **mean(x7AC,x8AC)** (clean-run delta +0.090) |
| hospital_load | pub_x11AC | 0.596 | mean(x7AC,x8AC) | 0.642 | 4/5 | **pub_x11AC** (clean-run delta -0.021 VETO) |

Held-out (warm-start folds): public 0.617 -> pick 0.644 (gain +0.0273, forecast public +0.0164).
Per fold (public -> pick): p1.hold_rec 0.714->0.696, p2.pulse120_280 0.634->0.666, p3.compose 0.624->0.642, p4.joint_hold 0.585->0.644, p6.voi 0.526->0.571.
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.625, selection before reverts 0.632 (+0.0065), per observable +0.013, +0.000; fold wins 4/5; nothing reverted.
p6.voi (LOO fold): public 0.526, selection before reverts 0.571; members pub_x11AC 0.526, x7AC 0.570, x10AC 0.556, x8AC 0.504, l0b 0.223.
Assembled `plans/zoo_epidemic_doc.json`: runtime vs manual combination max err 0.0e+00 sigma; 4,000-tick eval-like rollouts: sustained finite=True 2.52s outside=0.01, order finite=True 2.62s outside=0.00, recovery finite=True 2.57s outside=0.00, composition finite=True 2.51s outside=0.01.

## hospital_queue

Runs (8): p1.hold_rec, p2.pulse40, p2.mid40, p2.multilevel200, p3.compose, p4.pulse_long_recovery, p6.voi, p6.exam.

Shortlist: `pub_v9g` (u013 shipped), `v9w` (hospital_queue_hospital_queue_v9_v9w_doc.json), `v8b` (hospital_queue_hospital_queue_v8b_loo_doc.json), `min` (hospital_queue_hospital_queue_min_p3_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | wait_time | queue | discharges | mean |
|---|---|---|---|---|
| pub_v9g | 0.775 | 0.698 | 0.642 | 0.705 |
| v9w | 0.796 | 0.697 | 0.661 | 0.718 |
| v8b | 0.783 | 0.651 | 0.646 | 0.693 |
| min | 0.776 | 0.599 | 0.626 | 0.667 |
| l0b | 0.636 | 0.370 | 0.576 | 0.527 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| wait_time | pub_v9g | 0.775 | mean(v9w,min) | 0.801 | 4/7 | **mean(v9w,min)** (clean-run delta +0.005) |
| queue | pub_v9g | 0.698 | mean(pub_v9g,v9w) | 0.700 | 6/7 | **pub_v9g** (clean-run delta +0.014) |
| discharges | pub_v9g | 0.642 | v9w | 0.661 | 5/7 | **v9w** (clean-run delta +0.019) |

Held-out (warm-start folds): public 0.705 -> pick 0.720 (gain +0.0151, forecast public +0.0090).
Per fold (public -> pick): p1.hold_rec 0.838->0.807, p2.pulse40 0.681->0.750, p2.mid40 0.675->0.664, p2.multilevel200 0.655->0.686, p3.compose 0.569->0.577, p4.pulse_long_recovery 0.773->0.793, p6.voi 0.744->0.763.
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.693, selection before reverts 0.702 (+0.0090), per observable +0.003, +0.000, +0.024; fold wins 5/7; nothing reverted.
p6.voi (LOO fold): public 0.744, selection before reverts 0.763; members pub_v9g 0.744, v9w 0.787, v8b 0.760, min 0.708, l0b 0.667.
p6.exam (exam (never fitted; members fitted on all fit runs)): public 0.664, selection before reverts 0.661; members pub_v9g 0.664, v9w 0.646, v8b 0.659, min 0.641, l0b 0.461.
Assembled `plans/zoo_hospital_queue_doc.json`: runtime vs manual combination max err 0.0e+00 sigma; 4,000-tick eval-like rollouts: sustained finite=True 2.23s outside=0.00, order finite=True 2.24s outside=0.00, recovery finite=True 2.24s outside=0.00, composition finite=True 2.23s outside=0.00.

## market

Runs (8): p1.hold_rec, p2.pulse40, p2.mid40, p2.multilevel200, p3.compose, p4.rate_hold, p5.testlike, p6.voi.

Shortlist: `pub_x4` (u013 shipped), `x3` (market_market_x3_x7_doc.json), `v9g` (market_market_v9g_x7ref_doc.json), `v8t2` (market_market_v8t2_v8_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | price | volume | depth | mean |
|---|---|---|---|---|
| pub_x4 | 0.430 | 0.767 | 0.573 | 0.590 |
| x3 | 0.421 | 0.771 | 0.577 | 0.590 |
| v9g | 0.418 | 0.750 | 0.544 | 0.571 |
| v8t2 | 0.389 | 0.744 | 0.513 | 0.549 |
| l0b | 0.263 | 0.561 | 0.331 | 0.385 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| price | mean(pub_x4,l0b) | 0.368 | mean(pub_x4,v9g) | 0.443 | 6/8 | **mean(pub_x4,l0b)** (clean-run delta -0.151 VETO) |
| volume | pub_x4 | 0.767 | x3 | 0.771 | 4/8 | **pub_x4** (clean-run delta -0.000) |
| depth | pub_x4 | 0.573 | mean(x3,v9g) | 0.582 | 4/8 | **pub_x4** (clean-run delta +0.135) |

Held-out (warm-start folds): public 0.569 -> pick 0.569 (gain +0.0000, forecast public +0.0000).
Per fold (public -> pick): p1.hold_rec 0.766->0.766, p2.pulse40 0.499->0.499, p2.mid40 0.675->0.675, p2.multilevel200 0.443->0.443, p3.compose 0.551->0.551, p4.rate_hold 0.552->0.552, p5.testlike 0.517->0.517, p6.voi 0.551->0.551.
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.565, selection before reverts 0.565 (+0.0000), per observable +0.000, +0.000, +0.000; fold wins 0/8; nothing reverted.
p5.testlike (LOO fold): public 0.517, selection before reverts 0.517; members pub_x4 0.562, x3 0.554, v9g 0.472, v8t2 0.501, l0b 0.364.
p6.voi (LOO fold): public 0.551, selection before reverts 0.551; members pub_x4 0.526, x3 0.542, v9g 0.477, v8t2 0.450, l0b 0.338.

## power_grid

Runs (4): p1.hold_rec, p2.pulse60_120, p2.multilevel200, p6.exam.

Shortlist: `pub_v9c` (u013 shipped), `v9` (power_grid_power_grid_v9_v9_doc.json), `v8k` (power_grid_power_grid_v8k_v8k_lab_doc.json), `min` (power_grid_power_grid_min_v8ref_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | load | frequency | renewable_share | mean |
|---|---|---|---|---|
| pub_v9c | 0.744 | 0.674 | 0.782 | 0.734 |
| v9 | 0.743 | 0.659 | 0.782 | 0.728 |
| v8k | 0.721 | 0.660 | 0.799 | 0.727 |
| min | 0.668 | 0.607 | 0.776 | 0.684 |
| l0b | 0.516 | 0.443 | 0.730 | 0.563 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| load | pub_v9c | 0.744 | pub_v9c | 0.744 | 0/3 | **pub_v9c** (clean-run delta +0.000) |
| frequency | pub_v9c | 0.674 | pub_v9c | 0.674 | 0/3 | **pub_v9c** (clean-run delta +0.000) |
| renewable_share | pub_v9c | 0.782 | mean(v8k,min) | 0.819 | 2/3 | **mean(v8k,min)** (clean-run delta -0.002) |

Held-out (warm-start folds): public 0.734 -> pick 0.746 (gain +0.0122, forecast public +0.0073).
Per fold (public -> pick): p1.hold_rec 0.807->0.824, p2.pulse60_120 0.694->0.686, p2.multilevel200 0.700->0.728.
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.730, selection before reverts 0.743 (+0.0123), per observable +0.000, +0.000, +0.037; fold wins 2/3; nothing reverted.
p6.exam (exam (never fitted; members fitted on all fit runs)): public 0.657, selection before reverts 0.657; members pub_v9c 0.657, v9 0.657, v8k 0.656, min 0.633, l0b 0.520.
Assembled `plans/zoo_power_grid_doc.json`: runtime vs manual combination max err 0.0e+00 sigma; 4,000-tick eval-like rollouts: sustained finite=True 1.54s outside=0.00, order finite=True 1.51s outside=0.00, recovery finite=True 1.54s outside=0.00, composition finite=True 1.56s outside=0.00.

## reservoir

Runs (2): p1.hold_rec, p2.pulse200_200.

Shortlist: `pub_v8` (u013 shipped), `min2` (reservoir_reservoir_min2_doc.json), `min` (reservoir_reservoir_min_doc.json), `s1` (reservoir_reservoir_s1_cal4_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | level | inflow | outflow | quality | mean |
|---|---|---|---|---|---|
| pub_v8 | 0.838 | 0.738 | 0.857 | 0.548 | 0.745 |
| min2 | 0.746 | 0.828 | 0.822 | 0.574 | 0.743 |
| min | 0.713 | 0.816 | 0.766 | 0.619 | 0.729 |
| s1 | 0.782 | 0.816 | 0.835 | 0.626 | 0.765 |
| l0b | 0.534 | 0.372 | 0.300 | 0.588 | 0.449 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| level | pub_v8 | 0.838 | mean(pub_v8,s1) | 0.911 | 1/2 | **pub_v8**  |
| inflow | pub_v8 | 0.738 | min2 | 0.828 | 1/2 | **pub_v8**  |
| outflow | pub_v8 | 0.857 | mean(pub_v8,min2) | 0.861 | 1/2 | **pub_v8**  |
| quality | pub_v8 | 0.548 | mean(pub_v8,l0b) | 0.630 | 1/2 | **pub_v8**  |

Held-out (warm-start folds): public 0.745 -> pick 0.745 (gain +0.0000, forecast public +0.0000).
Per fold (public -> pick): p1.hold_rec 0.897->0.897, p2.pulse200_200 0.593->0.593.

## social_contagion

Runs (6): p1.hold_rec, p2.pulse200_200, p3.compose, p4.interior_holds, p5.testlike, p6.voi.

Shortlist: `pub_x7` (u013 shipped), `x2` (social_contagion_social_contagion_x2_xr5_doc.json), `x8` (social_contagion_social_contagion_x8_xr5_doc.json), `v8o` (social_contagion_social_contagion_v8o_xr5_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | adopters_a | adopters_b | mean |
|---|---|---|---|
| pub_x7 | 0.643 | 0.524 | 0.584 |
| x2 | 0.590 | 0.625 | 0.608 |
| x8 | 0.560 | 0.525 | 0.543 |
| v8o | 0.542 | 0.570 | 0.556 |
| l0b | 0.329 | 0.347 | 0.338 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| adopters_a | pub_x7 | 0.643 | pub_x7 | 0.643 | 0/6 | **pub_x7** (clean-run delta +0.000) |
| adopters_b | pub_x7 | 0.524 | mean(x8,v8o) | 0.626 | 3/6 | **pub_x7** (clean-run delta +0.551) |

Held-out (warm-start folds): public 0.584 -> pick 0.584 (gain +0.0000, forecast public +0.0000).
Per fold (public -> pick): p1.hold_rec 0.701->0.701, p2.pulse200_200 0.609->0.609, p3.compose 0.493->0.493, p4.interior_holds 0.625->0.625, p5.testlike 0.623->0.623, p6.voi 0.452->0.452.
p5.testlike (LOO fold): public 0.623, selection before reverts 0.623; members pub_x7 0.623, x2 0.643, x8 0.564, v8o 0.637, l0b 0.480.
p6.voi (LOO fold): public 0.452, selection before reverts 0.452; members pub_x7 0.452, x2 0.476, x8 0.412, v8o 0.272, l0b 0.235.

## supply_chain

Runs (3): p1.hold_rec, p2.pulse200_200, p3.hold_mid.

Shortlist: `pub_v8b` (u013 shipped), `v8c` (supply_chain_supply_chain_v8c_a_doc.json), `min` (supply_chain_supply_chain_min_v7b_doc.json), `min2` (supply_chain_supply_chain_min2_v7b_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | shipments | inventory_supplier | inventory_retail | mean |
|---|---|---|---|---|
| pub_v8b | 0.701 | 0.861 | 0.622 | 0.728 |
| v8c | 0.830 | 0.947 | 0.773 | 0.850 |
| min | 0.720 | 0.923 | 0.732 | 0.792 |
| min2 | 0.707 | 0.928 | 0.732 | 0.789 |
| l0b | 0.305 | 0.389 | 0.488 | 0.394 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| shipments | pub_v8b | 0.701 | v8c | 0.830 | 2/3 | **pub_v8b** (reverted by cold-start check) |
| inventory_supplier | pub_v8b | 0.861 | v8c | 0.947 | 3/3 | **pub_v8b** (reverted by cold-start check) |
| inventory_retail | pub_v8b | 0.622 | v8c | 0.773 | 2/3 | **pub_v8b** (reverted by cold-start check) |

Held-out (warm-start folds): public 0.728 -> pick 0.728 (gain +0.0000, forecast public +0.0000).
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.835, selection before reverts 0.725 (-0.1094), per observable -0.064, -0.181, -0.083; fold wins 0/3; reverted: shipments, inventory_supplier, inventory_retail.

## traffic

Runs (6): p1.hold_rec, p2.pulse40, p2.mid40, p2.multilevel200, p3.hold_mid, p5.testlike.

Shortlist: `pub_v8d` (u013 shipped), `v8c` (traffic_traffic_v8c_v8c_doc.json), `min` (traffic_traffic_min_p3_doc.json), `s1` (traffic_traffic_s1_cal4_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | flow_a | flow_b | speed_a | speed_b | mean |
|---|---|---|---|---|---|
| pub_v8d | 0.719 | 0.675 | 0.782 | 0.741 | 0.729 |
| v8c | 0.721 | 0.671 | 0.788 | 0.719 | 0.725 |
| min | 0.717 | 0.670 | 0.723 | 0.690 | 0.700 |
| s1 | 0.716 | 0.671 | 0.723 | 0.691 | 0.700 |
| l0b | 0.428 | 0.416 | 0.476 | 0.537 | 0.464 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| flow_a | pub_v8d | 0.719 | v8c | 0.721 | 4/6 | **pub_v8d**  |
| flow_b | pub_v8d | 0.675 | pub_v8d | 0.675 | 0/6 | **pub_v8d**  |
| speed_a | pub_v8d | 0.782 | v8c | 0.788 | 4/6 | **pub_v8d**  |
| speed_b | pub_v8d | 0.741 | pub_v8d | 0.741 | 0/6 | **pub_v8d**  |

Held-out (warm-start folds): public 0.729 -> pick 0.729 (gain +0.0000, forecast public +0.0000).
Per fold (public -> pick): p1.hold_rec 0.971->0.971, p2.pulse40 0.502->0.502, p2.mid40 0.841->0.841, p2.multilevel200 0.663->0.663, p3.hold_mid 0.726->0.726, p5.testlike 0.673->0.673.
p5.testlike (LOO fold): public 0.673, selection before reverts 0.673; members pub_v8d 0.673, v8c 0.675, min 0.651, s1 0.651, l0b 0.529.

## wildlife

Runs (4): p1.hold_rec, p2.pulse200_200, p3.compose, p4.corridor_habitat.

Shortlist: `pub_x13` (u013 shipped), `x6` (wildlife_wildlife_x6_a_doc.json), `x5` (wildlife_wildlife_x5_b_doc.json), `v8i` (wildlife_wildlife_v8i_v8i_loo_doc.json), `l0b` (l0b_lin).

Members alone, held-out per observable:

| member | prey_north | predator_north | prey_south | predator_south | mean |
|---|---|---|---|---|---|
| pub_x13 | 0.653 | 0.648 | 0.632 | 0.657 | 0.647 |
| x6 | 0.648 | 0.665 | 0.670 | 0.678 | 0.665 |
| x5 | 0.683 | 0.716 | 0.676 | 0.713 | 0.697 |
| v8i | 0.578 | 0.655 | 0.611 | 0.667 | 0.628 |
| l0b | 0.427 | 0.568 | 0.408 | 0.562 | 0.491 |

Per observable decision (best of singles and pair means):

| observable | public option | public | best option | best | wins | pick |
|---|---|---|---|---|---|---|
| prey_north | pub_x13 | 0.653 | mean(x6,x5) | 0.699 | 3/4 | **pub_x13** (reverted by cold-start check) |
| predator_north | pub_x13 | 0.648 | x5 | 0.716 | 4/4 | **pub_x13** (reverted by cold-start check) |
| prey_south | pub_x13 | 0.632 | mean(x6,x5) | 0.695 | 4/4 | **pub_x13** (reverted by cold-start check) |
| predator_south | pub_x13 | 0.657 | x5 | 0.713 | 4/4 | **pub_x13** (reverted by cold-start check) |

Held-out (warm-start folds): public 0.647 -> pick 0.647 (gain +0.0000, forecast public +0.0000).
Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): public 0.651, selection before reverts 0.635 (-0.0161), per observable -0.037, -0.003, -0.006, -0.018; fold wins 2/4; reverted: prey_north, predator_north, prey_south, predator_south.
