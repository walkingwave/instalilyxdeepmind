# Next step in the u019 direction? Public-history check with u019 included (Tue Sep 29)

Scripts: `scripts/lab_q_win_consist.py <sys> --seed {11,23}` (copy of lab_ph_consist with candidates = u019
+ near alternatives; out `plans/q_win_ph_<sys>[_s23].json`), `scripts/lab_q_win_table.py <sys> --ok 0.08`
(out `plans/q_win_fc_<sys>.json`). 80 eval-like episodes, calibrated sigma, u019 now a scored upload.
Forecast of X = pub(u019) + S_C(X) - S_C(u019), averaged over consistent truths C that are not X, u019 or a
clone of X (mutual implied > 0.93). Swap rule: forecast > pub(u019) + 0.005 AND LOO favours X.

Docs added: `plans/q_win_ad_hi1sc_doc.json` (win = med(hi1,v8b), spend = hi1, conv = hi1),
`plans/q_win_res_rr_doc.json` (= model_json of `plans/rr_fit_reservoir_reservoir_str9_rr.json`).

| system | u019 pub | candidate | mutual w/ u019 | usable truths | delta per truth | forecast | LOO vs u019 model | verdict |
|---|---:|---|---:|---|---|---:|---|---|
| reservoir | 0.8938 | str9 rr refit | 0.954 | u010b, u016, u017 | -0.006, -0.005, -0.006 | 0.888 | +0.008 (3/3 folds) | keep |
| traffic | 0.8473 | all_s | 0.943 | u017 | -0.031 | 0.816 | +0.016 (7/7) | keep |
| ad_auction | 0.8783 | hi1 alone | 0.962 | u010b, u013, u017 | +0.010, -0.031, -0.004 | 0.869 | 0.8641 vs 0.8674 | keep |
| ad_auction | 0.8783 | med;hi1;hi1 | 0.976 | u010b, u013, u017 | +0.012, -0.019, +0.008 | 0.879 | 0.8636 vs 0.8674 (loses 3/5) | keep |

Notes.
- Reservoir: only u019 and rr pass the strict consistency gate (gap_top 0.013 / 0.029). As truth, rr predicts
  u019 at 0.953 vs public 0.894; u019 as truth fits the old uploads better (gaps 0.007-0.022 vs 0.013-0.028).
  Every older upload sits slightly closer to u019 than to rr, and rr moves away from them: wrong side.
- Traffic: u017 -> u019 was a large step (mutual 0.898) for +0.009 public; all_s is a further step on the same
  line (u017-all_s 0.868) and u017-as-truth loses 0.031 on it. LOO gain does not show in public history.
- Ad: both hi1 variants are within noise of u019 (range -0.03..+0.02 across truths), and LOO prefers u019's
  median mix. The loose number 0.885 for med;hi1;hi1 includes hi1_all as truth (a clone): not used.
