# Public-history consistency: ad_auction, traffic, reservoir (Tue Sep 29, free, no gateway)

## Method
- Every distinct scored predictor of the system (deduplicated on predict.py + model.json; registry score
  averaged over uploads that share a hash) plus the candidates is rolled once on the same episodes:
  `gtlab.check.schedules` (eval-like sustained/order/recovery/composition, pinned lo/hi dropped),
  T = 4,000, y0 from the reset pool. 40 episodes x 2 seeds (11, 23) = 80 episodes, pooled.
- Each model C in turn is the noiseless truth. Implied score of Q:
  $S_C(Q) = \mathrm{mean}_{t,o,e}\, 1/(1+|Q-C|/\sigma_o)$, $\sigma$ = `plans/sigma_calibrated.json`.
- Compare $S_C(Q)$ with the public score of Q: mean |gap|, correlation, offset (pub - implied), and the same
  on the strong uploads only (pub >= 0.7), which are the ones that tell ODE-class truths apart.
  C's own upload is excluded. "no clones" also drops uploads with mutual implied score > 0.93 to C
  (a near-copy of C always gets an inflated implied score when C is the truth).
- Forecasts of final4 under truth C (C not final4 or a clone of it):
  offset: $S_C(f4) + \overline{(pub - S_C)}_{top}$; anchor: $pub(A) + S_C(f4) - S_C(A)$, A = best public model.
- Scripts: `scripts/lab_ph_consist.py <sys> [--seed]` -> `plans/ph_<sys>[_s23].json`;
  `scripts/lab_ph_table.py <sys>` pools seeds and prints the tables below.
- Candidate identity (hash of predict.py + model.json):
  ad_auction: final4 = alt1b (new, never scored); final1 = u014..u017 (pub 0.8656).
  traffic: final4 = alt1b = trhi (traffic_hi4, never scored); final1 = u016/u017 (pub 0.8385);
  `plans/traffic_hi_all_s_doc.json` is the same family with slightly different theta (rolled via infer).
  reservoir: final4 new (never scored); alt1b = final1 = u012..u017 (pub 0.8311).

Caveats: 2-4 strong uploads per system, so gap_top rests on few points; seed-to-seed noise in gap_top is
about 0.01-0.02 (traffic's ranking of the top three flips between seeds). A model that sits outside the
cluster of past uploads (no clones in history) is favored by the raw gap; the "no clones" column corrects
part of that.

## ad_auction (A = final1 = u017, pub 0.8656)
| truth C | gap | corr | offset | gap top | offset top | gap top, no clones | S_C(final4) | S_C(final1) | fc offset | fc anchor |
|---|---|---|---|---|---|---|---|---|---|---|
| u010b (u008 model) | 0.033 | 0.972 | +0.007 | 0.023 | +0.013 | 0.023 | 0.843 | 0.875 | 0.856 | 0.833 |
| u013 | 0.044 | 0.961 | -0.011 | 0.033 | -0.023 | 0.033 | 0.942 | 0.921 | (clone) | (clone) |
| final1 (=u017) | 0.050 | 0.973 | -0.019 | 0.055 | -0.055 | 0.055 | 0.934 | 1.000 | (clone) | - |
| final4 = alt1b | 0.053 | 0.972 | -0.027 | 0.055 | -0.055 | 0.013 (u010b only) | 1.000 | 0.934 | - | - |
| u003 | 0.102 | 0.927 | +0.091 | 0.164 | +0.164 | 0.164 | 0.699 | 0.694 | 0.864 | 0.871 |
| u004 | 0.141 | 0.842 | +0.141 | 0.231 | +0.231 | 0.231 | 0.629 | 0.619 | 0.860 | 0.876 |
| u002 | 0.184 | 0.774 | +0.184 | 0.306 | +0.306 | 0.306 | 0.542 | 0.538 | 0.849 | 0.870 |
| persistence | 0.292 | 0.510 | +0.292 | 0.407 | +0.407 | 0.407 | 0.444 | 0.445 | 0.851 | 0.865 |

- final4 and final1 are near-copies (mutual 0.934) and fit the history equally (gap 0.050 vs 0.053,
  corr 0.973 vs 0.972). History cannot separate them.
- Best-fitting truth is u010b (isolated old ODE); under it final4 is 0.03 below final1 (anchor 0.833).
  Under u013 (a near-clone of final4) final4 is 0.02 above. Distant truths: final4 - final1 = +0.004..+0.010.
- **Forecast final4 ad_auction: ~0.85 (range 0.83-0.87) vs final1 0.8656.** Not supported as an
  improvement: consistent with history, expected change about 0 with a -0.03 tail.

## traffic (A = final1 = u017, pub 0.8385)
| truth C | gap | corr | offset | gap top | offset top | S_C(final4) | S_C(final1) | fc offset | fc anchor |
|---|---|---|---|---|---|---|---|---|---|
| hi_all_s doc | 0.020 | 0.996 | +0.011 | 0.017 | +0.002 | 0.943 | 0.868 | (clone) | (clone) |
| final1 (=u017) | 0.023 | 0.997 | -0.001 | 0.020 | -0.020 | 0.898 | 1.000 | 0.878* | - |
| final4 = alt1b = trhi | 0.024 | 0.996 | -0.005 | 0.023 | -0.022 | 1.000 | 0.898 | - | - |
| u015 | 0.056 | 0.967 | -0.027 | 0.089 | -0.068 | 0.795 | 0.806 | 0.727 | 0.828 |
| u010b | 0.066 | 0.950 | -0.027 | 0.109 | -0.064 | 0.762 | 0.772 | 0.698 | 0.829 |
| u009 | 0.068 | 0.945 | -0.027 | 0.111 | -0.062 | 0.753 | 0.765 | 0.691 | 0.827 |
| u003 | 0.118 | 0.799 | +0.059 | 0.140 | +0.140 | 0.633 | 0.631 | 0.773 | 0.840 |
| u005 | 0.133 | 0.775 | +0.087 | 0.173 | +0.173 | 0.592 | 0.591 | 0.765 | 0.839 |
| u004 | 0.118 | 0.859 | +0.114 | 0.187 | +0.187 | 0.576 | 0.576 | 0.763 | 0.838 |
| persistence | 0.503 | 0.125 | +0.503 | 0.581 | +0.581 | 0.210 | 0.208 | 0.791 | 0.841 |

\* inflated: final1 and final4 are neighbours (0.898). Mirror check: with final4 as truth, final1 is
implied 0.898 but scored 0.8385 (overshoot +0.06); applying the same overshoot gives final4 ~0.84.

- The three traffic_hi4-class / final1 truths all reproduce the history to ~0.02 (corr 0.996-0.997);
  seed 11 ranks final1 first (0.009), seed 23 ranks hi_all_s first (0.012). Tie within noise.
- Older truths (u009/u010b/u015) fit much worse (gap top 0.09-0.11), so the truth is in the
  final1/final4 neighbourhood. From every non-clone truth the anchor says final4 = final1 - 0.01..+0.00.
- **Forecast final4 traffic: ~0.835 (range 0.82-0.86) vs final1 0.8385.** Supported as consistent with the
  history (as good a fit as final1), not supported as an improvement: expected change about -0.005.
  hi_all_s behaves like final4 (mutual 0.943); no reason to prefer it over final4 on this test.

## reservoir (A = final1 = alt1b = u017, pub 0.8311)
| truth C | gap | corr | offset | gap top | offset top | gap top, no clones | S_C(final4) | S_C(final1) | fc offset | fc anchor |
|---|---|---|---|---|---|---|---|---|---|---|
| final4 | 0.039 | 1.000 | +0.039 | **0.013** | +0.013 | **0.013** | 1.000 | 0.824 | - | - |
| u010b (u008 model) | 0.028 | 0.989 | +0.018 | 0.028 | +0.028 | 0.028 | 0.760 | 0.808 | 0.788 | 0.783 |
| u016 | 0.036 | 0.990 | -0.011 | 0.067 | -0.067 | 0.014 | 0.818 | 0.952 | 0.750** | (clone) |
| final1 = alt1b (=u017) | 0.042 | 0.991 | -0.018 | 0.075 | -0.075 | 0.026 | 0.824 | 1.000 | 0.749** | - |
| u003 | 0.245 | -0.059 | +0.034 | 0.278 | +0.278 | 0.278 | 0.477 | 0.537 | 0.755 | 0.771 |
| u004 | 0.246 | -0.065 | +0.033 | 0.279 | +0.279 | 0.279 | 0.476 | 0.537 | 0.754 | 0.770 |
| u002 | 0.229 | -0.098 | +0.125 | 0.352 | +0.352 | 0.352 | 0.402 | 0.473 | 0.754 | 0.760 |
| persistence | 0.349 | -0.982 | +0.349 | 0.537 | +0.537 | 0.537 | 0.279 | 0.274 | 0.815 | 0.836 |

\** offset dominated by the u016/u017 clone pair; with the no-clone offset (u010b only) these become
0.798 (truth u017) and 0.805 (truth u016).

- final4 as truth reproduces the strong uploads best: u017 0.824 (pub 0.831), u016 0.818 (0.829),
  u010b 0.760 (0.782); corr 1.000 over all 7 scored models, both seeds (gap top 0.013 / 0.012).
- The telling fact: u016 and u017 are near-copies (0.952) yet both scored ~0.83, so the truth sits
  ~0.17 from both. final4 sits at exactly that distance from both (0.82); u017-as-truth cannot explain it.
- Under the other consistent truths final4 lands at 0.78-0.81 (u010b 0.783-0.788; u017/u016 no-clone
  0.798-0.805), i.e. 0.02-0.05 below u017. Under the best-fitting truth (final4-like) final4 beats 0.831.
- **Forecast final4 reservoir: ~0.81 (0.78-0.80 if the truth is u017/u008-like, >0.83 if final4-like).**
  Supported: final4 is the most history-consistent model here (only system of the three where the
  candidate beats the incumbent on this test), but the downside if wrong is about -0.04.

## Summary
| system | best-matching truth | final4 gap top | incumbent pub | final4 forecast | verdict |
|---|---|---|---|---|---|
| ad_auction | u010b (0.023); final4/final1 tie at 0.055 | 0.055 | 0.8656 (final1) | ~0.85 (0.83-0.87) | not separable from final1; no gain expected |
| traffic | hi_all_s / final1 / final4 tie (0.017-0.023) | 0.023 | 0.8385 (final1) | ~0.835 (0.82-0.86) | consistent; no gain expected (~-0.005) |
| reservoir | final4 (0.013) | 0.013 | 0.8311 (u017) | ~0.81, two-sided (0.78 to >0.83) | supported by history; real -0.04 downside |
