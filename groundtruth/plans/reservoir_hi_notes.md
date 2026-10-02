# reservoir hi: stress test of str9 c before the final upload (Tue, no credits)

Question: str9 c (`plans/reservoir_str9_reservoir_str9_c_clean_doc.json`, LOO 0.8745 vs v8 0.812) gains
mostly on p8, the 700-tick long hold. A power_grid model with the same profile lost 0.016 on the public
test. Is the str9 c gain real without p8, and does it behave on 4,000-tick schedules?

Scripts: `scripts/lab_hi_res_fold.py` (one train/test split, neutral or given start),
`scripts/lab_hi_res_cross.py` (score one packaged predictor with another as truth on eval-like
4,000-tick episodes), `scripts/lab_hi_res_asym.py` (long-hold asymptotes per control corner).
Family `gtlab/ode/reservoir_hi.py` = str9 + delayed deposit stage $N_2$ + stage-tied reset profile
(parity with str9 c: max diff 7.5e-5 on quality, 0 elsewhere). Fold reports `plans/reservoir_hi_fold_<tag>.json`.
Score $\frac{1}{1+|e|/\sigma}$ at calibrated $\sigma$ = [110.4, 0.672, 1.849, 0.0113] (level, inflow, outflow, quality).

## 1. Refit with p8 removed entirely (neutral start = module defaults, 200 s, 8 starts)

Neither family ever sees p8 or any theta fitted on it.

| train | held-out run | v8 | str9 c | delta |
|---|---|---:|---:|---:|
| p1 + p2 | p8.longhold | 0.831 [.940 .851 .905 .628] | **0.880** [.938 .911 .934 .735] | +0.049 |
| p2 | p8.longhold | 0.826 [.917 .839 .927 .622] | **0.889** [.955 .908 .950 .741] | +0.062 |
| p2 | p1.hold_rec | **0.895** [.962 .899 .962 .757] | 0.863 [.962 .918 .967 .603] | -0.032 |
| p1 | p2.pulse200_200 | **0.640** | 0.598 | -0.042 |
| p1 | p8.longhold | **0.789** | 0.765 | -0.024 |

- The p8 gain is reproduced by fits that never saw p8: +0.05 to +0.06, on inflow (+0.06), outflow
  (+0.03) and quality (+0.11). It is not a p8 artefact.
- p1 alone (120-tick recovery hold, no drawdown) identifies neither model; both collapse on p2
  (0.60-0.64). Not informative, but v8 is less bad from a single hold.
- The one real loss: p1 quality when only p2 is trained (0.603 vs 0.757). str9 c overshoots quality
  during the fill (t20: 0.969 vs data 0.953) because $N$ starts at $n_0$ and lags the rising stage.
  It is the fill-phase leak already noted in the str9 notes; with p1+p8 in training it is 0.585-0.60
  for both families (v8 LOO 0.615).

## 2. Long-horizon behaviour (4,000 ticks)

Recovery-hold quality asymptote from the p8 reset (`lab_hi_res_asym.py`):

| fit | recovery | zero action | pulse (12,8,1,0) | release 12 + aeration (unseen) | depth 1 + aeration, full pool (unseen) |
|---|---:|---:|---:|---:|---:|
| data p8, t650-700 (still falling ~0.001 / 50 ticks) | 0.936 | | | | |
| v8 (p1+p2) | 0.952 | 0.942 | 0.931 | 0.952 | 0.942 |
| str9 c doc (all runs) | 0.937 | 0.925 | 0.939 | 0.965 | 0.922 |
| str9 c, no p8 (p1+p2) | 0.932 | 0.924 | 0.938 | 0.959 | 0.918 |
| str9 c, p2 only | 0.940 | 0.923 | 0.934 | 0.967 | 0.925 |
| str9 c LOO folds (3) | 0.928-0.937 | 0.918-0.925 | 0.933-0.941 | 0.965-0.968 | 0.912-0.923 |

- Every str9 fit, including the two that never saw p8, puts full-pool recovery quality at 0.928-0.940;
  p8 reads 0.936 at t675. v8 sits at 0.952 = +1.4 sigma. The slide is learned from p1 + p2 alone.
- No drift: every fit is flat from t700 to t4000; levels settle (full pool 941, pulse 263-285).
- Cross-score on eval-like episodes (3 per category, 4,000 ticks): the shipped final1 predictor scored
  with str9 c as truth = **0.813** [0.958 0.870 0.941 0.484]; its public score is **0.831**. Mean
  |diff| between them is 1.1-1.5 sigma on quality and <= 0.44 on the rest. The public test sees the
  shipped model about as wrong as it would be if str9 c were the truth, and nearly all of it is quality.
  Consistent with str9 c, not a proof.
- Risk corner: release 12 with aeration on (low stage, never observed). str9 c says 0.96-0.97 because
  the deposit stock $N$ follows the low stage; v8 says 0.952. If truth is v8-like this corner costs
  ~1 sigma on quality where it occurs. No data can settle it without credits.

## 3. Attempts to push further (LOO at 1.0 sigma, start `plans/reservoir_str9_c_theta0_clean.json`)

Same harness for all (k_ret 0, w_L 1, d0 0 fixed). The control reproduces str9 c at 0.867 (recorded
0.8745 with the lab script; optimizer spread ~0.007 per fold).

| fold | c (control) | hi1: + delay $N_2$ | hi2: $N_0 = n_0 + n_L L_0/L_{full}$ | hi3: $q_{aer}=0$ | hi4: delay, $q_{aer}=0$ |
|---|---:|---:|---:|---:|---:|
| p1.hold_rec | 0.856 | 0.856 | 0.858 | **0.868** | 0.867 |
| p2.pulse200_200 | 0.869 | **0.871** | 0.861 | 0.866 | 0.867 |
| p8.longhold | 0.878 | **0.885** | 0.848 | 0.883 | 0.874 |
| mean | 0.867 | 0.871 | 0.856 | **0.872** | 0.869 |

hi1 at 150 s scored 0.863 on p1 (vs 0.856 at 300 s): same model, different optimum. All gains are
<= 0.005 on the mean, inside the optimizer spread, and hi1's lean on p8 quality. Rejected: none wins
most folds by more than noise. Level (0.94-0.96) and outflow (0.93-0.96) held-out are within 0.02 of
the noise ceilings (0.97 / 0.975); inflow 0.91 vs 0.93; quality 0.73-0.75 on p2/p8 vs ceiling 0.74-0.77.
Only p1 fill-phase quality has room, and it is < 3% of a 4,000-tick episode.

## Verdict

**Ship str9 c as is: `plans/reservoir_str9_reservoir_str9_c_clean_doc.json`.** Nothing better found.
Unlike the power_grid case, the gain survives with p8 removed (+0.05-0.06 on p8 from p1+p2 or p2 alone),
the key long-run number (recovery quality ~0.935) is reproduced by every fit with or without p8, and
nothing drifts over 4,000 ticks. The loss is fill-phase quality on a short recovery hold (small weight
in long episodes) and one unobserved corner.

Forecast (public/final reservoir): v8 0.831. Quality carries the gap: if the full-pool asymptote is
right, quality error drops from ~1.4 to ~0.1-0.3 sigma on most ticks. Central **0.865**
(0.831 + 0.6 x the +0.055 no-p8 held-out gain), range 0.83-0.91 (upper end if str9 c is as close to
truth as the cross-score hints); downside if the low-stage aerated corner is v8-like and frequent: ~0.84.
On the 10-system mean: +0.003 central.
