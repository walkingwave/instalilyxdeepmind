# Public-history consistency: social_contagion, hospital_queue, wildlife (Tue Sep 29)

Free, no gateway. Scripts: `scripts/lab_ph2_history.py <sys> [--seed S --suffix _sS]` (rollouts),
`scripts/lab_ph2_analyze.py <sys>... [--suffix=_sS]` (fits, backtest, forecasts).
Raw: `plans/ph_<sys>_history{,_s23}.json`, `plans/ph_<sys>_analysis{,_s23}.json`.

## Method
Candidate $C$ = noiseless truth on 40 eval-like episodes (`gtlab.check.make_episodes`, T = 4,000,
10/10/9/9 per category, y0 from our data; seeds 11 and 23). Every scored past predictor $Q$ (deduped
on predict.py+model.json, u001 = persistence) is scored against $C$ with the calibrated sigma
(`plans/sigma_calibrated.json`): $\hat s_Q(C)=\tfrac14\sum_{cat}\text{mean}\,1/(1+|e|/\sigma)$.
Compared with the public score $s_Q$ (registry). Self pairs dropped.

- offset fit: $s_Q \approx \hat s_Q - d$, residual SD `sd_off`; $d\approx0$ also means sigma is consistent.
- affine fit: $s_Q \approx a + b\,\hat s_Q$, residual SD `sd_aff` (absorbs a sigma-scale mismatch).
- forecast of $X$ from truth $C\neq X$: $\hat s_X(C)-d_C$ (off) or $a_C+b_C\hat s_X(C)$ (aff);
  pooled over truths with weights $1/\text{sd}^2$.
- backtest: hide each scored $X$, refit, forecast it. Pooled MAE 0.016-0.033, bias <= 0.01;
  on the top-3 public predictors the off rule runs +0.01..+0.03 high, the aff rule -0.02..+0.005.

With 8-12 history points (many related), sd differences under ~0.01 are noise. A truth scores its
own close relatives too high (e.g. final4 as truth gives final1 0.95 on wildlife vs public 0.74):
that is the main source of the pooled bias at the top. Seeds 11 and 23 agree to <= 0.02 everywhere.

Identities (hash): hospital alt1b = canon2 alone; hospital final1 = u010b/u015/u016/u017 (public 0.690,
recovery queue 15.5, same as h9w1); social final1 = u016/u017 (0.645), social final4 = alt1/final3b;
wildlife final1 = u016/u017 (0.738), wildlife final4 = final3/final3b.

## hospital_queue
Recovery hold (4,000 ticks, y0 queue 41): h9w1 15.5, final1/u010b 15.5, u013 19, u012 23, u014 24,
u008 28, u009 44, u005 57, u004 61, **final4 60**, **canon2/alt1b 105**.

| truth | offset d (s11/s23) | sd_off | sd_aff | corr | band RMSE sus/seq (s11) |
|---|---|---|---|---|---|
| **canon2 (= alt1b)** | -0.003 / +0.008 | **0.034 / 0.021** | 0.015 / 0.012 | 0.99 | **0.070 / 0.030** |
| **final4** | +0.032 / +0.053 | 0.036 / 0.022 | 0.016 / 0.013 | 0.99 | 0.074 / 0.045 |
| u012 | +0.022 / +0.055 | 0.046 / 0.038 | 0.020 / 0.019 | 0.98 | 0.084 / 0.041 |
| u013 | +0.049 / +0.074 | 0.059 / 0.048 | 0.011 / 0.011 | 0.99 | 0.109 / 0.069 |
| final1 (= u010b) | +0.041 / +0.064 | 0.061 / 0.049 | 0.025 / 0.022 | 0.96 | 0.110 / 0.065 |
| h9w1 | +0.040 / +0.062 | 0.070 / 0.065 | 0.030 / 0.030 | 0.95 | 0.085 / 0.085 |

Best match: **canon2**, then final4 (close). h9w1 is the worst modern truth on every metric (largest
sd_off, sd_aff, sequence band gap); final1 (same recovery level) is next worst. canon2 is also the only
truth with offset ~0 (sigma consistent). So the history leans against the ~15 recovery queue and
towards the high side (>= 60). 60 vs 105 is not decided: no past upload had a recovery queue above 61,
and canon2 vs final4 differ by less than noise.

Forecast of final4: pooled 0.710 / 0.672 (s11), 0.720 / 0.683 (s23) (off / aff). The same rule gives
final1 0.705/0.678 and 0.689/0.674 vs its known 0.690, so **final4 ~0.70 (range 0.67-0.72), about
final1 + 0.01**. By single truth: canon2 says final4 0.775 vs final1 0.670 (+0.10); h9w1 says
0.732 vs 0.843 (-0.11). The better-fitting truth is canon2, so the expected sign is positive.

**Verdict: final4 supported.** It is the second-most-consistent truth and the hedge (median of
h9w1 and canon2) sits on the side the history favours. canon2 alone (= alt1b) fits slightly better
and would win big if right, but has the larger downside if the truth is near 15; not worth switching on
this evidence alone.

## social_contagion
All truths have a large negative offset (d = -0.08..-0.19: implied scores well below public, worst
for the old weak models), so the calibrated sigma looks too small for social or every model shares a
miss. Rank by sd_aff, which absorbs that.

| truth | offset d | sd_off | sd_aff (s11/s23) | corr | band RMSE sus/seq |
|---|---|---|---|---|---|
| u012 | -0.161 | 0.048 | **0.025 / 0.026** | 0.97 | 0.155 / 0.193 |
| alt1b | -0.112 | 0.076 | 0.027 / 0.029 | 0.97 | 0.136 / 0.159 |
| u008 | -0.185 | 0.030 | 0.029 / 0.028 | 0.97 | 0.123 / 0.219 |
| final4 | -0.082 | 0.114 | 0.036 / 0.038 | 0.95 | 0.141 / 0.157 |
| final1 | -0.101 | 0.097 | 0.044 / 0.045 | 0.91 | 0.134 / 0.152 |

Best match among candidates: **alt1b**, then final4, then final1 (all within ~0.02 of each other).
final4 as truth over-predicts its relatives u015 (0.76 vs 0.63) and final1 (0.73 vs 0.645).

Forecast of final4: pooled 0.693 / 0.668 (s11), 0.691 / 0.666 (s23). Backtest top-3 bias: off +0.024,
aff -0.006; final1 forecast 0.669/0.638 vs known 0.645. Debiased: **final4 ~0.665 (0.64-0.69), about
final1 + 0.02**. alt1b forecast 0.658/0.638, i.e. about final1.

**Verdict: final4 weakly supported.** Not the best-matching truth, but it beats final1 on consistency
and its forecast is the highest of the three (+0.02 over final1, inside the +-0.03 error). Keep.

## wildlife

| truth | offset d | sd_off (s11/s23) | sd_aff (s11/s23) | corr | band RMSE sus/seq |
|---|---|---|---|---|---|
| u012 | +0.012 | 0.047 / 0.046 | **0.023 / 0.019** | 0.99 | 0.062 / 0.043 |
| u015 (= u013/u014) | -0.002 | 0.044 / 0.043 | 0.032 / 0.028 | 0.98 | 0.063 / 0.048 |
| alt1b | +0.030 | 0.062 / 0.063 | 0.030 / 0.028 | 0.99 | 0.048 / 0.050 |
| final1 | +0.017 | 0.058 / 0.058 | 0.031 / 0.028 | 0.98 | 0.074 / 0.064 |
| final4 | +0.038 | **0.082 / 0.082** | **0.041 / 0.039** | 0.97 | 0.074 / 0.062 |

Best match: u012 (older), then u015/alt1b/final1 tied. **final4 is the least consistent modern truth**
(largest sd_off and sd_aff, largest offset). Its only large misses are its own relatives
(final1 0.95 vs 0.74, u012 0.84 vs 0.70), i.e. final4 is very close to final1 and both sit about the
same distance from the truth.

Forecast of final4: pooled 0.751 / 0.748 (s11), 0.751 / 0.750 (s23); final1's forecast 0.743 vs known
0.738. Debiased: **final4 ~0.745 (0.72-0.77), about final1 + 0.007**. alt1b ~0.73.

**Verdict: final4 not supported as a better model; expected change vs final1 ~0 (+0.007 +- 0.03).**
Low risk either way (final4 and final1 are near-identical on eval-like schedules). If a slot is
tight, final1 (known 0.738) is the safer pick; nothing here argues for spending one on final4.

## Summary

| system | best-matching candidate | final4 forecast | vs final1 public | final4 verdict |
|---|---|---|---|---|
| hospital_queue | canon2 (= alt1b), final4 close 2nd; h9w1 worst | ~0.70 (0.67-0.72) | 0.690, +0.01 | supported; queue ~15 disfavoured, 60 vs 105 open |
| social_contagion | alt1b (then final4, final1) | ~0.665 (0.64-0.69) | 0.645, +0.02 | weakly supported |
| wildlife | u012 / u015 / final1 / alt1b; final4 last | ~0.745 (0.72-0.77) | 0.738, +0.007 | not supported as better; neutral |
