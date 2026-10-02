# social_contagion: public-history judge with u019 added (Tue Sep 29)

Free, no gateway. Script `scripts/lab_q_soc_history.py --seed {11,23}` -> `plans/q_soc_history_s{11,23}.json`.
Same method as `lab_ph2_history.py` / `lab_ph2_analyze.py` (ph_soc_hosp_wild.md), now with u019
(final4 = median(z20, z21, canon3), public 0.6358) in the history. 15 truths (13 distinct scored
predictors + z21 + canon3), 40 eval-like episodes. Check: median of the rolled members = u019
predict.py exactly (max diff 0).

Per truth $C$: affine fit $s_Q \approx a_C + b_C\,\hat s_Q(C)$ over the scored history.
Delta forecast vs z20: $\Delta_X = b_C(\hat s_X(C) - \hat s_{z20}(C))$, pooled over truths with
weights $1/\text{sd}_\text{aff}^2$ ("aff") and $\exp(-\text{SSE}_C/2s^2)$ ("lik").

## Backtest (the judge judged)
Hide each scored predictor, refit, forecast it. Level MAE 0.028, bias +0.001. Delta vs z20 on
the >0.6 models: MAE 0.020, bias **+0.010**. The two direct tests of "move away from z20":

| hidden | public delta vs z20 | forecast (s11 aff/lik) | error |
|---|---|---|---|
| u015 (y10) | -0.0106 | +0.0060 / +0.0035 | +0.017 |
| **u019 (final4)** | **-0.0090** | **+0.0356 / +0.0357** | **+0.045** |

Even with u019 in the fit, final4 is still forecast +0.025 / +0.023 over z20. So this judge has a
systematic ~+0.03..+0.045 bias in favour of anything that pulls z20 toward canon3/z21, and a
+0.017 bias for pulling toward y10. The public test is the stronger evidence.

## Candidates (delta vs z20, pooled, seeds 11 / 23; aff / lik)

| candidate | s11 | s23 | calibrated (subtract judge bias) |
|---|---|---|---|
| perobs a=z20, b=med(z20,z21) | +0.013 / +0.023 | +0.014 / +0.019 | ~ -0.015 (canon/z21 axis, bias ~0.035) |
| med(z20,z21,y10) | +0.013 / +0.010 | +0.013 / +0.010 | ~ -0.015 |
| med(z20,y10,canon3) | +0.013 / +0.009 | +0.013 / +0.009 | ~ -0.015 |
| y10 alone | +0.007 / +0.005 | +0.007 / +0.006 | -0.011 (known public) |
| med(z20,y10,x7) | +0.004 / +0.000 | +0.004 / +0.001 | ~ -0.010 |
| mean(z20,y10) | +0.003 / +0.002 | +0.003 / +0.002 | ~ -0.010 |
| med(z20,z21) | +0.001 / +0.012 | +0.002 / +0.009 | ~ -0.01..-0.02 |
| perobs a=med(z20,z21), b=z20 | -0.012 / -0.010 | -0.012 / -0.011 | < -0.02 |
| perobs a=z21, b=z20 | -0.021 / -0.016 | -0.023 / -0.023 | < -0.03 |
| blend z20 lam 0.95 | -0.026 / -0.014 | -0.027 / -0.016 | < -0.02 |
| blend z20 lam 0.9 | -0.064 / -0.051 | -0.062 / -0.045 | < -0.05 |
| blend z20 lam 1.05 | -0.028 / -0.022 | -0.027 / -0.018 | < -0.02 |

Raw forecasts: nothing beats z20 by more than the judge's own +0.010 top-model bias except the
canon/z21-leaning mixes, which are exactly the direction u019 just showed to be wrong (-0.009 public
vs +0.036 forecast). Shrinking toward y0 loses on every rule (same as earlier blend tests).
z21 on adopters_a alone is forecast negative on every rule.

## Decision
**Keep z20 (u016 folder, 0.6448).** No cheap combination has a calibrated forecast above
0.6448 + 0.005, and the one history point that tested a hedge (u019) contradicts the judge by 0.045.
