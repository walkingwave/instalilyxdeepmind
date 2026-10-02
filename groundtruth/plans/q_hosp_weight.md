# hospital_queue: weight on canon2 (Tue Sep 29, free, no gateway)

Question: in $\hat y = w\,\text{canon2} + (1-w)\,\text{h9w1}$, which $w \in \{0.5, 0.65, 0.8, 1.0\}$?
Public: u017 (final1, h9w1-like) 0.6896; u019 (final4, $w=0.5$) 0.7294, +0.040.

Runtime: `ensemble` = median (2 members = mean), `blend` = shrink to y0, `perobs` = one member per
observable. No weighted mean kind, so only $w=0.5$ and $w=1$ can ship without a runtime change.

## (a) LOO, fold refits, 1.0 sigma (`scripts/lab_canon_sh_ens.py <h9w1fold> <canon2_hr> w`)

| fold | 0 | 0.5 | 0.65 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|
| hold_rec | 0.804 | 0.815 | 0.820 | 0.826 | **0.834** |
| pulse40 | 0.716 | 0.723 | 0.725 | **0.728** | 0.727 |
| mid40 | 0.657 | 0.660 | 0.661 | **0.662** | 0.659 |
| multilevel200 | 0.680 | 0.687 | 0.689 | **0.691** | 0.688 |
| compose | 0.616 | 0.631 | 0.634 | **0.636** | 0.635 |
| p4 pulse_long_recovery | 0.721 | 0.767 | 0.783 | 0.798 | **0.824** |
| p6 voi | 0.698 | 0.741 | 0.756 | 0.773 | **0.804** |
| p6 exam | 0.698 | **0.729** | 0.727 | 0.720 | 0.708 |
| **LOO** | 0.6987 | 0.7190 | 0.7244 | 0.7291 | **0.7348** |

Monotone in $w$. Per observable (mean over folds): wait 0.822 at 0.5 vs 0.816 at 1.0; queue 0.719 vs
**0.778**; discharges 0.616 vs 0.610. A per-obs pick (mean on wait and discharges, canon2 on queue)
gives LOO 0.739 vs 0.735: +0.004, far inside the fold noise.

## (b)+(c) Public history with truth $T_w = w\,C + (1-w)\,H$ (`scripts/lab_q_hosp_weight.py`)

40 eval-like episodes, seeds 23 / 11; all scored hospital predictors (incl. u019). off = mean
(implied - public), sd = its SD, sd_aff = affine residual SD. Public: u017 0.690, u019 0.729, gap +0.040.

| $w$ truth | off | sd | sd_aff | u017 | u019 | gap |
|---|---|---|---|---|---|---|
| 0.00 | +.062/+.041 | .062/.067 | .034/.034 | .907/.884 | .784/.772 | -.123/-.111 |
| 0.20 | +.067/+.046 | .042/.053 | .016/.020 | .816/.802 | .843/.834 | +.027/+.031 |
| 0.50 | +.073/+.054 | .066/.077 | .013/.016 | .748/.745 | 1 (self) | - |
| 0.65 | +.050/+.034 | .044/.055 | .023/.021 | .727/.728 | .904/.898 | +.177/+.170 |
| 0.80 | +.034/+.019 | .033/.044 | .016/.016 | .712/.719 | .843/.834 | +.131/+.114 |
| 1.00 | +.012/+.001 | .024/.035 | .012/.014 | .694/.711 | .784/.772 | +.090/+.061 |
| 1.10 | +.002/-.008 | .021/.033 | .013/.016 | .679/.699 | .760/.748 | +.081/+.048 |
| 1.20 | -.007/-.018 | .019/.031 | .014/.018 | .665/.681 | .739/.726 | +.074/+.045 |

- The two public points are matched best at $w \approx 1.1$-$1.2$ (u017 0.68-0.70, u019 0.73-0.76).
  Every $w \le 0.8$ puts u019 0.10-0.18 above its public score.
- The gap +0.040 is also hit near $w \approx 0.2$, but there u017 is implied 0.80-0.82 vs 0.690: out.
- Offset ~0 and smallest sd at $w \ge 1$; sd_aff is flat (0.012-0.016) for $w$ 0.5-1.2.
- Final4 build = exact mean of the two docs (max diff 0).

Break-even of canon2 alone vs the mean: implied P1.0 - P0.5 is -0.08 at truth $w=0.65$, +0.04 at 0.8,
+0.17-0.22 at 1.0-1.1. So canon2 alone wins whenever truth $w \gtrsim 0.72$, which is all of the
region the history supports. Per observable under these truths canon2 is closer on all three
observables, so the history does not support the per-obs split either.

## Verdict
$w = 1$: canon2 alone, doc `plans/hospital_queue_canon_hospital_queue_canon2_hr_doc.json`
(already packaged as `submissions/20260929-1840-alt1b/hospital_queue`, kind ode family
hospital_queue_canon2). Implied gain over final4 is +0.11..+0.17 under the best-fit truths; with the
usual single-truth optimism (~2x on u019 here) we expect about +0.04..+0.08, i.e. 0.77-0.80.
Downside only if the truth sits at $w \le 0.7$, which both public points reject.
