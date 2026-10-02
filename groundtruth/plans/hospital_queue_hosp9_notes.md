# hospital_queue hosp9: wait law on top of the public p3 states (Mon Sep 28)

No credits. 8 runs (1,770 ticks), calibrated sigma [22.94, 27.54, 2.149], lab `scripts/ode_lab.py`
at 1.0 sigma. Family `gtlab/ode/hospital_queue_hosp9.py` = p3 plus a wider wait law; with
eps_w = 1, c_w = 1, tau_dn = tau_w, b_ot = 0 it reproduces p3 exactly (max |dY| 3e-14 on all runs).

## 1. Idea

In p3 the reported wait is output only (wt never feeds back into W, A1, A2, T). So we keep every
queue/discharge parameter at the public doc (u010b, 0.6896) and refit only the wait law, leave-one-
run-out. Queue and discharges are then identical to the public predictor on every schedule.

    target = c_w * W / (f1 + eps_w) + b_ot * G / (W + w0)
    wt rises to target with tau_w, falls with tau_dn
    G: dG = staffing * overtime / 20 - G / tau_g   (overtime staff-hours stock)

## 2. Results (LOO vs the public doc's own per-run scores; the doc was fitted on the first five runs,
so those baselines are in-sample and favour it)

| tag | free | LOO | vs pub | hold_rec | pulse40 | mid40 | multilevel | compose | p4 | voi | exam |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pub doc | - | 0.6746 | 0 | 0.811 | 0.617 | 0.664 | 0.689 | 0.594 | 0.695 | 0.673 | 0.654 |
| h9w0 | tau_w | 0.6712 | -0.003 | 0.811 | 0.611 | 0.678 | 0.678 | 0.589 | 0.694 | 0.671 | 0.639 |
| h9w3 | tau_w, eps_w, tau_dn | 0.6774 | +0.003 | 0.805 | 0.652 | 0.648 | 0.686 | 0.591 | 0.705 | 0.682 | 0.650 |
| h9w5 | as h9w3, c_w = 1.35 fixed | 0.6913 | +0.017 | 0.804 | 0.685 | 0.654 | 0.694 | 0.606 | 0.713 | 0.690 | 0.686 |
| **h9w1** | tau_w, eps_w, c_w, tau_dn | **0.6987** | **+0.024** | 0.804 | 0.716 | 0.657 | 0.680 | 0.616 | 0.721 | 0.698 | 0.698 |
| h9w1d | h9w1, 12 starts, 2x budget | 0.6987 | +0.024 | same optimum | | | | | | | |
| h9w2 | h9w1 + overtime stock | 0.6736 | -0.001 | 0.812 | 0.671 | 0.636 | 0.671 | 0.561 | 0.703 | 0.681 | 0.654 |

Wait-only per fold (h9w1 vs pub): exam 0.825 vs 0.693, voi 0.854 vs 0.780, p4 0.857 vs 0.781,
pulse40 0.893 vs 0.598, compose 0.625 vs 0.557; losses hold_rec 0.934 vs 0.955, mid40 0.900 vs
0.920, multilevel 0.605 vs 0.631 (all three in-sample for pub).

h9w1 theta (wait only): tau_w 51.3, eps_w 1.50, c_w 1.70, tau_dn 19.8. Rest = public doc.

## 3. Overtime aftermath (compose t188-270, wait 21 -> 387 with queue 23)

The overtime staff-hours stock (h9w2) cannot be learned: only compose shows the pocket, so its own
fold loses (0.561 vs 0.594), and the full fit extrapolates (staffing-1 hold wait 1,020). Rejected.
Reading of the data: the pocket starts when W hits zero during the overtime drain (t188, queue 104),
the wait target then sits near 300-400 and keeps rising about 2 per tick for 60 ticks at constant
recovery controls, and collapses when the pulse refills the queue (t270). A quantity that grows
linearly at constant state looks like a cohort age, not a Little estimate; hold_rec (same controls,
same queue 23) shows wait 0.00, so the state that differs is history only. Not identifiable from
one run.

## 4. Long-run map (4,000-tick holds, wait end value; queue/discharges unchanged)

| hold | pub | h9w5 | h9w1 |
|---|---:|---:|---:|
| staffing 8 | 51.7 | 67.9 | 80.7 |
| mid (exam hold) | 43.5 | 57.4 | 68.9 |
| pulse | 137 | 173 | 190 |
| staffing 1 | 254 | | 304 |

The exam's mid hold reaches 52.9 at t80 and is still rising (0.25 per tick): p3's steady state 43.5
is below the data already, which supports a higher saturated wait (v9g went lower and lost
sustained). Agreement with p3 on the wait over 4,000-tick eval schedules: h9w1 0.54 sustained /
0.89 recovery; h9w5 0.65 / 0.90. That is the risk: the sustained wait level moves 1.5x.

## 5. Decision

Candidate: `plans/hospital_queue_hospital_queue_hosp9_h9w1_doc.json` (runtime check: 4,000-tick
episodes finite, 1.0-1.3 s each on a loaded machine). Lower-risk alternate:
`plans/hospital_queue_hospital_queue_hosp9_h9w5_doc.json` (+0.017, no fold below -0.010).
Forecast 0.6 x LOO gain: h9w1 +0.014 (about 0.704), h9w5 +0.010.
