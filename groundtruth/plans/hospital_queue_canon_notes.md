# hospital_queue canon: literal-code readings of the brief (Mon Sep 28, no credits)

Question: does a textbook discrete queue (hard min, round rates) plus the brief's three named
mechanisms written literally beat the pick (hosp9 h9w1: p3 queue/discharges + refitted wait law)?
Lab `scripts/lab_canon_sh_loo.py`, 8 runs (1,770 ticks), calibrated sigma [22.9, 27.5, 2.15] at
1.0x, queue capped at 333 as shipped. Every fold refit from the h9w1 theta (3 starts, 40 evals,
150 s). "pick" = h9w1's own LOO from `hospital_queue_hosp9_notes.md` (its queue/discharge
parameters were fitted on the first five runs, so those folds favour it).

## 1. What the tick data say

- Recovery hold (hold_rec): the queue settles at **23.0 exactly** with discharges **11.5** and wait
  **0.0**: arrivals 11.5/tick, every patient in service for exactly 2 ticks (Little: 23 = 11.5 x 2),
  nobody waiting. The pick settles at 15.5 / 10.6 / 0.6 (queue 0.73, discharges 0.74 on this run).
- After a pulse (p4, voi) the recovery drain is linear (~2.5/tick) and then stops at a **plateau
  of 92-100** from t~140 to t300 (falls 0.09/tick), wait ~5, discharges 11.1. The pick drains to
  15.5 (queue 0.56 on both runs).
- compose t188-270: queue 23.0 and discharges 11.5 exactly (as hold_rec), but the wait climbs
  78 -> 387 (~13/tick, then ~2/tick) and is diluted by the next pulse's arrivals (387 -> 61 while the
  queue refills). Looks like the mean age of a stuck cohort; one run only.
- exam t360-380: at recovery controls the queue jumps 275 -> 307 and discharges fall to 4: a delayed
  return of patients discharged during the t310-340 bursts.

## 2. Literal versions (families; mechanisms switched by parameters)

Base structure = hosp9. Added, each with an off switch:
- handover: effective staff $S_e$ rises to a staffing increase with lag $\tau_o$, follows decreases
  at once ($\dot S_e = (S - S_e)/\tau_o$ up, $4(S - S_e)$ down); off: $\tau_o = 0.3$.
- returning case mix: share $\rho$ of discharges returns to the waiting list through a two-stage
  delay (mean $\tau_r$); follow-up prevents share $p_f \cdot fu$ and diverts $k_{fs} \cdot fu$ of staff.
- admission drain $k_{dr}$ free (hosp9: 3), so an uncongested waiting list empties within the tick.
- canon2: treatment stage empties at $k_t$ (hosp9: 3), so the uncongested occupancy (23 = 11.5 x 2)
  is free of the congested capacities.
- canon3: fatigue as a slowly recovering overtime stock $\dot F = O - F/\tau_f$, eff / (1 + k F).

## 3. LOO at 1.0 sigma (mean of 3 observables)

| version (pair) | hold_rec | pulse40 | mid40 | multilevel | compose | p4 | voi | exam | LOO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pick h9w1 (notes) | 0.804 | 0.716 | 0.657 | 0.680 | 0.616 | 0.721 | 0.698 | 0.698 | 0.6987 |
| hosp9 refit all (same protocol) | 0.804 | 0.684 | 0.643 | 0.665 | 0.509 | 0.739 | 0.710 | 0.692 | 0.6809 |
| canon1 fatigue + returns | 0.798 | 0.691 | 0.635 | 0.646 | 0.586 | 0.795 | 0.745 | 0.711 | 0.7009 |
| canon1 handover + returns | 0.779 | 0.728 | 0.658 | 0.688 | 0.637 | 0.824 | 0.801 | 0.709 | 0.7280 |
| canon1 fatigue + handover | 0.611 | 0.723 | 0.694 | 0.669 | 0.559 | stopped | | | |
| canon2 fatigue + returns | 0.797 | 0.687 | 0.670 | 0.646 | 0.575 | 0.769 | 0.737 | 0.710 | 0.6987 |
| **canon2 handover + returns** | **0.834** | **0.727** | **0.659** | **0.688** | **0.635** | **0.824** | **0.804** | **0.708** | **0.7348** |
| canon2 h+r, $p_f = 1$ fixed | 0.617 | 0.725 | 0.645 | 0.664 | 0.615 | 0.798 | 0.761 | 0.710 | 0.6923 |
| canon3 slow fatigue + handover | 0.641 | 0.705 | stopped | | | | | | |
| canon3 slow fatigue + returns | 0.670 | 0.696 | stopped | | | | | | |

Named segments, canon2 h+r vs same-protocol base: compose pocket t180-270 0.568 vs 0.347;
p4 t130-300 (plateau) 0.883 vs 0.759; voi t130-300 0.863 vs 0.721; hold_rec t30-120 0.882 vs 0.851.

In-sample (full fit, `plans/hospital_queue_canon_hospital_queue_canon2_hr_doc.json`) vs pick:
hold_rec 0.849 / 0.804, pulse40 0.731 / 0.722, mid40 0.659 / 0.659, multilevel 0.692 / 0.689,
compose 0.652 / 0.618, p4 0.833 / 0.726, voi 0.810 / 0.704, exam 0.728 / 0.705.
Theta: lam0 9.99, rho 0.248, tau_r 110, pf 0.369, k_fs 0.024, tau_o 15.9, kt 1.63, kdr 4.28,
k_fat 0 (fatigue off); wait law c_w 2.00, eps_w 1.75, tau_w 36.5, tau_dn 18.1.

## 4. Reading

- The mechanism pairs separate cleanly: **handover + returns** wins (0.735 / 0.728), fatigue +
  returns 0.70, fatigue alone (pick structure) 0.68-0.70, fatigue + handover loses hold_rec badly.
  Returns are what explain the p4/voi plateau and the exam jump; the orientation lag (tau_o ~16)
  explains the slow start of the recovery drain after staffing 5 -> 20.
- canon2 h+r beats the pick on **every fold** (hold_rec +0.030, pulse40 +0.011, mid40 +0.002,
  multilevel +0.008, compose +0.019, p4 +0.103, voi +0.106, exam +0.010; mean +0.036). It passes the
  acceptance rule as stated (no pulse/recovery/exam fold lost).
- Fixing $p_f = 1$ (follow-up at 1 prevents every return) kills the hold_rec fold (0.617): the data
  want some returns under full follow-up.
- Slow-fatigue stock (canon3) cannot hold 23 on the hold_rec fold; stopped after two folds.

## 5. Risk before shipping (not tested by any run)

The return loop has a long-run consequence the runs never reach. In a clean recovery hold from
reset, canon2 h+r matches 23 through t120 (the hold_rec data end) and then **climbs to 105 by
t1000** (returns 0.25 x (1 - 0.37) of 11.4 discharges push the load up to the treatment capacity);
after a pulse it holds ~98-105 forever (data: 92 at t300, falling 0.09/tick). The pick stays at
15.5. Agreement with the pick on 4,000-tick eval-shaped schedules: queue 0.33 on recovery, 0.58 on
composition, 0.76 on sustained. So the LOO gain is real on 300-tick windows, but on 4,000-tick
episodes the queue level is a bet on an unobserved steady state. A long recovery hold (>= 400 ticks
from reset) would settle it. Until then: ship only as a hedge (e.g. median with the pick), or with
the return loop limited to pulse-era discharges.

Files: `gtlab/ode/hospital_queue_canon{1,2,3}.py`, lab jsons/docs
`plans/hospital_queue_canon_hospital_queue_canon{1,2}_{fr,hr}*.json`, `..._canon2_pf1*.json`,
base `plans/hospital_queue_canon_hospital_queue_hosp9_base*.json`.

## 6. Follow-up (same day): what 210 credits can decide, and safer variants

### 6.1 Pure recovery hold from a typical reset (y0 = 3.5 / 50 / 11.5), queue noise sd 2.36

| t | pick queue | canon2 h+r queue | gap (noise sd) | gap (calibrated sigma) |
|---|---:|---:|---:|---:|
| 120 | 15.5 | 22.6 | 3.0 | 0.26 |
| 150 | 15.5 | 23.0 | 3.2 | 0.27 |
| 180 | 15.5 | 23.4 | 3.4 | 0.29 |
| 210 | 15.5 | 26.1 | 4.5 | 0.39 |

Wait 0.5 vs 0.4 and discharges 10.6 vs 11.4 (0.7 noise sd) at t210. The gap is above 2 noise sd, but
it adds nothing: hold_rec already shows 23 through t120, so the pick is already refuted there. The
open question is whether canon2 keeps creeping upward (it only goes from 22.6 to 26.1 by t210, 1.3
noise sd above a flat 23 per tick), so a 210-tick recovery hold is **not** decisive.

### 6.2 Best 210-tick discriminator

Scored against three models (pick, canon2 h+r, canon4 elective-only returns); the queue gap is in
noise sd at t210:

| schedule | pick | c2 | c4 | pick-c2 | c2-c4 | pick-c4 |
|---|---:|---:|---:|---:|---:|---:|
| recovery 210 | 15.5 | 26.1 | 19.2 | 4.5 | 2.9 | 1.6 |
| recovery, follow-up 0, 210 | 15.5 | 70.1 | 19.2 | 23.2 | 21.6 | 1.6 |
| elective 30 (staff 20, fu 0) + recovery 180 | 15.5 | 101.8 | 103.0 | 36.6 | 0.5 | 37.1 |
| **elective 30 (fu 0) + recovery with fu 0, 180** | 15.5 | 131.3 | 102.8 | **49.2** | **12.1** | **37.1** |
| pulse 40 + recovery fu 0, 170 | 48.4 | 117.5 | 99.5 | 29.3 | 7.6 | 21.7 |
| pulse 40 + recovery 170 (as p4/voi) | 48.4 | 95.0 | 99.7 | 19.8 | 2.0 | 21.8 |

Chosen design: `plans/p9_hospital_alt.json` (exp `alt_el30_fu0`, 210 steps). It separates every pair
by more than 10 noise sd, and it tests the return loop directly. Alternate if only the recovery
creep matters: recovery with follow-up 0 for 210 ticks (c2 vs flat: 21 sd).

### 6.3 Safer variants, LOO at 1.0 sigma

| version | hold_rec | pulse40 | mid40 | multilevel | compose | p4 | voi | exam | LOO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pick h9w1 (fold refits, reproduced) | 0.804 | 0.716 | 0.657 | 0.680 | 0.616 | 0.721 | 0.698 | 0.698 | 0.6987 |
| canon2 h+r | 0.834 | 0.727 | 0.659 | 0.688 | 0.635 | 0.824 | 0.804 | 0.708 | 0.7348 |
| **median (= mean of 2) pick, canon2 h+r** | 0.815 | 0.723 | 0.660 | 0.687 | 0.631 | 0.767 | 0.741 | 0.729 | **0.7190** |
| 0.3 pick + 0.7 canon2 | 0.822 | 0.726 | 0.662 | 0.690 | 0.635 | 0.788 | 0.761 | 0.725 | 0.7260 |
| canon4: returns only from elective (pulse-period) cases | 0.615 | 0.721 | 0.652 | 0.656 | 0.564 | 0.792 | 0.753 | 0.700 | 0.6816 |

The median is at or above the pick on every fold (+0.020 mean) and is the best on exam (0.729); it
halves the long-run exposure (recovery-hold queue 20.8 at t210, ~60 at t1000 vs 105). canon4 fails
the same way the $p_f = 1$ run did: whenever returns cannot carry load under full follow-up, the
hold_rec fold breaks (queue 0.28). The data tie the 23-level and the post-pulse plateau to one loop.
Lab: `scripts/lab_canon_sh_ens.py`; fold fits `plans/hospital_queue_canon_hospital_queue_hosp9_h9w1fold.json`,
`..._canon4_c4f*.json`, full fit `..._canon4_c4_doc.json`.
