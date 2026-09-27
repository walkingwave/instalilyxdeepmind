# hospital_queue v9: restoring the sustained band (Sun Sep 27)

No credits spent. Same 6 runs (1,070 ticks). Calibrated in-sample = mean over runs of
`metric.score` at $\sigma$ = [22.94, 27.54, 2.149], queue capped at 333. LOO = `scripts/ode_lab.py`
leave-one-run-out at the fit scale (1.0x or 1.5x calibrated). No polish anywhere: every theta below is
the lab's cauchy fit.

## 1. Why v8f lost sustained (public 0.679 vs p3 0.690; sustained band 0.628 vs 0.677)

The first guess (fatigue never recovers, $\tau_f$ = 4,988) is not the mechanism. After 200 ticks of
overtime and 3,800 of recovery, v8f's end state equals the no-overtime end state (gap 0.01 sigma):
capacity under recovery exceeds arrivals, so a 13% fatigue loss does not show. Bounding $\tau_f$ to
[30, 400] (v9) changes nothing at the steady state.

What differs from p3 is the **long-run overtime law** and **discharges under stress** (holds at
staffing 8, other controls at recovery, end of a 4,000-tick hold; wait / queue / discharges):

| model | ot 0 | ot 1 | diag 0.8 | pulse hold |
|---|---|---|---|---|
| p3 (public sustained 0.677) | 51.7 / 313 / 4.56 | 51.4 / 313 / 4.60 | 116 / 320 / 1.52 | 137 / 329 / 1.20 |
| v8f (0.628) | 48.8 / 306 / 3.93 | 30.5 / 263 / 6.66 | 90 / 319 / 0.07 | 88 / 327 / 0.11 |

- v8f: $\dot F = O(1-F)/20 - F/\tau_f$ saturates $F \to 1$ for any $O > 0$, so the long-run factor is
  $(1 + g O)(1 - k_{fat})$ with $k_{fat}$ 0.13 < $g$ 0.62: sustained overtime is a permanent gain.
  p3: $F \to O$, $(1 + 0.55 O)(1 - 0.35 O) \approx 1$: neutral. The public ranking of three uploads with
  harmful / neutral / gainful overtime put neutral first.
- The wave discount $f_3 r_t^4/(r_t^4 + r_b^4)$ sends reported discharges to ~0 at every low-service
  hold (pulse, diag 0.8). Right for the bursty 40-tick pulses in the data (mode 0), wrong for long holds.
- Remaining gap to p3 in every v8/v9 fit: the wait at a saturated queue (~38-43 vs p3 ~52; pulse
  ~104-120 vs 137). The fits pick larger chairs/beds, so fewer patients count as "waiting". Not
  identified by our data (queue at cap only briefly in multilevel200).

## 2. Candidates

| id | structure | fit scale |
|---|---|---|
| v9 | v8f, $\tau_f \in [30, 400]$ | 1.0 / 1.5 / 1.0 warm (v9w, from v8f theta) |
| v9b | v8b, $\tau_f \in [30, 400]$ | 1.0 / 1.5 |
| v9d | p3 + wave discount only | 1.0 / 1.5 |
| v9e | v9, wait target $(W + A_1 + A_2)/(f_{2b} + 1)$ | 1.0 |
| v9f | v9, **neutral overtime**: $\dot F = (O - F)/\tau_f$, $\tau_f \in [10, 100]$, $k_{fat} = g/(1+g)$ | 1.0 / 1.5 |
| **v9g** | v9f **without the wave discount** (discharges = $f_3$) | 1.0 / 1.5 |

With $F \to O$ and $k_{fat} = g/(1+g)$, sustained full overtime ends at $(1+g)(1 - g/(1+g)) = 1$ exactly;
overtime only helps for ~$\tau_f$ ticks. At $O = 0.5$ the factor is $(1 + g/2)(1 - g/(2(1+g)))$, a small
gain (+5% at $g$ 0.64).

## 3. Results

Hold-agreement = mean calibrated score of the candidate against p3 on 8 eval-shaped sustained schedules
(SU) and 5 overtime-then-recovery schedules (O). OT gap = |end(ot 1) - end(ot 0)| in sigma units.

| model | cal in-sample | LOO (scale) | SU agree | O agree | ot1 end (w/q/d) | OT gap | disch diag0.8 / pulse |
|---|---:|---:|---:|---:|---|---:|---|
| p3 public doc | 0.678 | – | 1 | 1 | 51.4/313/4.60 | 0.0 | 1.52 / 1.20 |
| p3 lab refit | 0.695 | 0.634 (1.0) / 0.701 (1.5) | 0.923 | 0.942 | | 0.0 | |
| v8f (public 0.679) | 0.746 | 0.713 (1.5) | 0.730 | 0.763 | 30.5/263/6.66 | 1.6 | 0.07 / 0.11 |
| v9 | 0.728 | 0.667 (1.0) | 0.742 | 0.860 | 23.0/289/7.86 | 1.7 | 0.16 / 0.58 |
| v9w | 0.730 | 0.682 (1.0) | 0.744 | 0.868 | 23.3/290/7.81 | 1.7 | 0.17 / 0.57 |
| v9 s1.5 | 0.719 | 0.733 (1.5) | 0.747 | 0.881 | 23.2/290/7.89 | 1.7 | 0.23 / 0.77 |
| v9b | 0.702 | 0.643 (1.0) | 0.776 | 0.910 | 35.9/303/6.89 | 0.7 | 0.36 / 0.30 |
| v9b s1.5 | 0.711 | 0.721 (1.5) | 0.792 | 0.871 | 34.9/304/6.78 | 1.0 | 1.49 / 1.74 |
| v9d | 0.700 | 0.640 (1.0) | 0.886 | 0.944 | 46.4/312/5.01 | 0.4 | 0.40 / 0.25 |
| v9d s1.5 | 0.698 | 0.720 (1.5) | 0.860 | 0.930 | 64.3/313/3.73 | 0.4 | 1.05 / 0.28 |
| v9e | 0.725 | 0.666 (1.0) | 0.752 | 0.858 | 30.0/283/7.97 | 1.8 | 0.15 / 0.60 |
| v9f | 0.701 | 0.660 (1.0) | 0.774 | 0.833 | 43.4/310/4.64 | 0.0 | 0.15 / 0.04 |
| v9f s1.5 | 0.711 | 0.694 (1.5) | 0.801 | 0.872 | 39.1/312/4.40 | 0.0 | 0.28 / 0.09 |
| v9g | 0.713 | 0.639 (1.0) | 0.850 | 0.876 | 40.3/312/4.62 | 0.0 | 1.54 / 1.20 |
| **v9g s1.5** | 0.707 | **0.716 (1.5)** | 0.840 | 0.884 | 38.4/312/4.60 | 0.0 | **1.53 / 1.20** |

No candidate drifts to a different steady state after overtime ends (hysteresis vs the same schedule
without overtime: 0.00-0.01 sigma for all). All finite, 0.6-0.8 s per 4,000-tick episode.

## 4. Decision

**v9g at 1.5x** (`plans/hospital_queue_hospital_queue_v9g_v9gs15_doc.json`). It is the only candidate that
meets both long-hold requirements: overtime exactly neutral at the steady state (discharges 4.60 at
ot 1, p3 4.60; the gain decays over $\tau_f$ = 100) and discharges 1.53 / 1.20 under the diag-0.8 and
pulse holds (p3 1.52 / 1.20). Its LOO 0.716 (1.5x) is above p3's 0.701 and level with v8f's 0.713, so the
sequence side should hold. Theta: lam0 10.93, c_a 0.777, c_t 0.958, tau_a 1.09, chairs 17.8, beds 81.8,
ot_gain 0.642, k_l 0.0158, tau_w 35.5, q_cap 317.5, d_h 0.011, tau_o 8.0, tau_f 100 (at the bound we set),
tau_up 1.09; r_b unused.

Rejected: v9 / v9e (overtime still a permanent gain; stress discharges ~0); v9b s1.5 (discharges fine but
overtime gain 1 sigma); v9d (closest to p3 on generic holds but the discount zeroes pulse discharges);
v9f (neutral overtime but discount zeroes stress discharges): the discount alone costs the stress holds.

Open risks: (1) the saturated wait is still ~13 below p3 (38 vs 51; pulse 104 vs 137), ~0.5-1.4 sigma,
the same direction v8f was wrong in; (2) d_h 0.011 and beds 82 are weakly identified, so holds at
diag 0.1 extrapolate; (3) calibrated in-sample 0.707 is below v8f's 0.746 (the discount's in-sample
gain on the pulse runs is given up on purpose).
