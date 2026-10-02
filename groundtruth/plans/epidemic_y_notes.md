# Epidemic y: switch timing, mechanism pairs on the new run

Sun Sep 27, 19:00-21:40. No credits. Data: 5 runs, 1,411 ticks (`p1.hold_rec` 120,
`p2.pulse120_280` 400, `p3.compose` 291, `p4.joint_hold` 300, and the new `p6.voi` 300: mask 1
for 100 ticks, then vaccination 0.003 alone for 200). Organizer scale $\sigma = (12.95, 5.23)$.

## 1. Protocol

Every number comes from `scripts/ode_lab.py --system epidemic --budget 300 --starts 8 --nfev 50
--sigma-cal 1.0`: fit and score at $1.0\sigma$; LOO = fit on 4 runs (150 s per fold), score the
held-out run; in-sample = full fit on all 5. Module defaults + 7 LHS starts, no warm starts.
Reruns of the same command give the same LOO to 0.004 (x11 AC 0.622 / 0.618, y2 AC 0.683 / 0.683),
so the lab is reproducible; the start-set noise of the x notes (up to 0.08 on one fold) still applies
to structural comparisons, so we only trust differences that show on several folds at once.

**Baseline recomputed on the 5 runs: x11 AC LOO 0.622** (0.656 on the old 4 runs: the new p6 fold is
the weakest at 0.536, and the p4 fold drops 0.685 -> 0.605 once p6 enters the training set).

## 2. What p6 shows

| segment | truth | x11 AC (held out on p6) |
|---|---|---|
| first ticks under mask 1 | 217.9, 217.7, then +8.6, +7.1, +7.0, +4.8 (x1.031/tick) | 219.6, 223.5, 229.1 (moves at once) |
| first wave peak under mask 1 | 259 at t=13, beds peak 150.8 (below the 155 cap) | 284, beds sit on the cap for 40 ticks |
| vaccination-only switch at t=100 | 39.2, 39.0, then +3.1, +4.1, +3.5, +3.5 | smooth bend |
| second wave peak (vaccination on) | 150.5 at t~150 | 121 |
| tail t=290 | 87 cases / 46.5 beds | 74 / 50 |

Early growth per tick right after a start, by start control: none x1.165-1.169 (p1, p3 with
closure 0.85), joint 0.85 x1.075 (p4), pulse (1, 1, .003) x1.049 (p2), mask 1 alone x1.031 (p6).
Mask alone is lower than mask plus closure: closure seems to push contacts where masks act less.

Every start and every switch in every run has the same shape: two observations exactly on the old
trend, then the slope turns at once and stays turned. An Erlang-2 latent stage (x7/x11) bends the
slope over two to three ticks instead of turning it. A pure two-tick delay on a process whose slope
turns at once fits the shape: onsets leave one exponential latent stage (the onset slope turns at the
switch) and are reported two ticks later. Beds show no such delay (they move on the first tick).

## 3. Candidates (all numpy + math, batched, x11 vaccination bounds kept)

| family | change | params | states |
|---|---|---:|---:|
| `epidemic_y` | x11 + contacts split home / school / community; closure relocates school contacts home; masks act fully outside the home and `mask_home` inside; debt boosts community contacts (x8's split on x11) | 24 | 24 |
| `epidemic_y2` | **x11 with one latent stage and a reporting delay**: $\dot E_i = \lambda_i S_i - \sigma E_i$, onsets $o = N\sigma\sum_i n_i E_i$, reported through 6 stages of rate $6/2$: $\dot D_1 = 3(o - D_1)$, $\dot D_k = 3(D_{k-1} - D_k)$, cases $= D_6$, all $D_k(0) = y_0$ | 22 | 27 |
| `epidemic_y3` | y2 + four-stage referral pipeline (same mean $\tau_p$) for the three-tick bed dead time at every start | 22 | 29 |
| `epidemic_y4` | y2 with 8 delay stages and a fitted delay mean `dly` in [1.5, 4] (fit: 2.72) | 23 | 29 |
| `epidemic_y5` | y3 with y4's delay (fit `dly` 2.98) | 23 | 31 |
| `epidemic_y6` | y2 with `k_clinic` <= 20 (y2 sits at its bound 5); fit 7.4 | 22 | 27 |
| `epidemic_y7` | y2 with a curved mask response $1 - e\,m^{q}$ (fit $q$ = 1.96) | 23 | 27 |
| `epidemic_y8` | y2 with vaccine protection delayed through V for a fitted `tau_vd` in every pair (fit 10.3) | 23 | 27 |
| `epidemic_y9` | y7 + y6 | 23 | 27 |
| `epidemic_y10` | y7 + y8 | 24 | 27 |
| `epidemic_y11` | y2 + Erlang-2 infectious period (x10's change) | 22 | 30 |

## 4. Results (LOO and in-sample at $1.0\sigma$)

| model | pair | params | LOO p1 | p2 | p3 | p4 | p6 | **LOO** | in-sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x11 | AB | 22 | 0.705 | 0.616 | 0.612 | 0.609 | 0.605 | **0.629** | 0.727 |
| x11 | AC | 22 | 0.710 | 0.628 | 0.634 | 0.605 | 0.536 | **0.622** | 0.752 |
| x11 | AC rerun | 22 | 0.710 | 0.612 | 0.634 | 0.599 | 0.536 | **0.618** | 0.752 |
| x11 | BC | 22 | 0.609 | 0.356 | 0.639 | 0.525 | 0.592 | **0.544** | 0.682 |
| y | AC | 24 | 0.654 | 0.580 | 0.646 | 0.513 | 0.545 | **0.588** | 0.753 |
| **y2** | **AC** | 22 | 0.766 | 0.663 | 0.659 | 0.763 | 0.566 | **0.683** | 0.761 |
| y2 | AC rerun | 22 | 0.766 | 0.663 | 0.659 | 0.763 | 0.566 | **0.683** | 0.761 |
| y2 | AB | 22 | 0.673 | 0.617 | 0.626 | 0.587 | 0.605 | **0.622** | 0.725 |
| y2 | BC | 22 | 0.621 | 0.492 | 0.656 | 0.504 | 0.595 | **0.574** | 0.690 |
| y3 | AC | 22 | 0.767 | 0.664 | 0.652 | 0.753 | 0.564 | **0.680** | 0.764 |
| y4 | AC | 23 | 0.747 | 0.649 | 0.645 | 0.766 | 0.566 | **0.675** | 0.758 |
| y5 | AC | 23 | 0.742 | 0.653 | 0.640 | 0.766 | 0.565 | **0.673** | 0.760 |
| y6 | AC | 22 | 0.767 | 0.649 | 0.659 | 0.765 | 0.566 | **0.681** | 0.762 |
| y7 | AC | 23 | 0.790 | 0.662 | 0.741 | 0.609 | 0.592 | **0.679** | 0.773 |
| y8 | AC | 23 | 0.717 | 0.649 | 0.668 | 0.763 | 0.580 | **0.675** | 0.759 |
| y9 | AC | 23 | 0.797 | 0.652 | 0.741 | 0.597 | 0.594 | **0.676** | 0.777 |
| y10 | AC | 24 | 0.709 | 0.420 | 0.690 | 0.621 | 0.621 | **0.612** | 0.772 |
| y11 | AC | 22 | 0.765 | 0.654 | 0.657 | 0.762 | 0.574 | **0.682** | 0.759 |

What moved the score:

- **Reporting delay (y2): +0.061 LOO over x11 AC, on four of five folds** (p1 +0.056, p2 +0.035,
  p3 +0.025, p4 +0.158, p6 +0.030), same 22 parameters. In-sample score within 10 ticks of a start
  or switch (140 ticks): x11 0.704 -> y2 0.741; elsewhere 0.755 -> 0.759. The gain is in the
  transients, which is what the sequence categories score.
- **Pair AC stays best** on the new structure: AB 0.622, BC 0.574 (BC loses p2 and p4 as before;
  its fit puts `v_eld` and `tau_debt` on their bounds). On x11, AB (0.629) and AC (0.622) tie, but
  on y2 AC leads by 0.061 on four folds. Behavioural fatigue + postponed gatherings remains the
  identified pair.
- Setting-split contacts (y): the early-growth ordering in §2 points at it, but it loses p1 and p4
  (0.588 LOO) and lands `vac_eff` on its bound. Rejected a third time.
- Sharper referral pipeline (y3), sharper / fitted reporting delay (y4, y5; fitted mean 2.7-3.0
  ticks), looser clinic bound (y6), protection delay (y8), Erlang-2 infectious (y11): all within
  0.01 of y2. No gain; y2 is the simplest.
- Curved masks (y7, y9, $q \approx 2$): best in-sample (0.773-0.777) and +0.08 on the p3 fold,
  +0.03 on p1 and p6, but the p4 fold (the long 0.85 joint hold) falls 0.16 in both. With $q = 2$
  fitted on masks at 1 and 0.85-only switches, the long 0.85 hold is extrapolated wrongly. Not kept;
  one long mask hold at 0.5-0.7 would settle it.
- y10 (curve + protection delay) collapses the p2 fold (0.420) with four parameters on bounds.

## 5. Long holds (4,000 ticks from (150, 50)), mean of the last 500 ticks (cases / beds)

| hold | x11 AC (5 runs) | **y2 AC** | y3 AC | y7 AC | y11 AC |
|---|---|---|---|---|---|
| none | 105.1 / 78.8 | 104.6 / 78.5 | 104.4 / 78.7 | 104.7 / 78.3 | 104.8 / 78.5 |
| pulse (1, 1, .003) | 48.9 / 33.3 | 48.5 / 32.5 | 48.7 / 32.6 | 43.3 / 28.8 | 48.7 / 32.9 |
| vaccination .003 | 76.8 / 52.2 | 77.6 / 52.1 | 77.2 / 51.8 | 76.6 / 51.2 | 77.0 / 51.9 |
| closure 1 | 108.7 / 84.1 | 109.3 / 84.9 | - | 108.4 / 84.1 | 109.4 / 84.7 |
| mask 1 | 78.4 / 57.0 | 77.3 / 56.1 | 77.3 / 56.3 | 74.6 / 53.8 | 77.8 / 56.5 |
| closure + mask | 88.2 / 66.4 | 85.2 / 64.2 | 85.5 / 64.6 | 83.7 / 62.9 | 86.5 / 65.0 |
| (.85, .85, .00255) | 63.0 / 43.7 | 63.1 / 43.3 | 63.2 / 43.3 | 63.3 / 43.4 | 63.0 / 43.5 |
| min cases, vaccination 2,000 then pulse 2,000 | 10.6 | 13.0 | 12.9 | 10.2 | 12.4 |
| ticks with cases < 5, 12 test-shaped schedules | 0 % | 0 % | 0 % | 0 % | 0 % |

All settle on an endemic level set by the controls (vaccination holds ~77 cases), no die-out, no
runaway (max ~490 cases, the first free wave; beds stop at the 155 cap). y2's long holds match
x11's within 3 cases: the delay only moves transients.

Speed (one 4,000-tick episode, single thread, machine shared with 3 fits and other jobs): x11
0.9 s, y2 1.1-1.3 s at `N_SUB = 3`. The delay stages have rate 3, so `N_SUB = 2` is still stable:
same y2 theta on an order-shaped episode differs by at most 0.9 cases (mean 5e-5 $\sigma$) and
runs in 0.9 s. The lab doc keeps `n_sub` 3.

## 6. Decision

- **Recommend `epidemic_y2`, pair AC: `plans/epidemic_epidemic_y2_AC_doc.json`.** LOO 0.683 vs
  x11 AC 0.622 on the same 5 runs (+0.061, four of five folds up, reproducible), in-sample 0.761
  (+0.009), 22 parameters, clean long holds. Fitted: $N$ 18,690, $\beta$ 0.230, $\sigma$ 0.154,
  $\gamma$ = (0.44, 0.37, 0.36), sev = (0, 0.082, 0.153), los 9.9, $h_{cap}$ 154.2, school 1.0 (bound),
  mask 0.417, $k_{clinic}$ 5.0 (bound), $v_{eld}$ 4.7, $r_I$ 1.07, $\tau_p$ 6.3, $\tau_{fat}$ 59.5,
  $k_f$ 0.56, $\tau_{wane}$ 86.7, vac_eff 2.16, $k_D$ 0.22, $\tau_D$ 143.
- Target LOO 0.69 not reached (0.683). Forecast public: 0.6 x 0.061 = **+0.037**, about 0.687 -> 0.72,
  mostly in the sequence categories.
- Backups within noise: y11 AC (0.682), y6 AC (0.681), y3 AC (0.680).
- Still open: the p6 fold (0.566 for every AC variant) — a model trained without p6 under-predicts
  the second wave under vaccination alone (x11 held out: peak 121 vs 150; y2 fitted on p6 still 138) and over-predicts the first wave under
  mask 1. Curved masks fix the first part but break the p4 fold; a mid-level mask hold would decide it.
