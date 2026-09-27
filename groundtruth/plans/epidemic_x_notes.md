# Epidemic x: rebuild against the brief and the switch data

Sun Sep 27, 00:10-02:10. No credits. Data: the same 4 runs, 1,111 ticks (`p1.hold_rec` 120,
`p2.pulse120_280` 400, `p3.compose` 291, `p4.joint_hold` 300). Organizer scale
$\sigma = (12.95, 5.23)$ for (daily_cases, hospital_load), `plans/sigma_calibrated.json`.

## 1. Protocol

Every number here comes from `scripts/ode_lab.py --sigma-cal 1.0 --budget 400 --starts 8 --nfev 50`:
fit AND score at exactly $1.0\sigma$ (the v8 notes fitted at $1.5\sigma$). LOO = fit on 3 runs
(200 s per fold), score the held-out run; "in-sample" = full fit on 4 runs, scored on the same
runs. Starts are the module defaults plus 7 LHS points (spread 0.5). Runs tagged `w` start from a
full-data v8g theta (a mild leak into the folds), so only compare `w` with `w`.

Baselines recomputed with the same protocol:

| model | LOO p1 | p2 | p3 | p4 | **LOO** | in-sample |
|---|---:|---:|---:|---:|---:|---:|
| v8g AB (module defaults) | 0.734 | 0.515 | 0.473 | 0.649 | **0.593** | 0.755 |
| v8g AB `w` (from gAB theta) | 0.736 | 0.515 | 0.525 | 0.706 | **0.621** | 0.755 |
| v8g AC (module defaults) | 0.727 | 0.480 | 0.420 | 0.675 | **0.576** | 0.763 |

## 2. What the switches say (read straight off the data)

Let $\Delta$ = change in slope of daily_cases at a switch. Cases are onsets, so a jump in the
infection rate shows up as a slope change of $\sigma_{lat}\,\Delta\text{inc}$, and
$\Delta\text{inc}/\text{inc} \approx \Delta/(\sigma_{lat}\,\text{cases})$ with $\sigma_{lat} \approx 0.155$.

| switch | cases around the switch | slope before -> after | infection rate after / before |
|---|---|---|---:|
| p2 release after 120 ticks at (1, 1, .003) | 48.3, 48.2, 47.9, **51.7, 56.5, 61.0** | -0.2 -> +4.6 | 1.6 (restricted = 0.62 of free) |
| p3 mask 0.85 on (t=67) | 73.3, 71.2, **66.3, 60.8, 55.5** | -2.1 -> -5.3 | 0.72 |
| p3 mask off after 45 ticks (t=112) | 21.4, 21.5, **22.9, 24.7, 26.6** | 0.1 -> +1.8 | 1.54 (0.65) |
| p3 joint 0.85 on (t=201) | 118.6, 116.6, **110.6, 103.0** | -2.0 -> -7.0 | 0.70 |
| p3 joint off after 45 ticks (t=246) | 26.3, 25.9, **27.4, 29.2** | -0.3 -> +1.7 | 1.4 (0.71) |
| p3 closure 0.85 off (t=45) | 188.7, 179.3, 169.7, 161.8 | no change | ~1.0 |
| p1 start (no control) | y0 187.3; **188.0, 187.8**, 218.4, 255.3, 289.4 | flat 2 ticks, then +31, +37, +34, +27 | - |

Four facts follow.

1. **Two ticks of dead time everywhere.** The action at tick $t$ first moves the observation at
   $t+2$; every run starts with two observations equal to $y_0$ and then a slope jump. One
   exponential latent stage moves the slope inside the first tick. This is the largest single
   structural miss of v8g and it touches every switch in every test episode.
2. **Masks do the work, closure moves contacts.** Mask 0.85 alone gives about $\times 0.70$,
   joint 0.85 the same, closure alone nothing measurable. This is the brief's "closure changes
   where contacts occur": the net count barely changes.
3. **Compliance is intact after 120 ticks of full restriction.** The release jump in p2
   ($\times 0.62$ restricted/free) is what a mask efficacy of ~0.38 predicts with no fatigue at
   all. Fatigue fitted at $k_f \approx 0.9$, $\tau \approx 250$ makes that jump too small in
   every v8g/x fit (model 44.5 -> 54.7 in 6 ticks, truth 48.2 -> 68.5).
4. **Yet p4 climbs from 42 (t=132) to a flat 60 (t=250-300) under constant restriction.** Every
   fit without a fatigue-like term (pair B alone, BC) loses p1 (cases 0.55-0.58 in-sample): the
   early growth then needs a low $R_0$, and a low $R_0$ with a $\times 0.7$ restriction kills
   the p4 plateau. Fatigue is the only term we found that reconciles p1 and p4, so it stays.

## 3. Candidates

All are numpy+math, batched (`_PY/_NP`), `N_SUB = 2`, both observables anchor the state
(cases fix the E/I levels in the fixed age mix, beds fix $H$; waiting list, pipelines, fatigue,
debt start at 0).

| family | change vs v8g | params | states |
|---|---|---:|---:|
| `epidemic_x` | contacts split home / school / community (fixed matrices summing to v8g's); closure removes school contacts and relocates a share $\rho$ home; masks act fully outside the home, `mask_home` inside; debt boosts community contacts only; `home_k` scales home | 25 | 21 |
| `epidemic_x2` | x + adults and elderly have separate referral pipelines, waiting lists and beds (`los`, `los_e`) sharing one capacity (severity-weighted bed demand by age) | 26 | 25 |
| `epidemic_x3` | x + behaviour keeps two traces: fatigue on compliance and learned caution on contacts that persists after release (`hab_k`) | 26 | 21 |
| `epidemic_x4` | x with closure as pure relocation ($\rho = 1$, same contacts per child, now at home) | 24 | 21 |
| `epidemic_x5` | v8g with A = caution driven by bed pressure: $\dot F = (H/h_{cap} - F)/\tau_b$, contacts $\times 1/(1 + k_b F)$; no fatigue | 22 | 21 |
| `epidemic_x6` | x4 + x2's age-specific stays | 25 | 25 |
| `epidemic_x7` | **v8g with a two-stage (Erlang-2) latent period**: $\dot E_{1,i} = \lambda_i S_i - 2\sigma E_{1,i}$, $\dot E_{2,i} = 2\sigma(E_{1,i} - E_{2,i})$, cases $= N\,2\sigma\sum_i n_i E_{2,i}$ | 22 | 24 |
| `epidemic_x8` | x4 + Erlang-2 latent | 24 | 24 |
| `epidemic_x9` | x7 + bed-pressure caution inside A ($1/(1 + k_b H/h_{cap})$) | 23 | 24 |
| `epidemic_x10` | x7 + Erlang-2 infectious period (transmission from both stages, recovery and referrals from the second) | 22 | 27 |
| `epidemic_x11` | **x7 with physical vaccination bounds**: `vac_eff` $\le 3$, `k_clinic` $\le 5$ | 22 | 24 |

x0 for the Erlang families: $E_{1,i} = E_{2,i} = \text{cases}/(N\,2\sigma)\cdot m_i/n_i$ (m = fixed age
mix), $I_i = r_I E_{2,i}\,2\sigma/\gamma_i$, so the first two ticks of onsets equal $y_0$.

## 4. Results (LOO and in-sample at $1.0\sigma$)

| model | pair | LOO p1 | p2 | p3 | p4 | **LOO** | in-sample | 4,000-tick eval min cases | s/episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8g | AB | 0.734 | 0.515 | 0.473 | 0.649 | 0.593 | 0.755 | 10.4 | 0.9 |
| v8g | AC | 0.727 | 0.480 | 0.420 | 0.675 | 0.576 | 0.763 | 1.8 | 0.9 |
| x | AB | 0.711 | 0.521 | 0.455 | 0.703 | 0.597 | 0.753 | **0 (13 % of sustained ticks)** | 1.1 |
| x | AC | 0.744 | 0.479 | 0.395 | 0.672 | 0.573 | 0.766 | 0 | 1.1 |
| x | BC | 0.535 | 0.464 | 0.453 | 0.413 | 0.466 | 0.724 | 13.6 | 1.1 |
| x2 | AB | 0.700 | 0.555 | 0.470 | 0.709 | 0.608 | 0.768 | 0 | 1.2 |
| x2 | AC | 0.750 | 0.520 | 0.402 | 0.723 | 0.598 | 0.780 | 0 | 1.2 |
| x3 | AB | 0.727 | 0.532 | 0.455 | 0.719 | 0.608 | 0.755 | 0 | 1.1 |
| x4 | AB | 0.701 | 0.558 | 0.574 | 0.643 | 0.619 | 0.748 | 0 | 1.1 |
| x4 | AC | 0.714 | 0.494 | 0.505 | 0.579 | 0.573 | 0.765 | 0 | 1.1 |
| x4 | B | 0.607 | 0.570 | 0.610 | 0.479 | 0.567 | 0.697 | 14.0 | 1.0 |
| x5 | AB | 0.663 | 0.533 | 0.660 | 0.511 | 0.592 | 0.738 | 19.5 | 1.0 |
| x5 | AC | 0.666 | 0.411 | 0.595 | 0.491 | 0.540 | 0.738 | 18.6 | 1.0 |
| x6 | AB | 0.719 | 0.554 | 0.542 | 0.654 | 0.617 | 0.751 | 0 | 1.2 |
| x6 | B | 0.564 | 0.559 | 0.633 | 0.483 | 0.560 | 0.683 | 17.0 | 1.2 |
| x7 | AB | 0.708 | 0.488 | 0.658 | 0.672 | 0.632 | 0.762 | 10.0 | 0.9 |
| x7 | AB `w` | 0.756 | 0.489 | 0.660 | 0.669 | 0.643 | 0.762 | 10.1 | 0.8 |
| x7 | AC | 0.725 | 0.570 | 0.640 | 0.723 | **0.665** | 0.782 | **0 (see §5)** | 0.9 |
| x7 | AC `w` | 0.728 | 0.489 | 0.647 | 0.671 | 0.634 | 0.782 | 0 | 0.7 |
| x8 | AB | 0.705 | 0.567 | 0.610 | 0.651 | 0.633 | 0.752 | 0 | 0.9 |
| x8 | AC | 0.719 | 0.521 | 0.658 | 0.736 | 0.658 | 0.775 | 0 | 0.9 |
| x8 | B | 0.599 | 0.581 | 0.569 | 0.518 | 0.567 | 0.694 | 14.0 | 1.0 |
| x9 | AB | 0.707 | 0.488 | 0.652 | 0.647 | 0.624 | 0.771 | 12.2 | 0.8 |
| x10 | AB | 0.733 | 0.439 | 0.656 | 0.633 | 0.615 | 0.763 | 9.6 | 0.6 |
| x10 | AC | 0.706 | 0.560 | 0.664 | 0.693 | **0.656** | **0.782** | 10.9 | 0.6 |
| x10 | AC (10 starts) | 0.706 | 0.565 | 0.664 | 0.693 | 0.657 | 0.782 | 10.9 | 0.5 |
| x10 | B | 0.521 | 0.581 | 0.567 | 0.449 | 0.529 | 0.697 | 14.1 | 0.6 |
| x11 | AB | 0.719 | 0.477 | 0.645 | 0.698 | 0.635 | 0.752 | 11.8 | 0.5 |
| **x11** | **AC** | 0.729 | 0.568 | 0.641 | 0.685 | **0.656** | **0.782** | **11.8** | **0.5** |

s/episode = one 4,000-tick rollout in the lab (single thread, machine shared with 6-10 fits, so
upper bounds; the idle figure for v8g was 0.6-0.9 s).

What moved the score:

- **Erlang-2 latent period: +0.04 (AB) and +0.08 (AC) LOO, +0.007/+0.019 in-sample**, same 22
  parameters. The gain is almost all in the p3 fold (0.47 -> 0.66): p3 is the run with eight
  switches, and each switch now has the observed two-tick dead time. It also lifts the p1 fold.
- The p3 fold also rose with closure-as-relocation (x4: 0.574) and bed-pressure caution (x5:
  0.660), but both lost the p4 fold, and every home/school/community model (x, x2, x3, x4, x6,
  x8) eliminates the epidemic on some long holds (§5), so none of them is kept.
- Pair C (postponed gatherings) now beats pair A+B by 0.02-0.03 LOO once the latent period is
  right: it is the only term that speeds the rebound after the long p2 restriction (p2 fold
  0.56-0.57 vs 0.44-0.49 for AB). The fitted debt is small ($k_D \approx 0.07$, $\tau_D$ 45-52).
- Age-specific stays (x2) gave +0.013 in-sample on AB but nothing out of sample.
- Fold scores move by up to 0.08 between start sets (x7 AC 0.665 vs `w` 0.634, all of it in the
  p2 fold), so differences below ~0.02 LOO are noise.

Error budget, in-sample loss $\sum(1 - 1/(1+|e|/\sigma))$ (cases/beds):

| segment | v8g gAB | v8g AB (1.0σ) | x11 AC |
|---|---|---|---|
| p4 ticks 25-300 | 61/77 | 61/73 | 58/77 |
| p2 restriction 25-120 | 23/26 | 22/26 | 22/16 |
| p2 release 120-400 | 78/49 | 77/42 | 67/38 |
| p3 ticks 25-291 | 74/73 | 70/73 | 85/43 |
| p1 ticks 25-120 | 24/15 | 23/15 | 13/17 |
| first 25 ticks, all runs | 32/29 | 32/29 | 30/27 |
| **total** | **561** | **545** | **494** |

Still open: p4 (the 42 -> 60 climb is fitted by fatigue, which then overshoots, 135 loss) and
the p2 release rebound (truth +4.6/tick from tick 122, x11 +2.4).

## 5. Long holds (4,000 ticks from (150, 50))

Constant holds, mean of the last 500 ticks (cases / beds):

| hold | v8g gAB | x11 AC | x10 AC |
|---|---|---|---|
| none | 104.6 / 78.4 | 104.7 / 79.6 | 105.0 / 79.5 |
| pulse (1, 1, .003) | 77.6 / 56.2 | 60.9 / 42.6 | 64.3 / 45.1 |
| vaccination .003 | 84.9 / 61.0 | 75.3 / 52.6 | 76.1 / 53.2 |
| closure 1 | 102.3 / 79.4 | 104.9 / 82.2 | 105.4 / 82.1 |
| mask 1 | 83.8 / 61.9 | 85.8 / 64.4 | 86.6 / 64.7 |
| closure + mask | 99.7 / 75.2 | 97.2 / 74.5 | 98.9 / 75.3 |
| (.85, .85, .00255) | 76.0 / 55.6 | 67.6 / 48.3 | 69.9 / 50.0 |

All settle on an endemic level with no oscillation left; none runs away (max cases ~505, the
first unrestricted wave; beds stay at the 155 capacity). The p4 truth at tick 300 is 60.5 / 39.9
under (.85, .85, .00255); x11 AC gives 67.6 / 48.3 at the end of a long hold and 68.6 / 47.0 at
tick 300, v8g gAB 76.0 / 55.6.

Stress schedules (min cases over the episode / cases at tick 4,000):

| schedule | v8g gAB | x7 AC | x10 AC | **x11 AC** |
|---|---|---|---|---|
| vaccination .003 for 2,000, then pulse 2,000 | 0.0 / 0.8 | 0.0 / 0.0 | 3.6 / 64.3 | 5.6 / 60.9 |
| (.04, .02, .0024) for 2,900, then (.88, .92, .0026) | 10.8 / 76.5 | 0.0 / 0.0 | 10.9 / 69.7 | 11.8 / 67.2 |
| pulse 4,000 | 33.1 / 77.6 | 33.6 / 66.7 | 33.3 / 64.3 | 33.0 / 60.9 |
| 40 test-shaped schedules: share of ticks with cases < 5 | 1.0 % | 3.8 % | 0.0 % | 0.0 % |

x7 AC eliminates because `vac_eff` 7.0 times clinic availability $1/(1 + 15H/h_{cap})$ is fitted
where beds never fall below ~20; when beds empty, the clinic term goes to 1 and 2 % of
susceptibles are vaccinated per tick, faster than waning (1.1 %/tick) refills them, so a later
restriction pushes $R_{eff}$ far below 1 and the model never recovers. The doses are at most
0.3 % of the population per tick, so x11 bounds `vac_eff` $\le 3$ and `k_clinic` $\le 5$: the
fit lands at 2.0 and 4.9, the same LOO and in-sample, and no elimination. x10 AC gets there
without bounds (3.1 and 8.2).

Known structural quirk (all fatigue models): fatigue is driven by $(u_{clo} + u_{mask})/2$, so
adding closure (which does almost nothing) to a mask hold speeds fatigue and ends higher than
the mask alone (97 vs 86 cases). No data cover a long mask-only hold, so we cannot test it.

## 6. Decision

- **Recommend `epidemic_x11`, pair AC: `plans/epidemic_epidemic_x11_AC_doc.json`.** LOO 0.656
  (+0.063 over v8g AB, +0.080 over v8g AC, same protocol), in-sample 0.782 (+0.027), 22
  parameters, 0.5 s/episode, no elimination on any schedule we tried. Only `school` sits at its
  physical bound (closure removes all child-child school contacts). Fitted: $N$ 17,100,
  $\beta$ 0.651, $\sigma$ 0.116, $\gamma$ = (1.01, 1.46, 0.41), sev = (0, 0.091, 0.127), los 9.6,
  $h_{cap}$ 154.5, mask 0.41, $k_{clinic}$ 4.9, $v_{eld}$ 4.8, $r_I$ 0.85, $\tau_p$ 5.7,
  $\tau_{fat}$ 136, $k_f$ 0.76, $\tau_{wane}$ 92.7, vac_eff 2.03, $k_D$ 0.076, $\tau_D$ 45.5.
- Backup: `epidemic_x10` AC (`plans/epidemic_epidemic_x10_AC_doc.json`), identical scores; 3
  more states for the Erlang-2 infectious stage, no bounds needed.
- The target (LOO 0.68) was not reached. With v8g's 82 % transfer, +0.027 in-sample points to
  roughly +0.02 to +0.03 public, not +0.08; the LOO gain (+0.06) suggests transfer to held-out
  schedule shapes is better than in-sample shows.
- Rejected: the home/school/community split (x, x2, x3, x4, x6, x8: elimination on long holds,
  and its extra parameters did not beat v8g out of sample), bed-pressure caution in place of
  fatigue (x5: fits p3, loses p4), pair B alone and BC (lose p1), x7 AC (elimination).

## 7. What data would help most (1,189 credits left on epidemic)

1. **A long mask-only hold (mask 0.85-1.0, 300-400 ticks) from a fresh start.** It separates
   fatigue from the endemic climb in p4, pins how masks hold up over time (the sustained
   category's biggest unknown) and tests the closure-plus-mask quirk above. About 350 credits.
2. **A second pulse-release at a different level and length (e.g. (0.75, 0.75, .0022) for 60
   ticks, then 200 free).** The p2 fold is the weakest (0.56); only p2 shows a long
   restriction followed by release. It pins the debt (C) against the latent/fatigue terms and
   covers the recovery category's alpha range. About 260 credits.
3. **Vaccination alone from low bed occupancy** (e.g. mask 1 until beds < 20, then vaccination
   .003 for 200 ticks) to measure clinic availability at low pressure, which decides whether
   long vaccination holds can drive cases toward zero. About 300 credits.
