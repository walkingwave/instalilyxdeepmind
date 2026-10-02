# social_contagion: value of information (2026-09-26)

Free study, no credits. Script: `scripts/voi_social_contagion.py`, numbers: `plans/voi_social_contagion.json`.
Data: 3 runs, 811 ticks (hold_rec 120, pulse200_200 400, compose 291). $\sigma_{proxy}$ = (59.6, 32.4),
first-difference noise = (0.51, 0.30). Credits left 1,189. Public now: 0.553 (sustained 0.418, sequence 0.597).

## 1. Committee and validation

We rolled every `plans/social_contagion_*_doc.json` through `rollout_from_blob` on every ledger run and
compared per-observable scores with the lab json's `insample`.

| doc | mech | max abs diff vs lab | kept |
|---|---|---:|---|
| BC, BC2 | B, C ($k_C = 0$) | 0.000 | yes |
| AB, AB2 | A, B | 0.000 | yes |
| AC | A, C | 0.000 | yes (weak: in-sample 0.79) |
| min_v2 | v2 base | 0.000 | yes |
| min, min_p3, min_v3 | v1/v3 base | 0.39 / 0.81 / 0.15 | **dropped** (module changed since) |

Added: `l0b_lin` (all runs, clip margin 1), `public` (u010b model.json = median(min_v2, l0b_lin)), and a probe
`BC2_kC` = BC2 with $k_C$ raised to the largest value that keeps in-sample mean within 0.01 of BC2
(scan: $k_C$ 0.005 → 0.886, 0.01 → 0.877, 0.02 → 0.851, 0.1 → 0.756). So $k_C \le 0.005$: the data already
rules out any sizable bridge word-of-mouth term.

Disagreement set ("top"): BC2, AB2, min_v2, l0b_lin (distinct structures).

## 2. Disagreement on the test distribution

40 `eval_like` schedules (10 per category, $T = 4000$, seeds `1000·cat + k`), $y_0$ cycling the three observed
initials. Per tick $D(t) = \text{mean}_{pairs,obs} |Y_i - Y_j|/\sigma$; score gap $= 1 - \text{mean}\,1/(1+D)$.

| category | mean $D$ | adopters_a | adopters_b | score gap | A vs C (AB2–BC2) | public vs BC2 |
|---|---:|---:|---:|---:|---:|---:|
| sustained | 0.79 | 0.63 | 0.94 | 0.41 | 0.064 | **0.714** |
| order | 0.69 | 0.52 | 0.85 | 0.37 | 0.091 | 0.550 |
| recovery | 0.78 | 0.56 | 1.00 | 0.41 | 0.095 | 0.214 |
| composition | 0.74 | 0.55 | 0.94 | 0.39 | 0.092 | 0.399 |

By regime (share of all test disagreement mass):

| regime | tick share | mass share | mean $D$ |
|---|---:|---:|---:|
| seed+inc+bridge, interior levels | 0.355 | 0.304 | 0.64 |
| recovery after an interior joint hold | 0.224 | 0.231 | 0.77 |
| recovery at the start | 0.114 | 0.118 | 0.78 |
| recovery after seeding alone | 0.050 | 0.077 | 1.15 |
| seed+bridge | 0.024 | 0.048 | 1.48 |
| seeding alone (corner) | 0.068 | 0.047 | 0.52 |
| bridge alone, interior | 0.025 | 0.044 | 1.31 |

By dwell (ticks since the last control change), sustained: <50: 0.53, 50–200: 0.57, 200–1000: 0.73, ≥1000: 0.86.
Disagreement grows with hold length; sustained has 55% of its ticks at dwell ≥ 1000.

**Where the disagreement is.** Almost all of it is between the two old structures (min_v2, l0b_lin) and the
B-family ODEs. The two B-family ODEs agree with each other to 0.06–0.09 σ on every category: the A-vs-C question
is worth at most a few thousandths of score on the test.

**Why sustained is 0.42.** Taking each committee member as the truth and scoring the public model against it:

| truth | sustained | order | recovery | composition |
|---|---:|---:|---:|---:|
| BC2 | **0.627** | 0.686 | 0.830 | 0.746 |
| AB2 | **0.630** | 0.680 | 0.868 | 0.759 |
| min_v2 or l0b_lin | 0.697 | 0.720 | **0.609** | 0.652 |

The real public bands (sustained worst, recovery-heavy sequence better) match the pattern we get when a B-family
ODE is the truth, and contradict the one we get when min_v2/l0b_lin is the truth. Reading: the public model is
wrong mainly in long interior holds. Half of it is `l0b_lin`, whose equilibrium is linear in the controls; the
B-family relaxes to a saturating equilibrium ($r = \rho/(1+\rho/1.5)$, pool depletion), and the gap grows with
dwell. min_v2 lies between them. Every observed run sits at recovery, pulse or compose levels: no interior joint
hold has ever been observed, yet interior joint holds are 36% of test ticks.

## 3. Candidate experiments

Each schedule rolled under every member from the median observed $y_0$. Value
$V = \sum_t D(t)\, w(r_t)$, with $w(r)$ = share of test disagreement mass in regime $r$ (table above). Cost = ticks.

| # | experiment | credits | mean $D$ (top) | A vs C | C on | public vs BC2 | value | value/credit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E12 | three 100-tick interior holds: [5, 1, 0.5], [4.84, 0.71, 0.59], [6.94, 1.88, 0.58] | 300 | 0.58 | 0.063 | 0.063 | 0.71 | 52.6 | **0.175** |
| E10 | seed 5 + inc 1 + bridge 0.5 for 200, stop 100 | 300 | 0.46 | 0.061 | 0.056 | 0.57 | 40.8 | 0.136 |
| E01 | 0.5·pulse [4.5, 1, 0.3] 150, stop 50 | 200 | 0.36 | 0.057 | 0.033 | 0.44 | 21.1 | 0.105 |
| E11 | eval-style pulse train, 300 | 300 | 0.25 | 0.110 | 0.010 | 0.19 | 17.5 | 0.058 |
| E08 | seed 9 + bridge 1.0 150, stop 50 (local vs bridge) | 200 | 1.19 | 0.065 | 0.067 | 0.67 | 7.7 | 0.039 |
| E07 | seed 9 alone 200, stop 100 (queue build, A) | 300 | 0.57 | 0.073 | 0 | 0.29 | 11.2 | 0.037 |
| E04 | inc 2 then seed 9 + inc 2, stop | 250 | 0.54 | 0.051 | 0 | 0.46 | 3.4 | 0.014 |
| E05 | seed 9 then seed 9 + inc 2, stop | 250 | 0.35 | 0.060 | 0 | 0.36 | 2.8 | 0.011 |
| E13 | seed 9 + inc 2 200, stop 100 | 300 | 0.46 | 0.056 | 0 | 0.43 | 3.0 | 0.010 |
| E06 | bridge 1.0 alone 150, stop 50 | 200 | 0.54 | 0.096 | **0.213** | 0.50 | 0.8 | 0.004 |
| E09 | seed 9 + inc 2 100, inc 0.8 100, stop 100 | 300 | 0.68 | **0.184** | 0 | 0.64 | 0.4 | 0.001 |
| E02 | seeding 5 alone 200, stop 50 | 250 | 0.57 | 0.061 | 0 | 0.14 | 0 | 0 |
| E03 | incentive 0.8 alone 150, stop 50 | 200 | 0.48 | 0.090 | 0 | 0.17 | 0 | 0 |

(E02/E03 get zero weight because single interior controls never occur alone in `eval_like` schedules.)
All differences are far above observation noise (every row ≥ 28 noise units), so any of these runs resolves the
disagreement it carries. The best discriminators for mechanisms are E09 (A vs C, 0.18 σ, max 0.45) and E06
(bridge on, 0.21 σ), but both sit in regimes the test hardly visits.

## 4. Free model options (predictor scored against each member as truth, $\sigma_{proxy}$)

| predictor | mean over BC2/AB2 truths | worst over truths | sustained | recovery |
|---|---:|---:|---:|---:|
| **median(BC2, AB2, l0b_lin)** | **0.962** | 0.580 | 0.971 | 0.958 |
| BC2 or AB2 alone | 0.962 | 0.56 | 0.971 | 0.958 |
| median(BC2, AB2, min_v2, l0b_lin) (hedge) | 0.906 | 0.592 | 0.856 | 0.953 |
| per-obs (a: BC2, b: public) | 0.826 | 0.608 | 0.775 | 0.888 |
| public = median(min_v2, l0b_lin) | 0.728 | 0.669 | 0.628 | 0.849 |

Ranked free changes:
1. **Swap the public ensemble to median(BC2, AB2, l0b_lin).** Same expected score as BC2 alone under either
   B-family truth, and the l0b_lin member caps the damage if both ODEs drift the same way in a long hold
   (worst case 0.580 vs 0.562 for BC2 alone). LOO already favours BC (0.828 vs 0.698 for min_v2). Needs one
   upload slot, no credits; the band shift on sustained is the test.
2. If a slot is too dear to risk: the 4-member hedge median(BC2, AB2, min_v2, l0b_lin), +0.18 over public under
   the ODE truths with the best worst case after public.
3. Per-observable picks do not help: BC2 and AB2 agree on both observables; adopters_b carries 60% of the
   disagreement but no member is better there alone. Drop min, min_p3, min_v3 (no longer reproduce) and AC.

## 5. Recommended plan

1. Free first: upload median(BC2, AB2, l0b_lin). If sustained jumps (we expect it toward the other bands), the
   B-family structure is confirmed on the test distribution without spending.
2. Buy **E12 (300 credits)**: three 100-tick interior joint holds. It is the only design that observes the regime
   holding 36% of test ticks (plus the recoveries that follow it, another 22%), which no run has touched.
   It checks the one thing all ODE members share and the public score cannot isolate: the saturating interior
   equilibrium. Cheaper alternative: **E01 (200)**, one interior hold plus a stop.
3. Skip mechanism tests (E06, E09, E07) unless the upload surprises us: A vs C moves the test prediction by
   < 0.1 σ, and the $k_C$ scan already bounds bridge word-of-mouth near zero.

Balance after E12: 889.

## Caveats
- Value is measured in $\sigma_{proxy}$; the organizer's σ is smaller (public 0.553 vs 0.728 here), so real gaps are larger, the ranking should hold.
- The committee cannot value what every member assumes (e.g. the incentive split $\phi = c/(c+0.7)$, bridge at 1.0); E12 tests the shared interior structure, not those forms.
- Test $y_0$ = our three observed initials; the hidden initials may differ.
- `BC2_kC` is a probe (no refit), not a fitted C-active model.
- Regime weights come from our `eval_like` copy of the test generator, not the organizer's.
