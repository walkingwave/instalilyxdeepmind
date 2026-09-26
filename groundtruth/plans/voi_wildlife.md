# wildlife: value of information for the next purchase and the next model change

Date: 2026-09-26. Script: `scripts/voi_wildlife.py` (free, about 2 min). Numbers: `plans/voi_wildlife.json`.
Data: the same three runs (`p1.hold_rec` 120, `p2.pulse200_200` 400, `p3.compose` 291). Credits left: 1,189.
No credits spent.

## 1. Method

1. **Committee.** Every wildlife doc in `plans/` was re-rolled on the three ledger runs and compared with
   the `insample` block of its lab json (tolerance 1e-3). Reproducing docs: v7, AB, AB r2, AC, AC r2, BC
   (and min2_v6, a numerical copy of v7, dropped). Not reproducing: `wildlife_min` v1-v5, `min`, `min_p3`
   (their theta no longer matches the current `wildlife_min` module). Plus `l0b_lin` (no squares, no
   pairs, clip margin 1) fitted on all three runs. 7 members.
2. **Test distribution.** 40 schedules from `design.eval_like`, 10 per category, $T = 4000$, seed 20260926,
   $y_0$ cycling over our three observed initials. Every member rolled on every schedule.
3. **Score matrix.** $S_c[i,j]$ = mean score of member $i$ used as the prediction when member $j$ is the
   truth, over the category-$c$ schedules: $\frac{1}{T p}\sum_{t,k} 1/(1+|y^i_{tk}-y^j_{tk}|/\sigma_k)$.
4. **Anchor calibration (new).** Six wildlife models we uploaded have a public score (persistence 0.2227,
   l0b "relax" 0.2785, l1 u003 0.4292, l0b u004 0.5550, ode u008 0.6297, ode v7 u010b 0.6355). We rolled
   their shipped `predict.py` on the same 40 schedules. For each member taken as truth and each sigma
   multiplier $k$ ($\sigma = k\,\sigma_{proxy}$, $k \in \{.05,.1,.15,.2,.3,.5,1\}$) we compared the implied
   anchor scores with the public ones (RMSE over 6 anchors). Truth weights
   $w_j \propto \exp(-\tfrac12(\mathrm{RMSE}_j/0.03)^2)$ at each member's best $k$; one global $k^*$
   minimises the $w$-weighted RMSE.
5. **Control regimes.** Every test tick is labelled by its control vector against the six vectors our runs
   contain (recovery; pulse (7, .1, 1); (5.95, 1, 0); (0, .235, 0); (0, 1, .85); (5.95, .235, .85)):
   recovery, pulse-like (all three controls at $\alpha \ge 0.7$ of the way from recovery to pulse),
   a seen level, interior on the recovery-pulse line, interior off the line, corner/edge (a control at a
   bound, unseen combination); and by time since the last control change (<45, 45-200, >200 ticks; our
   longest hold is 200).
6. **Experiment value.** For each candidate schedule $E$ (from the mean observed $y_0$) we rolled every
   member. Residual misfit of the best members on our data gives an effective noise
   $s_{eff} = (10.6, 0.22, 8.5, 0.25)$; residuals are serially correlated, so the log likelihood is divided
   by $\kappa = 50$ ticks. Pair separation $\mathrm{sep}_{ij} = 1 - \exp(-\Delta^2_{ij}/8)$ with
   $\Delta^2_{ij} = \kappa^{-1}\sum_t \|(y^i_t - y^j_t)/s_{eff}\|^2$ (Bhattacharyya bound for two Gaussians).
   Value $V_E = \sum_{i<j} p_i p_j\, \mathrm{stake}_{ij}\, \mathrm{sep}_{ij}$, stake
   $= \tfrac12[(S_{ii}-S_{ji}) + (S_{jj}-S_{ij})]$ at the calibrated sigma. Also a Monte Carlo EVSI
   (members as truth hypotheses, decision = best predictor under the posterior among members and
   combinations) and a heuristic (member disagreement on $E$ weighted by the test frequency x test
   disagreement of the regime of each tick). Ranking by value per 100 credits.

## 2. Main finding: the 0.84 vs 0.64 gap is the sigma, not transfer

| | local score, $\sigma_{proxy}$ | local score, $0.15\,\sigma_{proxy}$ | public |
|---|---|---|---|
| v7 in-sample (3 runs) | 0.897 | **0.635** | **0.6355** |
| v7 LOO | 0.843 | | |
| AB r2 in-sample | 0.899 | 0.638 | |
| l0b_lin in-sample | 0.835 | 0.513 | 0.555 (older 2-run fit) |

- Anchor calibration puts the organizer's sigma at $k^* = 0.15\,\sigma_{proxy}$, i.e. about
  $\sigma \approx (8.1, 0.16, 6.7, 0.22)$ for (prey_north, predator_north, prey_south, predator_south).
  Independently, v7's own in-sample score at $k = 0.15$ is 0.635, the public number to three decimals.
- So v7 does *not* transfer poorly: it scores on the test what it scores on our own data once the
  sigma is right. Our `sigma_proxy` (the std of all observations) is about 7x too wide, which makes every
  local number look ~0.25 better than the board. **The lever is fit precision in the regimes the test
  spends its ticks in, at a scale of about 8 prey and 0.2 predators.**
- At this scale predators matter as much as prey: v7 in-sample per observable is 0.617 / 0.614 / 0.669 /
  0.639. The committee cannot see predator error at all: every member shares the same predator
  equation, so member disagreement on predators is 0.003-0.03 $\sigma_{proxy}$.

## 3. Committee disagreement on the test distribution

Off-diagonal mean of the score matrix (member vs member) and pairwise disagreement by observable
(mean $|y^i - y^j|/\sigma_{proxy}$):

| category | off-diag score at $\sigma_{proxy}$ | at $0.15\,\sigma_{proxy}$ | v7 if AB r2 were truth (0.15) | prey_N | pred_N | prey_S | pred_S |
|---|---|---|---|---|---|---|---|
| sustained | 0.935 | 0.772 | 0.862 | 0.167 | 0.016 | 0.126 | 0.014 |
| order | 0.934 | 0.764 | 0.837 | 0.158 | 0.024 | 0.131 | 0.018 |
| recovery | 0.938 | 0.788 | 0.901 | 0.167 | 0.003 | 0.139 | 0.010 |
| composition | 0.920 | 0.747 | 0.844 | 0.221 | 0.015 | 0.189 | 0.015 |

Composition and order carry the most disagreement (they visit unseen combinations); recovery the least.

By control regime (share of test ticks; mean pairwise disagreement, all pairs and v7 vs AB r2, in
$\sigma_{proxy}$; share of the v7-vs-AB r2 loss):

| regime | dwell | freq | d all | d v7-ABr2 | loss share |
|---|---|---|---|---|---|
| recovery | >200 | 0.321 | 0.071 | 0.014 | 0.121 |
| recovery | 45-200 | 0.134 | 0.074 | 0.023 | 0.083 |
| pulse ($\alpha \ge .7$) | >200 | 0.111 | 0.052 | 0.043 | 0.126 |
| interior on rec-pulse line | >200 | 0.102 | 0.082 | 0.041 | 0.109 |
| corner / edge | >200 | 0.073 | 0.143 | 0.081 | 0.142 |
| corner / edge | 45-200 | 0.060 | 0.152 | 0.069 | 0.101 |
| recovery (just after a release) | <45 | 0.052 | 0.153 | 0.101 | 0.115 |
| interior off line | >200 | 0.041 | 0.114 | 0.051 | 0.051 |
| pulse | 45-200 / <45 | 0.058 | 0.06 | 0.05 | 0.070 |
| everything else | | 0.048 | 0.10-0.14 | | 0.07 |

What the test contains that our runs do not:
- **65 % of test ticks sit more than 200 ticks into a hold**; our longest hold is 200. All members are
  settled there (last-quarter range of any member on any 800+ tick hold < 0.004 $\sigma_{proxy}$: nothing
  oscillates), so errors there are pure equilibrium-level errors.
- **46 % of test ticks use a control vector more than 5 % of range away from anything we observed**:
  interior levels on the recovery-pulse line (alpha 0.7-1 pulses and uniform draws), corners such as
  (8, 0, 1), interior off-line mixes. Intermediate habitat and corridor levels are here.
- **Recovery is 51 % of test ticks.** Its equilibrium level is therefore the single most valuable number.
  v7 settles at (120.6, 2.45, 100.5, 2.45); our long recovery segments end at 121.6-122.1 / 2.35-2.45 /
  96.5-98.7 / 2.30-2.45. prey_south is 3-4 high (0.5 $\sigma$) and predators 0.05-0.1 high (0.3-0.6
  $\sigma$) on half the test.
- The largest per-tick disagreement is in corners and in the first 45 ticks after a release (rebound
  height), but these are only 13 % of ticks together.

## 4. Candidate experiments (simulated under every member)

$V_E$ = pair value (uniform prior) in units of expected score; $V_E$/100 = per 100 credits. EVSI columns
are the Monte Carlo version (uniform / anchor-weighted prior). sep = pair separation.

| rank | experiment | credits | $V_E$ | $V_E$/100 | EVSI/100 unif | sep v7-ABr2 | sep AB-AC | sep AC-ACr2 |
|---|---|---|---|---|---|---|---|---|
| 1 | E02 corridor only: corr 0.5 60, off 30, corr 1 60, off 30 | 180 | 0.0912 | 0.0506 | 0.084 | 0.89 | 0.94 | 0.59 |
| 2 | E01 habitat 0.5 alone 150, recovery 50 | 200 | 0.0908 | 0.0454 | 0.074 | 0.33 | 0.99 | 0.51 |
| 3 | E03 brief: quota 6 at habitat 0.1 / habitat 1 / habitat 0.1 / quota 6 at habitat 1 | 230 | 0.0927 | 0.0403 | 0.067 | 0.51 | 1.00 | 0.49 |
| 4 | E04 brief: habitat 0.1 -> 1 with corridor closed, then open | 240 | **0.0964** | 0.0402 | 0.066 | 0.89 | 1.00 | 0.64 |
| 5 | E05 six full pulses of 10 ticks, gaps 10 | 200 | 0.0799 | 0.0400 | 0.074 | 0.41 | 0.78 | 0.52 |
| 6 | E14 habitat 0.5 100, then 3 short pulses | 230 | 0.0917 | 0.0399 | 0.065 | | | |
| 7 | E07 interior alpha 0.5 hold (3.5, .55, .5) 150 | 200 | 0.0796 | 0.0398 | 0.072 | | | |
| 8 | E09 corner (8, 0, 1) 100, recovery 100 | 200 | 0.0781 | 0.0390 | 0.068 | | | |
| 9 | E10 quota 3.5 alone 150 | 200 | 0.0732 | 0.0366 | 0.066 | | | |
| 10 | E08 alpha 0.7 pulse train | 260 | 0.0878 | 0.0338 | 0.059 | | | |
| 11 | E06 short pulses, gaps 40 | 260 | 0.0869 | 0.0334 | 0.059 | | | |
| 12 | E15 = E02 then habitat 0.5 100 + recovery 20 | 300 | 0.0954 | 0.0318 | 0.051 | **0.93** | 1.00 | **0.69** |
| 13 | E16 pulse 60, recovery 240 | 300 | 0.0814 | 0.0271 | 0.046 | 0.29 | | |
| 14 | E11 full pulse 250 | 300 | 0.0730 | 0.0244 | 0.024 | | | |

Reading: any 180-240 tick design already separates most member pairs, so value per credit mostly
rewards the shortest design that hits the pair that matters. The pair that matters for the next upload
is v7 vs the AB family (the two tied leaders); only designs with a **corridor block** separate it
(E02 0.89, E04 0.89, E15 0.93; habitat or pulse designs 0.3-0.5). The reason: after a corridor closes
the AB/AC/BC family releases its transit pool (end of E02: v7 prey_north 121, AB r2 143, AC 137, BC 142),
and v7 does not. Compose showed a 20-head rise after the corridor closed, so this is testable and real.

## 5. Free model options

Scored against every member as truth on the 40 test schedules at the calibrated sigma, and in-sample at
the calibrated sigma (column "ins"):

| option | uniform over truths | worst truth | ins (0.15) |
|---|---|---|---|
| median of all 7 | 0.842 | 0.616 | |
| AB r2 | 0.840 | 0.596 | 0.638 |
| median(v7, AB, AB r2) | 0.840 | 0.597 | |
| mean(v7, AB r2) | 0.826 | 0.598 | |
| v7 (public) | 0.816 | 0.598 | 0.635 |
| mean(v7, AB r2, l0b_lin) | 0.755 | 0.666 | |

Anchor-weighted numbers favour AC / AC r2 / BC (0.89-0.90), but that is an artefact: the anchors say the
truth is farther from v7 than any member (even the farthest member implies v7 = 0.79 vs 0.636 public),
and the members farthest from v7 fit that best. We do not use those weights for decisions.

Members do not oscillate (section 3), so averaging does not damp cycles here; mean/median behave as
level blends. Ranked free changes:

1. **Recovery equilibrium re-level** (prey_south about -3.3, predators about -0.05 once a recovery hold
   is older than ~60 ticks; better: refit with the late-segment recovery levels weighted up). In-sample at
   the calibrated sigma 0.6347 -> 0.6413 (+0.007) on 200 of 811 ticks; on the test recovery is 51 % of
   ticks, so we expect more there. Every member shares this bias.
2. **Pulse-level prey_north from AB r2** (8.4 vs v7 16.2, observed 7.0): +0.009 more in-sample
   (0.6413 -> 0.6503). Pulse-like holds are 17 % of test ticks. Equivalent: switch prey to AB r2 and keep
   the equilibrium fix (AB r2 + eq 0.6478).
3. **Fit and select at the calibrated sigma** ($0.15\,\sigma_{proxy}$) instead of $\sigma_{proxy}$. The
   Cauchy fit and LOO currently weight residuals at a scale 7x wider than the board; predator transients
   (0.49-0.56 on `hold_rec` at the board scale) and the 3-4 head equilibrium offsets are invisible at
   $\sigma_{proxy}$. A refit of v7 / AB r2 with residuals in units of $0.15\,\sigma_{proxy}$ targets exactly
   what the board scores. The committee cannot value this (all members share it).

## 6. Recommended plan

- **Free first**: changes 1-3 on AB r2 or v7 (or a per-observable pick: prey from AB r2, predators refit
  at the board sigma), judged by in-sample and LOO at $0.15\,\sigma_{proxy}$.
- **Buy E15, 300 credits** (corridor 0.5 for 60, off 30, corridor 1 for 60, off 30, habitat 0.5 for 100,
  recovery 20). It has the highest separation of the v7 vs AB r2 pair (0.93) and of AC vs AC r2 (0.69),
  it is the only design that measures both unseen intermediate levels (corridor 0.5, habitat 0.5), it
  tests the transit-pool rebound after a corridor closes, and it adds two more recovery settles for the
  equilibrium level. Balance after: 889.
- **Cheaper alternative: E02, 180 credits** (corridor part only), best value per credit (0.051/100).
- The brief's own tests (E03, E04) are worth the same in total but less per credit; E04 is the one to add
  later if the corridor result leaves C open.

Schedules (U as lists, controls [hunting_quota, habitat_protection, corridor_access]) are in
`plans/voi_wildlife.json` under `recommended` and `experiments.<id>.U`.

## 7. Caveats

- The truth is outside the committee, so separation and EVSI measure model identification, not the full
  gain; the calibration (section 2) says fit precision at the board sigma matters more than which pair.
- $k^* = 0.15$ rests on 6 anchors and 40 simulated schedules from our three initials; the hidden test
  initials and schedules may differ.
- Free-fix gains are in-sample on 811 ticks; they need LOO at the board sigma before an upload.
- $s_{eff}$ and $\kappa = 50$ set how easily pairs separate; with smaller $\kappa$ every design separates
  everything and the ranking reduces to length.
