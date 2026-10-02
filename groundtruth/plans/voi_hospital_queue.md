# hospital_queue: value of information (Sat Sep 26)

Script: `scripts/voi_hospital_queue.py` (free, ~140 s). Numbers and schedules: `plans/voi_hospital_queue.json`.
Data: 5 runs, 770 ticks. Credits left 1,230. No credits spent for this study.

## Method

**Committee.** p3 (current public, family `hospital_queue_p3`, no cohort), mech AB / AC / BC and min2 v4
(all four carry the stranded-cohort wait), plus `l0b_lin` refitted on all runs (clip 1x).
We rolled every doc on every ledger run and compared per-observable scores with the `insample`
block of its lab json (p3 against `hospital_queue_min_p3.json`, same theta). All five reproduce:
max deviation 0.000 / 0.002 / 0.002 / 0.003 / 0.003 (tolerance 0.01). None dropped.

**Test distribution.** 40 schedules from `design.eval_like`, 10 per category, $T=4000$, seeds
$[2026, c, k]$, $y_0$ cycled over the 5 observed initials. $\sigma$ = `sigma_proxy` =
[100.0, 120.1, 5.27] (wait, queue, discharges).
For members $i,j$: disagreement $|\hat y_i-\hat y_j|/\sigma$ and score loss
$\ell_{ij}=1-1/(1+|\hat y_i-\hat y_j|/\sigma)$, averaged over ticks, episodes, pairs.

**Regimes.** Each tick gets a label: staffing lo/mid/hi, overtime on / post (<=40 after) / no, follow-up
lo/hi, elective lo/hi, diagnostic lo/mid/hi; ticks at the exact recovery action are labelled by the
time since the last non-recovery tick (<50, 50-200, >=200, from reset). Test disagreement mass of a
regime $r$: $M_r = f_r\,\bar\ell_r$ (frequency x mean ODE-pair loss). $\sum_r M_r = 0.0866$.

**Experiment value.** Each candidate is rolled under every member from the median observed $y_0$.
Effective noise $\varepsilon$ = max(measurement noise, median in-sample RMS residual of the ODE members)
= [33.3, 19.6, 3.5]. Per regime touched by the experiment:
$z_r^2 = \frac{1}{\tau}\sum_{t\in r}\overline{\max_{obs}(\Delta_{ij,t}/\varepsilon)^2}$ with $\tau=10$ ticks
(residual autocorrelation), $P_r = 1-e^{-z_r^2/8}$ (z = 2 gives 0.39, z = 4 gives 0.86).
Value $V=\sum_r M_r P_r$, reported as a share of $\sum_r M_r$; ranked per credit and in total.
Cross-check $V_{pair}$: mean over ODE pairs of (test loss of the pair) x (probability the run separates
that pair). It saturates for almost every run (47-100%), so the regime value is the one that ranks.

## Committee disagreement on the test distribution

ODE members only (l0b_lin is an outlier everywhere: expected score 0.63 / 0.65 / 0.80 against the ODE members).

| category | wait (σ units) | queue | discharges | wait loss | queue loss | disch loss |
|---|---:|---:|---:|---:|---:|---:|
| sustained | 0.115 | 0.099 | 0.069 | 0.066 | 0.051 | 0.060 |
| order | 0.101 | 0.035 | 0.056 | 0.061 | 0.030 | 0.049 |
| recovery | **0.712** | 0.092 | 0.041 | **0.251** | 0.077 | 0.036 |
| composition | **0.779** | 0.139 | 0.062 | **0.235** | 0.076 | 0.048 |

Wait_time in recovery and composition is almost all of it. Where:

| regime | test freq | ODE-pair loss | share of mass |
|---|---:|---:|---:|
| recovery action, >=200 ticks after a disturbance | 0.156 | 0.146 | 26.3% |
| recovery action, 50-200 after | 0.116 | 0.137 | 18.4% |
| full pulse (S lo, OT on, FU lo, E hi, D hi) | 0.110 | 0.061 | 7.7% |
| recovery action, <50 after | 0.058 | 0.103 | 6.9% |
| overtime alone at S 20 | 0.023 | 0.185 | 5.0% |
| follow-up low alone at S 20 | 0.021 | 0.187 | 4.6% |

The mechanism: after any congestion ends, the four cohort members predict that once the ordinary
pool drains (queue ~15-25) the reported wait jumps to the cohort age, 300-400, some 150-200 ticks later,
and stays high for hundreds of ticks (AB 409 at t=2000 of a recovery episode; AC/BC/min2 ~350,
decaying to ~200 by t=3000). p3 says wait ~0.3 there. Real data saw this once: p3.compose, wait
78 -> 396 over ticks 190-270 with the queue at 23, cut off by the joint pulse. We have never observed
how long the tail lasts or when it starts after a plain pulse (p2.pulse40 had only 40 ticks of recovery).

Direct evidence on this regime (160 real recovery ticks after a disturbance, per-observable score):

| member | wait | queue | disch |
|---|---:|---:|---:|
| p3 | 0.778 | **0.884** | 0.762 |
| AB | 0.875 | 0.883 | 0.757 |
| AC | 0.870 | 0.881 | 0.757 |
| BC | 0.870 | 0.855 | 0.762 |
| min2v4 | 0.857 | 0.848 | 0.740 |
| median of 5 ODE | **0.876** | 0.878 | 0.761 |

## Candidate experiments (ranked by resolvable share of test disagreement)

Controls order: staffing, elective, diagnostic, urgent, overtime, follow-up. Recovery = (20, 0, 0.4, 0.6, 0, 1),
pulse = (5, 20, 0.75, 1, 1, 0).

| id | schedule | credits | V share | V per 100 cr | wait sep (mean / max, ε) | separates |
|---|---|---:|---:|---:|---|---|
| **E16** | pulse 40, recovery 260 | 300 | **50.1%** | 0.0144 | 2.07 / 12.2 | p3 vs all (1.3-1.9), AB vs min2 1.35 |
| E17 | S5 + elective 20 (no OT) 40, recovery 260 | 300 | 46.1% | 0.0133 | 2.12 / 11.7 | p3 vs all (1.5-1.7); cohort members close |
| E13 | pulse 40, recovery 200 | 240 | 34.8% | 0.0126 | 1.39 / 11.0 | onset only |
| E14 | S5 + E20 40, recovery 200 | 240 | 31.3% | 0.0113 | 1.45 / 11.2 | onset only |
| E15 | pulse 30, rec 60, pulse 30, rec 160 | 280 | 27.1% | 0.0084 | 0.53 / 11.1 | build-up, tail cut short |
| E11 | pulse 150, recovery 100 | 250 | 24.8% | 0.0086 | 0.31 / 1.3 | |
| E13b + E7b | pulse 40 + rec 160; follow-up 0 60 + rec 40 | 300 | 30.0% | | | |
| E7 | follow-up 0 alone 100, rec 50 | 160 | 20.7% | 0.0112 | queue 2.4 / 11.8 | BC vs rest only |
| E12 | S5+OT 30, S5 30, S20 100 | 160 | 13.0% | 0.0070 | | fatigue vs orientation on one rise |
| E6 | burst (S5+E20 40), follow-up 0 100, rec 40 | 180 | 11.5% | 0.0055 | | BC vs rest |
| E8 | alpha 0.5 hold 100, rec 60 | 160 | 10.0% | 0.0054 | | |
| E3 | overtime alone 40, rec 100 | 150 | 8.1% | 0.0047 | | |
| E1 / E2 / E10 | staffing steps without overtime | 160-180 | 5-6% | ~0.003 | | fatigue vs orientation |
| E4 | overtime spaced 10 vs 40 (brief test 1) | 320 | 5.5% | 0.0015 | | |
| E9 | recovery-shaped pulse train | 200 | 5.6% | 0.0024 | | |
| E5 | diagnostic 0.1 / 0.8 at S 10 (brief test 2) | 180 | 0.0% | 0 | | |

The brief's three tests (overtime spacing, diagnostic at fixed staffing, follow-up after a burst) and
the fatigue-vs-orientation staffing step are all low value for the score: the members already agree
on them to within 0.5 ε, and the regimes they probe carry little test disagreement. Per credit, the
short runs (E7b, E13) are close to E16, but E16 is the only one that reaches past the tail onset.

## Free model changes (ranked)

Expected score with each ODE member as the truth, averaged over the three truths no strategy uses
(AB, AC, min2v4), 40 test episodes. "Median" here is the median of the members other than the truth
(so it never contains the truth); the shipped version would be the median of all five:

| strategy | wait | queue | disch | mean | vs p3 |
|---|---:|---:|---:|---:|---:|
| p3 (public now) | 0.784 | 0.958 | 0.954 | 0.899 | |
| BC alone | 0.890 | 0.930 | 0.947 | 0.922 | +0.023 |
| wait BC, queue + disch p3 (Sunday plan) | 0.890 | 0.958 | 0.954 | 0.934 | +0.035 |
| **wait = median of ODE members, queue + disch p3** | 0.899 | 0.958 | 0.954 | **0.937** | **+0.038** |
| median of ODE members, all observables | 0.899 | 0.958 | 0.970 | 0.942 | +0.043 |

In-sample on all runs: p3 [0.876, 0.904, 0.736], median of 5 ODE [0.891, 0.904, 0.735].
LOO (lab json): p3 is best on every observable [0.854, 0.893, 0.719] vs cohort members 0.83-0.84 on wait,
but no LOO fold holds out a long post-congestion recovery, which is where the test disagreement lives.

1. **Wait_time from the per-tick median of the five ODE members (p3, AB, AC, BC, min2v4); queue and
   discharges stay p3.** +0.038 expected (committee as truth), +0.098 on wait in the 160 real
   post-disturbance recovery ticks. Keeps the public-validated queue/discharges. The median of five with
   four cohort members is cohort-shaped but damps the single most extreme tail (AB's 409).
2. **Full median of the five ODE members** (+0.043; the extra is discharges, 0.954 -> 0.970). Not the
   earlier ODE + l0b_lin median that lost both bands: l0b_lin is excluded here.
3. **Wait from BC** (the planned pick) is dominated by 1: same wait gain minus 0.009, and BC is the one
   member that disagrees with the rest on follow-up / queue (E7), so it is the riskiest single pick.

Runtime: five ODE rollouts per episode, ~0.4-0.6 s each, ~100-120 s for 40 episodes.

## Recommended plan

- **Free first (next hospital slot):** change 1 above. If public wait improves, change 2 next.
- **Buy E16, 300 credits** (pulse 40, recovery 260; U in the json under `recommended.primary`).
  It measures the onset tick and the first 60-100 ticks of the post-congestion wait plateau, which
  decides 50% of the resolvable test disagreement, separates p3 from every cohort member by 1.3-1.9 ε on
  average and ranks the cohort members (AB vs min2 1.35 ε). Refit the cohort family on it, then re-pick.
- Alternative at the same price: E17 (congestion without overtime). Pick it instead if we want to
  know whether the tail needs overtime at all; it separates cohort members less.
- Leave the brief's mechanism tests (overtime spacing, diagnostic at fixed staffing, follow-up after a
  burst) and the staffing step for later: 0-12% of the test disagreement each.

## Caveats

- Value is disagreement among our members, not error against the truth; if every member is wrong the same way no run shows it.
- The tail decay over 1,000+ ticks (the >=200 regime, 26% of the mass) is not seen by any <=300 run; E16 only fixes onset and level.
- `sigma_proxy` for wait (100) is our own scale; with a smaller organiser sigma the wait share grows further.
- Regime labels use the current action and time since the last disturbance, not the depth of the congestion.
- $\tau=10$ and $\varepsilon$ from in-sample residuals set $P_r$; the same five runs lead under the pair-based value (E16, E13, E17, E14, E15).
