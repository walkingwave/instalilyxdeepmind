# market: value of information for the next buys and free changes (Sat Sep 26)

Script: `scripts/voi_market.py` (simulation only, no credits). Numbers: `plans/voi_market.json`.
Data: 5 runs, 740 ticks. $\sigma_{proxy}$ = (9.59, 7.56, 24.4) for (price, volume, depth). Credits left: 1,260.

## Committee and validation

We re-rolled every candidate doc on every ledger run with the raw `core.rollout` and compared the per-observable
score with the lab json's `insample`. All five reproduce exactly (max |diff| = 0):

| member | family / mech | in-sample mean (clipped rollout) |
|---|---|---:|
| gate_s1 | market_gate AB (tax threshold) | 0.902 |
| min_p3 | market_min AB (floor) | 0.893 |
| AC2 | market_mech AC (v2 gate) | 0.910 |
| AB2 | market_mech AB (v2 gate) | 0.910 |
| BC2 | market_mech BC (v2 gate) | 0.908 |
| l0b_lin | l0b, linear features, clip margin 1, all runs | 0.852 |
| public (u010b) | median(min_p3, l0b_lin) | 0.881 |

The v1 AB/AC/BC docs were skipped (they do not match the current module).

## Method

**Test set.** 40 schedules from `design.eval_like` (10 per category, $T = 4000$, seeds 1000..1039), $y_0$ cycled
over the five observed initial states.

**Disagreement.** For members $i, j$: $d_{ij} = \mathrm{mean}_t |y_i - y_j| / \sigma$ per observable. Each tick gets a
regime label: rate band (0 / <0.07 / >=0.07), tax side (0 / interior / >= 0.047 frozen) and hold age (ticks since the
controls last changed: <60, 60-300, >=300).

**Score matrix.** $S(i, k)$ = competition score of predictor $i$ when member $k$ is the truth, mean of the four
categories. This gives every free option an expected score (mean over $k$) and a worst case (min over $k$).

**Experiment value (preposterior).** The members are the hypotheses, with equal prior. For an experiment $E$ we
simulate each member from each observed $y_0$ and draw data $y = Y_k^E + \varepsilon$ with
$\varepsilon \sim N(0, s^2)$, where $s^2$ = white noise$^2$ + model misfit$^2$ (median member in-sample RMS: $s$ = 3.05,
0.51, 4.45). We thin to every 25th tick, because price residuals stay correlated for 17-40 ticks. Posterior
$w_j \propto \exp(-\tfrac12 \sum_t \|(Y_j^E - y)/s\|^2)$. We then pick the predictor $i^* = \arg\max_i \sum_j w_j S(i, j)$:
$$\mathrm{VOI}(E) = \mathbb E_{k, \varepsilon}\big[S(i^*, k)\big] - \max_i \tfrac1H \textstyle\sum_k S(i, k)$$
We use 300 Monte Carlo draws (s.e. about 0.005). Two diagnostics sit beside it. *Resolve* is the
decision-weighted share of member pairs that $E$ separates: $\sum_{ij}(1-S_{ij})(1-e^{-z_{ij}^2/8}) / \sum_{ij}(1-S_{ij})$.
*Settle* is the share of each mechanism model's 4,000-tick price move that is already visible when $E$ ends.
Settle guards against the case where none of the members is the truth.

## Where the test lives

Tick shares on the 40 test schedules:

| category | rate >= 0.07, tax free | same, hold age >= 300 | rate mid, age >= 300 | tax frozen | zero controls |
|---|---:|---:|---:|---:|---:|
| sustained | 0.30 | **0.26** | 0.36 | 0.12 | 0.17 |
| order | 0.36 | 0.20 | 0.13 | 0.20 | 0.20 |
| recovery | 0.05 | 0 | 0 | 0.02 | 0.93 |
| composition | 0.13 | 0.05 | 0 | 0.31 | 0.56 |

A quarter of the sustained category, and a fifth of order, is a high rate held past 300 ticks. That is exactly the
regime where the floor question lives. Another 36% of sustained is a mid rate held long, where the same question
arises in milder form.

## Committee disagreement

Mean pairwise $|\Delta y|/\sigma$ (price, volume, depth):

| category | mean | worst pair |
|---|---|---|
| sustained | 1.15, 0.01, 0.25 | 1.89, 0.03, 0.51 |
| order | 1.31, 0.01, 0.22 | 2.20, 0.03, 0.44 |
| recovery | 0.86, 0.01, 0.05 | 1.58, 0.04, 0.08 |
| composition | 1.04, 0.01, 0.10 | 1.66, 0.04, 0.18 |

Price carries almost all of it. Volume is settled (all members agree to 0.01 sigma). By regime, the largest shares of
total disagreement are: zero controls held >= 300 (18%), high rate + interior tax held >= 300 (16%), zero controls
60-300 (12%), mid rate held >= 300 (11%), and high rate, no tax, held >= 300 (9%).

Long-hold price from the mean initial state (price at t = 120 / 300 / 3999):

| hold | gate_s1 | min_p3 | AC2 | AB2 | BC2 | l0b_lin |
|---|---|---|---|---|---|---|
| r 0.1 | 80 / 79 / 79 | 80 / 79 / 79 | 67 / 42 / 37 | 64 / 41 / 39 | 64 / 47 / 43 | 87 / 77 / 64 |
| r 0.07 | 81 / 80 / 79 | 80 / 79 / 79 | 74 / 58 / 54 | 74 / 59 / 57 | 76 / 64 / 60 | 87 / 77 / 64 |
| r 0, x 0.05 | 99 / 99 / 99 | 99 / 99 / 99 | 98 / 98 / 97 | 99 / 99 / 99 | 99 / 99 / 100 | 106 / 112 / 119 |

The committee splits into two clusters. The floor cluster (gate, min, $S$ = 0.993 between them) sits at 79. The
mechanism cluster (AC2/AB2/BC2, $S$ = 0.90-0.93) falls to 37-60. Across the clusters $S$ is 0.75-0.78. l0b_lin is its
own case: its price sits on the clip floor (64.2) for any rate >= 0.04, and a tax held alone pushes its price up to
119, which no ODE supports.

## Score matrix (predictor vs hypothesis as truth)

| predictor | gate | min | AC2 | AB2 | BC2 | l0b | mean | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| public = med(min, l0b) | 0.840 | 0.840 | 0.817 | 0.814 | 0.827 | 0.840 | 0.830 | 0.814 |
| med(gate, AC2, l0b) | 0.826 | 0.824 | 0.898 | 0.852 | 0.856 | 0.907 | **0.861** | **0.824** |
| med(gate, AC2, AB2) | 0.785 | 0.784 | 0.971 | 0.947 | 0.909 | 0.813 | 0.868 | 0.784 |
| med(all ODE) | 0.792 | 0.791 | 0.944 | 0.912 | 0.946 | 0.816 | 0.867 | 0.791 |
| med(all 6) | 0.796 | 0.795 | 0.898 | 0.887 | 0.900 | 0.853 | 0.855 | 0.795 |
| AB2 alone | 0.755 | 0.754 | 0.918 | 1 | 0.934 | 0.819 | 0.863 | 0.754 |
| per-obs pick (P: med(gate,AC2,AB2), V: min, D: med ODE) | | | | | | | 0.868 | 0.784 |

## Candidate experiments (ranked by value per 100 credits)

| # | schedule | ticks | VOI | VOI/100 | P(truth) | resolve | settle |
|---|---|---:|---:|---:|---:|---:|---:|
| E3 | r 0.1 hold | 120 | +0.113 | 0.095 | 0.61 | 0.90 | 0.57 |
| E13 | r 0.1 + x 0.044 60 -> x 0.049 60 | 120 | +0.102 | 0.085 | 0.57 | 0.82 | 0.40 |
| E14 | r 0.1 hold | 150 | +0.124 | 0.083 | 0.69 | 0.93 | 0.67 |
| E1 | r 0.1 hold | 200 | +0.125 | 0.063 | 0.72 | 0.95 | 0.80 |
| E15 | r 0.1 120 -> 0 80 | 200 | +0.124 | 0.062 | 0.71 | 0.94 | n/a |
| E6 | r 0.1 + x 0.044 100 -> x 0.049 100 | 200 | +0.123 | 0.061 | 0.70 | 0.94 | 0.60 |
| E12 | r 0.1 + x 0.03 hold | 200 | +0.122 | 0.061 | 0.66 | 0.94 | 0.80 |
| E4 | r 0.07 hold | 200 | +0.120 | 0.060 | 0.68 | 0.93 | 0.80 |
| E7 | rate/tax alternating 25 x 8 | 200 | +0.119 | 0.059 | 0.70 | 0.92 | n/a |
| E9 | 6 rate pulses, short gaps | 200 | +0.108 | 0.054 | 0.61 | 0.80 | n/a |
| E8 | interior (0.05, 0.025) hold | 200 | +0.106 | 0.053 | 0.63 | 0.85 | 0.80 |
| E10 | pulse ref alpha 0.85 hold | 250 | +0.124 | 0.050 | 0.72 | 0.94 | 0.86 |
| E5 | r 0.1 150 -> 0 100 | 250 | +0.123 | 0.049 | 0.70 | 0.95 | n/a |
| E2 | r 0.1 hold | 300 | **+0.134** | 0.045 | **0.85** | **0.98** | **0.94** |
| E11 | r 0.1 200 -> 0 100 | 300 | +0.130 | 0.043 | 0.78 | 0.96 | n/a |

No pair of experiments within 320 ticks beats E2 alone (best pair E1 + E13: +0.129).

Readings:
- Any rate hold of 120 ticks or more separates the floor cluster from the mechanism cluster. That split is worth
  about +0.11 of the +0.13 available. Extra ticks mostly add the settle fraction and separate the mechanism members
  from each other.
- The tax threshold probes (E6, E13) and the interior hold (E8) come last per unit of what they settle. All fits
  already agree on the threshold.
- Per credit, 120 ticks looks best. But it shows only 57% of the mechanism models' fall. If neither cluster is right
  (the likely case), we would again be extrapolating the level that decides a quarter of sustained.

## Free changes (ranked)

1. **Swap the public median to med(gate_s1, AC2, l0b_lin).** Expected 0.861 vs 0.830 (+0.031), and the worst case
   rises too (0.824 vs 0.814). It is the best worst case of every option. It carries both clusters plus the
   data-driven member, so whichever cluster is right, the median sits between them.
2. **Price from med(gate, AC2, AB2), volume and depth unchanged.** Price is the only observable with room (0.59 ->
   0.69 mean over hypotheses), but the worst case drops to 0.784. Only worth it once the buy confirms the
   mechanism cluster.
3. **Fix l0b_lin's price before it votes.** Its price rides the clip floor (64.2) for any rate >= 0.04, and a tax
   alone raises the price to 119. Dropping the tax -> price feature (price-only) is a free fix to screen by LOO.

## Recommended plan

- **Buy E2: interest_rate 0.1, tax 0, hold 300 ticks (300 credits)**, from a reset near the usual 93-109 price.
  It has the highest value (+0.134), a posterior on the truth of 0.85, and 94% of the mechanism models' fall is
  observed directly. It answers the floor question with data. Cheaper option: E1, the same hold for 200 ticks
  (+0.125, settle 0.80), which keeps 100 credits.
- If the price stops near 79: keep the floor cluster, and the median in change 1 still holds. If it keeps falling
  past 60: refit AC2/AB2 on 6 runs and move price toward change 2.
- Apply free change 1 now (before the buy): +0.031 in expectation and a higher worst case; it gives up 0.014 only if the floor cluster is exactly right.
