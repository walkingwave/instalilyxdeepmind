# Round-2 value of information: the next buy across the five weakest systems

Sun Sep 27. Simulation only, no credits. Script: `scripts/voi_round2.py` (per system ~100 s, then
`--portfolio` ~15 s). Numbers and every schedule: `plans/voi_round2.json`.
Public now (u013): social 0.626, market 0.627, hospital 0.672, epidemic 0.687, wildlife 0.724.
Balances: social 489, market 660, hospital 930, epidemic 889, wildlife 889. Reserve 150 each.

## 1. Method

- **Committee** per system: the public model plus the four strongest alternatives. A document is kept
  only if it reproduces its lab json's in-sample scores (lab sigma, every run the lab saw) within 0.01;
  the two without a lab json (market `x4p`, hospital `p3`) must reproduce the shipped `model.json`
  exactly on every ledger run.
- **Prior weights** from public evidence: $w_j \propto \exp((s_j - s_{\max})/0.02)$ for members with a
  public score $s_j$, $e^{-0.5}$ relative weight for unscored ones. Hospital: p3 (0.690, best sustained
  0.677) leads; v8f 0.679 and v9g 0.672 behind it.
- **Test distribution**: 40 `eval_like` schedules (10 per category, T = 4,000), $y_0$ cycled over the
  ledger initials. $S_{ij}$ = test score of predictor $i$ if member $j$ were the truth, at the organizer's
  sigma (`plans/sigma_calibrated.json`), averaged over the 4 categories. Predictors = members + median.
- **Preposterior value** of a run (or set of runs) $U$:
  $V(U) = \mathbb{E}_{k\sim w,\ \epsilon}\big[S_{\hat\imath(w'),k}\big] - \max_i \sum_j w_j S_{ij}$,
  $\hat\imath(w') = \arg\max_i \sum_j w'_j S_{ij}$, $w'_j \propto w_j\, \mathcal{L}_j$. The run is simulated
  under member $k$ with a per-run level offset $\sim N(0, \text{misfit})$ plus white noise
  $N(0, \text{sd})$, $\text{sd} = \sqrt{\text{noise}^2 + \text{misfit}^2}$ (misfit = median member RMS on
  our runs), every 40th tick. The offset stands in for "the truth is not a member". 800 draws;
  sets of runs multiply likelihoods (common random numbers across sets).
- **Starts**: reset $y_0$ is only ±20 % from the ledger initials, so "high start" = mean initial × 1.2.
- **Portfolio**: every feasible set per system (≤ 3 runs, within balance − 150), then a knapsack over
  the five systems in 10-credit steps; we take the smallest spend within 0.002 of the best value.
- **Scale**: $V$ is on the committee scale (truth = a member). Real gains come through refits and the
  truth is further from every member than they are from each other; §28 rebuilds realised roughly a
  quarter to a half of their held-out targets. We use a **transfer factor 0.25** to turn committee value
  into expected public gain (range 0.1–0.5). Mean-score gain = system gain / 10.
- Robustness: 3 seeds (SE ≈ 0.004), a uniform prior, and no offset. The ranking held under all three.

## 2. Committee check

All 25 documents pass. Calibrated in-sample on all runs (organizer sigma):

| system | members (prior weight) | in-sample | max dev vs lab | expected test score (committee) |
|---|---|---|---|---|
| social | x7 .29, x2/x6/x8/v8o .18 | .724/.725/.718/.679/.724 | 0.000 | x7 **.753**, x6 .745, median .742 |
| market | x4p .34, x4/x3/v8t2 .20, v9g .05 | .625/.664/.668/.621/.655 | 0.000 | x4 **.787**, x3 .780, median .776, x4p .771 |
| hospital | p3 .31, v9d/min2 .19, v8f .18, v9g .13 | .678/.698/.649/.746/.707 | ≤ 0.003 | p3 **.851**, median .850, v9d .824, v9g .815 |
| epidemic | x11 .29, x10/x7/x8/v8g .18 | .782/.782/.762/.775/.749 | 0.000 | median **.825**, x10 .821, x11 .819 |
| wildlife | x13 .32, x6/x10/x16 .20, v8h .08 | .734/.735/.734/.734/.709 | 0.000 | median **.947**, x13 .945 |

Free readings (no credits): hospital p3 over v9g (+0.036 committee), as the public already says;
market x4 alone edges x4p only because the committee does not see x4p's exam price win (0.417 vs 0.330),
so no change there; epidemic and wildlife medians are within 0.006 of the public member.

## 3. Where the committees disagree on the test

Mean pairwise |Δy|/σ by category (per observable):

| system | sustained | order | recovery | composition | largest regime (share, d, share of all disagreement) |
|---|---|---|---|---|---|
| social | 0.91 / 1.36 | 1.36 / 1.80 | 0.61 / 1.48 | 0.91 / 1.73 | zero controls 150+ (.28, 1.19, .26); all on 150+ (.30, 0.76, .18) |
| market (P, V, D) | 11.9 / .03 / 1.65 | 8.7 / .06 / 2.1 | 7.4 / .05 / .47 | 7.0 / .06 / 1.1 | **rate and tax both on, 150+** (.31, 4.76, .44); zero 150+ (.32, 2.68, .25) |
| hospital (W, Q, D) | .84 / .47 / .36 | .77 / .21 / .62 | 2.34 / .94 / .23 | 2.61 / .79 / .28 | at recovery 150+ (.24, 1.38, .39): the tail after pulses |
| epidemic (C, H) | .67 / 1.24 | 1.28 / 2.31 | .11 / .17 | 1.05 / 1.85 | **all three on, 150+** (.30, 1.84, .52) |
| wildlife | ≤ 0.49 | ≤ 0.42 | ≤ 0.17 | ≤ 0.23 | all on 150+ (.32, 0.27, .39) |

- Market price is the widest split anywhere (7–12 σ): the members disagree on long interior joint holds
  most, more than on zero-control holds.
- Wildlife's five members agree to within 0.5 σ everywhere: the x-family variants are near copies on
  the data we own. Nothing we can buy moves the choice among them.
- Epidemic recovery is settled (0.1 σ); order and composition carry the risk.

## 4. Candidates

Credits = ticks. Value = committee-scale preposterior gain for that system; per 100 credits.
"Regime" = the disagreement-times-test-share index of `voi_epidemic.md` (run-to-run relative only).

| system | run | credits | value | per 100 | regime/100 |
|---|---|---:|---:|---:|---:|
| social | S1 zero hold 330 (proposed) | 330 | 0.045 | 0.014 | 11.3 |
| social | S1b zero hold 200 | 200 | −0.002 | −0.001 | 3.1 |
| social | S2 seeding (4,0,0) 150 | 150 | 0.089 | 0.060 | 1.1 |
| social | S3 seeding (4,0,0) 150 → +incentive (4,1.5,0) 150 | 300 | 0.175 | 0.058 | 1.7 |
| social | S4 bridge (5,1,0) 100 → (5,1,0.8) 100 | 200 | 0.158 | **0.079** | 2.7 |
| social | S5 zero hold 330, high start | 330 | 0.045 | 0.014 | 13.4 |
| social | S6 exam mixed 330 | 330 | 0.146 | 0.044 | 7.6 |
| social | SJ1 joint mid (5,1,0.5) 300 | 300 | −0.011 | −0.004 | 3.1 |
| social | **SJ2 joint (9.03, 0.135, 0.673) 300** | 300 | **0.202** | 0.068 | 29.5 |
| market | M1 zero hold 350, high start (proposed) | 350 | 0.144 | 0.041 | 71.6 |
| market | M1b zero hold 350, typical start | 350 | 0.063 | 0.018 | 27.0 |
| market | M1c zero hold 200, high start | 200 | 0.094 | 0.047 | 23.9 |
| market | M2 mid rate (0.035, 0) 200 | 200 | 0.050 | 0.025 | 0.8 |
| market | M3 tax step (0, 0.043) 100 → (0, 0.045) 100 | 200 | 0.117 | 0.059 | 2.0 |
| market | M4 frozen (0, 0.05) 300 | 300 | 0.129 | 0.043 | 13.6 |
| market | M5 exam mixed 400 | 400 | 0.145 | 0.036 | 5.3 |
| market | **MJ3 joint mid (0.05, 0.025) 200** | 200 | 0.144 | **0.072** | 64.2 |
| market | MJ1 joint mid (0.05, 0.025) 300 | 300 | 0.146 | 0.049 | 150.6 |
| market | MJ2 joint (0.090, 0.0034) 300 | 300 | 0.147 | 0.049 | 70.0 |
| hospital | H1 exam mixed 400 (proposed) | 400 | 0.069 | 0.017 | 3.3 |
| hospital | H2 staffing 20 → 8, no overtime, 150 | 150 | −0.006 | −0.004 | 0.1 |
| hospital | **H3 pulse 40 + recovery 260** | 300 | **0.076** | **0.025** | 42.0 |
| hospital | H4 overtime hold (staffing 8, ot 1) 300 | 300 | 0.035 | 0.012 | 0.0 |
| hospital | H5 exam mixed 250 | 250 | 0.030 | 0.012 | 2.1 |
| hospital | HJ1 joint mid 300 | 300 | −0.009 | −0.003 | 1.8 |
| hospital | HJ2 joint (18.2, 1.35, .57, .47, .68, .04) 300 | 300 | 0.049 | 0.016 | 7.8 |
| epidemic | E1 mask-only (0, 0.9, 0) 350 (proposed) | 350 | 0.131 | 0.037 | 1.3 |
| epidemic | E1b mask-only (0, 0.9, 0) 200 | 200 | 0.118 | **0.059** | 0.8 |
| epidemic | E2 pulse-release (.75,.75,.0022) 60 → free 200 | 260 | 0.074 | 0.028 | 1.6 |
| epidemic | **E3 mask 1 100 → vaccination .003 200** | 300 | **0.155** | 0.052 | 1.2 |
| epidemic | E4 exam mixed 400 | 400 | 0.129 | 0.032 | 1.2 |
| epidemic | EJ1 / EJ2 / EJ3 joint holds | 200–300 | 0.03–0.06 | 0.013–0.019 | 2.8–7.8 |
| wildlife | W1 hold (4,1,0) 300 (proposed) | 300 | 0.001 | 0.000 | 1.7 |
| wildlife | W1b hold (4,1,0) 200 | 200 | 0.005 | 0.002 | 1.1 |
| wildlife | W2 habitat 0.05 100 → +hunting 4 100 | 200 | 0.010 | 0.005 | 0.2 |
| wildlife | W3 repeated pulse 30/70 × 2 | 200 | 0.004 | 0.002 | 0.9 |
| wildlife | W4 habitat 0.5 no corridor 200 | 200 | −0.015 | −0.008 | 0.2 |
| wildlife | W5 exam mixed 400 | 400 | 0.022 | 0.006 | 1.2 |
| wildlife | WJ1–WJ3 joint holds | 200–300 | −0.03–0.00 | ≤ 0.000 | 2.1–9.2 |

Negative values are runs that mislead more than they teach: with the misfit offset, a run where the
members differ by less than their misfit tilts the weights toward the wrong member.

Readings:
- **Social**: the proposed zero hold separates the members poorly (0.47 σ spread over 330 ticks, below
  misfit). A high-seeding joint hold with bridge (SJ2) or the bridge step (S4) splits them 1.5–2.5 σ.
- **Market**: the joint interior hold (0.05, 0.025) for 200 ticks gets the same value as the proposed
  350-tick high-start zero hold in 57 % of the credits; it sits in the regime holding 44 % of the test
  disagreement. The tax step (M3) is a close second per credit. After one of these, a second market run
  adds little (+0.03).
- **Hospital**: a second pulse 40 + recovery 260 from a new start beats the exam run (0.025 vs
  0.017 per 100): the post-pulse wait tail is still where members split (recovery 2.3 σ on wait).
- **Epidemic**: the mask-then-vaccination run (E3) is worth the most; the 200-tick mask hold (E1b) gets
  90 % of the 350-tick version's value. Only one epidemic run is worth buying.
- **Wildlife**: nothing is worth credits (best 0.006 per 100, inside MC noise). Hold all 889.

## 5. Portfolios

Knapsack optimum (committee value; expected public mean = committee sum / 10 × 0.25):

| budget | used | runs | committee sum | mean gain (committee) | expected public mean gain |
|---:|---:|---|---:|---:|---:|
| 1,000 | 1,000 | SJ2 300, MJ3 200, H3 300, E1b 200 | 0.534 | +0.053 | **+0.013** |
| 1,730 / 1,750 | 1,700 | SJ2 300, MJ3 200 + MJ2 300, H3 300 + HJ2 300, E3 300 | 0.636 | +0.064 | +0.016 |
| 2,500 | 2,400 | the 1,700 set + EJ1 300 + W5 400 | 0.665 | +0.066 | +0.017 |
| proposed five | 1,730 | S1 330, M1 350, H1 400, W1 300, E1 350 | 0.383 | +0.038 | +0.010 |
| hold | 0 | – | 0 | 0 | 0 |

- Marginal value falls fast: the first 1,000 credits give 0.534; the next 700 give 0.10 (0.014 per 100);
  the last 700 give 0.03 (0.004 per 100, noise level). Past ~1,100 credits, holding is as good as buying.
- **Recommended: 1,100 credits** = the 1,000 set with epidemic's E3 (300) in place of E1b (200):
  +0.038 committee for 100 credits, better than any other marginal step. Committee sum 0.572, expected
  public mean gain ≈ +0.014 (range +0.006 to +0.029 over transfer 0.1–0.5).
  Balances after: social 189, market 460, hospital 630, epidemic 589, wildlife 889 (every one above the
  150 reserve; 2,757 left in total for a Monday exam or targeted refits).

| system | run | U (controls in brief order) | start | credits |
|---|---|---|---|---:|
| social | SJ2 joint hold | (9.0317, 0.1354, 0.6727) × 300 | any reset | 300 |
| market | MJ3 joint mid hold | (0.05, 0.025) × 200 | any reset | 200 |
| hospital | H3 pulse then recovery | pulse ref (5, 20, 0.75, 1, 1, 0) × 40, then recovery (20, 0, 0.4, 0.6, 0, 1) × 260 | any reset | 300 |
| epidemic | E3 mask then vaccination | (0, 1, 0) × 100, then (0, 0, 0.003) × 200 | any reset | 300 |
| wildlife | none | – | – | 0 |

## 6. Are the proposed five the right set?

No. For the same 1,730 credits the optimum is worth 0.636 against 0.383 (+66 %), and the 1,100-credit
recommendation beats the proposed five with 630 fewer credits. By system:
- **social**: zero hold → high-seeding joint hold (0.045 → 0.202).
- **market**: keep the value, cut the cost: 350-tick high-start zero hold → 200-tick joint mid hold
  (0.144 at 350 → 0.144 at 200). The high start helps the zero hold a lot (0.144 vs 0.063 from a typical
  start), but we cannot choose it beyond +20 %.
- **hospital**: exam 400 → pulse 40 + recovery 260 (0.069 at 400 → 0.076 at 300).
- **epidemic**: mask hold 350 → mask 100 then vaccination 200 (0.131 → 0.155, 50 fewer credits).
- **wildlife**: hold (4,1,0) 300 → nothing (0.001: the committee already agrees there).

## 7. Caveats

- Value is member selection on the committee scale. The real gain comes from refits and a truth outside
  the committee; the 0.25 transfer factor is a planning number, not a measurement.
- Social's zero-control level (the long-hold audit wants ≈ 65, every member sits at 80–100) is a
  hypothesis no member carries, so the zero hold's value is understated if that audit is right.
- SJ2 and MJ2/HJ2 use one random "uniform" level; a different interior level of the same kind should
  score similarly for market (MJ1 ≈ MJ2), not for social or hospital (their mid levels score ≤ 0).
- H3 repeats the shape of purchase 4's hospital run from a new start; it pays because the members still
  split on the post-pulse wait, and p3 (which did not see that run) leads the prior.
- The misfit offset is white per run; misfit that drifts inside a run would make every run less
  informative, the long ones most.
