# Epidemic: value of information for the next buy and the next free change

Sat Sep 26. Simulation only, no credits. Script: `scripts/voi_epidemic.py` (about 2 min).
Numbers: `plans/voi_epidemic.json`. Data: 3 runs, 811 ticks (`p1.hold_rec` 120,
`p2.pulse120_280` 400, `p3.compose` 291). Balance 1,189. Public epidemic 0.545 (sustained 0.587,
sequence 0.531) with `p3BC`.

## 1. Method

- **Committee.** We roll every model document in `plans/epidemic_*_doc.json` on the 3 ledger runs.
  We keep a document only if its per-run, per-observable in-sample score matches the number in its
  lab json (`insample`) to within 0.01. Then we add `l0b_lin` fitted on all runs (clip margin 1).
- **Test distribution.** 40 schedules from `design.eval_like` (10 per category, T = 4,000, seeds
  $1000c + i$). $y_0$ is drawn from the 3 observed initials.
- **Disagreement.** $d_t = \frac{1}{|\text{pairs}|}\sum_{i<j} \frac{1}{p}\sum_k |P_{i,t,k} - P_{j,t,k}| / \sigma_k$,
  with $\sigma$ = `metric.sigma_proxy(runs)` = (97.1, 39.2). "Score at risk" is $1 - 1/(1+d)$,
  the score we lose if the truth sits $d$ sigma from us.
- **Regimes.** Each tick gets a label: which controls are on (C closure, M mask, V vaccination;
  "on" means above 5 % of range), the hold age (0-50, 50-150, 150+ ticks since the last change),
  and whether any active control is interior (below 60 % of range). We also flag vaccination
  after a restriction, restriction after vaccination, and release (within 150 ticks of a
  restriction). A regime's weight is its share of test ticks.
- **Experiment value.** For a candidate schedule $U$ (100-300 ticks), averaged over the 3 initials:
  $V(U) = \sum_t d_t(U)\cdot \frac{w(r_t)}{\bar w}\cdot\frac{\bar d_{\text{test}}(r_t)}{\bar d_{\text{test}}}$.
  So a tick counts when the members disagree on it, the test spends time in that regime, and the
  test is uncertain there. We rank by $V/\text{credits}$. A regime the test never visits gets
  a quarter of its control-combination share.
- **Committees.** `ode3` = {p3BC, p3AB, sirs_p3} is the primary one: the three grey-box fits on
  all 3 runs. `top4` = ode3 + l0b_lin. `AB_vs_BC` = {p3BC, p3AB} is the mechanism-pair question.

## 2. Committee validation

| doc | reproduces lab | fitted on | in-sample (clipped, own sigma) | kept |
|---|---|---|---:|---|
| p3BC (public) | yes | all 3 | 0.906 | yes (ode3) |
| p3AB | yes | all 3 | 0.907 | yes (ode3) |
| sirs_p3 | yes | all 3 | 0.852 | yes (ode3) |
| AB, BC | yes | p1+p2 only | 0.902, 0.880 | yes (prior only) |
| AC | yes | p1+p2 only | 0.618 | yes, ignored (bad fit) |
| sirs_smoke | yes | p1+p2 only | 0.817 | yes, ignored |
| sirs | no lab reference | - | 0.819 | dropped |
| l0b_lin (new fit) | - | all 3 | 0.724 / 0.593 per obs; LOO 0.646 / 0.497 | yes (top4) |

Every document that has a lab reference still reproduces it, so no module has drifted.

## 3. Where we are least certain on the test

Mean pairwise disagreement (sigma units) and score at risk, per category:

| category | ode3 d | cases | hospital | at risk | top4 d | AB vs BC d |
|---|---:|---:|---:|---:|---:|---:|
| sustained | 0.27 | 0.20 | 0.35 | 0.18 | 0.55 | 0.21 |
| order | **0.42** | 0.32 | 0.53 | 0.24 | 1.04 | 0.22 |
| recovery | 0.08 | 0.05 | 0.11 | 0.07 | 0.24 | 0.07 |
| composition | **0.41** | 0.32 | 0.51 | 0.23 | 0.68 | 0.26 |

- **Recovery (pulse trains) is settled.** Every member agrees to within 0.1 sigma. More pulse
  data buys nothing.
- **Hospital load is the uncertain observable** in every category (about 1.6x cases).
- **Order and composition are uncertain for the same reason as sustained.** Their blocks are long:
  order blocks are 4000/(2nb+1), about 450-1,300 ticks. So the uncertainty is about where the
  system settles under a long restriction.

Regimes by contribution (share x d), ode3:

| regime | test share | d | cases | hospital | contribution |
|---|---:|---:|---:|---:|---:|
| C+M+V, held 150+, high levels | 0.120 | **0.68** | 0.51 | 0.85 | **0.082** |
| recovery, held 150+ | 0.303 | 0.12 | 0.08 | 0.16 | 0.036 |
| C+M+V, held 150+, interior | 0.146 | 0.22 | 0.17 | 0.27 | 0.032 |
| recovery 50-150 after a change | 0.091 | 0.30 | 0.22 | 0.37 | 0.027 |
| recovery 0-50 after a change | 0.056 | 0.31 | 0.27 | 0.34 | 0.017 |
| closure only, 150+ | 0.051 | 0.30 | 0.21 | 0.39 | 0.015 |
| vaccination only, 150+ | 0.037 | 0.38 | 0.29 | 0.47 | 0.014 |
| mask only, 150+ | 0.031 | 0.44 | 0.39 | 0.49 | 0.013 |
| closure+vaccination, 150+ | 0.021 | 0.60 | 0.44 | 0.76 | 0.012 |

The brief's order questions are small in the test: vaccination after restriction is 1 % of ticks
(d 0.57), restriction after vaccination 2 % (d 0.82), release 12 % (d 0.30).

**Why the long joint restriction dominates.** On a 1,500-tick hold, the members agree up to
about t = 120 (that part is covered by `p2.pulse120_280`). Then they split and stay split.
Cases / hospital:

| hold | model | t=50 | t=120 | t=200 | t=300 | t=1,000 |
|---|---|---|---|---|---|---|
| joint, alpha 1 | p3BC | 108/122 | 16/22 | 6/5 | 4/3 | 14/11 |
| | p3AB | 112/118 | 34/31 | 48/29 | 82/57 | 77/52 |
| | sirs_p3 | 123/110 | 30/35 | 9/11 | 0/1 | 0/0 |
| joint, alpha 0.85 | p3BC | 128/145 | 23/27 | 19/13 | 42/26 | 43/31 |
| | p3AB | 126/135 | 34/32 | 55/33 | 84/60 | 79/54 |
| | sirs_p3 | 134/123 | 34/38 | 24/19 | 5/6 | 0/0 |

The three members are three stories of the same data:
- **p3BC**: suppression holds.
- **p3AB**: fatigue brings a rebound to about 80 cases by t = 300.
- **sirs_p3**: elimination.

The plateau is also sensitive to alpha under BC (13 at alpha 1 against 43 at alpha 0.85). The
split has opened by t = 200 and is final by t = 300. Measured noise is about 1.2 cases and 0.7
beds (first-difference MAD), so one 300-tick run separates the three stories without ambiguity.
Under the recovery action the members agree to within 5 cases and 6 beds at every horizon.

## 4. Candidate experiments

13 plus 4 schedules. Credits = ticks. Value is from the primary committee (ode3); the other two
columns show robustness.

| rank | experiment | credits | d (ode3) | cases / hosp | test share covered | value | value/credit | top4 v/c | AB-BC v/c |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | joint pulse hold, alpha 1, 300 | 300 | 0.32 | 0.24 / 0.40 | 0.166 | 594 | **1.98** | 1.60 | **3.47** |
| 2 | joint hold, alpha 0.85, 300 | 300 | 0.29 | 0.21 / 0.36 | 0.166 | 518 | **1.73** | 1.46 | 2.68 |
| 3 | joint pulse hold, alpha 1, 250 | 250 | 0.25 | 0.19 / 0.31 | 0.166 | 313 | 1.25 | 1.24 | 2.22 |
| 4 | joint alpha 0.85 250, then release 50 | 300 | 0.31 | 0.23 / 0.38 | 0.222 | 323 | 1.08 | 1.16 | 1.68 |
| 5 | recovery only, 250 | 250 | 0.13 | 0.07 / 0.19 | 0.450 | 84 | 0.34 | 0.96 | 0.19 |
| 6 | closure 60 / rec 60 / mask 60 / rec 60 | 240 | 0.20 | 0.17 / 0.22 | 0.175 | 59 | 0.24 | 0.61 | 0.23 |
| 7 | vaccination 100, then C+M 0.85 40, rec 60 | 200 | 0.22 | | 0.163 | 43 | 0.22 | 0.43 | 0.05 |
| 8 | closure + vaccination, 300 | 300 | 0.25 | | 0.024 | 62 | 0.21 | **5.68** | 0.27 |
| 9 | vaccination only, 200 | 200 | 0.25 | | 0.049 | 38 | 0.19 | 0.37 | 0.33 |
| 10 | joint pulse 80, then release 120 | 200 | 0.13 | | 0.193 | 32 | 0.16 | 0.30 | 0.17 |
| 11 | interior hold alpha 0.5, 200 | 200 | 0.13 | | 0.166 | 28 | 0.14 | 0.47 | 0.12 |
| 12 | closure only, 250 | 250 | 0.16 | | 0.068 | 35 | 0.14 | 3.58 | 0.06 |
| 13 | pulse train alpha 0.7 (5 pulses) | 290 | 0.11 | | 0.173 | 40 | 0.14 | 0.52 | 0.10 |
| 14 | mask 60 / rec / closure 60 / rec | 240 | 0.15 | | 0.175 | 33 | 0.14 | 0.64 | 0.19 |
| 15 | C+M 0.85 40, then vaccination 160 | 200 | 0.26 | | 0.052 | 21 | 0.10 | 0.19 | 0.13 |
| 16 | closure + vaccination, 200 | 200 | 0.21 | | 0.024 | 20 | 0.10 | 3.26 | 0.12 |
| 17 | mask only, 250 | 250 | 0.16 | | 0.041 | 24 | 0.10 | 0.10 | 0.02 |

- **All three committees put the long joint restriction hold (300 ticks) at or near the top.**
  The top4 ranking puts closure-only and closure+vaccination holds first. That comes from
  l0b_lin alone: a linear model extrapolating a closure step it has seen for 60 ticks, 2-4 sigma
  from the grey-box members, which agree among themselves there (0.16-0.25). l0b_lin's leave-one-run-out score
  (0.65 / 0.50) is far below the grey-box fits, so we do not buy against it.
- **The brief's two comparisons rank mid-table** (closure vs mask 0.24/credit, vaccination vs
  restriction order 0.22 and 0.10). The members already agree on them, and the test holds them
  for very few ticks.
- 250 ticks is not enough. The p3AB rebound finishes between t = 200 and t = 300, which is why
  the 250-tick version gets half the value at 83 % of the price.

## 5. Free model options

Nothing in-sample separates the members. They were fitted on the same 811 ticks, and between the
grey-box pairs the errors straddle the truth on only 13-36 % of ticks. The mean of two is never
better than the best single one:

| option (in-sample, per obs cases / hospital) | score | vs best single |
|---|---|---|
| p3AB | 0.914 / 0.901 | |
| p3BC | 0.908 / 0.905 | |
| mean(p3BC, p3AB) | 0.912 / 0.909 | -0.002 / +0.004 |
| median(p3BC, p3AB, sirs_p3) | 0.913 / 0.901 | about 0 |
| median(p3BC, p3AB, l0b_lin) | 0.918 / 0.903 | +0.004 / -0.002 |

So the free decision has to be made on the test distribution, under model uncertainty. We score
each predictor against each plausible member taken as the truth, with a uniform prior, averaged
over the 40 test schedules:

| predictor | truth in {p3BC, p3AB, sirs_p3, AB, BC} | truth in ode3 only |
|---|---:|---:|
| **median(p3BC, p3AB, sirs_p3)** | **0.902** | **0.908** |
| p3AB | 0.900 | 0.893 |
| median(p3BC, p3AB, l0b_lin) | 0.899 | |
| mean(p3BC, p3AB) | 0.893 | |
| p3BC (public) | 0.885 | 0.893 |
| mean of the three | 0.878 | |
| sirs_p3 | 0.845 | 0.876 |
| l0b_lin | 0.646 | |

Ranked free changes:

1. **Ship the median of p3BC, p3AB and sirs_p3.** It gains +0.015 to +0.017 in expectation over
   p3BC under both priors, and it is best in every category. The median takes the middle story on
   the long restriction plateau instead of betting on one mechanism. The runtime already has an
   `ensemble` kind (median of members). This costs one upload slot, and the three rollouts are
   about 0.6 s each per 4,000-tick episode.
2. **If we test one member on its own, test p3AB, not the mean of two.** Between the single models it is
   the minimax choice (0.900 against p3BC 0.885). The mean of two loses to the median, because the
   score rewards being close to one story over sitting between two.
3. **Keep l0b_lin out of epidemic.** Its test disagreement with the grey-box models (1.5 sigma on
   order) is extrapolation error, not information. Its leave-one-run-out score is 0.65 / 0.50.

## 6. Recommended plan

- **Buy one run: joint restriction hold at alpha 0.85, 300 ticks = 300 credits** (U = [0.85, 0.85,
  0.00255] for all 300 ticks; `recommended.schedules.joint_a085_long_300` in the json). Balance
  after: 889.
  - Why alpha 0.85 rather than 1: the alpha-1 hold scores 13 % higher here, but we already own
    120 ticks at alpha 1. 0.85 is the mean of the test's U(0.7, 1) pulse levels. The BC plateau
    moves threefold between the two, so a second level pins the dose response the test actually
    uses. If Dylan prefers the higher raw score, the alpha-1 version is the same price
    (`joint_pulse_long_300`).
  - Any reset initial works: the members converge within 200 ticks whatever the start. Reset-shop
    only to avoid an extreme initial.
  - What it decides: rebound near 80 means fatigue (AB). Near 40 means BC suppression. Near 0 means
    SIRS elimination. Then we refit all three on 4 runs and re-run this script.
- **Before the buy (free):** upload the median-of-three ensemble as the epidemic factor. That hedges
  the plateau today, and the buy then tells us whether to collapse to one member.
- **Hold for later, only if the refit still disagrees:** vaccination-only 200, then closure vs mask
  240 (the brief's comparison). Each is under 0.35 value per credit today.

## Caveats

- Disagreement is not error. All the members could share a wrong structure; the public
  sequence score (0.531) sits well below the in-sample 0.9.
- `sigma_proxy` is a stand-in for the organizer's sigma. Rankings are robust to its scale, but the
  absolute "score at risk" is not.
- The regime weights come from our `eval_like` imitation of the test, not the real episodes.
- The Bayes-pick gain assumes the truth is one of our members. It measures hedging value, not
  expected leaderboard gain.
