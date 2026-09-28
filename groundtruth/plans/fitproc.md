# Fitting procedures on fixed families (Sun night → Mon, no credits)

Question: with the model structure frozen (the u015 family of each system, started from its public
parameters), does a different way of *fitting* generalise better than our standard fit?
Code: `scripts/fitproc.py` (`fit`, `eval`, `docs`); numbers: `plans/fitproc.json`.

## Protocol

- Scale: organizer $\sigma$ (calibrated, 1.0×). Score per tick $1/(1+|e|/\sigma)$, mean over ticks and
  observables, after the public doc's clip vectors and post rules.
- Member: the u015 ODE (for ad_auction and power_grid the main member, v8b and v9c, alone).
- Held-out: fold $k$ = every fit run except $k$, score run $k$. Recovery-shaped folds = the pulse runs.
  Exams: the `p5.testlike` fold (market, social, traffic); `p6.exam` (hospital, ad, power), never
  fitted, scored from fits on every fit run.
- Every fit starts at the public $\theta$ and has the same budget (75 s, ≤ 20 Jacobians per start).
- A procedure **wins** if its mean held-out gain over P0 is > 0.01, the mean change on the pulse folds is ≥ −0.002,
  and the exam change is ≥ −0.002.

| | procedure |
|---|---|
| P0 | `fit_ode`: Cauchy loss on $(\hat y-y)/\sigma$, 2 starts (public $\theta$ + one LHS start, spread 0.1) |
| P1 | P0's loss with importance weights $w=p_{test}(b)/p_{train}(b)$. Bins $b$ = (ticks since the last control change: 0/3/10/30/100/300+) × (controls at the recovery action or not). $p_{test}$ = the four `eval_like` categories at 25 % each, 4,000-tick schedules. Ratio clipped to [0.1, 10], mean weight 1 |
| P1h | P1 with tempered weights $w^{1/2}$ |
| P2 | L-BFGS-B on $\frac1N\sum\rho(e_i)+\lambda\frac1P\lVert z-z_{P0}\rVert^2$ with $\rho(e)=\frac{s}{1+s}$, $s=\sqrt{e^2+0.05^2}$ (smooth $1-$score), $z$ = the fitter's [0,1] coordinates (log for log-parameters), starting at the P0 solution on the same runs, 25 evaluations. $\lambda\in\{0.1,1,10,100,1000\}$ chosen per fold by the mean score on the *other* folds |
| P3 | bagging: models fitted with one more run left out ($K-1$ of them) plus the fold's P0. `P3med` = per-tick median (ships as `ensemble`), `P3mean` = mean (no runtime support), `P3tree` = balanced tree of pairwise means (shippable), `P3m2`/`P3m3` = P0 plus the 1 or 2 members nearest the bag's parameter mean, `P3par` = one model at the bag's mean $z$ |
| P4 | $y_0' = y_0 - a\,(y_0-\bar y_0^{reset})$, $a=\nu^2/(\nu^2+s^2)$, where $\nu$ = diff-based noise and $s$ = spread of the reset observations in `resets.jsonl` |

Cost: 560 fits (P0 on every run subset with one or two runs removed, plus the other procedures on subsets with at most one removed).

## Results (held-out gain over P0: mean over folds (pulse folds / exam))

| system | folds | P0 | P1 | P1h | P2 | P3med | P3mean | P3m3 | P4 | winner |
|---|---:|---:|---|---|---|---|---|---|---|---|
| social_contagion | 6 | 0.570 | −0.004 (−0.000 / −0.007) | −0.001 (−0.005 / −0.003) | −0.004 (−0.006 / −0.008) | **+0.014 (+0.042 / +0.008)** | +0.024 (+0.062 / +0.007) | −0.005 (+0.023 / −0.033) | 0.000 | P3med* |
| market | 8 | 0.615 | −0.013 (+0.022 / −0.032) | +0.002 (+0.014 / −0.001) | +0.000 (+0.002 / +0.001) | −0.000 (−0.003 / +0.004) | +0.002 (−0.000 / +0.005) | +0.001 (+0.011 / +0.002) | 0.000 | – |
| hospital_queue | 7 | 0.666 | +0.000 (+0.015 / −0.007) | −0.003 (+0.008 / −0.004) | −0.001 (−0.001 / −0.002) | +0.001 (+0.000 / +0.002) | +0.001 (+0.002 / +0.004) | +0.002 (−0.001 / +0.001) | 0.000 | – |
| epidemic | 5 | 0.684 | +0.001 (+0.019) | +0.003 (+0.012) | **+0.010 (+0.042)** | −0.017 (−0.019) | −0.012 (+0.041) | −0.005 (−0.005) | 0.000 | P2 |
| wildlife | 4 | 0.649 | **+0.021 (+0.064)** | +0.007 (+0.020) | −0.005 (+0.010) | −0.002 (−0.028) | −0.022 (−0.080) | −0.002 (−0.012) | 0.000 | P1 |
| traffic | 6 | 0.725 | +0.006 (+0.016 / +0.004) | +0.001 (−0.004 / +0.002) | +0.002 (+0.004 / +0.000) | +0.002 (+0.006 / −0.000) | −0.003 (+0.006 / −0.000) | −0.000 (+0.001 / −0.001) | 0.000 | – |
| power_grid | 3 | 0.734 | −0.022 (+0.003 / +0.000) | +0.001 (+0.003 / +0.000) | +0.010 (+0.010 / +0.001) | −0.003 (+0.001 / +0.002) | −0.051 (−0.051 / +0.001) | −0.003 (+0.001 / +0.005) | 0.000 | – (0.0100, not > 0.01) |
| ad_auction | 3 | 0.821 | +0.002 (+0.009 / +0.011) | +0.004 (+0.012 / +0.005) | −0.001 (−0.006 / +0.011) | −0.017 (−0.038 / −0.003) | −0.022 (−0.051 / −0.010) | −0.017 (−0.038 / −0.008) | 0.000 | – |
| supply_chain | 3 | 0.820 | −0.046 (−0.064) | −0.000 (−0.002) | −0.001 (−0.004) | −0.043 (+0.052) | −0.092 (−0.031) | −0.043 (+0.052) | 0.000 | – |
| reservoir | 2 | 0.769 | **+0.024 (+0.049)** | +0.010 (+0.021) | +0.001 (+0.000) | n/a | n/a | n/a | 0.000 | P1 |

\* social's winner is not shippable under our 1.5 s per episode rule (below).

Per-fold scores for the winners (P0 → winner):

| system | folds |
|---|---|
| social P3med | hold_rec 0.682→0.706, pulse200_200 0.492→0.535, compose 0.562→0.582, interior_holds 0.570→0.564, testlike 0.630→0.637, voi 0.486→0.482 |
| epidemic P2 (λ=1) | hold_rec 0.766→0.747, pulse120_280 0.665→0.707, compose 0.659→0.671, joint_hold 0.764→0.759, voi 0.566→0.588 |
| wildlife P1 | hold_rec 0.683→0.684, pulse200_200 0.698→0.762, compose 0.614→0.620, corridor_habitat 0.601→0.614 |
| reservoir P1 | hold_rec 0.897→0.896, pulse200_200 0.642→0.691 |

## Winners and docs

| system | procedure | doc | held-out gain | forecast public (0.6×) | check (4,000 ticks, 4 categories) |
|---|---|---|---:|---:|---|
| wildlife | P1 | `plans/fitproc_wildlife_doc.json` | +0.021 | **+0.012** | finite, 0.37–0.41 s |
| reservoir | P1 | `plans/fitproc_reservoir_doc.json` | +0.024 | **+0.014** | finite, 0.37–0.40 s |
| epidemic | P2, λ = 1 | `plans/fitproc_epidemic_doc.json` | +0.010 | **+0.006** | finite, 0.56–0.66 s |
| social_contagion | P3med (7 members) | not written | +0.014 | (+0.008) | finite but 2.9–3.2 s per episode |

Each doc is the u015 doc with only the ODE parameters replaced. Clip vectors and post rules are unchanged.
The runtime reproduces the study's prediction exactly (max error 0 σ). Against the public doc on
eval_like schedules the mean |Δ| is 0.08–0.16 σ, and sustained end levels move by at most 5 units (wildlife prey 46 → 41
on one hold), so no long-hold defect appears. In-sample on every fit run: epidemic 0.761 → 0.767, wildlife 0.734 → 0.736,
reservoir 0.896 → 0.892 (the weighted fit gives up a little in-sample on purpose).

The social bag needs 7 rollouts per episode (≈ 3 s under tonight's machine load, ≈ 120 s per 40 episodes).
That is inside the 1,200 s limit but over our 1.5 s rule. To write it anyway:
`python scripts/fitproc.py docs --max-sec 5`. Smaller bags do not carry the gain (P3m2 +0.011 but exam −0.021;
P3m3 −0.005). A single-model average of the bag's parameters fails everywhere (P3par, mean −0.049).

## What generalises

| procedure | mean gain (10 systems) | systems > 0 | systems > +0.01 | mean pulse-fold change |
|---|---:|---:|---:|---:|
| P1 test-mix weights | −0.003 | 6 | 2 | **+0.013** |
| P1h tempered weights | +0.002 | 7 | 1 | +0.008 |
| P2 score + ridge | +0.001 | 5 | 1 | +0.005 |
| P3med bag median | −0.007 | 3 | 1 | +0.001 |
| P4 denoised $y_0$ | 0.000 | – | 0 | 0.000 |

1. **Test-mix weighting (P1) is the one procedure that moves the recovery-shaped fold up almost everywhere**
   (+0.013 on average, up on 8 of 10 systems). It helps where our runs are dominated by short segments
   (wildlife, reservoir, traffic, epidemic's pulse). It hurts where one long run gets most of the weight (market: hold_rec
   carries weight ≈ 5, and the testlike exam drops 0.032) or where it pulls away from well-fitted transients
   (supply_chain −0.046, power_grid −0.022). The tempered version (P1h) never loses more than 0.004 but gains less.
   Rule: use P1 only after checking held-out results for each system.
2. **Score-shaped loss with a ridge (P2)** gives nothing systematic. The inner choice of λ runs from 0.1 to 1,000
   with no pattern, and the λ grid changes the fold score by at most 0.01. With 25 evaluations from the P0 solution the
   short run itself is the regularizer. Epidemic's +0.010 comes almost entirely from its pulse fold (+0.042), while
   hold_rec loses 0.019. This matches §25: the score-polished models transferred worst. We treat it as marginal.
3. **Bagging** helps only with ≥ 6 runs, and only where the fit is unstable across subsets (social: 6 of 6 folds up for
   the mean). With 3 runs the bag members are fitted on one run each and lose badly (ad, power, supply). Market and
   hospital, with 7–8 runs, stay flat: their fits are already stable.
4. **Initial-state denoising (P4) is worth nothing.** Reset spread is 10–100× the observation noise on every system, so
   the shrink factor is ≤ 0.03 except on hospital (0.28 and 0.21 on two observables), where the noise is 2 % and 47 %
   of σ. The first-20-tick change is between −0.0007 and +0.0003 everywhere.

## Caveats

- Every fold fit starts from the public θ, which was fitted on the held-out run too. This leak is the same for all
  procedures, but it favours procedures that stay near the start (P2, P3 members).
- For P2, λ is chosen on the other folds. Their fits include the scored run as training data (never as validation).
- Wildlife, reservoir and epidemic owe most of their gain to one pulse fold. Reservoir has only 2 runs, so each fold
  is fitted on a single run.
- Fits are budget-capped (75 s), so P0 here is below the public fits' full polish. The gains compare equal budgets.
