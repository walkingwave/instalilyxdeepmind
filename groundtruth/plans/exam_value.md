# Is a 400-tick exam run worth it? (Sun Sep 27)

Script: `scripts/exam_value.py` (free, no gateway). Numbers: `plans/exam_value.json`.

## Method

For each system we replay every distinct scored public predictor (deduplicated on identical
predictions over all owned runs) on every owned ledger run and score it at the calibrated sigma
(`plans/sigma_calibrated.json`, field `sigma`):

$$s_{i,r} = \frac{1}{T_r p}\sum_{t,j} \frac{1}{1 + |\hat y_{i,t,j} - y_{r,t,j}|/\sigma_j}$$

Agreement with the public board:
- Spearman $\rho(s_{\cdot,r}, \text{pub})$ (also vs the sustained and sequence bands);
- pairwise-order accuracy $PA$ = share of predictor pairs with $|\Delta\text{pub}| > 0.003$ that
  the run orders like public; $PA_{hard}$ = same on close pairs, $|\Delta\text{pub}| \le 0.05$.

A run is *unseen* by a predictor when its reset timestamp is after the upload stamp (persistence
is always unseen). OOT = agreement over unseen predictors only. In-sample inflation check:
correlation between the share of owned runs a predictor had seen and its residual in
$\text{local} \sim a + b\,\text{pub}$.

## Realised value of the three exams we own

Same predictor subset (the ones that had not seen the exam), PA:

| system | exam | mean of owned runs | best owned run (post hoc) | exam - mean | exam - best |
|---|---|---|---|---|---|
| market (n=8) | 0.857 | 0.714 | 0.929 compose | +0.14 | -0.07 |
| traffic (n=7) | 0.952 | 0.810 | 0.905 hold_rec / mid40 | +0.14 | +0.05 |
| social_contagion (n=8) | 0.821 | 0.929 | 0.929 pulse200_200 | -0.11 | -0.11 |
| **mean** | **0.877** | 0.818 | 0.921 | **+0.06** | **-0.04** |

On close pairs the exams are no better than owned runs: $PA_{hard}$ 0.50 / 0.75 / 0.50 on 4
pairs each. So an exam lifts a weak mean-of-runs proxy to about $PA \approx 0.88$, but it does not
beat the best owned run and it does not settle close calls. Its real asset is that it is held
out: every owned run is in-sample for every future candidate.

## Owned runs per system

PA over all distinct public predictors (n); OOT = best owned run over predictors that had not seen
it; inflation = corr(share of runs seen, local-minus-public residual).

| system | n | ticks owned | mean-of-runs PA / rho | best owned run PA | PA_hard (pairs) | best OOT run PA (n) | inflation | exam PA | verdict |
|---|---|---|---|---|---|---|---|---|---|
| epidemic | 7 | 1111 | 1.00 / 1.00 | 1.00 pulse120_280 | - (0) | 0.90 joint_hold (5) | 0.12 | - | **skip** |
| market | 9 | 1340 | 0.75 / 0.63 | 0.92 compose | 0.60 (5) | 0.87 compose (6) | 0.19 | 0.89 (OOT 0.86) | owned |
| traffic | 7 | 1240 | 0.81 / 0.82 | 0.90 hold_rec | 0.50 (4) | 0.60 hold_mid (5) | 0.11 | 0.95 (OOT 0.95) | owned |
| power_grid | 7 | 500 | 0.90 / 0.93 | 0.95 hold_rec | 0.67 (6) | none | 0.68 | - | **buy** |
| supply_chain | 8 | 970 | 1.00 / 0.95 | 1.00 hold_rec | - (0) | 0.62 hold_mid (6) | 0.10 | - | **skip** |
| wildlife | 8 | 1111 | 0.96 / 0.98 | 0.96 corridor_habitat | 1.00 (2) | 0.93 corridor_habitat (6) | 0.26 | - | **skip** |
| reservoir | 6 | 520 | 1.00 / 1.00 | 1.00 pulse200_200 | 1.00 (1) | none | 0.60 | - | optional |
| ad_auction | 6 | 500 | 0.80 / 0.77 | 0.87 pulse60_120 | 0.75 (4) | none | 0.61 | - | **buy** |
| social_contagion | 9 | 1511 | 0.92 / 0.93 | 0.92 interior_holds | 0.60 (5) | 0.86 interior_holds (7) | 0.03 | 0.86 (OOT 0.82) | owned |
| hospital_queue | 9 | 1070 | 0.75 / 0.57 | 0.83 compose | 0.50 (10) | 0.73 compose (6) | 0.54 | - | **buy** |

Other proxies (PA): the calibration-time in-sample score ranks no better than the mean of runs
(market 0.62, traffic 0.73, hospital 0.71, ad_auction 0.80); tick-weighted mean is the same as
the plain mean except social (0.94) and hospital (0.81).

## Expected value of a new exam

Prior from the three exams: $E[PA_{exam}] \approx 0.88$ (range 0.82 to 0.95), no gain on close
pairs. Expected gain vs the current proxy $= 0.88 - PA_{own}$:

| system | PA_own | expected gain | held-out evidence today | reason |
|---|---|---|---|---|
| hospital_queue | 0.75 | +0.13 | weak (compose OOT 0.73) | ranks worst; 10 close pairs at coin flip; inflation 0.54 |
| ad_auction | 0.80 | +0.08 | none | 3 short runs, every ODE-era predictor saw all of them; inflation 0.61 |
| power_grid | 0.90 | -0.03 | none | ranks fine today, but no held-out run at all and inflation 0.68; late subset PA 0.67 |
| reservoir | 1.00 | -0.12 | none | ranks perfectly, but only 2 runs / 6 predictors / 1 close pair; inflation 0.60 |
| wildlife | 0.96 | -0.08 | good (0.93, n=6) | owned runs already rank like public |
| supply_chain | 1.00 | -0.12 | mixed (hold_mid OOT 0.62) | hold_rec + pulse rank perfectly; only hold_mid is a poor ranker |
| epidemic | 1.00 | -0.12 | good (0.90, n=5) | owned runs already rank like public |

For power_grid and reservoir the case is not the ranking we see today but the absence of any run
that a new candidate has not trained on: the positive inflation correlation says local scores
already flatter predictors that saw more runs.

## Which run shapes rank best

| shape | best-ranking run in | worst |
|---|---|---|
| compose / joint | market, hospital, wildlife (tie), epidemic OOT | - |
| long pulse (pulse120_280, pulse200_200, pulse60_120) | epidemic, reservoir, ad_auction, social | - |
| hold_rec (120) | power_grid, supply_chain, traffic | market, hospital, social |
| short p2 pieces (pulse40, mid40, multilevel200) | - | market, hospital |
| long mid hold (hold_mid 400-450) | - | traffic OOT 0.60, supply OOT 0.62 |
| test-shaped (eval_like mixed 400) | traffic | social (below owned pulse run) |

So the exam should stay `eval_like(spec, "mixed", 400, rng)`: it mixes compose blocks and pulse
trains, the two shapes that rank best, and avoids long single-level holds.

## Recommendation

| system | exam | credits |
|---|---|---|
| hospital_queue | buy | 400 |
| ad_auction | buy | 400 |
| power_grid | buy | 400 |
| reservoir | optional (held-out only; buy if we pick among close ODE variants there) | (400) |
| epidemic, supply_chain, wildlife | skip | 0 |
| market, traffic, social_contagion | already own one | 0 |

**Total 1,200 credits (1,600 with reservoir).** Rule for any exam we buy: keep it out of every fit
until the candidates for that system are ranked; fold it into training only afterwards.

Caveats: 6 to 9 predictors per system and 0 to 10 close pairs, so one flipped pair moves PA by
0.03 to 0.17; "best owned run" is chosen after the fact and is optimistic; the prior for the new
exams rests on three systems, one of which (social) did not beat its owned runs.
