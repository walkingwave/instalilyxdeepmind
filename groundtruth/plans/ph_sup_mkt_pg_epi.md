# Public-history consistency: supply_chain, market, power_grid, epidemic (Tue Sep 29)

Free, no gateway. Scripts: `scripts/lab_ph3_history.py` (rollouts + implied scores), `scripts/lab_ph3_pairs.py`
(offset-free, pairwise, leave-one-out). Raw: `plans/ph_<sys>.json` (seed 11), `plans/ph_<sys>_s23.json` (seed 23),
`plans/ph_pairs_sup_mkt_pg_epi{,_s23}.json`.

## Method
Candidate C is taken as the noiseless truth. 40 episodes from `gtlab.check.make_episodes(spec, 42, 4000, seed)`
(the two pinned lo/hi episodes dropped; initial states from our reset pool). Every distinct scored public predictor
Q (grouped by predict.py+model.json hash) is rolled on the same episodes and scored against C:
$s(Q|C) = \frac14 s_{sus} + \frac34 \overline{s_{ord,rec,comp}}$, $s = \text{mean } 1/(1+|e|/\sigma)$,
$\sigma$ = `plans/sigma_calibrated.json`. Then implied $s(Q|C)$ vs public $p(Q)$:
- common set = history predictors identical to NO candidate (so every candidate is judged on the same points);
- stats: MAE, corr, offset $\overline{p-s}$, and the one that matters, **LOO**: fit $p = a + b\,s$ on the
  common set minus one, predict the held-out one; "frontier" = LOO on the top-4 public predictors;
- calibrated forecast of X under C ($C \ne X$): $a + b\,s(X|C)$ with the full common-set fit;
- known-candidate check: when a candidate is byte-identical to a scored upload, its forecast under another truth
  can be checked against its real public score.

**Key fact: final4 is byte-identical to scored public uploads on all four systems** (final4 = final1):
supply = u012..u017 (v8b, **0.8217**), market = u014..u017 (y3, **0.6339**), power_grid = u014..u016
(perobs v9c, **0.7745**), epidemic = u014..u017 (y2, **0.7166**). Those public scores ARE the calibrated
forecast of final4 (same public episodes); the method's own forecasts are listed as a check.
alt1b: supply v8c, market canon4 (hard hysteresis gate), power_grid median(v9c, w5) per obs, epidemic mean(y2, c).

## Candidate-as-truth tables (seed 11 / seed 23)

| system | truth | common MAE | corr | offset | LOO rmse | frontier LOO | known-candidate check |
|---|---|---|---|---|---|---|---|
| supply | final4 (v8b) | .091/.096 | .995/.997 | -.09 | .016/.025 | **.016/.013** | – |
| supply | alt1b (v8c) | .085/.080 | .990/.994 | -.08 | .025/.023 | .028/.024 | final4 fc .812/.810 vs pub .8217 |
| market | final4 (y3) | .065/.065 | .939/.939 | +.04 | .056/.056 | .012/.011 | – |
| market | alt1b (canon4) | .063/.063 | .940/.940 | +.04 | .056/.056 | .013/.012 | final4 fc .78 vs pub .634 (closeness inflation) |
| power_grid | final4 (perobs v9c) | .092/.091 | .996/.994 | -.08 | .015/**.016** | .019/.018 | w5 fc .726 vs pub .7585 |
| power_grid | alt1b (med v9c,w5) | .061/.068 | .996/.993 | -.05 | **.013**/.017 | .013/.012 | final4 .768/.765 vs .7745; **w5 .783/.780 vs .7585 (wrong order)** |
| power_grid | w5 (u017) | .055/.061 | .994/.991 | -.04 | .016/**.020** | .015/.013 | final4 .771/.766 vs .7745 |
| epidemic | final4 (y2) | .035/.037 | .997/.997 | .00 | **.020/.020** | **.022/.020** | – |
| epidemic | alt1b (mean y2,c) | .039/.040 | .995/.996 | .00 | .026/.026 | .032/.032 | final4 fc .787/.790 vs pub .7166 |

Reference (history predictors as truth, best ones): LOO rmse supply u010b .027/.018; market u013 .056, u012 .100;
power_grid u012 .021/.023, u013 .020/.022; epidemic u013 .021/.020, u012 .035/.038. Old models (u001-u005) .10-.65.

Caveat that dominates everything: **closeness inflation.** A truth scores near-identical predictors near 1, so
under C = alt1b the final4 predictor is implied at .84-.92, far above its public score. Candidate pairs are very
close (mutual implied score: market .917, supply .84, epidemic .87, power_grid .81-.87), all far from the real
truth (public .63-.82). The history resolves "which of two neighbours is nearer the truth" only weakly: LOO
differences between candidates are .001-.01, the same size as seed-to-seed changes.

## Forecasts of alt1b (calibrated, linear fit; seed 11 / 23)

| system | final4 public (exact) | alt1b under final4-truth | under best external truth (not a candidate) | read |
|---|---|---|---|---|
| supply | 0.8217 | .807/.795 | u010b: alt1b .786/.772 vs final4 .802/.795 (Δ -.015/-.023) | alt1b ≈ 0.80-0.81 |
| market | 0.6339 | .784/.778 (inflated) | u013: .645/.653 vs .647/.656; u012: .621/.619 vs .624/.619 (Δ ≈ -.002) | alt1b ≈ 0.63, tie |
| power_grid | 0.7745 | .735/.736 | u012: .753/.756 vs .776/.775 (Δ -.022); u008: .732/.735 vs .769/.768 (Δ -.036) | alt1b ≈ 0.755-0.765 after correcting the -.015..-.04 under-prediction these truths show on w5 |
| epidemic | 0.7166 | .797/.799 (inflated) | u013: .722/.726 vs .705/.709 (Δ +.017); u012: .659/.660 vs .659/.661 (Δ 0); u010b Δ +.003 | alt1b ≈ 0.72-0.73, slight edge, weak |

## Verdicts
- **supply_chain: keep final4 (v8b).** v8b-as-truth has the better frontier LOO on both seeds (.016/.013 vs
  .028/.024); the best external truth (u010b) puts v8c .015-.023 below v8b. Forecast alt1b ≈ 0.80 vs 0.8217 known.
- **market: tie, keep final4 (y3).** y3 and canon4 are indistinguishable to the history (LOO .056 both, frontier
  .011-.013 both) and to every external truth (Δ ≤ .003). The hysteresis gate barely moves eval-shaped
  episodes. y3 has a known 0.6339; no evidence to take the unscored model.
- **power_grid: keep final4 (perobs v9c).** All three truths fit the history equally (LOO .013-.020); every
  external truth ranks final4 > alt1b > w5, and the calibrated alt1b forecast is 0.735-0.765 < 0.7745.
  alt1b-as-truth gets the one directly checkable ordering wrong (predicts w5 .78 > final4 .77; public is
  .7585 < .7745).
- **epidemic: keep final4 (y2), low confidence.** y2-as-truth fits the history better on both seeds
  (LOO .020 vs .026, frontier .020 vs .032), but the best-calibrated external truth (u013, frontier LOO .008)
  gives the y2+c mean +.017. Net: no reliable edge; y2 has a known 0.7166, alt1b is an unscored ±.02 bet.

## w5 sanity check (validates the method?)
Expected: w5-as-truth fits the history worse than final4-as-truth (w5 scored .7585 < .7745).
- LOO rmse: seed 11 w5 .016 vs final4 .015 vs alt1b .013; seed 23 w5 **.020** vs final4 .016 vs alt1b .017.
  Right direction, but the margin (.001-.004) is inside seed noise.
- MAE/offset: w5 fits BEST (.055 vs .092). MAE here measures the offset, i.e. how the sigma calibration sits
  relative to each truth, not truth-likeness; we do not use it for decisions.
- External truths (u008, u012, u013) all order final4 > w5, as public does, but overstate the gap (.02-.06 vs .016).

**Conclusion: the check comes out in the expected direction only on the calibration-free metric (LOO) and only
weakly.** The method separates good from bad model classes cleanly (old l0/l1 truths: LOO .10-.65 vs ODE
truths .013-.03) but cannot rank two close neighbours at the .01-.02 level. Treat the four verdicts above as
"no evidence to switch away from final4's known public scores", not as measured gains.
