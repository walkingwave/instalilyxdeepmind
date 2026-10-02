# Late horizon: where candidates agree early but split late (Tue Sep 29, no credits)

Question: the hidden episodes run 4,000 ticks and our data mostly stops at 400. Where do final3 and the
other good candidates agree on the data window but part ways later? Those are the places the hidden score
depends on extrapolation. And does a horizon switch help: final3 up to $T_0$, then the median of the
candidates?

## Protocol

- Candidates = every distinct shipped `predict.py` + `model.json` (hashed) among final3, alt1, u017, final2,
  final1, u016, u015. final3 always comes first. Traffic, market, epidemic, supply and ad_auction have only
  one alternative, and for traffic and reservoir that alternative is an older, weaker model (u015/u016).
- 20 eval-like schedules of 4,000 ticks (`design.eval_like`, 5 per category, seeds 5100+), with y0 cycled
  through our runs. Units are the calibrated sigma (`plans/sigma_calibrated.json`).
- Disagreement = mean $|y_a - y_b|/\sigma$ in the bands [0,400), [400,1000) and [1000,4000).
- Scenario score: take each candidate in turn as truth and score the others with $1/(1+|e|/\sigma)$.
  "exp" = uniform prior over the candidates, "worst" = worst truth.
- Data check: every run longer than 400 ticks, scored on [0,400) and [400,T). Caveat: every shipped model
  was fitted on all runs, the long holds included, so these numbers are in-sample. Held-out numbers for the
  long holds are in the ens9 / tm fold tables.
- Scripts: `scripts/lab_lh_disagree.py` (writes `plans/lh_disagree.json`), `scripts/lab_lh_table.py`
  (prints it), `scripts/lab_lh_combo.py` (switch combos, writes `plans/lh_combo.json`). The run took about
  6 min on 3 processes.

## A structural point first

The score is convex and decreasing in $|e|$. If the truth is one of the candidates, a median of two
candidates at distance $d$ scores $1/(1+d/2)$, while picking one scores $\tfrac12(1 + 1/(1+d))$ in
expectation. The pick does better on average and the median does better in the worst case (d = 0.3:
pick 0.885 expected / 0.769 worst, midpoint 0.870 / 0.870). So a median across only **two** candidates
buys worst-case safety at a cost in expectation. It pays only when the truth tends to fall between the
candidates, not on one of them. With three or more candidates, the median helps in expectation only when
final3 sits off-centre. The scenario tables below measure exactly that.

## Per-system table (ranked by expected gain)

Late disagreement = final3 vs the most credible alternative, band [1000,4000), all categories (worst
category in brackets). "Long hold" is the [400,T) score on data (in-sample).

| # | system | early / late disagreement ($\sigma$) | agrees early, splits late? | long hold favours | recommended late-horizon combination | expected effect |
|---|---|---|---|---|---|---|
| 1 | social_contagion | final3-alt1 0.26 / 0.50 (comp 0.66, rec 0.49); final3-z20 0.54 / 1.01; final3 vs median of 4 0.26 / 0.58 | **yes, the largest** | no data past 400 | **ship alt1's social = median(z20, z21, canon3) for the whole horizon** (existing `ensemble` kind, no switch needed) | scenario exp +0.037 / +0.072 / +0.074 per band, worst +0.05 / +0.13 / +0.12; held-out LOO tie (0.623 vs 0.622, exam +0.024, compose -0.015). Realistic: +0.01 to +0.03 on social |
| 2 | supply_chain | 0.10 / 0.25 (sustained 0.51; inventory_retail 0.42) | yes (sustained and order) | p3.hold_mid [400,450): alt1 0.764 vs final3 0.578; p7: tie (0.973 / 0.972) | keep final3. Median of 2 loses on expectation (0.956 vs 0.938, late band); only 50 late ticks favour alt1 and they are in-sample, while ens9 held-out has v8c losing hold_mid by 0.069 | 0 (the switch costs -0.018 expected, gains +0.027 worst) |
| 3 | hospital_queue | final3-alt1 0.17 / 0.41 (rec 0.59; queue 0.76) | yes | no data past 400 | keep final3: it is already the per-tick midpoint of hosp9 and canon2, so median(final3, alt1, final2) = final3 exactly | 0. Note the recovery steady queue: hosp9 15, final3 60, canon2 105 (3.3 $\sigma$ apart). This is the biggest open extrapolation question in hospital, and final3 is already the minimax answer |
| 4 | ad_auction | 0.13 / 0.16 (conversions 0.26) | mildly | p8 [400,700): alt1 0.880, final3 0.876, median 0.882 | keep final3 (switch400: exp -0.008, worst +0.046; p8 +0.006 in-sample) | ~0 |
| 5 | wildlife | final3-alt1 0.10 / 0.20 (sustained 0.41, prey 0.34) | mildly | p7 [400,700): alt1 0.797, final3 0.791, median 0.787 | keep final3: it is itself a median of 5 and sits at the centre (final3 vs build-median 0.03 $\sigma$) | ~0 (+0.003 exp with the build median) |
| 6 | power_grid | final3-alt1 0.14 / 0.12 | no | p8: alt1 0.926 vs final3 0.910 (load 0.881 vs 0.834, in-sample) | keep final3 (final3 vs median 0.03) | 0 |
| 7 | epidemic | 0.14 / 0.15 (peak 0.22 at 400-1000) | no | no data past 400 | keep final3 | 0 |
| 8 | market | 0.42 / 0.29 (order 0.46, comp 0.49; price 0.75) | no: the split is early, not a horizon effect | no data past 400 | keep final3 | 0 |
| 9 | traffic | final3-u015 0.24 / 0.33 (order 0.66) | yes, but the alternative is weak | p7 [400,600): final3 0.769, u015 0.565, median 0.663 | keep final3: the long hold rejects the median (-0.106) | a switch would **lose** ~0.1 on the late segment |
| 10 | reservoir | final3-u017 0.21 / 0.36 (quality 1.21) | yes, but the alternative is weak | p8 [400,700): final3 0.903, u017 0.835, median 0.837 (quality 0.759 vs 0.519) | keep final3: the long hold rejects the median (-0.066) | a switch would lose |

## What the long holds say about the switch

Six systems have data past tick 400 (traffic p7, power_grid p8, supply p3.hold_mid/p7, wildlife p7,
reservoir p8, ad_auction p8). On these segments switch(400) = median scores:

| system | final3 [400,T) | median [400,T) | switch - final3 |
|---|---:|---:|---:|
| traffic p7 | 0.769 | 0.663 | -0.106 |
| reservoir p8 | 0.903 | 0.837 | -0.066 |
| power_grid p8 | 0.910 | 0.909 | -0.001 |
| wildlife p7 | 0.791 | 0.787 | -0.004 |
| ad_auction p8 | 0.876 | 0.882 | +0.006 |
| supply p3.hold_mid | 0.578 | 0.647 | +0.069 (50 ticks) |
| supply p7 | 0.972 | 0.974 | +0.002 |

A median across all candidates hurts wherever the alternatives are older models, and it is flat elsewhere.
In the only seen case, the long hold picks the better-validated model rather than the centre. This matches
ens9 (horizon switches after control changes never won a nested comparison).

## Recommendation

1. **social_contagion: swap final3's `mean(z20, z21)` for alt1's `median(z20, z21, canon3)`**, whole
   horizon. It is already built (`submissions/20260928-2330-alt1/social_contagion`) and uses the existing
   `ensemble` kind. On held-out data it ties (0.623 vs 0.622), and it is more central on every band and in
   every category. The reason: final3 carries half of z21's documented long-hold defect, where b's churn goes
   to 0.0014 and b climbs above a with outreach stopped (z_notes §4). On the recovery hold, b at t = 4000 is
   96 for final3, 88 for alt1 and 79 for z20. This is the only system where the late-horizon analysis changes
   the pick.
2. Every other system: keep final3. hospital, wildlife and power_grid are already medians or midpoints of
   their credible alternatives. Traffic and reservoir have no credible alternative, and the long holds reject
   the median. Supply and ad_auction have only two candidates, where a median trades expectation for
   worst-case safety, and the evidence is split.
3. No new runtime kind is needed. If one is ever wanted, the exact change to `gtlab/runtime/infer.py` is:

   ```python
   def _roll_switch(blob, doc, y0, U, ctx):
       lo, hi = ctx["lo"], ctx["hi"]
       E = finalize(_dispatch(blob["early"], doc, y0, U, ctx), y0, lo, hi)
       L = finalize(_dispatch(blob["late"], doc, y0, U, ctx), y0, lo, hi)
       t0, w = int(blob.get("t0", 400)), int(blob.get("width", 0))
       t = np.arange(U.shape[0])
       a = np.clip((t - t0) / max(1, w), 0, 1) if w > 0 else (t >= t0).astype(float)
       return (1 - a)[:, None] * E + a[:, None] * L
   ```

   Register it as `"switch": _roll_switch` in `_KINDS`, and add it to the kinds list in the module docstring
   so that flatpack's dead-code pruning keeps it. The same logic, as pure functions, is `switch` /
   `blend_switch` in `scripts/lab_lh_disagree.py`. Use `width` of about 100 so there is no step at $t_0$.

## Caveats

- Scenario scores assume the truth is one of the candidates. When all candidates miss by more than their
  mutual distance, the difference between pick and median shrinks toward zero.
- Candidate pools are thin: one alternative on five systems, and a weak one on traffic and reservoir. There
  are no doc-only candidates (e.g. traffic tm1 neutral), because those files are not packaged as builds.
- Long-hold scores are in-sample for every candidate.
