# supply_chain str9 (Mon, no credits): diagnosis only, no change

Shipped v8b (final1), score by segment at the calibrated sigma [9.0, 98.1, 234.7]
(shipments / supplier / retail):

| run | t0-50 | t50-200 | t200-300 | t300-end |
|---|---|---|---|---|
| p1.hold_rec | 0.949 0.873 0.995 | 1.00 0.991 1.00 | | |
| p2.pulse200_200 | 0.860 0.881 0.914 | 0.951 0.914 0.959 | **0.640** 0.992 0.919 | 1.00 0.99 1.00 |
| p3.hold_mid | 0.874 0.976 0.993 | 0.988 1.00 0.986 | **0.680** 1.00 **0.858** | **0.544 0.846 0.745** |
| p7.longhold | 0.940 0.881 0.995 | 1.00 0.991 1.00 | 1.00 0.991 1.00 | 0.978 0.964 0.989 |

p7's 0.98 is a hold at the **recovery** action; it says nothing about interior holds. The loss is in:
1. **Recovery after a pulse** (recovery category): after p2's switch at t200 the data drain the
   conveyor backlog at a limited rate (arrivals 29.9 / 14.5 / 7.8 / 0.1 per 20-tick bin), the model
   empties it exponentially (44.5 / 22.0 / 1.2 / 0.07) and starts the burst early (t190-210:
   24.7 vs 10.4). Pinned by one run (p2 is the only pulse -> recovery run), so no LOO can test it.
2. **Interior holds past ~200 ticks** (sustained with interior levels): p3 at (40, 0.6, 0.65, 1.25,
   0.925, 0.5): shipments 34.5 -> 37.4 (29/45 alternation) after t205, supplier leaves 0 at t360
   (+2.3/tick), retail 1,107 vs 950 at t420. Production creeping from 34.5 to ~40 = order quantity
   (adaptive production commitment) fits the sum 37.4 + 2.3 ~ 40. Earlier attempts (v8c slow commitment)
   were identified by p3 alone and lost that fold.
Both leaks are pinned by one run each: the lever is data (an interior hold >= 400 ticks at a
different interior level, and a second pulse -> recovery), not a refit.
