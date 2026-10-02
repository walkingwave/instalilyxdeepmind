# supply_chain canon: literal-code readings of the brief (Mon Sep 28, no credits)

Question: if the simulator is textbook code (hard min/max, round constants, dt = 1), do literal
versions of the brief's sentences fix the two named misses of the pick (final1 = v8b, u010b):
(1) the post-pulse backlog drains at a limited rate, the model empties it exponentially;
(2) interior holds past ~200 ticks. Lab `scripts/lab_canon_sh_loo.py`, 4 runs (p1 hold_rec 120,
p2 pulse200_200 400, p3 hold_mid 450, p7 longhold 800), calibrated sigma [9.0, 98.1, 235] (1.0x),
every fold refit from the final1 theta (3 starts, 40 evals, 150 s), same protocol for the base.

## 1. What the tick data say (read at single-tick resolution)

- Discrete events. Pulse start: shipments 0, 0, 0, 17.2, 21.2, then 19.2 flat (= 54.9 x receiving
  0.35): a pure 3-tick transport delay, then a hard terminal cap. p7 switch at t769: supplier drops by
  exactly 40.0 (= order_quantity) in one tick, then flat for 10 ticks, then -18/tick; arrivals start
  at t772 (3 ticks). Recovery hold: retail drains 27.6/tick linearly (sales = min(R, 27.6)), supplier
  fills 11.5/tick linearly to a hard 362 cap. Noise is tiny (0.12 on shipments).
- Post-pulse (p2 t200-263): arrivals are piecewise flat, not exponential: 5.3 -> 9 (t201-204),
  15-17 (t205-209), 23-27 (t210-221), a 4-tick burst 56/56/53/50 (t223-226), 15-18 (t227-236),
  12.4 (t237-246), 11.7 flat (t247-262), then 0.5 -> 0 at t263. Total ~1,135 units vs ~1,700 in v8b.
  Arrivals drop at the switch (9.4 -> 5.3) although receiving rises to 1.5: the binding limit in
  recovery is not the terminal; maintenance = 1 is the only control that tightens.
- Interior hold p3: shipments 34.5 flat to t205, a smooth decline to 30.7 (t235), then a strict
  period-2 alternation 29/45 (mean 37.4) to the end. A period-2 cycle is the signature of a
  dt = 1 threshold rule (bang-bang at a cap); an ODE can only give its mean. Note for scoring: at
  sigma 9 a constant at one mode (29) scores ~0.68 on those ticks vs ~0.53 at the mean.
- Wear timing: the pulse's throughput halves at t101 with maintenance 0. With heat += (1 - m)/100
  per tick and no cooling, the interior hold (m = 0.5) would cross at t200, which is where p3 bends.

## 2. Literal versions tried (all keep v8b's structure otherwise)

| family | literal reading | params |
|---|---|---|
| canon1 | "maintenance takes productive treatment time": terminal also capped at $c_T(1 - k_{mT} m)$ (hard server cap, linear drain); wear accumulates $(1-m)/100$ with cooling free down to 1e-4; wear loss $w_l$ free | 16 |
| canon2 | canon1 + conveyors start empty at reset (brief) + travel commitment fixed at departure: dispatch splits into a rush lane (share $1-\ell$, rate $k_f$) and a normal lane (share $\ell$, rate $k_n$) per class, goods in transit keep their lane rate | 17 |
| canon3 | canon2 with the maintenance cap through its value at $m = 0.5$ ($c_h \ge 40$, p7 passes 39.4/tick there) and $m = 1$ ($c_f$) | 17 |
| canon3 w8b | canon3 with v8b's wear fixed (kcool 0.013, loss 0.5) | 15 |
| canon2 AB / AC / BC | mechanism pairs: AB no commitment ($k_c = 0$), AC no wear ($w_l = 0.05$, kcool 1), BC no congestion ($q_m = 5000$) | |

## 3. LOO at 1.0 sigma (mean of 3 observables; named segments in brackets)

| version | hold_rec | pulse (t200-300) | hold_mid (t200-300 / t300-450) | longhold (t760-800) | LOO mean |
|---|---:|---:|---:|---:|---:|
| **v8b refit (base)** | **0.977** | 0.866 (0.814) | **0.674** (0.613 / 0.501) | 0.981 (0.740) | **0.8744** |
| canon1 | 0.961 | 0.875 (0.848) | 0.649 (0.663 / 0.560) | 0.979 (0.710) | 0.8658 |
| canon2 | 0.968 | **0.882** (0.849) | 0.624 (0.604 / 0.527) | 0.980 (0.708) | 0.8633 |
| canon3 | 0.968 | 0.882 (0.843) | 0.623 (0.604 / 0.525) | 0.980 (0.708) | 0.8630 |
| canon3 w8b | 0.968 | 0.874 (0.794) | 0.653 (0.658 / 0.570) | 0.980 (0.710) | 0.8688 |
| canon2 AB (no commitment) | 0.965 | 0.874 | **0.428** | 0.982 | 0.8124 |
| canon2 AC (no wear) | 0.968 | 0.867 | 0.637 | 0.981 | 0.8634 |
| canon2 BC (no congestion) | 0.968 | **0.789** | **0.395** | 0.980 | 0.7829 |

In-sample (canon2 full fit): hold_rec 0.982, pulse 0.923, hold_mid 0.856, longhold 0.982.

## 4. Reading

- The maintenance cap does what it should on the recovery segment (+0.03 on p2 t200-300, pulse fold
  +0.009 to +0.016), but every literal version loses the hold_mid fold (-0.02 to -0.05), mostly on
  its first 200 ticks: without p3 in training the new caps and lanes are only bounded by p2 and the
  31 interior ticks of p7, and the fits pick values that bite at (lead 0.6, m 0.5). hold_rec also
  loses 0.01 (supplier fill rate).
- Literal wear (no cooling) lets the p3 fold predict a halving at t200 that the data do not show
  (kcool -> 1e-4 on that fold); p3 bends but its mean arrival rises to 37.4.
- Mechanism pairs: dropping commitment or congestion breaks the interior hold (0.43 / 0.40) and
  dropping congestion breaks the pulse (0.79). Dropping wear costs nothing (0.8634 vs 0.8633).
  So the data favour **congested transport + adaptive commitment** as the active pair, with heat/wear
  the inactive one. Our v8b still carries a wear term (the t101 halving); it is not needed for LOO.
- Acceptance (LOO >= pick on every fold, pulse/recovery folds not lost): **no canon version passes**;
  best mean is the base. Pick unchanged: final1 v8b.

## 5. Not identifiable here / next evidence

The recovery drain shape (flat 11.7 segment, 4-tick burst) and the p3 alternation each live in one
run. The p7 switch (supplier -40 in one tick, 10 flat ticks) is the cleanest literal event and v8b
misses it (supplier -20/tick at once; t760-800 segment 0.74); a lumped order withdrawal
(dispatch = min(order, S) paid at once, then production refills) is the next literal form to try,
but p7 is its only instance. Data that would decide: a second pulse -> recovery with maintenance
0.5, and an interior hold at maintenance 1.
