# Long-hold audit of our scored predictors (Sun Sep 27)

Free, no credits. Script: `scripts/longhold_audit.py`; numbers: `plans/longhold_audit.json`.

## What we did
For every system we loaded each scored upload's `predict.py` (deduplicated on predict.py+model.json;
u001 = persistence) and rolled it for 4,000 ticks on:
- 20 sustained-style schedules (`design.eval_like(spec, "sustained", 4000, rng)`, seeds 7000-7019,
  initial states cycled over our ledger runs' `y0`);
- 10 constant holds at uniform interior controls (5-95 % of each range, seed 9100) from a typical `y0`;
- one-control sweeps (5 levels, others at mid-range) for the monotone pairs the brief implies;
- init-memory pairs: the recovery hold and the mid hold from the lowest and the highest `y0`.

All distances are in calibrated organizer sigma (`plans/sigma_calibrated.json`). Per predictor we record
disagreement with u012 over ticks [0,450) and [450,4000), drift (mean of 3800-4000 minus mean of 3300-3500),
settle time (last tick with any observable more than 0.25 sigma from its final value), steady-state (ss)
spread across holds, init memory, and sweep signs.

**Two ways we use the public sustained band as evidence about truth beyond tick 450**
1. *Persistence check.* Persistence (u001) has a public sustained band. If predictor $P$ were the truth,
   persistence would score $\hat s_{\text{pers}}(P)=\text{mean}\,1/(1+|y_0-P_t|/\sigma)$ on our sustained
   schedules. $P$ is consistent with the truth when $\hat s_{\text{pers}}(P)\approx$ u001's band. A lower
   value means $P$ moves further from $y_0$ than the truth does, a higher value means it moves less far.
   Caveat: this assumes our reset initial states look like the organizer's.
2. *Truth consistency.* Take $P$ as the truth, compute the implied sustained score of every other scored
   predictor $Q$, and compare with $Q$'s public band (RMSE, rank correlation). The implied scores are
   biased low for every $P$ (our predictors disagree with each other more than with the truth), so we
   use the rank correlation more than the RMSE.

**Global result: no predictor drifts, oscillates or diverges on 4,000-tick holds** (max drift below 0.03
sigma everywhere except reservoir, where 0.07 sigma is the seasonal inflow cycle and physical). Every
predictor settles within 76-1,660 ticks. So the sustained losses come from *wrong steady-state
levels and maps*, not from instability. Columns in the tables: disagreement = mean over observables of
mean |P - u012|/sigma; pers = $\hat s_{\text{pers}}$.

---

## market (u012 sustained 0.491, lowest)

| upload | kind | sustained | dis. <450 | dis. >450 | settle med/max | pers |
|---|---|---:|---:|---:|---:|---:|
| u012 | ode v9g | 0.491 | 0 | 0 | 158/1659 | **0.243** |
| u009 | ensemble | 0.485 | 4.7 | 6.2 | 728/3091 | 0.066 |
| u003 | l0b_lin | 0.474 | 6.9 | 10.1 | 611/707 | 0.067 |
| u010b | ensemble | 0.472 | 4.8 | 6.8 | 1323/2189 | 0.068 |
| u008 | doc ode | 0.367 | 5.3 | 7.4 | 792/3415 | 0.105 |
| u001 | persistence | 0.129 | — | — | — | 1 |

What the holds show (typical $y_0$: price 102.4):

| hold (rate, tax) | u003 | u009 | u008 | u012 |
|---|---:|---:|---:|---:|
| 0, 0 | 94.4 | 98.4 | 102.4 | **102.4 (= y0)** |
| 0, 0.05 | 143.7 | 110.1 | 102.4 | **102.4** |
| 0.05, 0 | 57.0 | 71.9 | 73.4 | 101.5 |
| 0.05, 0.05 | 106.2 | 89.8 | 73.4 | **102.4** |
| 0.1, 0 | 19.5 | 71.9 | 73.4 | 74.7 |
| 0.1, 0.05 | 68.8 | 71.9 | 73.4 | **102.4** |

- **u012's price is frozen at its initial value** on four of the six holds: whenever the tax is positive or the
  rate is zero, price never leaves $y_0$. Init memory is 18.3-18.8 sigma, i.e. the whole difference in
  initial price survives to tick 4,000. Its ss spread across the 10 random holds is 6 sigma, against
  11-29 sigma for the others.
- The persistence check rules this out: if u012 were the truth, persistence would score 0.243 (its price
  component alone would be 0.57), but it scored **0.129**. Volume (~0.007) and depth (~0.12) are the same
  under every predictor, so the truth's price persistence component must be about
  $3(0.129) - 0.007 - 0.12 \approx 0.26$. That is between u012 (0.57) and u003/u009/u010b (0.07-0.08),
  and close to u008 (0.205).
- u009's sustained score equals u012's (0.485 vs 0.491) even though its sequence score is far lower
  (0.505 vs 0.620). So u009's long-hold price is worth about as much as u012's whole short-horizon
  advantage.

**Finding.** Trust u012 for the first ~150 ticks and for volume and depth. Do not trust its long-hold
price. **The next model must let price settle, within about 450-1,000 ticks, to a control-dependent level
$P^*(r,\tau)$ that does not depend on the initial price.** The best-scored movers imply
$P^*(0,0)\approx 95\text{-}100$, $\partial P^*/\partial r<0$ (about $-30$ from $r=0$ to $0.05$, then
saturating near 72), and $\partial P^*/\partial \tau>0$ (about $+10$ to $+40$ at $\tau=0.05$). The mean
distance from $y_0$ should be about half of u009's (the target is pers ≈ 0.26). Cheapest test: blend
u012 early into u009's price after tick ~450 (price only).

## social_contagion (u012 0.571)

| upload | kind | sustained | dis. <450 | dis. >450 | settle | pers |
|---|---|---:|---:|---:|---:|---:|
| u012 | ode | 0.571 | 0 | 0 | 242/351 | **0.077** |
| u009 | ensemble | 0.485 | 4.2 | 5.1 | 296/339 | 0.122 |
| u008 | doc ode | 0.444 | 3.5 | 4.0 | 346/404 | 0.105 |
| u010b | ensemble | 0.418 | 3.8 | 4.3 | 182/227 | 0.111 |
| u003 | l0b_lin | 0.410 | 5.4 | 7.0 | 220/240 | 0.167 |
| u004 | l0b | 0.351 | 6.6 | 8.4 | 206/235 | **0.192** |
| u001 | persistence | 0.187 | — | — | — | 1 |

Adopters (a, b) at tick 4,000 from $y_0=(48.6, 35.9)$:

| hold | u003 | u009 | u008 | u012 |
|---|---|---|---|---|
| recovery (no outreach) | 49.8, 39.6 | 71.3, 53.5 | 92.9, 67.5 | **105.3, 100.9** |
| pulse | 188.7, 119.2 | 194.4, 130.2 | 200.1, 141.1 | 194.2, 132.5 |
| mid | 129.8, 85.4 | 154.3, 103.4 | 178.7, 121.4 | 180.3, 124.0 |

- Every predictor agrees under the pulse hold (within about 2 sigma). They disagree most **with outreach
  stopped**: u012 more than doubles both communities (b triples: +65, about 13 sigma) with zero seeding.
- Persistence check: u012 implies 0.077 against the actual **0.187**. Only the low-amplitude models
  (u003 0.167, u004 0.192) reproduce the persistence band. So the truth stays much closer to $y_0$ on
  long holds than u012 says.
- u012 is also the only predictor whose incentive sweep lowers adopters_a at the steady state
  (2.2 sigma range, the wrong sign for the brief).

**Finding.** Keep u012's dynamics and pulse/mid levels. **The next model must have no spontaneous growth
under the recovery hold:** with seeding = 0, adopters should stay within about 1-2 sigma of $y_0$
(b ≲ 40-50, not 100), or drift down slowly (disappointed members). Adopters should rise monotonically
with incentive at the steady state. This is the largest and cleanest sustained error we found after
market.

## hospital_queue (u012 0.628; u010b scored 0.677)

| upload | kind | sustained | dis. <450 | dis. >450 | settle | pers |
|---|---|---:|---:|---:|---:|---:|
| u010b | ode minimal | **0.677** | 0.56 | 0.75 | 56/76 | 0.301 |
| u012 | ode | 0.628 | 0 | 0 | 60/146 | 0.295 |
| u008 | doc ode | 0.574 | 0.92 | 1.10 | 84/314 | 0.258 |
| u009 | ensemble | 0.468 | 2.2 | 2.7 | 134/287 | 0.240 |
| u001 | persistence | 0.311 | — | — | — | 1 |

Steady state (wait, queue, discharges) at tick 4,000, staffing 8, other controls at recovery:

| hold | u008 (0.574) | u010b (0.677) | u012 (0.628) |
|---|---|---|---|
| overtime 0 | 49.4, 315, 5.16 | 51.8, 313, 4.56 | 48.8, 306, 3.93 |
| overtime 0.5 | 68.6, 319, 3.54 | 49.6, 312, 4.80 | 42.8, 302, 4.92 |
| overtime 1.0 | 168.7, 324, 0.91 | **51.4, 313, 4.60** | **30.5, 263, 6.66** |
| diag 0.8 | 70.3, 321, 3.03 | 116.2, 320, 1.52 | 90.1, 319, **0.07** |
| pulse | 207.8, 333, 0.42 | 137.1, 329, 1.20 | **88.0, 327, 0.11** |

- The three ODEs give three different long-run overtime laws. u008: fatigue makes sustained overtime very
  harmful. u012: overtime is a permanent gain (wait −18, discharges +2.7) with no fatigue. u010b:
  overtime is **neutral at the steady state**. The public ranking is u010b > u012 > u008, so both
  extremes lose and the neutral law wins.
- u012 also drives gross discharges to about 0 under the pulse hold and at diag 0.8. u010b keeps
  them at 1.2-1.5 and has a higher wait (137 vs 88) under the pulse.
- All three agree the queue sits near capacity (~310-330). The persistence check cannot separate
  u010b from u012 (0.301 vs 0.295, band 0.311). All three settle within 150 ticks.

**Finding.** Trust u010b's long-hold map. **The next model must make sustained overtime roughly
neutral: fatigue cancels the overtime gain within about 100 ticks** (u012 still gains at 450). Also,
**discharges must not collapse below ~1.2/tick under heavy diagnostic allocation or the pulse
hold**, and the pulse-hold wait should be ~135, not ~90. Cheapest step: u012 dynamics for the first
~100 ticks, then u010b's steady state. The public gap is 0.049 on sustained.

## power_grid (u012 0.698; u008 scored 0.715)

| upload | kind | sustained | dis. <450 | dis. >450 | settle | pers |
|---|---|---:|---:|---:|---:|---:|
| u008 | doc ode | **0.715** | 0.24 | 0.25 | 87/89 | 0.379 |
| u012 | ode v8g | 0.698 | 0 | 0 | 88/92 | 0.393 |
| u003 | l0b_lin | 0.499 | 1.0 | 1.2 | 2/3 | 0.375 |
| u001 | persistence | 0.412 | — | — | — | 1 |

Frequency at the steady state (recovery base; sigma 0.31):

| interconnector | reserve 0 (u008 / u012) | reserve 150 (u008 / u012) |
|---|---|---|
| 0.0 | 50.34 / 50.25 | **50.63 / 49.88** |
| 0.5 | 50.34 / 50.25 | 51.47 / 51.33 |
| 1.0 | 50.34 / 50.25 | **52.31 / 51.55** |

- Load and renewable share agree within 0.2 sigma. The only material difference is **frequency under
  reserve dispatch**. u012 lowers frequency with full reserve at interconnector 0 (wrong sign for the
  brief) and under-shoots at interconnector 1 by 2.5 sigma.

**Finding.** Trust u008's frequency map. **Reserve dispatch must raise the steady-state frequency at
every interconnector setting, by about +0.3 Hz at 0, +1.1 at 0.5 and +2.0 at 1.0 for 150 units**
(u008), with no negative effect at a closed interconnector. The known gap is 0.017.

## traffic (u012 0.635; u008 0.647, u010b 0.633)

| upload | kind | sustained | dis. <450 | dis. >450 | settle | pers |
|---|---|---:|---:|---:|---:|---:|
| u008 | doc ode | **0.647** | 0.24 | 0.27 | 98/175 | 0.212 |
| u012 | ode v8d | 0.635 | 0 | 0 | 62/101 | 0.224 |
| u010b | ode | 0.633 | 0.17 | 0.20 | 172/350 | 0.218 |
| u001 | persistence | 0.308 | — | — | — | 1 |

- The three structural models agree to within 0.3 sigma on flows. Both alternatives have **speeds about
  0.4-0.5 sigma (~2-2.5 km/h) below u012** across the 10 holds.
- Under the pulse hold u012 alone makes speed_b 2.5 higher than speed_a (10.8 vs 8.3). u008 and u010b
  keep them symmetric (b ≈ a − 0.4).
- All three have a lane-closure sweep with no steady-state effect on speed_a, and ramp metering raising
  flow_a. These are shared structural choices, and nothing in the bands tests them.

**Finding.** Weak evidence, small stake (0.012). Next model: **slightly lower long-hold speeds
(about −2 km/h at mid-range holds) and symmetric route speeds under the pulse hold**. No flow change.

## wildlife (u012 0.656, best)

| upload | kind | sustained | dis. <450 | dis. >450 | settle | pers |
|---|---|---:|---:|---:|---:|---:|
| u012 | ode | **0.656** | 0 | 0 | 278/295 | 0.105 |
| u008 | doc ode | 0.646 | 0.89 | 1.09 | 142/148 | 0.101 |
| u010b | ode | 0.611 | 1.1 | 1.3 | 82/83 | 0.124 |
| u004 | l0b_lin | 0.597 | 1.2 | 1.3 | 93/93 | 0.089 |
| u001 | persistence | 0.191 | — | — | — | 1 |

- Every structural model sends predators from ~11 to ~1.6 (about 45 sigma) and prey_north from ~83 to
  ~26. The truth-consistency rank correlation is 0.89 for u012 and u008.
- Persistence check: every predictor implies about 0.10 against the actual 0.191. The two predator
  components are near 0.02 under every model, so a band of 0.19 needs the prey components near 0.35.
  Either prey stays much closer to $y_0$ than any model says, or our initial states differ from the
  organizer's. This is unresolved.
- u012 and u008 differ mainly on predators (u008 +0.7 sigma, rising with corridor access).

**Finding.** Keep u012. Open question for the next model: **prey collapse may be too deep on long holds.**
Test a shallower prey steady state (prey_north ss nearer 40 than 26) only if a cheap public slot is spare.

## epidemic (u012 0.660, best)

| upload | kind | sustained | dis. <450 | dis. >450 | settle | pers |
|---|---|---:|---:|---:|---:|---:|
| u012 | ode | **0.660** | 0 | 0 | 414/621 | 0.306 |
| u010b | ode | 0.587 | 1.5 | 1.8 | 554/629 | 0.247 |
| u008 | doc sirs | 0.469 | 1.8 | 1.1 | 419/438 | 0.292 |
| u001/u003 | persistence | 0.273 | — | — | — | 1 |

- u012 passes both checks: pers 0.306 vs band 0.273, and the best truth-consistency (RMSE 0.058, rank
  0.94). It is the slowest system to settle (median 414 ticks, and still moving 0.2-0.3 sigma after 450).
- u010b's hospital_load steady state is 2.2 sigma below u012 and depends strongly on vaccination.
  u008's steady states rise with vaccination (wrong sign). Both scored worse.
- u012's school-closure sweep has a slightly positive ss slope (0.5 sigma, negligible).

**Finding.** Trust u012. **The long-hold property to keep: a slow (~400-600 tick) approach to a
vaccination-dependent endemic level, with hospital load ~59 at interior holds.** Faster-settling or
lower-hospital variants scored worse.

## supply_chain (u012 0.754; u010b 0.759)
u010b and u012 differ by 0.1-0.4 sigma at the steady state (inventory_supplier, tied to production
effort and maintenance). The band difference (0.005) is inside the noise. The doc models (0.58) and
l0b (0.51) move far less. **Finding: no long-hold defect visible. Keep u012.** Inventory is the only
observable still moving after 450 ticks (late move 0.37 sigma; max settle ~730-800 ticks).

## reservoir (u012 0.870, best) and ad_auction (u012 0.846, best)
Both have u012 best and u008 second, within 0.2-0.3 sigma. Reservoir level never becomes stationary
under either model (seasonal inflow; drift 0.07 sigma per 500 ticks), which is physical.
**Finding: trust u012; nothing to fix for sustained.**

---

## Ranking by expected sustained gain
1. **market**: unfreeze price. Settle to $P^*(r,\tau)$ independent of $y_0$ (largest gap; band 0.491).
2. **social_contagion**: no spontaneous growth with outreach stopped (u012 doubles adopters at seeding 0).
3. **hospital_queue**: overtime neutral at the steady state, discharges ≥ ~1.2 under stress (u010b map; +0.05 known).
4. **power_grid**: reserve raises frequency at every interconnector setting (u008 map; +0.017 known).
5. **traffic**: speeds ~2 km/h lower on long holds (u008; +0.012 known).
6. wildlife: open question on prey depth. epidemic, supply_chain, reservoir, ad_auction: keep u012.
