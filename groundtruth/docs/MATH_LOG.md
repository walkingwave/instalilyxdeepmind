# Math log

Lab notebook for the Toronto 26 forecasting challenge. One dated entry per working session.
Each entry records the question, the math, the evidence, the decision, and what we rejected.
Everything needed to rebuild a model or defend a number lives here.

Conventions: $y_t \in \mathbb{R}^p$ observables after action $u_t \in \mathbb{R}^m$; $T = 4000$ ticks per
scored episode; $\sigma_j$ the organiser's fixed scale per observable; per-tick score
$s(e) = 1/(1+|e|/\sigma)$.

---

## 2026-09-24 (Thu) — Metric analysis, round-one design, full math audit

State at start: 0 of 2,000 credits spent on every system. One public upload (u001, persistence on
all 10): mean 0.3416, ~14th. Leader 0.7284.

### 1. What the metric rewards

$s(e) = 1/(1+|e|/\sigma)$ per observable per tick, averaged over ticks, observables, 40 episodes,
then over four equally weighted categories.

Properties we rely on:

1. $s$ is symmetric, decreasing in $|e|$, and **convex** in $|e|$ (second derivative
   $2/(\sigma^2(1+|e|/\sigma)^3) > 0$). Convexity means mass near $e = 0$ is worth more than the
   same mass spread out: a forecast that is exactly right half the time and badly wrong the other
   half beats one that is moderately wrong all the time.
2. For a symmetric unimodal predictive belief $q$ about $y_t$, the point forecast maximising
   $\mathbb{E}_q[s(y - \hat y)]$ is the centre of $q$ (Anderson's lemma: $s$ is symmetric and
   decreasing, so the expectation is maximised at the symmetry point). Mean, median and mode
   coincide there. **No shrinkage is justified by the metric itself.**
3. Shrinkage toward persistence $y_0$ (our "blend": $\hat y = y_0 + \lambda(\text{model} - y_0)$)
   is only justified when the belief is bimodal: "model right" vs "model broke". When the two modes
   are far apart ($|\text{model} - y_0| \gg \sigma$), the optimum is one of the modes, not a convex
   combination: a point between two distant modes scores badly against both. Per tick the blend
   beats both ends only where the model error $e_m$ and the persistence error $e_0$ have opposite
   signs, with optimum $\lambda^* = e_0/(e_0 - e_m)$. So one global $\lambda$ is a weak knob.
4. **Oscillations must not be damped to the mean.** We previously argued (strategy report §3.5)
   that a phase-uncertain oscillation should be shrunk by $e^{-v/2}$ ($v$ = phase variance). That is
   the $L_2$-optimal forecast, wrong for this metric. Monte Carlo (400k draws, truth
   $A\sin\phi$, forecast $cA\sin(\phi+\delta)$, $\delta \sim N(0,v)$):

   | amplitude | best $c$ | $e^{-v/2}$ | note |
   |---|---|---|---|
   | $A = \sigma$ | 0.65 – 0.98 | 0.14 – 0.78 | damping barely matters |
   | $A = 3\sigma$ | 0.83 – 0.98 | | even with fully decorrelated phase |
   | $A = 10\sigma$ | ~1 | | wrong-phase oscillator 0.211 beats flat mean 0.192 |

   Reason: the wrong-phase oscillator's error passes through zero twice per beat; the flat mean
   never does. Convexity of $s$ rewards the former. **Decision:** keep full amplitude on wildlife
   and power_grid cycles; fix the strategy report.
5. The organiser's $\sigma$ comes from "frozen reference trajectories". Ours (`sigma_proxy`) is the
   pooled std of our collected runs, floored at $3\times$ a first-difference noise estimate. It scales
   with our experiment design, not theirs. If their references are long holds, their $\sigma$ is
   smaller than ours. We hedge selection over $\sigma \times \{0.5, 1, 2\}$ but that band may not
   cover it. Treat $\hat\sigma$ as a weight, never as a truth.
6. Surrogate loss: we fit l1/l2 with Cauchy $\log(1 + r^2)$ on $r = (y-\hat y)/\hat\sigma$. The
   metric is linear-then-flat in $|e|$; Cauchy is quadratic-then-flat, so it under-weights errors
   below $\sigma$. Acceptable, not exact.

### 2. Round-one experiment design

Question: with 2,000 steps per system and no data yet, what is the best first purchase for
choosing the rest of the week's strategy?

Plan on the table: `hold_rec` (reset, hold recovery action 120 ticks) + `hold_pulse` (reset,
pulse 60 ticks, recovery 60 ticks). 240 per system, 2,400 total.

What each buys:

- `hold_rec`: the reset transient (present in every scored episode since every episode starts
  from reset), the recovery equilibrium, the dominant time constant $\tau$, and the measurement
  noise from the stationary tail. This is the timescale probe: it tells us the size of a "step".
- `hold_pulse` at $\alpha = 1$: one joint step on and one off. Response size and on/off asymmetry.
  All $m$ controls move together, so **zero per-control information**. Nothing for composition
  or order.

Identifiability: the two runs contain exactly two distinct control vectors, $u_{rec}$ and
$u_{pulse}$. Any equilibrium map $W \in \mathbb{R}^{(m+1)\times p}$ (linear features) has $m+1$
unknowns per observable and two equations. Ridge returns the minimum-norm split across controls,
which is arbitrary. Composition and order test episodes query control vectors never seen. So after
round one the honest model is "persistence + one shared step response", nothing finer. That is fine
for round one; it is not fine for the final.

Weaknesses of the 60/60 dwell: it is a guess made before $\tau$ is known. If $\tau \geq 100$
(reservoir: one tick is one day and inflow is seasonal; wildlife: predator–prey cycles;
supply_chain: inventories are integrators), 60 ticks show only the initial slope. If $\tau \approx 5$,
55 of each 60 are wasted.

**Decision (pending approval): split the round.**

1. Buy `hold_rec` only, all systems: 120 each, 1,200 total.
2. Estimate $\tau$ per observable by fitting $y_t = y_\infty + (y_0 - y_\infty)e^{-t/\tau}$ to the
   hold, and the noise $\hat\sigma_n$ from the last 40 ticks' first differences
   ($\hat\sigma_n = 1.4826\,\mathrm{MAD}(\Delta y)/\sqrt 2$).
3. Set the pulse dwell per system to $D = \mathrm{clip}(3\tau_{\max}, 30, 120)$ and buy
   `hold_pulse` with that $D$ on and $D$ off. Fast systems keep the spare credits for a third hold
   level (interior, `mid`), which starts the equilibrium map toward the sustained category.

Cost of splitting: one extra collect command. Value: the dwell matches measured physics, not
mock guesses. Total stays near 240 per system.

Rejected: buying `long_train` (450) first. Slow modes matter, but we cannot size gaps or lengths
before $\tau$ is known, and the run would be one initial condition.

### 3. Audit of the model ladder (l0 / l1 / l2 / ensemble / runtime)

Method: hand-derived each forward recursion and compared training code against the numpy
runtime; checked transform inverses; checked stability bounds; read the parity tests.

Correct:
- l0b: $v_t = a^t v_0 + \sum_{k<t} a^{t-1-k}\,\phi(u_k)W$ matches the runtime recursion.
- l1: $v_t = c + \sum_k z^{(k)}_{t+1}$ with the history multiplier $(1+\gamma h_t)$ applied before
  the state update; delays applied to $u$ consistently.
- l2: block-diagonal $A$ with complex modes as $2\times2$ rotations, $|{\rm eig}| \leq 0.9995$;
  `lfilter` initial state $z_0 = \lambda x_0$ gives $x_1 = A x_0 + B f_0$; runtime identical.
- Transforms: logit, log1p and affine all invert exactly; logit argument clamped to $\pm 60$ sd.
- Alignment: $y_t$ is the observation after $u_t$, as the rules state.
- Stability: l0b $0<a<1$; l1 $a = \sigma(x)$, $x \in [-4, 9.2]$; l2 radius $\leq 0.9995$;
  non-finite outputs replaced by clipped persistence. Nothing can blow up.

Findings (severity order) and decisions:

| # | finding | effect | decision |
|---|---|---|---|
| 1 | Per-control $W$ unidentifiable from hold-only data (§2) | order + composition guesses | buy compose/order runs before trusting $W$; until then `l0b_lin` with per-control linear features only, blended |
| 2 | `soft_clip` floors the allowed output range at $10^{-3}\lvert\max\rvert$ of what was observed. With 120-tick holds, a slow integrator that moved 2 % is capped at ~4 % for all 4,000 ticks | caps a correct model; persistence unhurt; hits sustained | loosen for stable-by-construction kinds (margin 3–5 or hard bounds only), keep the $50\times$ divergence gate |
| 3 | l1/l2 initial state $z_0 = E\,g(y_0) + b$ with $E \in \mathbb{R}^{K\times p}$ free, fitted from a handful of $y_0$ vectors. Columns $a^t v_{0,i}$ are collinear, so $E$ is overfit; at an unseen $y_0$ the offset persists for hundreds of ticks when $a \approx 0.99$ | early-tick error in every episode | tie $z_0$ to $g(y_0)$ (identity $E$, one scalar shrink) |
| 4 | $y_0$ is noisy and never denoised. Resets randomise only observables, so the across-run mean and variance of $y_0$ are known | bias at every tick for slow observables | shrink $\hat y_0 = \bar m + \frac{v_{prior}}{v_{prior}+v_{noise}}(y_0 - \bar m)$ |
| 5 | l0b fits $a$ and $W$ by SSE in transformed space, no $\sigma$; logit space over-weights values near 0/1 | mild metric mismatch on the kind we ship first | fit with $\sigma$-weighted loss in physical space |
| 6 | Early-tick weight applied as $\sqrt w$ on the residual inside Cauchy: $\log(1+wr^2) \neq w\log(1+r^2)$, so it changes the effective $\sigma$ instead of the weight | minor | apply as a loss weight or drop |
| 7 | Noise estimate uses pooled first differences including transients | inflated $\hat\sigma_n$ | estimate from the stationary tail of `hold_rec` only |
| 8 | Clip inside the residual gives zero gradient on saturated ticks | fits can stick | guarded by "keep init if worse"; leave |

Test gaps: no blend parity test, no noisy-$y_0$ sensitivity test, log1p/logit paths not covered,
l1 with delays > 0 not exercised through the runtime, `sigma_proxy` untested, `select.py` untested.

### 4. Audit of the strategy report against the rules and the code

Errors in the report:
- §3.5 oscillation shrinkage: wrong, see §1.4 above.
- §3.3 "the mean is not the target": for symmetric unimodal beliefs it is; see §1.2.
- §3.6 calls Cauchy "quadratic-then-flat" as if it matched the metric; the metric is
  linear-then-flat.
- §4 "40 short runs from reset are nearly as informative as one long run" is asserted and
  contradicted by our own credit study (long runs are the only thing that helped at 1,000 steps).
- Minor: 15 µs per RHS evaluation is optimistic (expect 50–100 µs; still within 1,200 s);
  "diagonalisable" should read "diagonalisable over $\mathbb{C}$".

Report says implemented, code does not:
- Soft box margin 20 % → code uses 1.0 (0.2 hurt on mocks).
- T-optimal mechanism discrimination → not implemented; p2 mechanism experiments are placeholders.
- Multiple shooting for the ODE fit → absent; multistart plus truncation only.
- "Validation runs never trained on" → every run is tagged `train` by autotune and the builder.
- Stress gate with plausibility box → default margin is $50\times$ data range; the autotune
  propose path skips the gate entirely.
- Per-horizon damping → blend $\lambda$ is a constant, never per tick.
- Budget table and timeline → stale (240 first, not 1,000).

**Decision:** fix the report before any presentation. Until then this log is the reference.

### 5. Audit of the leaderboard loop (`autotune`)

- Public score is deterministic on 40 fixed episodes, so the issue with tuning $\lambda$ from
  uploads is shape, not noise: $S(\lambda) = \text{mean}_t\, s(e_{0,t} + \lambda(e_{m,t} - e_{0,t}))$
  is a sum of cusped bumps, not a parabola. Three uploads (a full day's slots) for one scalar on
  one kind, and any data purchase invalidates the points, so the search restarts daily.
- **Decision:** tune $\lambda$ locally on leave-one-run-out rollouts (free, unlimited, can be per
  observable and per horizon). Spend uploads on comparing *kinds*, plus one confirmation.
- Local screen: 2–6 held-out scores, no variance, scored against noisy $y$ with $\hat\sigma$ from
  the same runs. **Decision:** use paired per-run differences against persistence; veto only on a
  sign test (loses on every run) or mean difference $< -0.02$ with paired SE below it; print the SE.
- Mild overfitting to the public set: ~15 comparisons per system on one fixed 40-episode set over
  five days. Final uses different episodes. Keep the number of public-driven picks small.

### 6. Audit of the credit study (`scripts/design_study.py`, CREDIT_PLAN.md)

What it did: mock ODEs (our own templates, parameters perturbed by $\exp U(-0.25, 0.25)$), two
worlds (AB seed 0, BC seed 5), 6 designs × 5 budgets, models `l0b_lin` and `l1` only, 16 episodes
per system, $T = 2000$ not 4,000, $\sigma$ = across-episode std of noiseless truth, no standard
errors, a different RNG per budget rung (so rungs use different initial conditions), and the
"plan" design at 420 truncates `long_train` mid-train.

What it shows: `l0b_lin` and `l1` (a linear equilibrium map plus three first-order lags per
observable, a few dozen parameters) saturate at about 240 mock ticks. The 240 → 420 drop
(0.67 → 0.65) is the noise floor.

What it does not show: how much data a richer model (l2, the ODE templates) needs; any
data-versus-model interaction; calibration to reality (mock persistence scores 0.49–0.52 versus
real 0.34, market 0.18, so the mocks are *less* dynamic than the real systems in exactly the
quantity being studied).

**Decision:** the CREDIT_PLAN sentence "after 240 the model is the bottleneck" is downgraded to
"l0b/l1 saturate at ~240". Re-run the curve on real data before deciding the remaining 1,760 per
system. Keep the round-one purchase; it is cheap and needed regardless.

### 7. Environment

Python 3.12 is not installed on the workstation (only 3.9 and 3.14). The scoring sandbox is 3.12
with numpy 2.3.5 / scipy 1.16.3. Install 3.12 and build the venv from it so local parity checks
mean something.

### 8. Open items carried forward

1. Approval and purchase of `hold_rec` (1,200 credits total), then sized `hold_pulse`.
2. Implement findings 2, 3, 4 of §3 before the first fitted upload.
3. Move $\lambda$ tuning local (§5).
4. Fix `split="train"` tagging so validation runs are actually held out.
5. Route the propose path through the stress gate.
6. Correct the strategy report (§4).
7. ODE templates: demoted to optional ensemble member (§9). No credits for mechanism tests.

### 9. Audit of the grey-box ODE templates (`gtlab/ode/`)

Method: ran the fast test suite (interface, numpy-only import scan, 4,000-step stress on all three
mechanism pairs × six schedules, batch/scalar parity, mocks): all pass. Then a real fit experiment
on the mocks, where the template *is* the truth and the true parameters sit within ±25 % of the
initial guess: the most optimistic setting possible.

Findings, severity order:

1. **Six of ten `x0` rules ignore part of the observed initial.** ad_auction anchors nothing;
   traffic starts from an all-zero state; market ignores volume and depth; supply_chain ignores
   shipments; hospital_queue ignores discharges; reservoir ignores outflow. Tick-one error on a
   metric that rewards staying near $y_0$. Persistence beats these on sustained.
2. **Mechanism pair is not identifiable from hold data.** market mock, truth AB, 240 ticks of
   hold_rec + hold_pulse: the wrong pair AC reached lower training cost (51.7) than AB (61.3).
   With no validation runs the selector falls back to training cost, so it picks pairs by noise.
3. **Fitting on 240 ticks can be worse than the prior.** market: untouched initial parameters
   scored 0.67 / 0.66 on eval-shaped pulses / mixed; fitted parameters 0.44 / 0.37 (training fit
   0.90). wildlife: 0.88 / 0.82 fitted vs 0.81 / 0.85 initial (wash). ad_auction: 0.98 / 0.95 vs
   0.78 / 0.85 (works, simplest template). Cause: 19–28 free parameters per system against two
   excited modes; the optimiser walks along flat directions (slow $	au$ in $[10, 5000]$,
   mechanism gains) and extrapolates worse than the prior. Cauchy loss does not fix this; only
   freezing most parameters or strong priors would.
4. **Fixed-step RK4 can go unstable inside the parameter bounds.** reservoir with one substep:
   mixing coupling eigenvalue up to ~2.8 (RK4 limit 2.78), flushing rates ~10, $\nu\,\mathrm{inflow}/V$
   unbounded as $V 	o 0$. power_grid burst rate up to 6 per tick at $dt = 0.5$. supply_chain
   dispatch rate 5 at $dt = 0.5$. Clipping turns blow-ups into finite garbage instead of NaN, so
   the fitter sees a rugged, flat residual landscape.
5. Minor: `export()` writes `param_names` strings into model.json (check flatpack strips it);
   `lead_time_buy` mapped to a rush share with no basis in the brief; traffic speed modelled by
   occupancy while the brief defines it from journey times and class mix.

Per-system plausibility against the briefs: power_grid, hospital_queue, epidemic, wildlife map the
hints well; market, supply_chain, social_contagion miss named couplings (placement speeds, shared
cooling/drive service, three audience types); reservoir collapses stratification to one scalar and
is numerically the weakest; traffic and ad_auction do not anchor to $y_0$.

Runtime: 0.21–0.69 s per 4,000-tick episode on this PC, ~175 s for all 10 systems × 40 episodes.
Fits the 1,200 s limit with a $6	imes$ margin.

**Decision.** The ODE code stays but is not the main bet. Next three days go to the data-driven
ladder (l0b_lin / l1 / l2 with blend), which anchors at $y_0$ by construction. ODE work, if any,
is limited to: fix `x0` anchoring for the six systems, fit 3–6 parameters with the rest frozen,
use only as an ensemble member behind blend. **No credits are spent on "mechanism test" runs
to feed this fitter.** The strategy report's "ODE is the main bet" is withdrawn.

### 10. Setup and checks run tonight (no credits)

- Installed Python 3.12.10, built `.venv` with numpy 2.3.5 / scipy 1.16.3 (sandbox pins).
- Tests: 141 pass, 2 fail. The two failures are the packaging checker's sandboxed run, which
  uses `preexec_fn` (Linux only). Expected on Windows; the checker must run on Linux.
- Gateway budget call (free): 2,000 remaining on every system. Auth works.
- Dry run of `hold_rec` (free): 120 ticks per system, 1,200 total. Not purchased.

### 11. Purchase 1: `hold_rec`, 120 ticks × 10 systems = 1,200 credits (22:00–22:15)

Server balance after: 1,880 on every system. Ledger agrees, no drift.

Per-observable summary (`scripts/probe_summary.py`): $y_0$ = noisy initial, $y_{end}$ = mean of
the last 40 ticks, $\tau$ from a single-exponential fit $y_t = y_\infty + (y_1 - y_\infty)e^{-t/\tau}$
(fit % = RMSE of that fit relative to scale; large means non-exponential), noise = tail
first-difference MAD $\times 1.4826/\sqrt2$.

| system | observable | $y_0$ | $y_{end}$ | change | $\tau$ | fit % | noise % |
|---|---|---:|---:|---:|---:|---:|---:|
| epidemic | daily_cases | 187 | 51 | −73 % | 73 | 46 | 0.19 |
| epidemic | hospital_load | 60 | 47 | −0.3 % of scale, still falling | >1000 | 1 | 0.02 |
| market | price | 93.4 | 94.3 | +1 % | 20 | 0.7 | 0.36 |
| market | volume | 91.6 | 1.8 | −98 % | 2.7 | 0.1 | 0.01 |
| market | depth | 117 | 91 | −22 % | 16 | 0.8 | 0.25 |
| traffic | flow_a / flow_b | 36 / 33 | 0 / 0 | −100 % | 2 / 17 | 0 | 0 |
| traffic | speed_a / speed_b | 36 / 42 | 49 / 49 | +27 / +13 % | 3.4 | 0.2 | 0.2 |
| power_grid | load | 106 | 92 | −13 % | 22 | 6.6 | 0.31 |
| power_grid | frequency | 49.84 | 50.40 | +1.1 % | 23 | 0.6 | 0.04 |
| power_grid | renewable_share | 0.29 | 0.37 | +24 % | 30 | 5.8 | 0.22 |
| supply_chain | shipments | 26 | 0 | −100 % | 1.4 | 0 | 0.01 |
| supply_chain | inventory_supplier | 98 | 362 | +72 %, still rising | 12 | 3.5 | 0.25 |
| supply_chain | inventory_retail | 105 | 0 | −100 % | 1.5 | 1.3 | 0 |
| wildlife | prey_north | 86 | 121 | +41 % | 343 | 23 | 0.85 |
| wildlife | predator_north / south | 8.7 / 12.6 | 2.5 / 2.6 | −71 / −80 % | 18 | 1.6 | 0.15 |
| wildlife | prey_south | 96 | 97 | +1 % | 96 | 14 | 0.40 |
| reservoir | level | 481 | 941 | +47 %, integrating | 30 | 1.8 | 0.60 |
| reservoir | inflow | 10.5 | 11.4 | +8 % | 21 | 13 | 0.79 |
| reservoir | outflow | 6.4 | 10.1 | +12 % | 253 | 7.9 | 0.17 |
| reservoir | quality | 0.87 | 0.95 | +9 % | 3.8 | 0.6 | 0.83 |
| ad_auction | win_rate | 0.44 | 0.26 | −40 % | 20 | 1.9 | 0.95 |
| ad_auction | spend | 10.6 | 13.1 | +22 % | 50 | 8.5 | 0.61 |
| ad_auction | conversions | 0.87 | 3.2 | +67 % | 7.4 | 14 | 0.54 |
| social_contagion | adopters_a / b | 53 / 22 | 59 / 44 | linear growth, no saturation | >1000 | 0.2 | 0.01 |
| hospital_queue | wait_time | 3.0 | 0.003 | −100 % | 10 | 2 | 0.12 |
| hospital_queue | queue | 66 | 23 | −65 % | 6.5 | 7.3 | 0.17 |
| hospital_queue | discharges | 8.2 | 11.5 | +28 % | 1.2 | 28 | 0.61 |

Reset pools (free): initial observables vary about ±20 % around their medians, e.g. epidemic
daily_cases 104–235, reservoir level 401–560, wildlife predators 8–15.

What this changes:

1. **Noise is negligible** (0.01–1 % of scale). Errors will come from hidden state and wrong
   dynamics, not measurement noise. $y_0$ denoising (§3 finding 4) is low priority.
2. **The systems are far more dynamic than the mocks.** Persistence loses 40–100 % of scale on
   many observables within 120 ticks. The mock design study under-stated the value of data
   (§6 already flagged this).
3. **The recovery action is a degenerate corner for four systems**: traffic flows, supply_chain
   shipments and retail inventory, hospital wait time, market volume all go to zero under it.
   The recovery equilibrium therefore says little about interior control levels, which the
   sustained category holds. Interior levels are needed early.
4. **Integrators and slow modes are real**: reservoir level doubled in 120 ticks and is still
   climbing; supply_chain supplier inventory ×3.7 and climbing; social_contagion adopters grow
   linearly; wildlife prey_north $\tau \approx 340$; epidemic is a wave (non-exponential). Over
   4,000 ticks these must saturate somewhere we have not seen. A long run is required for these
   five systems, and the $\pm 20\%$ initial spread means the saturation level, not the initial,
   dominates the sustained score.
5. Timescales split cleanly: fast ($\tau < 20$: traffic, hospital_queue, market), medium
   ($\tau$ 20–50: power_grid, ad_auction, reservoir quality), slow (the five above).

### 12. Purchase 2 design (materialized as phase `p2`, `plans/p1b_pulse.json`, not yet bought)

Per system 120 ticks, 1,200 total:

- Fast systems (traffic, hospital_queue, market): `pulse40` = pulse 40 on, recovery 40 off, then
  `mid40` = a fresh reset held 40 ticks at the bounds midpoint. Three equilibria instead of one,
  and the on/off response at $2\tau$ or more.
- All others: `pulse60` = pulse 60 on, recovery 60 off. For the slow systems this gives the
  initial slopes of both responses; saturation is deferred to the long run.

Rejected: pulse 120 on only for slow systems (loses the off-response that the recovery category
scores on every pulse).

### 13. Free public probe built (not yet uploaded)

`submissions/20260924-2232-20260924-relax`: `l0b_lin` (relax from $y_0$ toward a learned
equilibrium with one learned rate per observable) fitted on `hold_rec` for the six systems whose
recovery equilibrium is not degenerate: epidemic, wildlife, social_contagion, power_grid,
reservoir, ad_auction. Omitted systems keep persistence. In-sample on `hold_rec` (own
$\hat\sigma$): model 0.57–0.72 vs persistence 0.29–0.56. Out of sample unknown; that is the point
of the upload. 4,000-tick rollouts finite, within plausible ranges, 0.02 s each. Today's three
upload slots are unused and expire at midnight, so this costs nothing.

### 14. Upload u002 (public, Sep 24 ~23:00): relax model on six systems

| system | persistence (u001) | relax `l0b_lin` (u002) | delta |
|---|---:|---:|---:|
| ad_auction | 0.4711 | 0.6044 | +0.133 |
| reservoir | 0.3471 | 0.4625 | +0.115 |
| social_contagion | 0.3052 | 0.3672 | +0.062 |
| power_grid | 0.4867 | 0.5448 | +0.058 |
| wildlife | 0.2227 | 0.2785 | +0.056 |
| epidemic | 0.2720 | 0.1562 | −0.116 |

Ten-system mean: 0.3416 → 0.3724 (other four unchanged at persistence).

Reading: one 120-tick hold from reset, turned into "relax from $y_0$ toward a fixed level at a
fixed rate", is worth +0.06 to +0.13 wherever the recovery hold is representative of the dynamics.
Epidemic is the exception: its hold was a wave (single-exponential fit error 46 %), the learned
"equilibrium" of 51 cases is a point on a falling curve, and interventions in the test schedules
change the wave itself. Epidemic reverts to persistence until it has a wave-capable model
(SEIR-like, or at least a second-order lag). Decision for the next upload: keep u002 on the five
gainers, persistence on epidemic, and add fitted models for the four degenerate systems once
purchase 2 gives interior levels.

### 15. Simulations on the real hold_rec data (`scripts/sim_next.py`, free)

Two questions. (a) Extrapolation: fit each kind on ticks 1–80 of the hold, score ticks 81–120
with our $\hat\sigma$. (b) Committee disagreement: 18 members per system (l0b_lin and l1 × three
$\sigma$ multipliers × three block bootstraps of the run), rolled on 300-tick candidate
schedules; spread = median absolute deviation from the committee median, in $\hat\sigma$ units,
averaged over ticks and observables. Large spread means the data would settle something the
models cannot.

| system | persist | l0b_lin | l1 | highest-spread candidates (σ units) |
|---|---:|---:|---:|---|
| epidemic | 0.66 | 0.38 | 0.73 | all ≈ 0.4 (composition, mid hold, order) |
| market | 0.29 | 0.68 | 0.80 | mid40 0.29, pulse40 0.23 |
| traffic | 0.05 | 0.87 | 0.85 | mid40 1.50, pulse40 0.77, multilevel 0.38 |
| power_grid | 0.33 | 0.79 | 0.24 | long pulse hold 0.70, pulse60 0.67, sustained 0.61 |
| supply_chain | 0.09 | 0.87 | 0.92 | pulse60 0.26, multilevel 0.17 |
| wildlife | 0.43 | 0.56 | 0.86 | pulse60 0.36, composition 0.35 |
| reservoir | 0.38 | 0.59 | 0.54 | sustained 1.05, long pulse hold 1.03, mid hold 0.91 |
| ad_auction | 0.33 | 0.43 | 0.74 | long pulse hold 2.29, sustained 2.02, pulse60 1.75 |
| social_contagion | 0.43 | 0.43 | 0.98 | every long schedule ≈ 1.35 |
| hospital_queue | 0.32 | 0.92 | 0.98 | ≈ 0 everywhere (all members collapse to the same zero state) |

Reading:
- The lag model l1 extrapolates the tail of the hold far better than relax-to-equilibrium on 7
  of 10 systems; l0b_lin wins on power_grid (l1 over-fits the oscillatory rebound with 80 ticks)
  and reservoir. Epidemic: nothing beats persistence by much; the wave is not a lag.
- Committee spread points at long holds and pulse-level holds for ad_auction, reservoir,
  social_contagion and power_grid: the models do not know the saturation level. For traffic and
  market the spread is at interior levels (mid40): the recovery corner is degenerate and the
  models have never seen a non-zero flow under a held control.
- Caveat: every member has seen one control vector. Spread on control-moving schedules is a
  floor. It cannot yet rank pulse-train gaps or order effects.

### 16. Strategy research and the decision (full text: `docs/study/strategy_synthesis.md`,
sources `docs/study/research_input_design.md`, `docs/study/research_models.md`)

Literature check on our shape (space-filling first, committee-adaptive after): right shape, two
corrections. (1) For slow modes and integrators, a segment must last 3–5 $\tau$ to pin the gain
(Ljung ch. 13; Rivera/Braun switching-time rules); 200-tick random excitation with ~14-tick
segments returns slopes, not gains, for the five slow systems. (2) The largest documented lever for
open-loop rollout error is the training loss (simulation / multi-step error with multiple
shooting: Ribeiro 2020, Forgione & Piga 2021), not cleverer excitation; adaptive design gains are
constant-factor (Wagenmaker 2020), so it gets a capped slice, not the majority.

**Purchase 2 revised (phase `p2`, `plans/p2_v2.json`, dry-run 3,720):**

| group | systems | experiments | per system |
|---|---|---|---|
| fast | traffic, hospital_queue, market | pulse 40/40, mid hold 40, multilevel 200 (market dwell 15–60) | 320 |
| medium | power_grid, ad_auction | pulse 60 on / 120 off, multilevel 200 with 40–70 dwell | 380 |
| slow | wildlife, reservoir, social_contagion, supply_chain | pulse 200 on / recovery 200 off | 400 |
| slow | epidemic | pulse 120 on / recovery 280 off | 400 |

Balance after: 1,480–1,560 per system. Then P3 Sat (650: recovery-shaped train, single-vs-joint,
held-out validation run; swap to a 400-tick midpoint hold where the 200-on segment did not
settle), P4 Sun (adaptive committee picks, capped), 300 reserve to Tue 09:00.

Model order: Fri audit fixes + l1/l0b_lin per-control refit and upload; Sat l2 by multiple
shooting with eigenvalue 1.0 allowed inside the physical box; Sun per-system structure (SIR wave
for epidemic, full-amplitude modes for wildlife, integrator priors, log1p floors for the
degenerate-corner systems), median ensemble except on oscillatory systems.

Oscillation dispute closed with a derivation (synthesis §D): under $s(e)=1/(1+|e|/\sigma)$ with
phase uncertainty, full amplitude is optimal whenever $A \gtrsim 3\sigma$; the $e^{-v/2}$ shrink
is an $L_2$ result and does not apply. Do not average ensemble members with different phases.

## 2026-09-25 (Fri, early) — Purchase 2 done, model fixes, screening

### 1. Purchase 2 (phase `p2`): 3,720 credits, 22:50–00:10

Bought exactly as in the Sep 24 §12/§16 table. Server balances after: epidemic, supply_chain,
wildlife, reservoir, social_contagion 1,480; power_grid, ad_auction 1,500; market, traffic,
hospital_queue 1,560. Ledger agrees. Total spent so far 4,920 of 20,000.

### 2. Model fixes from the audit (Sep 24 §3), all tests pass

- `soft_clip` margin 1.0 → 3.0 × observed range (still inside hard bounds). Reason: integrators
  and slow modes leave the short-run range over 4,000 ticks; the 50× stress gate still catches
  divergence.
- l1 initial state anchored: $z_{0,k} = (v_{0,j} - c)/K + b_k$ so $v_0 = v_{0,j} + \sum_k b_k$;
  only $K$ offsets in $[-1, 1]$ sd are free instead of a $K \times p$ matrix $E$. Linear init
  solves for $W, c$ with the anchored decay $D_t = \frac1K\sum_k a_k^t$: target
  $V_t - v_{0,j}D_t$, columns $[\mathrm{lfilt}(a_k, F), (1 - D_t)]$. Export folds the anchor into
  the runtime's $z_0 = E v_0 + b$ form ($E = e_j/K$, $b \leftarrow b - c/K$), verified by the
  parity test to $10^{-8}$. Bug found on the way: least_squares drifted the unused $E$ entries,
  so $E$ is pinned after the fit.
- l0b picks its relaxation rate $a$ by the competition score in physical units
  ($\sum w^2 (1 - s(e))$ with our $\hat\sigma$) instead of SSE in transformed space; $W$ still by
  ridge.
- Held-out runs (`val.*`) are no longer relabeled as training by autotune and the builder; the
  final refit passes `--include-val` / `--final` to use them.
- Not done: stress gate on the propose path (finalize already replaces non-finite output with
  clipped persistence, and the clip bounds the rest); early-tick loss weighting.

### 3. Leave-one-run-out screen (`scripts/screen.py`) and upload u003

Robust score on the held-out run, paired against persistence (mean ± SE over folds):

| system | runs | persistence | l0b_lin | l1 | pick |
|---|---:|---:|---:|---:|---|
| epidemic | 2 | 0.554 | 0.484 (−0.07) | 0.447 (−0.11) | persistence |
| market | 4 | 0.454 | 0.770 (+0.32 ± 0.05) | 0.736 | l0b_lin |
| traffic | 4 | 0.412 | 0.605 (+0.19 ± 0.03) | 0.510 | l0b_lin |
| power_grid | 3 | 0.559 | 0.707 (+0.15 ± 0.02) | 0.487 | l0b_lin |
| supply_chain | 2 | 0.323 | 0.715 (+0.39 ± 0.04) | 0.598 | l0b_lin |
| wildlife | 2 | 0.396 | 0.706 | 0.721 (+0.33 ± 0.06) | l1 |
| reservoir | 2 | 0.423 | 0.597 (+0.18 ± 0.05) | 0.406 | l0b_lin |
| ad_auction | 3 | 0.551 | 0.609 | 0.698 (+0.15 ± 0.07) | l1 |
| social_contagion | 2 | 0.695 | 0.707 (+0.01 ± 0.02) | 0.615 | l0b_lin |
| hospital_queue | 4 | 0.571 | 0.631 | 0.645 (+0.07 ± 0.08) | l1 |

With per-control linear features and 3–4 control vectors, the relax model now beats persistence
by 0.15–0.39 on six systems. l1 wins only where the dynamics are visibly second-order
(wildlife, ad_auction rebound) or fast with queues (hospital). Epidemic still beats nothing:
the wave needs its own model.

u003 = these picks, all 10 systems, `submissions/20260924-2301-u003-screen`. 4,000-tick rollouts
on all four categories finite, 17 s projected for the whole evaluation. Public-upload policy from
here: one factor per system per upload, three uploads per day = thirty free held-out tests on the
true test distribution; the next upload is designed from u003's per-system deltas.

### 4. u003 public scores (Thu 23:05) and the u004 hypothesis test

| system | u002/u001 | u003 | Δ | LOO estimate |
|---|---:|---:|---:|---:|
| traffic | 0.264 | 0.626 | +0.36 | 0.61 |
| market | 0.185 | 0.481 | +0.30 | 0.77 |
| supply_chain | 0.420 | 0.702 | +0.28 | 0.72 |
| wildlife | 0.279 | 0.429 | +0.15 | 0.72 |
| social_contagion | 0.367 | 0.440 | +0.07 | 0.71 |
| reservoir | 0.463 | 0.529 | +0.07 | 0.60 |
| power_grid | 0.545 | 0.593 | +0.05 | 0.71 |
| ad_auction | 0.604 | 0.628 | +0.02 | 0.70 |
| hospital_queue | 0.443 | 0.451 | +0.01 | 0.65 |
| epidemic | 0.272 | 0.272 | 0 | 0.55 |

Mean 0.372 → **0.515**. Where LOO and public agree (traffic, supply_chain) the model transfers.
Where LOO sits 0.2–0.3 above public (market, wildlife, social_contagion, hospital_queue,
power_grid) the test schedules exercise dynamics our four runs do not contain: LOO folds hold out
one of our own hold/pulse/multilevel runs, which are all from reset and short, while the test has
4,000-tick holds and trains. Those five are the priority for Saturday's data.

u004 (`submissions/20260924-2305-u004-test`), one factor per system:

| system | u003 | u004 | question |
|---|---|---|---|
| epidemic | persistence | l0b_lin blend λ = 0.5 | does half relaxation beat persistence on the wave? |
| market | l0b_lin | l1 | which of the two LOO-close kinds transfers? |
| wildlife | l1 | l0b_lin | same |
| ad_auction | l1 | l0b_lin | same |
| hospital_queue | l1 | l0b_lin | same |
| traffic, power_grid, supply_chain, reservoir, social_contagion | l0b_lin | l0b (quadratic + pairwise control features) | do nonlinear equilibrium maps help? |

### 5. u004 results (Thu ~23:40)

| system | u003 | u004 | verdict |
|---|---:|---:|---|
| wildlife | 0.429 | 0.555 | l0b_lin transfers; l1 over-fit the cycle phase (LOO 0.72 was on our own runs) |
| epidemic | 0.272 | 0.327 | half-strength relaxation beats persistence; tune λ |
| hospital_queue | 0.451 | 0.478 | l0b_lin |
| ad_auction | 0.628 | 0.586 | l1 stays |
| market | 0.481 | 0.336 | l0b_lin stays; l1 does not transfer |
| traffic | 0.626 | 0.556 | quadratic features hurt |
| power_grid | 0.593 | 0.550 | quadratic hurt |
| social_contagion | 0.440 | 0.412 | quadratic hurt |
| reservoir | 0.529 | 0.527 | tie |
| supply_chain | 0.702 | 0.700 | tie |

Best-of per system: mean **0.536**. Rules learned: linear control features only (quadratic and
pairwise terms over-fit with 4 short runs); the simple relax model transfers better than the lag
model wherever the two are close locally; blend strength is a real knob on epidemic.
Friday hypotheses (one per system per upload): epidemic λ ∈ {0.3, 0.7}; blend λ = 0.8 on the
systems whose LOO sits far above public (market, power_grid, social_contagion, hospital_queue);
l1 with anchored state vs l0b_lin on ad_auction is settled (l1).

### 6. u005 prepared for Friday morning (`submissions/20260924-2309-u005-lam`)

Single factor: blend strength $\lambda$ in $\hat y = y_0 + \lambda(\text{model} - y_0)$, per
system, against the u003/u004 winner at $\lambda = 1$ (epidemic against $\lambda = 0.5$).

| λ | systems | reason |
|---|---|---|
| 0.7 | epidemic | 0.5 beat 0 (persistence); is more relaxation better still? |
| 0.8 | market, power_grid, social_contagion, hospital_queue, wildlife, reservoir, ad_auction (l1) | local LOO sat 0.1–0.3 above public: over-confident models, shrink toward $y_0$ |
| 1.2 | traffic, supply_chain | LOO and public agreed: test mild extrapolation |

Per-tick optimum is $\lambda^* = e_0/(e_0 - e_m)$ (Sep 24 §1.3); a global $\lambda$ can only
help where model and persistence errors have opposite signs on most ticks. This upload measures
whether that holds per system. Friday's second and third uploads bracket the winner.

### 7. u005 results and end-of-day record

| system | λ | before | u005 | Δ |
|---|---:|---:|---:|---:|
| hospital_queue | 0.8 | 0.4776 | 0.4925 | +0.015 |
| traffic | 1.2 | 0.6259 | 0.6116 | −0.014 |
| supply_chain | 1.2 | 0.7016 | 0.6444 | −0.057 |
| market | 0.8 | 0.4806 | 0.3422 | −0.138 |

Shrinkage toward $y_0$ is a loss wherever persistence is catastrophically wrong (market volume:
$|e_0| \approx 88$ vs $|e_m|$ of a few units, so $0.2\,|e_0|$ of added error every tick). Rule:
$\lambda = 1$ except where a paired test shows opposite-sign errors (epidemic 0.5, hospital 0.8).

**Best public score per system, end of Thu Sep 24** (mean 0.537, from 0.342):
supply_chain 0.7016 l0b_lin · ad_auction 0.6283 l1 · traffic 0.6259 l0b_lin · power_grid 0.5932
l0b_lin · wildlife 0.5550 l0b_lin · reservoir 0.5286 l0b_lin · hospital_queue 0.4925 l0b_lin λ0.8 ·
market 0.4806 l0b_lin · social_contagion 0.4404 l0b_lin · epidemic 0.3265 l0b_lin λ0.5.

The action-by-action account with the mathematics of each effect is in
`docs/study/actions_and_effects_2026-09-24.md`.

### 8. Architecture work (Fri 00:30–01:30, no credits)

Question: does hidden state help on the data we have? Leave-one-run-out, robust score:

| system | l0b_lin | l2 (modal state space) | l0b_lin + delay search |
|---|---:|---:|---:|
| market | 0.770 | 0.757 | 0.770 (delay 0) |
| traffic | 0.605 | 0.549 | 0.625 (delay 8 on 3 of 4 folds) |
| power_grid | 0.707 | 0.551 | 0.707 |
| supply_chain | 0.715 | 0.560 | 0.721 (delay inconsistent) |
| wildlife | 0.706 | **0.748** | |
| reservoir | 0.597 | 0.442 | |
| ad_auction | 0.609 | 0.627 (± 0.17) | |
| social_contagion | 0.707 | 0.597 | 0.707 |
| hospital_queue | 0.631 | 0.593 | 0.645 (inconsistent) |
| epidemic | 0.484 | 0.552 | |

Generic hidden state (2 complex modes + 3 real per observable, rollout-error fit) loses to the
first-order relax model on 7 of 10 with four short runs: more parameters than the data can pin.
It wins on wildlife, the one system with a measured cycle. A global pure delay on the controls
gains ≤ 0.02 and is only consistent on traffic (8 ticks). Conclusion: on the current data the
model class is not the bottleneck; long-horizon and history data are (Saturday's purchase).

Changes made anyway, all tested, parity to $10^{-8}$:
- l2 initial hidden state now $x_0 = C^{+}(v_0 - c - D\phi(u_0))$, the minimum-norm state
  consistent with the observed initial, replacing a free $E v_0 + e_0$; pole radius cap 1.0.
- Runtime "post" rules, exact physics after the model: `le_control` (ad_auction spend ≤ 1.04 ×
  max(budget_cap$_t$, budget_cap$_{t-1}$); the fitted model violated the cap on 73 % of ticks on a
  sustained schedule) and `integrate` (capped stock: reservoir level$_{t}$ = clip(level$_{t-1}$
  + 0.921 inflow − 0.905 outflow − 1.10, 0, 956); pooled fit of Δlevel, residual ≈ measurement
  noise). The reservoir rule is mixed leave-one-run-out (hold 0.84 → 0.64, pulse 0.53 → 0.58) so
  it ships as a public hypothesis, not a default. Inflow has no season at any period 200–730 and
  no trend: forecast its mean.
- `scripts/build_cfg.py`: build from an explicit per-system JSON (kind, cfg, λ, clip margin,
  post rules), copied into the build folder for reproducibility.

### 9. u006 prepared for Friday's first slot (`submissions/20260924-2326-u006`, config in `plans/u006.json`)

One factor per system against the best-of:

| system | change | basis |
|---|---|---|
| ad_auction | spend ≤ 1.04·max(cap$_t$, cap$_{t-1}$) rule on top of l1 | exact physics; model broke it on 73 % of ticks |
| wildlife | l2 modal state space | only system where hidden state won LOO (0.748 vs 0.706) |
| reservoir | level = capped integral of inflow − outflow | mass balance holds in data; mixed LOO |
| epidemic | λ = 0.35 | parabola through λ ∈ {0, 0.5, 1} peaks at 0.37 |
| hospital_queue | λ = 0.6 | bracket the 0.8 winner from below |
| traffic | 8-tick control delay | chosen on 3 of 4 folds, +0.02 LOO |
| supply_chain, social_contagion | clip margin 1× (was 3×) | integrators: does the tighter box help or hurt? |
| market, power_grid | clip margin 10× (near hard bounds only) | same question, other direction |

Rollouts finite on all four categories; cap rule holds with zero violations; full evaluation 14 s.

### 10. Score audit: where the 0.54 comes from (Fri morning, no credits)

Question: our best-of mean is 0.537 against a leaderboard top of 0.728. Which part of the pipeline
loses the points? Method (scripts and raw output in `docs/study/audit_2026-09-25/`): re-fit
`l0b_lin` in-sample on every run we own and score it per run and per observable with our own
$\hat\sigma$; roll the *uploaded* `predict.py` files (best-of from u003/u004/u005) over eval-shaped
4,000-tick schedules of all four categories and count ticks predicted outside the observed data
range; test the data for exogenous structure.

**1. The in-sample fit is already the ceiling.** Per-run scores of `l0b_lin` on its own training
data (own $\hat\sigma$, mean over observables): epidemic 0.47 / 0.71, market 0.80–0.95, traffic
0.66–0.93, power_grid 0.69–0.82, supply_chain 0.73 / 0.96, wildlife 0.82 / 0.87, reservoir 0.70 /
0.71, ad_auction 0.76–0.82, social_contagion 0.77 / 0.89, hospital_queue 0.74–0.90. A model that
cannot fit what it was trained on cannot transfer. The one-pole map
$v_{t+1} = a v_t + (1-a)\,W\phi(u_t)$ has no representation for what the data show:

| system | shape in the data | what the one-pole fit does |
|---|---|---|
| epidemic | wave: cases 188 → 512 (t=20) → 69 (t=60); second wave after release at t=120 | $\tau = 88$ decay through the hump; ≈ persistence |
| power_grid | load 93 → 76 (t=20) → 99 (t=60) under a constant hold (thermostat rebound) | picks $a = 0.4$: an instantaneous map, no rebound |
| reservoir | level is a capped integrator (481 → 941 then flat; −1.3/tick under release 12) | relax with $\tau = 40$ toward a linear level |
| traffic | exit flows rise 20–40 ticks *after* the admission pulse ends (pipeline) | $\tau = 1.7$ on flows: attributes flow to the wrong control sign |
| hospital_queue | queue saturates at ≈ 330 (beds), wait_time then explodes 0 → 220 | log1p-linear map, no capacity |
| wildlife | prey 86 → 195 (t=20) → 122: overshoot | monotone relax |

Fri §8 concluded "the model class is not the bottleneck" from a leave-one-run-out comparison of
two models of the *same* class; the in-sample residuals say otherwise.

**2. Extrapolation of the equilibrium map is the largest single leak.** Uploaded models on
eval-shaped schedules, fraction of the 4,000 ticks predicted more than 5 % of range outside the
observed data range:

| system (upload) | observable | data range | predicted | ticks outside |
|---|---|---|---|---:|
| hospital_queue (u005) | wait_time | 0–322 | up to 1,030 | 22–38 % (order) |
| hospital_queue (u005) | queue | 23–333 | down to 7 | 28–39 % (order) |
| traffic (u003) | speed_a / speed_b | 11–49 | up to 84 / 89 | 74 % (one sustained) |
| market (u003) | price | 78–110 | 143 / below 78 | 32–79 % |
| power_grid (u003) | load | 64–158 | below 64 | 72 % (one sustained) |

Mechanism: $W$ is linear in $\tilde u$ in log1p / logit space, so a control corner we never held
multiplies out exponentially, and the 3× clip margin (Fri §2) lets it run to the box edge
(wait_time clip = 1,287). Every test category holds control vectors we do not own: sustained
holds at unseen levels, recovery at $\alpha \in [0.7, 1]$ per control, composition with single
controls moved. We have 2–16 distinct control vectors per system, mostly 2–4.

**3. Reservoir inflow is a deterministic season, not noise.** Runs 0 and 1 agree tick for tick
(correlation 0.991, mean |difference| 0.25 on an sd of 1.6): the phase is locked to reset, so every
test episode sees the same inflow$(t)$. Two-harmonic least squares on the 400-tick run, period
scanned 30–1,200 at 0.1:

$$\text{inflow}(t) = 11.43 + 2.154\sin\tfrac{2\pi t}{67.8} + 0.293\cos\tfrac{2\pi t}{67.8}
 - 0.018\sin\tfrac{4\pi t}{67.8} - 0.006\cos\tfrac{4\pi t}{67.8}$$

Residual sd 0.19 in-sample, 0.145 on the held-out 120-tick run. Score on that run (own
$\hat\sigma = 1.57$): sinusoid 0.919, constant 0.553, persistence 0.535. Fri §8 searched periods
200–730 only and reported "no season". Inflow is one quarter of the reservoir observables and the
input to the level integrator. No other observable shows a clean exogenous period on the segments
we own (power_grid load's 45-tick component is the rebound, not a driver).

**4. $y_0$ carries little information on several observables.** market volume 90–110 → 16–19 at
t=5 → 2–3 at t=20 in all four runs; traffic flows → 0 at t=5 in all four; supply_chain retail
105 → 0; hospital discharges → 0 at t=5 in three of four. The hidden state, not the reported
initial, sets the first 20 ticks. Anchoring $v_0 = g(y_0)$ with one pole is the right shape for a
decay, wrong for a burst (discharges, flows).

**5. The local screen is not calibrated.** $\hat\sigma$ is the std over everything we collected
(epidemic 86 cases, reservoir level 258, hospital queue 127, wait 73), so local scores read
0.1–0.3 above public; with 2–4 runs the fold SE is ±0.05, so most screen "wins" are ties.

**6. Data.** 440–520 ticks per system, 2–4 runs, nothing past tick 400, 15,080 credits (75 %)
unspent. The sustained category holds for thousands of ticks we have never observed.

Decisions:

1. Reservoir: ship inflow$(t)$ as an exogenous post rule (table or the formula above, phase from
   tick 0 = first action after reset) and feed it into the level integrator. Expected ≈ +0.09 on
   reservoir from inflow alone.
2. Extrapolation guard before any further upload: clip margin ≤ 1× on hospital_queue, traffic,
   market, power_grid, and bound the equilibrium $W\phi(u)$ in transformed space to the observed
   range of $v$ (plus a small margin), so an unseen corner relaxes to the nearest observed level
   instead of past it. The u006 factors (clip 1× vs 10×) test only the box, not the map.
3. Structural models where the residual shape is known: epidemic SIR-type on the 520 ticks we own
   (β, γ, hospitalisation lag are identifiable from one wave plus one restricted wave); traffic
   transport delay on flows; power_grid second-order (rebound); hospital_queue capacity clamp.
   These are the four systems whose in-sample fit is worst.
4. Saturday's purchase, in this order: one long hold (≥ 1,000 ticks) per slow system for the
   sustained level; single-control pulses at $\alpha \approx 0.85$ for composition; interior
   levels for the corner-degenerate systems. Data for the map first, more model kinds second.
5. Report per-observable local scores; treat leave-one-run-out differences under 0.05 as ties.

Rejected: blend $\lambda$ as a lever. u005 showed shrinkage toward $y_0$ costs up to −0.14 where
persistence is far off; the leak is in the equilibrium map and the missing dynamics, not in the
mix. Still to read: the per-category split of our uploads in the portal (sustained vs sequence
transfer), which would rank items 1–4 by category.

### 11. Category bands from the portal receipts, and the plan they imply (Fri morning)

Every receipt is JSON at `/submissions/<id>` with `bands = {overall, id, extrapolation}`;
`overall = 0.25·id + 0.75·extrapolation`, so `id` is the sustained category and `extrapolation`
is the mean of order, recovery and composition ("sequence transfer" on the standings). Saved in
`tune/bands.json`. Our standing after Thursday's sweep: #16, 0.5047 overall, 0.5279 sequence,
hence sustained 0.435. #1: 0.751 overall, 0.760 sequence, sustained ≈ 0.72.

Best-of per system (receipt id, kind), with the persistence bands from u001 for reference:

| system | best | overall | sustained | sequence | pers. sustained | pers. sequence |
|---|---|---:|---:|---:|---:|---:|
| supply_chain | 843 l0b_lin | 0.702 | 0.508 | 0.766 | 0.462 | 0.406 |
| ad_auction | 836 l1 | 0.628 | 0.537 | 0.659 | 0.484 | 0.467 |
| traffic | 844 l0b_lin | 0.626 | 0.446 | 0.686 | 0.308 | 0.249 |
| power_grid | 840 l0b_lin | 0.593 | 0.499 | 0.625 | 0.412 | 0.511 |
| wildlife | 856 l0b_lin | 0.555 | 0.597 | 0.541 | 0.191 | 0.233 |
| reservoir | 841 l0b_lin | 0.529 | 0.563 | 0.517 | 0.361 | 0.342 |
| hospital_queue | 859 l0b_lin λ0.8 | 0.493 | 0.370 | 0.533 | 0.311 | 0.487 |
| market | 839 l0b_lin | 0.481 | 0.474 | 0.483 | 0.129 | 0.204 |
| social_contagion | 842 l0b_lin | 0.440 | 0.410 | 0.451 | 0.187 | 0.345 |
| epidemic | 848 l0b_lin λ0.5 | 0.327 | 0.297 | 0.336 | 0.273 | 0.271 |

Where the gap to #1 sits, by weight: sustained $(0.72 - 0.435) \times 0.25 = 0.07$; sequence
$(0.76 - 0.53) \times 0.75 = 0.17$. **Seventy percent of the gap is in the sequence categories**,
i.e. in the transient dynamics after a change (order, recovery, composition), not in the long-hold
level. That matches §10 finding 1 (the one-pole map has no transient shape) more than finding 2.
Sustained is still the weakest band on 8 of 10 systems, and on supply_chain (0.51 vs 0.77),
traffic (0.45 vs 0.69) and hospital_queue (0.37 vs 0.53) the sustained band is barely above
persistence: the fitted equilibrium is wrong at the held levels (§10 finding 2).

Other reads from the bands:
- Blend λ < 1 loses sustained hardest (market λ 0.8: 0.474 → 0.323; supply_chain λ 1.2:
  sequence 0.766 → 0.691). λ stays at 1 except epidemic and hospital, as decided in §7.
- Quadratic features (u004) lost both bands on traffic and power_grid; l1 on market lost both.
- Epidemic and social_contagion are below 0.45 in both bands: those two need a new model, not tuning.

Plan, ordered by expected points per unit of work, credits separate:

**Friday (no credits; 3 slots per system):**
1. Extrapolation guard in the runtime: clamp the equilibrium $W\phi(u)$ to the observed range of
   $v$ plus 0.25 × range, and clip margin 1× on hospital_queue, traffic, market, power_grid.
   Replace u006's "clip 10×" factors on market and power_grid with this. Local check: the
   out-of-range fraction on eval-shaped schedules (§10 table) must go to ≈ 0.
2. Reservoir: inflow$(t)$ exogenous rule (§10) and the level integrator driven by it, with the
   irrigation control in the balance ($\Delta\text{level} = c_1\,\text{inflow} - c_2\,\text{outflow}
   - c_3\,\text{irrigation} + b$, pooled least squares on both runs).
3. Epidemic: SIR-type grey box. Two observables, two runs, two waves; fit $\beta(u), \gamma$,
   hospitalisation fraction and lag by least squares on the 520 ticks; ship as the epidemic model
   if it beats λ = 0.5 on both runs in-sample and stays finite over 4,000 ticks.
4. Hospital_queue: queue capacity clamp (≈ 333) and wait_time as a function of queue above
   capacity; traffic: exit flow = delayed, smoothed admission (l1 with the 8-tick delay from u006);
   power_grid: second-order load (rebound). Each is one factor for a Friday or Saturday slot.
5. Autotune: store the bands with every score; screen on both bands; treat leave-one-run-out
   differences under 0.05 as ties; add the out-of-range gate to the build.

**Saturday purchase (needs a go; 15,080 credits left):** the sequence categories need history
data and the sustained category needs held levels we own. Per system, in this order:
- composition: single-control blocks at $\alpha \approx 0.85$, 60 ticks each, then the joint
  pulse, from one reset ($60(m+1)$ ticks: 180–420 per system);
- recovery: a pulse train with gaps 10 / 40 / 150 (about 350 ticks);
- sustained: one long hold, 600–1,000 ticks, at an interior level ($\alpha = 0.5$) for the five
  slow systems (reservoir, supply_chain, social_contagion, wildlife, epidemic) and 400 for the rest.
Roughly 1,100–1,600 per system, 300–500 kept in reserve. Priority by band gap: supply_chain,
traffic, hospital_queue, power_grid, then the rest.

**Sunday–Monday:** refit every kind on the full data, one-factor public tests on the remaining
slots, then the Monday 12:00 final upload of the best-of, and keep one slot per system for a
Tuesday correction.

### 11. Acting on the score audit (Thu 23:40, no credits)

Two runtime/model changes tested leave-one-run-out with clip margin 1×:

| system | l0b_lin | + equilibrium bound | note |
|---|---:|---:|---|
| epidemic | 0.484 | 0.537 | |
| market | 0.770 | 0.805 | |
| social_contagion | 0.718 | 0.733 | |
| hospital_queue | 0.638 | 0.650 | |
| others | tie | tie | |

Equilibrium bound: $W\phi(u)$ is clipped in transformed space to the observed range of $v$ ±25 %
of that range, so an unseen control corner relaxes to the nearest observed level. On by default
in l0b (`eq_margin=0.25`, exported as `q_lo`/`q_hi`, runtime clips the same way).

Reservoir, per observable, held-out run:

| rule set | level | inflow | outflow | quality | mean (hold / pulse) |
|---|---:|---:|---:|---:|---|
| none | 0.84 / 0.53 | 0.55 / 0.56 | 0.53 / 0.46 | 0.70 / 0.65 | 0.654 / 0.548 |
| inflow season | same | **0.92 / 0.91** | same | same | 0.746 / 0.634 |
| + level integrator | 0.70 / 0.57 | | | | 0.710 / 0.646 |

Inflow season (period 67.8, two harmonics, phase from the first tick after reset) is a default
post rule now: +0.09 on the reservoir mean in both folds. The level integrator is net negative
(−0.05 mean) and stays optional. u006 rebuilt (v2) with the bound everywhere, clip 1× on the four
leaking systems, the inflow rule, the spend cap, l2 on wildlife, λ at the known winners.

### 12. Traffic transport delay; power_grid second order (Thu 23:50, no credits)

Traffic, l0b_lin with one global control delay $d$ (clip 1×, equilibrium bound), leave-one-run-out:

| $d$ | 0 | 8 | 16 | 24 | 32 | 40 |
|---|---:|---:|---:|---:|---:|---:|
| LOO | 0.623 | 0.625 | 0.663 | **0.690** | 0.659 | 0.583 |

At $d = 24$ all four observables improve (flows 0.63 → 0.70/0.71, speed_a 0.55 → 0.65). This is the
pipeline the brief describes: vehicles admitted now exit 20–40 ticks later. u006 rebuilt (v3) with
$d = 24$ on traffic. A per-observable delay (flows vs speeds) is the next refinement.

Power_grid, state space with fewer modes: l2 with one lag bank and one complex mode 0.654, two
complex modes 0.613, two banks and one mode 0.709, against l0b_lin 0.707. The rebound is real
(poles near $r = 0.97$–$0.99$ with rotation appear in every fit) but the extra freedom costs as
much as it gains with three runs. Parked until Saturday's long holds.

### 12. Epidemic: minimal SIRS + hospital grey box (Fri 00:00, no credits), u007

Question: §10 showed the one-pole map cannot produce the wave; can a five-state compartment model
fit the 520 ticks we own and stay sane over 4,000? Family `gtlab/ode/epidemic_sirs.py`
(numpy-only, inlined into predict.py by flatpack; fit script `scripts/fit_epidemic_sirs.py`).

States $S, E, I, R$ (fractions) and $H$ (beds), one population, RK4 with two substeps per tick:

$$\lambda = \beta\,(1 - c_s\,u_{\text{closure}})(1 - c_m\,u_{\text{mask}})\,I,\qquad
v = k_v\,u_{\text{vac}} \big/ (1 + k_c H / h_{\text{ref}})$$
$$\dot S = -\lambda S - vS + R/\tau_w,\quad \dot E = \lambda S - \sigma E,\quad
\dot I = \sigma E - \gamma I,\quad \dot R = \gamma I + vS - R/\tau_w,\quad
\dot H = f_h\,N\,\gamma I - H/L$$
$$\text{daily\_cases} = N\sigma E,\qquad \text{hospital\_load} = H.$$

Reset rule: $E_0 = \text{cases}_0/(N\sigma)$, $I_0 = E_0\sigma/\gamma$, $R_0 = r_0$ fixed,
$S_0 = 1 - E_0 - I_0 - R_0$, $H_0$ = observed. Thirteen parameters, log-space bounds, cauchy
least squares with ten Latin-hypercube starts (existing `gtlab.ode.fit`), 200 s budget.

| fit on | scored on | cases | hospital | persistence |
|---|---|---:|---:|---|
| pulse120_280 (400) | hold_rec (120), held out | 0.712 | 0.621 | 0.42 / 0.47 |
| hold_rec (120) | pulse120_280 (400), held out | 0.458 | 0.377 | 0.79 / 0.53 |
| both (520) | hold_rec | 0.871 | 0.687 | |
| both (520) | pulse120_280 | 0.877 | 0.886 | |

Own $\hat\sigma$ = (86, 33). The one-pole `l0b_lin` in-sample was 0.49 / 0.45 and 0.81 / 0.60
(§10). The wave, the crash and the endemic plateau are reproduced; the 120-tick unrestricted
wave alone cannot identify the intervention gains (second row), so both runs are used.

Fitted values: $N = 23{,}434$, $\beta = 2.37$, $\sigma = 0.281$, $\gamma = 1.5$ (upper bound),
$\tau_w = 72.7$, $c_s = 0.079$, $c_m = 0.094$, $k_v = 0$, $f_h = 0.0233$, $L = 29.4$,
$r_0 = 0.069$; the clinic term is inactive ($k_v = 0$). Caveats: $\gamma$ on its bound and the
small intervention gains say the restricted wave in our data is explained mostly by its lower
initial cases, not by the controls; the recovery category (repeated closure + mask + vaccination
pulses) is where this model is least trusted. That is what u007 measures.

4,000-tick rollouts on all four categories: finite, cases within 32–456, hospital 27–190,
0.3 s per episode. Packaged predict.py reproduces the in-sample scores above to three decimals.

u007 = u006 v3 (nine systems unchanged, `submissions/20260924-2341-u006v3`) + epidemic as this
ODE at $\lambda = 1$: `submissions/20260924-2346-u007`, registered. One factor (epidemic) against
u006. Next for epidemic: a second restricted run of 200+ ticks from a different initial in
Saturday's purchase, so the control gains are pinned by data rather than by the initial cases.

### 13. Hospital queue capacity rule (Fri 00:00, no credits)

Queue never exceeds 333 in 440 ticks across three runs (top values 332.3–333.0), and the brief says
overflow is referred elsewhere: a hard bed capacity. Runtime rule `le_const` (queue ≤ 333) as the
hospital_queue default. Leave-one-run-out on l0b_lin (clip 1×, bound): 0.650 → 0.667, no fold
worse. Not in u006/u007 (built earlier); it is the hospital factor for u008.

### 13. Saturday purchase design, phase `p3` (Fri 00:30; materialized, NOT bought)

Target: the two bands of §11. Sequence transfer (75 % of the score) needs history and
composition data; sustained needs held levels we own. Generator `scripts/make_p3.py` (seed 3),
schedules in `plans/p3.json`, materialized into `data/<sys>/plan.json`; `p3` added to
`design.PHASES`; `data/<sys>/budget.json` now caps `p3` at balance − 300 and keeps 300 in reserve.

Per system, three runs from three resets:

| run | schedule | category | ticks |
|---|---|---|---:|
| `p3.compose` | each control alone at $\alpha = 0.85$ (recovery + 0.85·(pulse − recovery)) for $D$, recovery $D/2$; then the joint $\alpha = 0.85$ level for $D$, recovery $D$. $D$ = 60 (2–3 controls), 45 (3), 40 (4), 30 (6) | composition | 291–330 |
| `p3.train` | recovery baseline, 6 pulses with $\alpha \sim U(0.7, 1)$ per control, lengths 10–30, log-uniform gaps 10–150, lead 20 | recovery | 350 |
| `p3.hold_mid` | one hold at $\alpha = 0.5$ | sustained | 450 slow (reservoir, supply_chain, social_contagion, wildlife, epidemic), 400 others |

Why these: the composition run gives $m + 1$ new control vectors on the same reset, so the
additivity of the equilibrium map (§10 finding 2) is tested directly and the map gets its first
interior single-control levels; the train has gaps spanning the $\tau$ range we measured (2–150),
which is what the recovery category scores; the interior hold gives one level between recovery
and pulse for the sustained band and, at 450 ticks, the first look past tick 400 on the slow
systems. Rejected: a 1,000-tick hold per slow system (would leave < 150 in reserve on five
systems); $\alpha = 1$ single-control blocks (the tests use 0.7–1, and 1 is already in p2).

Cost (server balance after p2 → after p3):

| system | compose | train | hold | total | balance | left |
|---|---:|---:|---:|---:|---:|---:|
| epidemic | 291 | 350 | 450 | 1,091 | 1,480 | 389 |
| market | 300 | 350 | 400 | 1,050 | 1,560 | 510 |
| traffic | 330 | 350 | 400 | 1,080 | 1,560 | 480 |
| power_grid | 320 | 350 | 400 | 1,070 | 1,500 | 430 |
| supply_chain | 330 | 350 | 450 | 1,130 | 1,480 | 350 |
| wildlife | 291 | 350 | 450 | 1,091 | 1,480 | 389 |
| reservoir | 320 | 350 | 450 | 1,120 | 1,480 | 360 |
| ad_auction | 291 | 350 | 400 | 1,041 | 1,500 | 459 |
| social_contagion | 291 | 350 | 450 | 1,091 | 1,480 | 389 |
| hospital_queue | 330 | 350 | 400 | 1,080 | 1,560 | 480 |

Total 10,844 credits; 4,236 stay in reserve for Sunday/Monday. Command, once approved
(USES 10,844 CREDITS):

```powershell
$env:GT_ALLOW_SPEND = "1"
python -m gtlab.cli collect --all --phase p3 --spend --max-steps 1200 --yes
Remove-Item Env:GT_ALLOW_SPEND
```

Order of purchase if we split it: compose first on every system (the composition band and the
map), then train, then hold_mid.

Addendum (Fri 00:50): p3 is bought in two halves. Half 1, Saturday: `compose` + `train` only
(641–680 per system, 6,594 total), the two categories with no data at all. Half 2, Sunday:
the sustained buy sized after the refit and Saturday's public bands: 1,000-tick holds on the
slow five if the long-horizon residual is the issue, interior holds where the committee
disagrees otherwise; 300 stays in reserve. `hold_mid` stays materialized in p3 but is not run
on Saturday. Reason: one 450-tick hold at one level does not answer the "nothing past tick 400"
gap, and spending 72 % of the balance before u006/u007 land forecloses the adaptive slice.
Half-1 command (USES 6,594 CREDITS): `collect --all --phase p3 --only compose,train --spend
--max-steps 700 --yes` with `GT_ALLOW_SPEND=1`.

### 14. Minimal grey-box ODE per system (Fri afternoon, no credits)

Question from §10–11: the one-pole map cannot fit its own runs and the sequence categories hold
70 % of the gap. Does a minimal mechanistic model per system (≤ 12 parameters, ≤ 8 states,
every observed initial used, RK4 with two substeps) beat it on leave-one-run-out with the data
we own? Method: one shared harness, `scripts/ode_lab.py`, fits a family with the existing
`gtlab.ode.fit` (log-space bounds, cauchy least squares, ten Latin-hypercube starts, 240 s),
scores each held-out run per observable against `l0b_lin` fitted on the same folds (clip 1×,
equilibrium bound) and persistence, then rolls 4,000 ticks on eval-shaped schedules of all four
categories. Gates in `scripts/assemble_picks.py`: LOO mean above `l0b_lin` by more than 0.05
(our $\hat\sigma$ inflates scores, so less is a tie), never below persistence on a fold, finite,
under 1 s per episode, excursion beyond the observed range under 1× the range (the runtime clip).
Equations, parameter tables, rejected structures and caveats per system are in
`plans/<system>_min_notes.md`; families in `gtlab/ode/<system>_min.py`; lab reports and model
documents in `plans/<system>_<system>_min.json` / `_doc.json`.

| system | states / params | LOO ode | LOO l0b_lin | LOO pers. | in-sample | excursion | gate |
|---|---|---:|---:|---:|---:|---:|---|
| ad_auction | 3 / 12 | 0.849 | 0.613 | 0.553 | 0.910 | 0.00 | pass |
| supply_chain | 7 / 12 | 0.857 | 0.716 | 0.31 | 0.910 | 0.56 | pass (fold 2 tie) |
| hospital_queue | 6 / 12 | 0.827 | 0.655 | 0.572 | 0.855 | 0.00 | pass |
| power_grid | 6 / 11 | 0.812 | 0.713 | 0.561 | 0.852 | 0.63 | pass |
| reservoir | 4 / 12 | 0.810 | 0.601 | 0.42 | 0.903 | 0.12 | pass |
| traffic | 8 / 12 | 0.788 | 0.626 | 0.41 | 0.823 | 0.20 | pass (all 4 folds) |
| wildlife | 6 / 10 | 0.772 | 0.691 | 0.39 | 0.918 | 0.04 | pass |
| market | 5 / 12 | 0.860 | 0.814 | 0.45 | 0.926 | 0.16 | tie (+0.046) |
| social_contagion | 6 / 12 | 0.769 | 0.739 | 0.70 | 0.915 | 0.08 | tie (+0.03) |
| epidemic (§12) | 5 / 13 | 0.53 | 0.54 | 0.55 | 0.830 | 0.00 | in-sample only |

What the structures are, one line each (full derivations in the notes files):

- **reservoir**: level = capped integrator of season(t) − head-limited delivery
  $c_{out}(L/500)^{p}$ (smooth min with release + irrigation) − seepage, hard cap 941.3; outflow =
  delivery + algebraic spill when full; quality first-order in aeration; irrigation return flow.
- **supply_chain**: production = effort·(p₀ + committed), two-stage commitment following orders;
  dispatch = min(order, stock, capacity); two-stage conveyor at rate $k_q/(1 + k_l\,\text{lead})$;
  arrivals gated by a heat/wear state driven by (1 − maintenance); retail drains at a demand rate;
  supplier stock capped at 362.
- **traffic**: per route three first-order stages approach → junction → exit (mean transit
  ≈ 12 ticks), admission = demand·ramp cut by a finite approach buffer, junction service split by
  signal timing, exit capacity cut by lane closure and raised by clearance, speed relaxes to
  $v_{free}/(1 + n/n_{ref})$. Toll and freight fitted to zero effect.
- **power_grid**: desired demand $d_0 - d_1\,\text{price}$; a price step kicks a damped oscillator
  (period 62, ζ 0.28, tanh-saturated impulse) = the thermostat rebound; reserve request lags and is
  delivered ∝ interconnector; frequency = 50 + asymmetric droop on the balance; renewable share
  ∝ interconnector, suppressed by reserve. Physical clips: rebound ±32, frequency ±2.5 Hz.
- **wildlife**: per region delayed-logistic prey $P' = rP/(1+P/c) - d_l L P - h\,\text{hunting}\,P$
  with a lagged density $L$ (gives the overshoot), predators relax to a weakly prey-dependent
  capacity; south = north scaled. Habitat and corridor fitted to zero effect (never varied alone).
- **hospital_queue**: waiting → two assessment stages (finite chairs) → treatment (finite beds)
  → discharges; throughput = staffing·(1 + overtime gain)·(1 − fatigue); arrivals gated at the
  322 queue cap; wait = filtered W/(admissions); fatigue integrates overtime.
- **ad_auction**: win probability $w = b^n/(b^n + b_0^n)$, impressions = min(w·audience,
  cap/price), spend = price·impressions ≤ cap, reachable pool depletes with impressions and
  returns with τ 48, two-stage purchase queue with a fulfilment ceiling. Reset state is empty
  (conversions are 0 at t = 0–1 in every run; the observed initial is a reading, not a state).
- **market**: reset order backlog draining with ratio 0.69/tick (volume), dealer capacity
  first-order toward $d_0 e^{-a_t \tau - a_r r}$ with asymmetric shrink/refill, price
  $P' = (m - k_r r)e^{-a_x \tau}$ with momentum $m$ and a floor at 73.4 (tax freezes trading).
- **social_contagion**: per community loyal / incentive-led / onboarding compartments with a pool;
  recruitment ∝ seeding·(1 + k·incentive)·share + word of mouth, saturating; incentive-led members
  churn fast when the incentive stops. Mix half-point fixed at 0.7 (unidentifiable from
  incentive ∈ {0, 2}); a free half-point drained the community on long low-incentive holds.

All ten documents package through flatpack into predict.py + model.json; the packaged predictors
reproduce the in-sample scores above and run 0.28–0.38 s per 4,000-tick episode, finite on every
category. Common limits across the notes: two-run systems have one uninformative fold (the fold
fit on the recovery hold alone cannot identify control gains), several controls were never varied
alone (habitat, corridor, followup, charging, toll, freight: fitted to zero), and event-like
observables (discharges, flows, shipments) cap near 0.6–0.85 under any smooth model. Saturday's
composition blocks are exactly the data that pins those gains.

Decision: the seven gate passes replace `l0b_lin` as the candidate for their systems in the next
public upload; market and social_contagion (ties locally) go as public one-factor tests; epidemic
stays the SIRS. Picks in `plans/picks_ode.json`. Ensembles (median of ODE and l0b) are the fallback
where the public bands disagree with the LOO.

### 14. u008: grey-box models on all ten systems (Fri 19:30, no credits)

`submissions/20260925-1925-u008-ode`, config `plans/u008.json`: every system's factor against
u006 is its own minimal grey-box ODE (≤ 12 parameters, fitted in the lab harness, gated on
leave-one-run-out against l0b_lin with clip 1× and the equilibrium bound, same folds). Seven pass
the +0.05 gate (ad_auction 0.849 vs 0.613, hospital_queue 0.827 vs 0.655, power_grid 0.812 vs
0.713, reservoir 0.810 vs 0.601, supply_chain 0.857 vs 0.716, traffic 0.788 vs 0.626, wildlife
0.772 vs 0.691); market (0.860 vs 0.814) and social_contagion (0.769 vs 0.739) are local ties and
go in because the public score is the only tie-breaker; epidemic is the SIRS model from u007.
Runtime rules (spend cap, queue cap, inflow season) stay on top. All 4,000-tick rollouts finite on
the four categories, none more than half a data range outside the observed range, 0.27–0.37 s per
episode, 127 s projected for the full evaluation.

## 2026-09-26 (Sat, early)

### 1. u008 public scores and u009 design

| system | best before | u008 (grey-box) | Δ |
|---|---:|---:|---:|
| ad_auction | 0.6283 | 0.8304 | +0.202 |
| reservoir | 0.5286 | 0.7819 | +0.253 |
| traffic | 0.6259 | 0.7556 | +0.130 |
| power_grid | 0.5932 | 0.7535 | +0.160 |
| supply_chain | 0.7016 | 0.7526 | +0.051 |
| wildlife | 0.5550 | 0.6297 | +0.075 |
| hospital_queue | 0.4925 | 0.6047 | +0.112 |
| social_contagion | 0.4404 | 0.5176 | +0.077 |
| epidemic | 0.3265 | 0.4779 | +0.151 |
| market | 0.4806 | 0.4228 | −0.058 |

Mean 0.537 → **0.653**. Structure from the briefs with ≤ 12 parameters beat the data-driven
ladder on nine of ten. Market is the exception: its grey box loses to the relax model.

u009 (`submissions/20260926-0031-u009`, `plans/u009.json`), one factor per changed system:
market, social_contagion, hospital_queue → per-tick median of {grey box, l0b_lin (clip 1×,
bound)}; supply_chain → grey box v6 (retail cap, level-dependent sales); wildlife stays on the
grey box alone (a two-member median is a mean, and a mean across phases damps a cycle, which the
metric penalises: Sep 24 §16 D). Others unchanged.

### 2. u008 category bands (portal: sustained / sequence transfer)

| system | sustained | sequence | gap |
|---|---:|---:|---:|
| ad_auction | 0.826 | 0.832 | 0 |
| reservoir | 0.833 | 0.765 | +0.07 |
| traffic | 0.647 | 0.792 | −0.15 |
| power_grid | 0.715 | 0.767 | −0.05 |
| supply_chain | 0.577 | 0.811 | −0.23 |
| wildlife | 0.646 | 0.624 | +0.02 |
| hospital_queue | 0.574 | 0.615 | −0.04 |
| social_contagion | 0.444 | 0.542 | −0.10 |
| epidemic | 0.469 | 0.481 | −0.01 |
| market | 0.367 | 0.441 | −0.07 |

Friday sweep: #1 0.7693, #10 0.7081; 0.653 sits about 13th. Sequence transfer (order, recovery,
composition) is now level with or above sustained on the strong five, so the remaining hole is
sustained on supply_chain, traffic, market, social_contagion, epidemic: held levels we have never
observed (nothing past tick 400, interior levels only from the short multilevel segments).
Decision: the Saturday purchase includes the interior hold (`hold_mid`, 400–450 ticks at α = 0.5)
on those five, on top of composition blocks and pulse trains everywhere. Total ≈ 8,800.

## 2026-09-26 (Sat) — u008 scored; strategy from the bands

### 15. u008 public result and the per-system plan

u008 (grey box on all ten, §14) scored Sat 00:20. Bands from the receipts (sustained / sequence),
previous best-of in brackets:

| system | u008 | sustained | sequence | before | Δ |
|---|---:|---:|---:|---:|---:|
| ad_auction | 0.830 | 0.826 | 0.832 | 0.628 | +0.20 |
| reservoir | 0.782 | 0.833 | 0.765 | 0.529 | +0.25 |
| traffic | 0.756 | 0.647 | 0.792 | 0.626 | +0.13 |
| power_grid | 0.754 | 0.715 | 0.767 | 0.593 | +0.16 |
| supply_chain | 0.753 | 0.577 | 0.811 | 0.702 | +0.05 |
| wildlife | 0.630 | 0.646 | 0.624 | 0.555 | +0.08 |
| hospital_queue | 0.605 | 0.574 | 0.615 | 0.493 | +0.11 |
| social_contagion | 0.518 | 0.444 | 0.542 | 0.440 | +0.08 |
| epidemic | 0.478 | 0.469 | 0.481 | 0.327 | +0.15 |
| market | 0.423 | 0.367 | 0.441 | 0.481 | −0.06 |

Mean 0.653 (best-of 0.658), from 0.537. Friday's sweep: #1 0.7693, #10 0.7081; we sit near 13th.
The mechanistic models transferred: nine of ten up, and the sequence band (75 % of the score) is
now 0.77–0.83 on the five systems whose data covered their controls. The local LOO ordering held
everywhere except market, where the price floor (73.4) that the fit chose from one multilevel run
is wrong on long holds (sustained 0.37).

Where the remaining points are, by band:

1. **Sustained on supply_chain (0.58 vs 0.81 sequence), traffic (0.65 vs 0.79), power_grid
   (0.72 vs 0.77)**: the model is right after changes and wrong at held levels we never
   observed. The interior hold (`p3.hold_mid`, α = 0.5) is the direct fix; every model's
   equilibrium at interior controls is currently set by structure, not data.
2. **Both bands low on market, social_contagion, epidemic, hospital_queue**: the notes for all
   four say the same thing: controls never varied alone (incentive, bridge, closure, mask,
   followup, interest rate at interior levels) fitted to zero or to a guess. The composition
   blocks and the pulse train pin those.
3. **wildlife (0.65 / 0.62)**: habitat and corridor fitted to zero effect (never moved alone);
   both bands suffer equally. Composition blocks.
4. **Strong five**: ad_auction is at 0.83 both bands; reservoir's sequence 0.77 is the
   irrigation / withdrawal-depth attribution (never varied alone); traffic and power_grid are
   sustained-limited as above.

Decision (Saturday buy, needs the go): compose + train on all ten (6,594) plus `hold_mid` on the
seven systems that are sustained-limited or both-band-limited (supply_chain, traffic,
power_grid, market, social_contagion, epidemic, hospital_queue: 450/400 each, 3,000), total
≈ 9,600, reserve ≥ 300 per system. After the buy: refit all ten families through the lab,
compare LOO with and without the new runs, rebuild as the day's third slot. Sunday: the reserve
goes to whichever system's post-refit bands still lag, one run each.

No-credit factors for today's second slot (u009, built by the build lane): market, social,
hospital as a per-tick median of the ODE and `l0b_lin` (the two disagree most where the ODE
extrapolates), supply_chain on the v6 family (retail cap), wildlife unchanged (averaging a cycle
damps its amplitude; §16 D).

### 16. u009: medians of ODE and l0b_lin (Sat 01:00)

| system | u008 ODE (sust / seq) | u009 median (sust / seq) | verdict |
|---|---|---|---|
| market | 0.423 (0.367 / 0.441) | **0.500** (0.485 / 0.505) | median beats both members on both bands |
| social_contagion | 0.518 (0.444 / 0.542) | 0.500 (0.485 / 0.505) | sustained up, sequence down; ODE stays |
| hospital_queue | 0.605 (0.574 / 0.615) | 0.538 (0.468 / 0.561) | median loses both bands; ODE stays |
| supply_chain v6 | 0.753 (0.577 / 0.811) | 0.755 (0.583 / 0.812) | tie; v6 keeps the retail cap |

Reading: the per-tick median of two members pays only where their errors have opposite signs on
most ticks (market: the ODE's floor undershoots, l0b_lin overshoots). Where one member is simply
better (hospital) the median halves its lead. Rule from here: ensembles per observable and only
where the local fold predictions of the members straddle the truth; otherwise the better member
alone. Best-of after u009: mean 0.661.
