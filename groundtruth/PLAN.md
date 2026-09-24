# GroundTruth Toronto 26: architecture and build plan

Written 2026-09-23. Kit source: `kit/`.

## 0. Facts from the kit that drive the design

- **2,000 paid steps per system, and they never come back.** `reset`, `brief`, `documents` and `budget` are free. Every `step` costs 1.
- **Every eval episode starts from a reset.** Hidden state is a deterministic function of the true initial observables, with no random prehistory. The reset transient is therefore part of every one of the 40 scored episodes. Data that starts at reset is the most valuable data you can buy.
- **Free resets give free information.** Each reset returns a noisy initial observation drawn from the eval-like distribution. Use them to (a) learn the distribution of initial observations and (b) "reset-shop": reset repeatedly and only buy steps on runs whose initial observation fills a gap in coverage.
- **The eval horizon is 4,000 open-loop steps, but the total budget is 2,000.** The model must be stable and have a sensible equilibrium for any held action. Slow modes (fatigue, immunity, wear, seasons) need at least one long run per system.
- **Scoring:** `1/(1+|e|/σ)` against the *noiseless* truth, with σ fixed by the organizer per observable. The metric is bounded and robust: a wild miss costs at most 1, while a small bias costs a little on every tick. That favors predictions near the median/mode, clipped to plausible ranges, and it heavily punishes divergence.
- **Pulse definition:** `pulse = recovery + α·(pulse_ref − recovery)`, with α ~ U(0.7, 1) drawn independently per control. Recovery-history episodes alternate recovery and pulses with varied gaps. Our designs should sample exactly this.
- **Runtime:** 40 × 4000 = 160k steps in 1200 s is about 7.5 ms/step, which is plenty even for a pure Python loop. Target is under 300 s locally so there is 4× margin.
- **Uploads:** 3 per system per Toronto day, shared between public and final. The public leaderboard is a free out-of-sample validation signal on real eval schedules, so treat it as the best selector available (see §8).
- **Response format:** the `budget` response has both `remaining` and `simulator_steps_remaining`. Kit `Client._post` sends an `Idempotency-Key`, and a retry with the same key must not double-charge. We make keys deterministic (§2).
- **Gemma:** optional. No component below depends on it.
- **The simulators are differential equations (confirmed).** Each family is a deterministic ODE system: hidden compartments/queues/temperatures evolve as `dx/dt = f(x, u; θ, mechanisms)`, sampled once per tick, plus observation noise. Consequences:
  - The right model class is **grey-box ODE**, not a generic sequence model. If the structure is right, a dozen parameters fit from ~1,500 steps extrapolate cleanly to 4,000 steps, which a black-box model cannot.
  - Deterministic + noiseless truth + fixed reset rule means the transient from reset is perfectly repeatable, so **integrate the ODE from the same deterministic initial hidden state** that the brief describes (e.g. "waiting lists start empty", "warehouses half full").
  - Mechanism identification becomes model selection: fit all three 2-of-3 combinations `{AB, AC, BC}` and pick by validation. The briefs' suggested comparisons are exactly the experiments that make these three fits disagree.
  - The ODE is integrated with fixed-step RK4 and sub-steps inside `predict`, so the same code serves as mock simulator (true θ) and as forecaster (fitted θ̂). The mocks double as an identifiability test: fit θ̂ from a mock's 2,000 steps and check the recovered parameters.

---

## 1. Repo layout

```
gt/                          # repo root (git)
  pyproject.toml             # package "gtlab"; deps: numpy, scipy, httpx; extras[train]: torch (CPU)
  Makefile                   # thin wrappers over the CLI
  kit/                       # vendored copy of the participant kit (client.py, briefs.md, ...)
  gtlab/
    __init__.py
    cli.py                   # `gt ...` entrypoint (argparse)
    systems.py               # parse kit/briefs.md -> systems.json (observables, bounds, recovery, pulse, hard ranges)
    gateway.py               # Gateway protocol; RealGateway(kit Client); MockGateway; SpendGuard wrapper
    ledger.py                # append-only JSONL ledger, deterministic request ids, resume logic, lockfile
    budget.py                # per-system budget plan (phases/caps), reconciliation vs server
    mocks/
      base.py                # MockSystem ABC: reset(seed)->obs, step(u)->obs, .truth (noiseless), mechanism flags
      epidemic.py market.py traffic.py power_grid.py supply_chain.py
      wildlife.py reservoir.py ad_auction.py social_contagion.py hospital_queue.py
    design.py                # schedule generators (eval-like + identification experiments), experiment plans
    collect.py               # executes a plan phase against a Gateway, via ledger
    data.py                  # ledger -> Run objects (x0, U[T,m], Y[T,p], exp tags), splits
    metric.py                # exact score, sigma proxy, category-wise reporting
    features.py              # action normalization, input nonlinearities φ(u), output transforms (log1p/logit)
    models/
      base.py                # Model API: fit(train_runs, cfg) ; rollout(x0, U) -> Y ; to_npz() / from_npz()
      l0_persist.py          # persistence + "initial-to-equilibrium" variant
      l1_lag.py              # multi-timescale Hammerstein lag
      l2_lss.py              # stable Hammerstein-Wiener linear state space
      l3_nss.py              # linear backbone + small NN residual state space (torch train, numpy infer)
      ensemble.py            # per-observable median / weighted blend of top models
    runtime/                 # numpy-only code copied into every submission folder
      infer.py               # rollout implementations for L0..L3 + ensemble (no torch, no scipy needed)
      predict_template.py    # predict() -> loads model.npz + meta.json relative to __file__
    select.py                # CV/validation, model selection, report
    package.py               # build submission/<system>/ + zip
    check.py                 # local contract checker (§6)
    report.py                # HTML/PNG plots of fits vs data per system (matplotlib, dev only)
  data/<system>/             # ledger.jsonl, resets.jsonl, plan.json, budget.json (gitignored? NO: commit it, it is irreplaceable; also back up)
  artifacts/<system>/<model_id>/  # fitted params, val scores, plots
  submissions/<date>-<tag>/  # built folders + zip + checker report (keep every uploaded version)
  tests/                     # pytest: ledger resume, guard, mocks, checker, end-to-end offline
```

### CLI (all commands default to **mock/dry-run**)

| Command | What it does |
|---|---|
| `gt init` | parse briefs, write `systems.json`, create `data/<sys>/budget.json` from the default plan |
| `gt docs SYS\|--all` | fetch free `brief` + `documents` (real gateway, free), save to `data/<sys>/docs.json` |
| `gt resets SYS --n 40` | free resets only, log initial observations (never steps) |
| `gt plan SYS --phase p1` | materialize the phase's experiments (deterministic schedules) and print the step cost |
| `gt collect SYS --phase p1 [--spend --max-steps N --yes]` | run the phase; without `--spend` it runs against MockGateway |
| `gt fit SYS [--models l0,l1,l2,l3]` | fit the ladder on train split, write artifacts |
| `gt validate SYS` | score every artifact on the validation runs (exact metric), write leaderboard.json |
| `gt package SYS\|--all [--pick best\|<model_id>]` | build the submission folders |
| `gt check submissions/<tag>` | contract checker, required before any upload |
| `gt pipeline SYS --phase p1 [--spend ...]` | collect → fit → validate → package → check, **one command** |
| `gt mock-bench SYS\|--all` | whole pipeline against the mock, scored vs noiseless truth on eval-like schedules |
| `gt status` | per system: spent / planned / server remaining / best val score / last upload |

`gt pipeline --all --phase p1 --spend --max-steps 1000 --yes` is the "run overnight while at work" command. It processes systems sequentially and survives crashes via resume.

---

## 2. Budget safety (never overspend, never re-buy)

Layers, all enforced in code:

1. **Mock by default.** `collect` builds `MockGateway` unless `--spend` is passed. `RealGateway` construction additionally requires env `GT_ALLOW_SPEND=1`, so the key alone is not enough. Dry-run prints the full plan with exact step counts.
2. **Explicit cap.** `--spend` requires `--max-steps N`. The run refuses if the phase needs more than N. There is no default N.
3. **Budget plan file** `data/<sys>/budget.json`: `{"total":2000,"phases":{"p1":1000,"p2":500,"val":250,"reserve":250},"floor":0}`. A phase can never exceed its cap, and cumulative ledger steps can never exceed `total`.
4. **Pre-flight reconciliation.** Before any spend, call free `budget(sys)` and require `server_remaining == 2000 − ledger_steps_charged`. On mismatch (steps spent outside this tool, e.g. the kit's collect.py), abort unless `--reconcile` is passed, which records an `external_spend` ledger line. Also require `planned ≤ server_remaining − floor`.
5. **SpendGuard wraps the gateway.** `step()` is the only paid call, and the guard is its only caller. Before every step it checks `charged+1 ≤ min(phase_cap_left, max_steps_left, server_remaining_cached)`. Every 50 steps it re-reads the free server `budget` and aborts on drift.
6. **Write-ahead ledger** `data/<sys>/ledger.jsonl`, append + `fsync` per line:
   - `{"t":"reset","exp":"p1.hold_rec","run_id":..,"req":..,"obs":{..},"raw":{..},"ts":..}`
   - `{"t":"intent","exp":..,"run_id":..,"tick":k,"req":R,"action":{..}}` is written **before** sending.
   - `{"t":"step","exp":..,"run_id":..,"tick":k,"req":R,"action":{..},"obs":{..},"raw":{..}}` is written after.
7. **Deterministic idempotency keys.** `req = uuid5(NS, f"{sys}/{exp_id}/{run_idx}/{tick}")`, and the reset key is the same with `tick=-1`. On resume, an `intent` without a `step` is re-sent with the **same key**. The server dedupes it, so a crash can never cost a double purchase. The kit client already retries TransportError with the same key.
8. **Never re-buy.** An experiment's schedule is materialized once into `plan.json` (seeded, hashed). `collect` skips completed experiments and resumes partial ones at `last_tick+1` on the stored `run_id`. Changing a materialized schedule requires a new `exp_id`.
9. **Single writer.** A lockfile `data/<sys>/.lock` (fcntl) blocks two concurrent collectors on the same system.
10. **Backups.** After each phase, auto-copy the ledger to `data/<sys>/backup/ledger.<ts>.jsonl`. Commit it to git: the data is irreplaceable.
11. **Tests (pytest, mock).** Cover: a kill at a random tick followed by resume ends with exactly the planned count and no duplicate ticks; a cap is hit with an exception and nothing is sent; drift is detected; dry-run makes zero `RealGateway` calls (the `RealGateway` constructor raises in tests).

**Offline mode.** `MockGateway(system, variant, seed)` implements the kit Client surface (`reset/step/budget/brief/documents`) over `mocks/<family>.py`. It keeps its own 2,000-step budget and honors idempotency keys, so the guard, ledger and resume paths are exercised exactly as in real use. Mock data is written to `data_mock/<sys>/...` and never to `data/`.

---

## 3. Mock simulators

The goal is realistic *difficulty*, not fidelity: a way to exercise the pipeline and rank model classes.

The common `MockSystem` has hidden state `h`, `step(u) → y_true`, and a noisy observation `y = y_true·(1+ε₁) + ε₂` (≈3% multiplicative plus a small additive term). `reset(rng)` randomizes the observables within a range; hidden state is a fixed deterministic function of the true observables. `variant ∈ {AB, AC, BC}` switches on 2 of 3 mechanisms so you can test whether experiments discriminate them. Every mock includes at least one pure delay/pipeline, one saturation, a fast mode and a slow mode (τ≈300–1500), and clipping to physical ranges. Integrate with `dt=1`, with sub-steps where the dynamics are stiff. Keep each mock to about 60–120 lines.

| Family | Mock hidden state and dynamics (sketch) | Mechanisms A/B/C |
|---|---|---|
| epidemic | 3-age SEIR + H with bed cap; β_ij scaled by (1−0.6·closure on school row)·(1−0.4·mask); vaccination flow = rate·N·clinic, clinic = 1/(1+H/Hcap); waiting list when H>cap; cases = σE; hospital_load = H+wait | A: behavioral fatigue (compliance decays with cumulative restriction, τ=200); B: waning immunity R→S τ=400; C: postponed gatherings (a closure accumulates a "debt" that raises β on release) |
| market | producer/consumer inventories+cash, order pipeline (4-tick delay), dealer book; price from excess demand, tanh-saturated; volume = min(buy, sell, depth); depth = dealer capacity − tied funding | A: settlement tie-up (funding recovers τ=30); B: risk capacity drops after adverse moves (τ=150); C: momentum investors (EMA of returns shifts demand) |
| traffic | 2 routes × (approach buffer, junction, exit) queues with finite capacity; arrivals = demand(toll mix)·(1−ramp); signal splits junction service; flows = exits; speed = vfree/(1+k·occupancy) mixed by class | A: route learning (split drifts toward the faster route, τ=100); B: crew fatigue (clearance effectiveness decays under use); C: spillback front (hysteretic: capacity drops once the queue exceeds a threshold and recovers slowly) |
| power_grid | 200 thermostatic loads aggregated as a 20-bin temperature histogram (price shifts the deadband) giving oscillatory rebound; reserve battery SOC with charging limit; governor 1st order; frequency = 50 + k·(gen−load) filtered; renewable_share = ren_used/total, curtailed when interconnector is low | A: interconnector thermal (capacity derates with an EMA of flow, τ=300); B: reserve thermal/duration limits; C: load heterogeneity dispersion. Keep all three simple. |
| supply_chain | 2-class supplier stock, conveyor (8-tick delay), treatment with activity decaying without maintenance, retail stock; shipments = min(order, available) + secondary grade | A: congestion+rework loop; B: machine heat/wear (slow, reset by maintenance); C: adaptive production commitment (EMA of orders) |
| wildlife | 2 regions × {prey, predator} with 3 patches collapsed to a 2-patch refuge; logistic prey with resource renewal (habitat), Holling II predation, harvest = min(quota, prey); transit pipeline (5-tick delay) gated by corridor | A: juvenile stage (delay 10); B: resource depletion τ=100; C: settlement-space competition |
| reservoir | level integrator; inflow = seasonal sine (period 365, fixed phase at reset) + return flow (delay 20); outflow = min(release+irrigation, fouling-limited cap); 2-layer quality with algae/oxygen | A: fouling biomass (slow); B: groundwater/contaminant return; C: sediment remobilization by aeration |
| ad_auction | win = sigmoid(k(bid − rival price)); impressions ∝ breadth; spend = min(cap, win·bid·impr); purchase pipeline with fulfillment capacity; converted customers in cooldown | A: rival capital shift (rival price follows our spend, τ=50); B: exposure fatigue pool; C: broad-introduction priming |
| social_contagion | 2 communities, Bass-like interest from seeding/incentive/bridge; onboarding queue with workforce capacity shared with members; churn → refractory pool (τ=60) → susceptible | A: credibility (unkept promises lower adoption); B: incentive expectation (adoption drops when incentive falls below its EMA); C: cross-community ties grow with bridge usage |
| hospital_queue | 3 case types, assessment chairs and beds finite, service rate = staffing·(1+0.5·overtime)·effectiveness; wait = queue/throughput (Little) smoothed; discharges = completions; overflow referred | A: fatigue (overtime EMA lowers effectiveness later); B: handover/orientation (staff increases are effective after a ramp τ=20); C: returning case mix (discharges return after 30–60 ticks unless followup) |

`mock-bench` generates 40 eval-like episodes of 4000 steps per system (the §4 generators) and scores against `.truth` using σ = std of truth across those episodes. This is the offline equivalent of the leaderboard.

---

## 4. Experiment designer

### Primitive schedule generators (`design.py`)

All return a `U[T, m]` in physical units, clipped to bounds, from a seeded RNG.

- `hold(level, T)`: levels are drawn from {recovery, pulse(α), mid-bounds, uniform, random corner}.
- `pulse_train(n, L∈[3,40], gaps log-uniform[5,300], α~U(.7,1) per control)`: the baseline is the recovery action. This matches the eval recovery category.
- `order_pair(blocks)`: returns two schedules with the same multiset of blocks in different orders (e.g. ABC vs CBA), plus the A→B vs B→A swap.
- `single_vs_joint(base)`: steps each control alone from base to its pulse level (dwell D), then all controls jointly, then pairwise in a different sequence.
- `multilevel(T, dwell log-uniform[3,120])`: random levels (50% interior uniform, 30% corners, 20% recovery/pulse). This is the general-coverage excitation, and it beats the kit's fixed 16-block design.
- `eval_like(category, T)`: the four categories, used for mock-bench and to generate validation-run schedules in miniature.

### Initial conditions

Initial conditions are free. Do `gt resets SYS --n 40` once and store the pool. For each paid run, "reset-shop": reset up to 10 times (free), then pick the candidate that maximizes the minimum distance, in normalized observable space, to the initial conditions already bought. Record the extra resets in `resets.jsonl`. They tell you the initial-observation spread and give a noise estimate, because the observations are noisy draws.

### Default per-system allocation (2,000 steps)

All runs start from reset. p1 is fixed, p2 is adaptive, val is held out and never trained on until the final refit, and reserve is kept for late fixes.

| Phase | Exp | Steps | Content |
|---|---|---:|---|
| p1 | `hold_rec` | 120 | reset → hold recovery 120. Gives the reset transient, the recovery equilibrium, and noise σ from the tail. |
| p1 | `hold_pulse` | 120 | reset → hold pulse(α=1) 60 → recovery 60. Step response on and off. |
| p1 | `long_train` | 450 | reset → recovery 40 → pulse_train(gaps 5…300, 8–12 pulses) → recovery tail ≥100. Slow modes and history. **Reservoir: 700** (season), taken from p2. |
| p1 | `compose` | 180 | single_vs_joint from recovery; dwell = max(10, 150/(m+1)). |
| p1 | `order_AB` / `order_BA` | 2×65 | same blocks, swapped order, each from a fresh reset (fresh reset makes them directly comparable). |
|  |  | **1000** |  |
| p2 | `mech` | ~150 | the brief's discriminating experiment (§7), designed after looking at p1 residuals |
| p2 | `multilevel` | ~200 | general coverage, interior levels |
| p2 | `fix` | ~150 | targeted at the worst-scoring category/observable on validation-in-train CV |
| val | `val1`, `val2` | 2×125 | eval_like mixed (hold → order → pulses → joint), fresh resets, never seen in training |
| reserve | | 250 | spent on Sep 27–28 on the systems with the biggest val/public gap, or on a second long run. Folded into the final fit. |

Why this split:
- About 70% of steps go to many short-to-medium runs from reset, because every eval episode starts at reset.
- About 25% goes to long runs for the slow modes. The eval is 4000 steps long, but a model with a correct equilibrium map plus correct slow τ extrapolates. Spending 2000 steps on one trajectory would teach only one initial condition.
- The 250-step validation holdout is the only honest local selector with real noise. The public leaderboard is the second selector.

For the final model, refit on **all** data (train + val + reserve) with the selected configuration.

---

## 5. Model ladder

Everything runs at inference in numpy only, with a per-step Python loop. Each model is a function `rollout(y0, U) → Ŷ[T,p]`.

### Shared pieces

- **Normalization.** Action `ũ = (u−lo)/(hi−lo) ∈ [0,1]`. Observable transform `g` per observable: `log1p(y/s)` for positive, heavy-dynamic series (cases, populations, adopters, queue, volume), `logit` with ε for [0,1] shares (quality, win_rate, renewable_share), and affine otherwise (frequency: `(y−50or60)/s`). The choice is auto-picked by a rule: bounded in [0,1] → logit; min>0 and max/min>20 → log1p; else affine. Rules can be overridden in `systems.json`.
- **Input features** `φ(ũ) = [ũ, ũ², pairwise ũ_iũ_j for m≤4 (else top-6 by CV), 1]`. There is also an optional per-control "pulse indicator" `σ(k(ũ−0.5))`.
- **Physical clipping at output:** hard bounds (≥0, ≤1) plus a soft range of `[min−0.2·range, max+0.2·range]` of all observed data. Clipping is cheap insurance against divergence.
- **Initial state encoder.** The hidden state is a deterministic function of the true initial observables, so every model gets a learned linear map `x0 = E·g(y0) + e0`, fit jointly. The model also implicitly denoises `y0`: step 1 is not forced to equal the noisy `y0`.
- **Training objective (all levels): full open-loop simulation error, matched to the metric.** The metric is ` 1/(1+|e|/σ)`. Use the surrogate `ρ(e) = log(1 + |e|/σ̂)`, which is tangent to `1−score` at 0, robust, and concave in |e|. Implementation:
  - scipy `least_squares(loss='cauchy', f_scale=σ̂)` on residual `(ŷ−y)` gives `ρ = log(1+(e/σ̂)²)`, which is close enough and uses a trust-region solver. This is for L1/L2.
  - For torch (L3), use `log1p(|e|/σ̂)` directly, with a warm-up of 20% of epochs on MSE.
  - Weight each run equally (not each tick) and optionally up-weight the first 50 ticks ×2, since the transient appears in all 40 episodes.
- **σ proxy.** `σ̂_j = std over all collected observations of observable j` in physical units, floored at `max(3·noise_std_j, 1e-3·|mean_j|)`. Noise is estimated as `std(diff(y))/√2` on hold tails. The organizer's σ is "from frozen reference trajectories", most likely a per-observable spread across trajectories, so this is the closest proxy. For robustness, select models by the mean of scores computed at σ̂×{0.5, 1, 2}.
- **Multiple shooting / curriculum.** For L2/L3, train on windows of length 32 → 128 → 512 → full. The initial state of each non-reset window is a free parameter (L2) or an encoder of the previous 10 observations and actions (SUBNET-style, L3). A continuity penalty is added. Finish with 2–3 epochs of full rollouts from reset. This smooths the loss landscape and avoids the exploding gradients of long rollouts ([Ribeiro et al. 2020](https://doi.org/10.1016/j.automatica.2020.108777); [Beintema, Tóth & Schoukens, L4DC 2021](https://proceedings.mlr.press/v144/beintema21a.html); [Beintema et al., Automatica 2023](https://www.sciencedirect.com/science/article/pii/S0005109823003710)).
- **Why output-error:** equation-error/one-step (ARX) fits are biased under output noise and are not what is scored ([Ljung 1999, System Identification]; [Schoukens & Ljung 2019, "Nonlinear system identification: a user-oriented road map", IEEE CSM]). One-step fits are used only as initializations.

### L0: persistence (plus the equilibrium variant)
- `L0a`: `ŷ_t = y0`. This is the kit baseline and the floor.
- `L0b`: `ŷ_t = y0 + (y*(ũ_t) − y0)·(1−a^t)`, with `y*` a ridge regression of hold-tail means on φ(ũ). It is cheap and often surprisingly strong on sustained operation.

### L1: multi-timescale Hammerstein lag
- The state is K=3 banks per observable, `z^k_{t+1} = a_k z^k_t + (1−a_k)·W_k φ(ũ_t)`, with `a_k = sigmoid(θ_k)` initialized at (0.5, 0.95, 0.995) (τ ≈ 2, 20, 200). The output is `ŷ = g⁻¹(c + Σ_k z^k)` followed by clipping.
- Optional **pure delay** d per control (grid-searched 0…10 on CV) for the pipelines in market, supply_chain and wildlife.
- Optional **history state**: a slow EMA `h` of the "pulse-ness" `‖ũ − ũ_rec‖`, which multiplies the gains, `W_k → W_k(1+γ_k h)`. This cheaply captures fatigue, immunity and wear.
- Init `z0` comes from the encoder so that `ŷ_0 ≈ g(y0)`.
- Parameters number about 3·p·|φ| + small: roughly 50–300. Fit with `least_squares` (cauchy) with finite-difference or analytic Jacobian over all runs. It takes 1–5 min/system on a laptop. **This is the default submission for Sep 24.**
- It is stable by construction (|a_k|<1).

### L2: stable Hammerstein–Wiener linear state space
- `x_{t+1} = A x_t + B φ(ũ_t)`, `ŷ = g⁻¹(C x_t + D φ(ũ_t) + c)`, order n ∈ {4, 6, 8, 12}, picked by CV.
- **Stability by parametrization:** A is block-diagonal in real 2×2 modal blocks `r·[[cos θ, −sin θ],[sin θ, cos θ]]` plus real 1×1 blocks, with `r = r_max·sigmoid(ρ)`, `r_max = 0.9995`. Complex poles are required for rebound/oscillation (power_grid TCLs, wildlife). This is the same idea as the LRU eigenvalue parametrization ([Orvieto et al. 2023](https://arxiv.org/abs/2303.06349)).
- **Near-integrators** (supply_chain inventories, reservoir level): allow `r_max = 0.99995` on one mode, and rely on output clipping plus the saturating Wiener output for the bounds.
- **Init:** N4SID on the transformed data with the φ-lifted input ([Van Overschee & De Moor 1994](https://doi.org/10.1016/0005-1098(94)90230-5); SIPPY [Armenise et al. 2018](https://github.com/CPCLAB-UNIPI/SIPPY) as the reference implementation; a ~80-line numpy N4SID is enough). Then convert to modal form, project eigenvalues inside r_max, and refine with simulation error by `least_squares` (cauchy). If N4SID fails, init from L1.
- **Exogenous time input for reservoir:** add a `[sin, cos](2πt/P)` input with P fit by grid search on the long run. P is fixed from reset, and this is the only system where t matters.

### L3: grey-box linear backbone + small NN residual (torch-trained, numpy-exported)
- `x_{t+1} = A x_t + B φ(ũ_t) + W₂ tanh(W₁[x_t; ũ_t] + b₁)`, `ŷ = g⁻¹(C x_t + c + V tanh(U x_t))`, with n=8–16 and hidden size 32. A and B come from L2 (warm start), and the residual is initialized near 0.
- **Stability tricks:** spectral normalization of W₂W₁ via a power-iteration penalty; state clipping `x ← clip(x, ±5·std_train)` at every step, in both train and inference; weight decay; gradient clip 1.0; the multiple-shooting curriculum above. Stable/contracting RNN theory supports a contraction constraint ([Miller & Hardt, ICLR 2019](https://arxiv.org/abs/1805.10369); [Revay, Wang & Manchester, RENs, IEEE TAC 2023](https://arxiv.org/abs/2104.05942)). Use the cheap penalty and not a full REN.
- **Alternative L3b:** a GRU with 16–32 units, same curriculum. Keep it only if it wins validation.
- Train 5 seeds and keep them as an ensemble (median).
- Export: weights go to `model.npz`, and `runtime/infer.py` re-implements the forward pass in numpy. A unit test asserts numpy == torch to 1e-6 on 4000 random steps.

### L4: grey-box ODE per family (the main bet, because the simulators are ODEs)
- `gtlab/ode/` holds one module per family: `f(x, u, θ, mech) -> dx/dt`, `x0(y0, θ) -> hidden state` (the brief's deterministic reset rule), `h(x, θ) -> y`, a `PARAMS` table (name, init, lower, upper, log-scale flag) and `MECHS = {A, B, C}` switches.
- Integrator: fixed-step RK4 with `n_sub` sub-steps per tick (default 4), vectorised numpy, state clipped to physical ranges after each sub-step. Pure numpy, so it ships in the submission as-is.
- The **same module powers the mock** (`MockSystem` = ODE with hidden "true" θ, a secret mechanism pair and noise) **and the forecaster** (fitted θ̂). One code path, tested once.
- Fit: `scipy.optimize.least_squares(loss='cauchy', f_scale=σ̂)` on full open-loop residuals across all runs, in log-parameter space, bounded. Multi-start (8 starts from Latin hypercube in the bounds), then keep the best. Run for each mechanism pair {AB, AC, BC} and choose by validation score. Multiple shooting for long runs if the loss surface is rough.
- The mock ODE structures are guesses from the briefs. Once real data arrives, the per-family template gets edited to match what residuals show: this is where research time goes on Sep 24–27. L1/L2 stay as fallbacks and as ensemble members.
- Optional **hybrid**: ODE backbone + L2 linear residual state space fit on the ODE's residuals, for systems where the template is partially wrong.

### Stress gate (every model, before selection)
Roll out 200 random `eval_like` schedules of 4000 steps from random pool initial conditions. The model is rejected if any output is nonfinite, any output leaves the soft range for more than 1% of ticks, or the rollout is slower than 1.5 ms/step.

### Selection (`gt validate`)
1. Score = the exact metric `mean_{obs,t} 1/(1+|ŷ−y|/σ̂)` on val1/val2 (noisy targets add a similar penalty to every model, so ranking is fair). Also report 5-fold leave-runs-out CV on train, and per-category scores by tagging segments of the val schedules.
2. The candidate set is each ladder model that passed the stress gate, plus a **per-observable median ensemble** of the top 3 (the median is the right aggregator for this mode-seeking metric).
3. Pick the highest mean over σ̂×{0.5,1,2}. On ties within 0.005, pick the simpler model.
4. Offline, `mock-bench` confirms the val-based choice correlates with noiseless-truth ranking. If it doesn't, weight CV more.
5. **Selection can be per observable.** A folder can hold L1 for one observable and L3 for another, because `infer.py` supports a per-observable source map.

---

## 6. Packaging and the local contract checker

`gt package` writes `submissions/<YYYYMMDD-HHMM>-<tag>/<system>/{predict.py, infer.py, model.npz, meta.json}`. `meta.json` holds the family, observables, transforms, clip ranges, model ids and the git sha. It then builds `submission.zip` with system folders at the root, using `zip -X`, excluding `__pycache__`/.DS_Store, and deterministic.

`predict.py` template rules:
- It loads via `Path(__file__).with_name(...)`, caches the weights in a module global (allowed), and keeps **no trajectory state in globals**.
- It checks `context['family']`, normalizes with `context['intervention_bounds']` (falling back to meta), and returns `[{name: float(v)} ...]`.
- It wraps everything in `try`: if the model path raises or produces a nonfinite value, fall back per-tick to L0b, then L0a. A crash scores 0, and the fallback scores more.

`gt check <dir|zip>` hard-fails on any of:
1. The zip opens. Root entries are only directories named exactly from the 10 system ids. Each has `predict.py`. There is no enclosing folder. Zip is ≤30 MiB and expanded size ≤300 MiB.
2. **Secret scan:** no file contains `GROUNDTRUTH_KEY`, `GEMMA_API_KEY`, `Bearer `, `AIza`, or the actual key strings read from env. There are no `.env`, `venv/`, `site-packages` or `*.pyc` files.
3. **Import whitelist (AST scan of every .py):** numpy, scipy, sklearn, joblib, and stdlib {json, math, pathlib, os.path, functools, itertools, typing, dataclasses, warnings, collections}. Anything else fails, and the banned list includes torch, httpx, requests, socket, urllib, subprocess, multiprocessing, `open(` with mode 'w'.
4. **Sandbox run:** extract to a temp dir. In a fresh subprocess, use a **py3.12 venv pinned to numpy==2.3.5 scipy==1.16.3 scikit-learn==1.7.2 joblib==1.5.2** (`uv venv` once). Run with cwd set elsewhere (tests relative loading), `socket.socket` monkeypatched to raise, `RLIMIT_AS=3 GiB`, `taskset -c 0,1`, and `OMP/OPENBLAS_NUM_THREADS=2`.
5. The runner makes **40 episodes × 4000 steps** of `eval_like` schedules (all 4 categories, plus schedules pinned at bounds extremes), with initial conditions from the reset pool or mock, and a `context` dict with exactly the 7 keys (docs from `docs.json`).
6. **Asserts:** it returns a `list` of length 4000; each item is a `dict` with keys == observables; each value is a python/numpy real and finite; hard bounds are respected.
7. **Fresh state:** call episode A, then B, then A again, and require A's outputs to be byte-identical. Also call A with the list copied and check the input is not mutated.
8. **Timing:** total wall time for load + 40 episodes must be ≤ **400 s** locally (the organizer limit is 1200 s; the 3× margin covers slower hardware). Peak RSS must be ≤ 1.5 GiB.
9. **Output:** `check_report.json`, saved beside the zip. `package` refuses to mark a build "uploadable" without a passing report.

---

## 7. Per-system notes

Hard bounds apply to the outputs you return. "2-of-3" means the brief says exactly two of three candidate mechanisms apply.

**epidemic** (`daily_cases`, `hospital_load` ≥0)
- Age-structured SEIR-H with a bed-capacity waiting list (saturation). Vaccination is throttled by hospital pressure (a feedback loop). Cases span orders of magnitude, so use log1p. Hospital lags cases by roughly 5–15 ticks.
- History carriers: behavior/fatigue, developing immunity, postponed gatherings (rebound when closures lift). Long horizons can burn out or re-wave, so the long run is essential.
- Discriminating experiment: closure-only vs mask-only pulses started at similar case counts (reset-shop to matched initial cases), and a vaccination block before vs after a closure pulse. Track hospital recovery.

**market** (`price` >0 → log, `volume` ≥0, `depth` ≥0)
- An order pipeline means policy changes act with a pure delay, and orders already committed are not cancelled. Fit delay d. Price is likely mean-reverting to a tax/rate-dependent level.
- 2-of-3 memory: settlement funding tie-up, risk-capacity loss after adverse moves, momentum investors. Momentum can produce overshoot and oscillation (complex poles in L2).
- Experiment: pulse → reversal vs pulse → hold, comparing the depth response to the reversal (the brief hints that depth composition changes what a reversal does).

**traffic** (`flow_a/b` ≥0, `speed_a/b` ≥0 and capped by the max observed free-flow)
- A queue network with finite buffers means strong saturation and hysteresis (congestion is not symmetric in on and off). One route can block the other through the shared junction. Speed is a nonlinear function of mix and occupancy, so fit per-route output nonlinearities.
- 2-of-3: route learning, crew fatigue/switching cost, persistent spillback fronts.
- Experiments: toll-prep vs ramp-prep with similar totals; after stopping arrivals (ramp_metering=1, which is also the pulse level) compare clearance_effort vs signal reversal. Six controls make composition the most expensive to learn, so keep φ pairwise terms sparse.

**power_grid** (`load` ≥0, `frequency` near nominal, `renewable_share` ∈ [0,1] → logit)
- Thermostatic load synchronization gives a **damped oscillatory rebound** after price pulses. L1 cannot represent this and L2 needs complex poles. Resets start at price 0.8 while recovery uses 1.5, so every eval starts with a price-step transient.
- Frequency deviations are tiny relative to their mean. σ̂ is probably small, so frequency may dominate the error budget. Model the deviation from nominal directly (affine transform around the nominal value; detect 50 vs 60 from data).
- Reserve energy limits (depletion with long reserve_dispatch) and interconnector thermal derating are slow states. Test with a long reserve pulse (does the supply fade?) and with charging_allowance=0 vs 1 during recovery after it.

**supply_chain** (`shipments`, `inventory_supplier`, `inventory_retail` all ≥0)
- Inventories are **integrators** (near unit root) with capacity saturation. Allow near-1 poles plus clipping. Conveyors carry pure delays, and orders withdraw only available stock (a `min` nonlinearity).
- 2-of-3: congestion/rework, machine heat/wear, adaptive production commitments.
- Experiments: at fixed product_mix, maintenance-block vs idle-pause (production 0) of the same length; and equal total orders in opposite production sequences (an order-pair design).

**wildlife** (4 populations ≥0 → log1p)
- Predator–prey in two regions with transit delay through the corridor. Oscillations or limit cycles are likely. Open-loop phase drift over 4000 ticks is the main risk. If the val score shows phase errors, the median ensemble naturally damps toward the cycle mean, and at this metric that beats a confident wrong phase. Consider an explicit "amplitude shrink with horizon".
- Hidden: patch occupancy, juvenile condition, in-transit animals (a delay pipeline gated by corridor). Hunting quota (max 8) is a `min(quota, prey)` harvest.
- Experiments: a habitat_protection recovery block with corridor 0 vs 1; hunting pulse before vs after a protection block.

**reservoir** (`level`, `inflow`, `outflow` ≥0; `quality` ∈ [0,1] → logit)
- A tick is one day, and seasonal inflow likely has a period of about 365 with a fixed phase from reset. Model inflow as a function of t (harmonics) plus a delayed return flow. This needs a long run of **≥700 steps** (budget shifted from p2). Level is an integrator: level_{t+1} ≈ level + inflow − outflow, so hard-code this structure in L1/L2 as a physics prior (it is cheap and big).
- Outflow = min(requested release + irrigation, fouling-limited capacity). Fouling is a slow state reduced by aeration and flushing.
- Experiments: deep release (withdrawal_depth=1) block, then shallow, comparing later quality; aeration on at low vs high quality (dilution vs remobilization).

**ad_auction** (`win_rate` ∈ [0,1] → logit, `spend` ∈ [0, budget_cap], `conversions` ≥0)
- Mostly a static auction map (win vs bid), with **hard constraint spend ≤ budget_cap(t)**. Enforce it in predict. It is a free win.
- Conversions are delayed by the purchase pipeline and capped by fulfillment work (saturation). Customer cooldown means repeated pulses yield less.
- 2-of-3 (not labeled so, but three listed): rival capital shift, exposure fatigue, broad-introduction priming. Experiment: broad (1.0) then narrow vs narrow-only at equal spend; repeated pulses with short vs long gaps.

**social_contagion** (`adopters_a/b` ≥0, bounded by unknown populations → log1p; learn the upper asymptote)
- Bass/SIS-like adoption with an onboarding-capacity queue (saturation) and churn → refractory → re-eligible. Growth is logistic, so long-run levels matter most for sustained-operation scores.
- History: credibility (promises vs capacity), incentive expectations (removing an incentive may cause churn below baseline), cross-community ties.
- Experiments: local (bridge=0) vs bridge (bridge=0.6) with equal seeding; incentive before vs after a seeding burst; recovery with seeding=0 to measure churn/refractory time.

**hospital_queue** (`wait_time`, `queue`, `discharges` ≥0)
- A finite-capacity queue network with blocking. Little's law `wait ≈ queue / throughput` is a structural prior: add `log(queue) − log(discharges)` as a feature or constraint. Staffing changes work rate, not counts directly.
- 2-of-3: fatigue (overtime → later slowdown), handover (a staff change → temporary efficiency loss), returning case mix (discharges return later unless followup).
- Experiments: equal staff-hours with overtime concentrated vs spread; a diagnostic_allocation change at fixed staffing; followup on vs off after the same discharge burst.

---

## 8. Day-by-day plan (Toronto time)

Upload rule of thumb: at most 2 of 3 slots per system per day are planned, which keeps one spare for crash fixes. Every upload passes `gt check` first. Keep every zip.

**Wed Sep 23 (tonight)**
- Build the skeleton: `systems.py`, `gateway.py` (Mock + guard), `ledger.py`, `budget.py`, `design.py`, `metric.py`, L0a/L0b, `runtime/`, `package.py`, `check.py`. Mocks for 2 families first (power_grid, hospital_queue), then the rest as generic templates.
- Write the pytest suite for resume/caps/drift. Run `gt mock-bench power_grid` end to end.
- Free calls only: `gt docs --all`, `gt resets --all --n 40`.
- **Upload #1 (public, all 10): L0a persistence.** This validates the packaging and earns a nonzero floor on every system. That matters because missing systems score 0.

**Thu Sep 24 (before work, ~20 min of attention)**
- Run `gt pipeline --all --phase p1 --spend --max-steps 1000 --yes` (sequential, resumable), at about 10k steps total. Check `gt status` at lunch.
- Evening: implement L1, then fit and validate using CV on p1 (val is not collected yet).
- **Upload #2 (public, all 10): best of L0b/L1 per system.**

**Fri Sep 25**
- Implement L2 (N4SID + modal refinement) and the stress gate. Mock-bench L0/L1/L2 on all mocks.
- Look at p1 residual plots (`report.py`) and write the `mech` + `fix` experiments per system (§7). Run `gt collect --all --phase p2 --spend --max-steps 500`, then `--phase val --max-steps 250`.
- **Upload #3 (public): systems where L2 beats L1 on val by more than 0.01.** Compare against Thursday's public score. This calibrates local val against public.

**Sat Sep 26**
- Implement L3 (torch train, numpy infer, parity test). Train it only on systems where L2 shows structured residuals (likely power_grid, wildlife, traffic, hospital_queue, market). Add the median ensemble and per-observable selection.
- **Upload (public): up to 2 per system** for the new winners, and A/B two variants on the systems where local val is ambiguous. The public score is the tiebreaker.

**Sun Sep 27**
- Spend the **reserve** (≤250/system) where public or val is weakest: a second long run or a targeted history experiment. Refit final configs on all data (train + val + reserve).
- **Upload (public): final candidates.** Freeze code tonight. After this only params and selection change.

**Mon Sep 28 (final opens 12:00)**
- 12:00–13:00: **submit Final #1 for all 10 in the Final tab**, the best public-validated models. This is insurance. Public uploads do NOT carry over.
- Evening: at most one public upload per system for last experiments. Keep ≥1 slot for final.

**Tue Sep 29**
- Final #2 only for systems where a change clearly improved public or local val (≥0.01) that same day. Remember final scores are hidden, so rely on public and val.

**Wed Sep 30 (close 12:00)**
- By 10:00 at the latest: last Final upload for any changed system, then confirm the receipts list all 10 as accepted. After that there are no changes and nothing left to do but wait.

### Automation summary (for a full-time job)
Per system, one command: `gt pipeline SYS --phase <p> --spend --max-steps N --yes`, which runs collect → fit (ladder) → validate → package → check. The `--all` form runs the whole portfolio overnight. `gt status` shows one table. Uploading stays manual in the portal (3 min/day): pick the latest `submissions/*/submission.zip` with a passing `check_report.json`.

---

## 9. Opinionated defaults and risks

- **Default winner guess:** L1 or L2 for epidemic, market, ad_auction, social_contagion, reservoir (L1 with physics priors: level integrator, season, spend cap). L2/L3 for power_grid, wildlife, traffic, hospital_queue, supply_chain.
- **The biggest score risk is divergence or bias over 4000 ticks, not missing fine detail.** Always clip, always pass the stress gate, and prefer the median ensemble.
- **The biggest budget risk is a crash mid-run causing a re-buy, or the kit's collect.py being run by habit.** It is handled by deterministic idempotency keys, the ledger, and pre-flight reconciliation. Delete or rename the kit's `collect.py` in the working copy.
- **Unknown response shapes:** log `raw` everything. `data.py` extracts only `observation`, so extra fields are harmless.
- **σ uncertainty:** selection is averaged over σ̂×{0.5,1,2}, and the public leaderboard is the calibrator.

### References
- Ljung, *System Identification: Theory for the User*, 2nd ed., 1999 (output-error vs equation-error).
- Schoukens & Ljung, "Nonlinear system identification: a user-oriented road map," IEEE Control Systems Magazine, 2019.
- Van Overschee & De Moor, "N4SID: Subspace algorithms for the identification of combined deterministic-stochastic systems," Automatica, 1994.
- Armenise et al., "An open-source system identification package for multivariable processes" (SIPPY), UKACC 2018.
- Ribeiro, Tiels, Umenberger, Schön, Aguirre, "On the smoothness of nonlinear system identification," Automatica, 2020 (multiple shooting).
- Beintema, Tóth, Schoukens, "Nonlinear state-space identification using deep encoder networks," L4DC 2021, https://proceedings.mlr.press/v144/beintema21a.html ; "Deep subspace encoders for nonlinear system identification," Automatica 2023.
- Forgione & Piga, "Continuous-time system identification with neural networks: model structures and fitting criteria," Eur. J. Control, 2021.
- Miller & Hardt, "Stable recurrent models," ICLR 2019.
- Revay, Wang, Manchester, "Recurrent equilibrium networks," IEEE TAC, 2023.
- Orvieto et al., "Resurrecting recurrent neural networks for long sequences" (LRU), ICML 2023.
