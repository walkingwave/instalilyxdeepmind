# Ground Truth — demo brief

Instalily x Google DeepMind Toronto hackathon, Sep 23–30 2026. Team: Dylan Ho (walkingwave).
Full lab notebook with every equation, experiment and number: `docs/MATH_LOG.md` (§1–46).

## 1. The problem

- 10 hidden simulators (ODE black boxes): epidemic, market, traffic, power_grid, supply_chain,
  wildlife, reservoir, ad_auction, social_contagion, hospital_queue. Each brief names 3 mechanisms;
  exactly 2 are active.
- Budget: 2,000 simulator steps per system for the whole week. Reset, brief and documents are free.
- Submission: one `predict(initial, interventions, context)` per system, open loop, 4,000-step
  forecasts, numpy/scipy only, 1,200 s for 40 episodes.
- Score per observable per tick: $1/(1+|err|/\sigma)$ against the noiseless truth, averaged over
  40 hidden episodes in 4 equal categories: sustained, order, recovery, composition.
- 3 uploads per system per day, shared by the Public and Final tabs. Final scores hidden until close.

## 2. Result

| | mean |
|---|---:|
| persistence baseline (u001) | 0.342 |
| first grey-box models (u008) | 0.653 |
| final entry, `submissions/20260929-2130-final5` (Public-verified per system) | **0.769** |

Final entry per system (each one byte-identical to its best Public-scored version):

| system | model | public |
|---|---|---:|
| reservoir | str9 c: groundwater inflow + slow water-quality stock | 0.894 |
| ad_auction | hi1: audience split into 10 breadth bins | 0.878 |
| traffic | dln_s: 7-tick travel delay, lane closure slows/shrinks route b | 0.847 |
| supply_chain | v8b (congestion + commitment) | 0.824 |
| power_grid | v9c per-observable mix | 0.775 |
| wildlife | median of 5 predator-prey models | 0.750 |
| hospital_queue | 50/50 mean of two queue models | 0.729 |
| epidemic | SIRS + hospital + reporting delay (y2) | 0.717 |
| social_contagion | z20 two-community diffusion, short relationship memory | 0.645 |
| market | y3 dealer/inventory model with freeze gate | 0.634 |

## 3. Method in one paragraph

Grey-box modelling: for each system we wrote a small mechanistic ODE from the brief's wording
(stocks, delays, capacity caps, the named mechanisms), fitted it with robust least squares
(Cauchy loss, multistart) to the runs we bought, and judged every change by leave-one-run-out at
the organizer's scale. Uploads were used as experiments: one factor per system per upload, so
each Public score is a clean A/B test on the hidden physics.

## 4. Timeline and score history

| upload | mean | what changed |
|---|---:|---|
| u001 | 0.342 | persistence, all 10 (baseline) |
| u003 | 0.515 | linear relax-to-equilibrium models from the first 1,200 credits |
| u008 | 0.653 | grey-box ODE on every system (wins 9/10) |
| u010b | 0.687 | refits on purchase 3 (composition blocks) |
| u012 | 0.731 | refits at the organizer's scale; structural pass on the weak five |
| u013–u016 | 0.739 → 0.755 | reporting delay (epidemic), toll-elastic demand (traffic, from a 600-tick hold), library picks |
| u017 | 0.754 | power_grid w5 — **lost** 0.016 despite a held-out gain |
| u019 (final4) | 0.768 | reservoir str9 c, ad hi1, traffic dln_s, hospital 50/50, wildlife median-5 |
| u020 (final6) | 0.768 | hospital canon2 alone — **lost** 0.012 |
| final5 | 0.769 | best Public-scored version of every system |

## 5. Key decisions and why

1. **Grey-box over black-box.** Linear and state-space models capped at ~0.54: the true dynamics
   (epidemic waves, queue capacity, integrators, delays) sit outside their class. Mechanistic ODEs
   took us to 0.65 in one step.
2. **Spend credits on regimes nobody had observed.** After the first purchases, refitting on the same
   data stopped paying. The biggest single gains came right after long holds in new regimes
   (traffic +0.063 after a 600-tick harsh hold).
3. **Uploads as A/B tests.** One change per system per upload, all 10 systems at once, so each
   upload answered 10 questions.
4. **Acceptance rule for any change:** mean held-out score up AND the pulse/recovery-shaped run and the
   exam run not worse. Added after social y13 won on average and lost the recovery category publicly.
5. **Gains must be spread over runs, not driven by one.** power_grid w5 won held-out mostly on one long
   run and lost 0.016 publicly. After that, only broad wins were shipped.
6. **The final entry uses Public-verified pieces only.** Held-out and history-based forecasts missed
   by 0.02–0.05 on single systems; the Public score is the only reliable measurement.

## 6. Findings worth telling

- **The simulators read like literal code.** Hard switches and integer delays, not smooth laws:
  - market's freeze is a switch with memory: an open book closes above tax ≈ 0.043 and a closed book
    reopens only below ≈ 0.034–0.040; it starts closed;
  - hospital patients spend exactly 2 ticks in service (recovery hold settles at 23.0 / 11.5 / 0);
  - supply shipments alternate 29.35 / 44.74 on even/odd ticks;
  - traffic flows are exactly 0 for 11 ticks after a start;
  - epidemic vaccination capacity is $1-H/h_{cap}$.
  The constants are not round (reservoir inflow period 67.75), only the structure is literal.
- **Reservoir inflow is a deterministic season locked to reset** (period 67.8): one rule took the
  inflow score from 0.55 to 0.92.
- **ad_auction:** the audience must be tracked per targeting breadth; one pooled audience smeared the
  depleted core over new people when targeting widened.
- **Held-out wins don't always transfer.** Three of four late held-out "wins" lost on Public
  (power_grid w5, social 3-model median, hospital canon2 alone).

## 7. What we'd do differently

- Buy long sustained holds (1,000+ ticks) early on every system: 75%+ of each test episode is past
  the horizon our early data covered.
- Save the free `documents` text on day one.
- Keep a dedicated Public slot each day purely for testing, from the start.

## 8. Where things are

- Code: `groundtruth/gtlab/` (ODE families in `gtlab/ode/`, runtime in `gtlab/runtime/infer.py`).
- Every uploaded model: `groundtruth/submissions/<stamp>-<tag>/` (predict.py + model.json + scores).
- Score history: `groundtruth/tune/registry.json`.
- Notebook: `groundtruth/docs/MATH_LOG.md`; math report: `groundtruth/docs/report/strategy.pdf`.
