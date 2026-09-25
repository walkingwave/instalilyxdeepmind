# Credit plan

Measured offline on the 10 mock simulators, two different hidden "worlds" (mechanisms AB seed 0,
BC seed 5). Six experiment designs x five budgets (120-1000 steps/system), scored against
noiseless test-shaped episodes. Script: `scripts/design_study.py`.

## Result (mean over 10 systems, best of the fast models)

| budget/system | plan (hold_rec + hold_pulse first) | best other design |
|---:|---:|---:|
| 0 (persistence) | 0.49-0.52 | - |
| 120 | 0.60-0.62 | 0.61-0.63 |
| **240** | **0.67-0.68** | 0.65-0.67 |
| 420 | 0.65 | 0.65-0.66 |
| 700 | 0.66-0.68 | 0.68 |
| 1000 | 0.71 | 0.72-0.73 (long test-shaped runs) |

- The first 240 steps are worth about +0.17. The next 460 are worth roughly nothing with the
  current models. Past that, long runs add about +0.04.
- So after 240, the bottleneck is the model, not the data. Extra steps only pay once richer
  models (ODE, state space) exist that can use them.
- Long runs beat many short ones at high budgets (slow effects only show up over long holds).
  Sustained operation is also the leaderboard's weak category.

## Schedule

| When | Steps/system | What | Why |
|---|---:|---|---|
| Thu | 240 | hold_rec 120 + hold_pulse 120 | best value per credit measured |
| Thu-Fri | 0 | richer models, public feedback | model is the bottleneck |
| Fri/Sat | ~500 | one long test-shaped run per system, only where residuals show slow drift | the +0.04 at 1000 came from long runs |
| Sat | ~150 | mechanism test from the brief, only where ODE variants disagree | T-optimal: buy only what separates hypotheses |
| Sun-Mon | <=1100 left | reserve: spend where public/local shows the largest gap | never before evidence |

Caveat: the mocks are guesses at the real physics. The shape of the curve (steep then flat) is
the robust part, not the exact numbers. Re-check after Thursday's real data.
