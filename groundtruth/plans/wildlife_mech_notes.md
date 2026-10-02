# wildlife: mechanism-pair search (`gtlab/ode/wildlife_mech.py`)

Date: 2026-09-26. Data: the same three runs as v7 (`p1.hold_rec` 120, `p2.pulse200_200` 400,
`p3.compose` 291; 811 ticks). Fitter: `scripts/ode_lab.py --budget 300 --starts 10 --nfev 60`,
Cauchy least squares in units of `sigma_proxy` = [54.3, 1.10, 44.8, 1.48]. No credits spent.

## Why

v7 (`wildlife_min`) scores LOO 0.843 / in-sample 0.897 on our runs but 0.636 public (sustained 0.611).
Its lagged density $L$ is neither of the brief's mechanisms, so we built one family where each named
mechanism is a real state and fitted the three pairs, AB / AC / BC, on identical data.

## Model

Per region $i \in \{N, S\}$ (scale $s_N = 1$, $s_S = k_s$), observed prey $= P_i + J_i$ (v1) or $P_i$ (r2):

$$b_i = r\,F_i\,P_i,\qquad F_i = R_i \text{ if B else } 1,\qquad
\text{take}_i = h_q u_{hunt} (1 - e_i u_{hab}),\ e_N = e_n,\ e_S = 0$$

- **A (juvenile condition)**: $\dot J_i = b_i - m_J J_i - \text{take}_i J_i$; of the juveniles leaving the
  nursery a fraction $1/(1 + J_i/(c_J s_i))$ matures, the rest die (food competition), so recruitment
  $= m_J J_i / (1 + J_i/(c_J s_i))$. Without A, $J = 0$ and recruitment $= b_i$.
- **B (finite food renewal)**: $\dot R_i = w\,(1 - b_h(1-u_{hab}))(1 - R_i) - k_R (P_i/s_i) R_i$, $R \le 1$.
  Without A the births carry no crowding, so B alone must regulate.
- **C (settlement competition)**: arrivals settle with probability $\max(0, 1 - P_{dest}/(k_{sc} s_{dest}))$,
  unsettled arrivals are lost. Without C all arrivals settle.
- Always on: adult death $d_0 P_i$; emigration $e_m u_{cor} P_i$ into a transit pool per direction,
  arriving at rate $1/\tau$ (travellers still arrive after the corridor closes); predators
  $\dot Q_i = s(q_0 - q_c u_{cor} - Q_i)$ as in v7.

$$\dot P_i = \text{recruit}_i - d_0 P_i - \text{take}_i P_i - e_m u_{cor} P_i + \text{settle}_i$$

Reset: $P, Q$ from $y_0$; $J = 0$, $R = 1$, transit pools empty. 10 states, 16 parameters
(base 10: $r, d_0, k_s, h_q, e_n, s, q_0, q_c, e_m, \tau$; A: $m_J, c_J$; B: $w, k_R, b_h$; C: $k_{sc}$),
RK4 with 2 substeps. Parameters of the inactive mechanism have no effect (their values in a pair's
theta are meaningless, including "at bound").

## Results (three runs, LOO = leave one run out, own `sigma_proxy`)

| model | LOO hold_rec | LOO pulse | LOO compose | **LOO mean** | in-sample | full cost | at bound (active params) |
|---|---|---|---|---|---|---|---|
| l0b_lin | 0.739 | 0.819 | 0.822 | 0.793 | | | |
| v7 `wildlife_min` | 0.855 | 0.848 | 0.827 | 0.843 | 0.897 | | none |
| AB (obs $P+J$) | 0.845 | 0.857 | 0.824 | 0.842 | 0.896 | 103 | $d_0$ (0.5 hi), $m_J$ (1.5 hi), $k_{sc}$ inactive |
| AC (obs $P+J$) | 0.813 | 0.864 | 0.824 | 0.834 | 0.870 | 187 | $d_0$ (hi), $\tau$ (100 hi) |
| BC | 0.802 | 0.859 | 0.793 | 0.818 | 0.874 | 132 | none |
| AB r2 (obs $P$) | 0.855 | 0.839 | 0.838 | **0.844** | **0.899** | 101 | $d_0$ (hi), $m_J$ (hi) |
| AC r2 (obs $P$) | 0.798 | 0.862 | 0.849 | 0.836 | 0.866 | 201 | $\tau$ (hi), $m_J$ (hi), $k_{sc}$ (20 lo) |

Files: `plans/wildlife_wildlife_mech_{AB,AC,BC,ABJ_r2,ACJ_r2}.json` (+ `_doc.json`), logs
`plans/wildlife_mech_*.log`.

Ranking: AB r2 $\approx$ AB $\approx$ v7 $>$ AC $>$ BC. The top three are within 0.002 LOO: a tie.

### Fitted theta, AB r2 (best)

| $r$ | $d_0$ | $k_s$ | $h_q$ | $e_n$ | $s$ | $q_0$ | $q_c$ | $e_m$ | $\tau$ | $m_J$ | $c_J$ | $w$ | $k_R$ | $b_h$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.737 | 0.500 | 0.813 | 0.0208 | 0.405 | 0.0641 | 2.446 | 0.759 | 0.0443 | 70.3 | 1.50 | 352 | 0.0411 | 1.0e-4 | 0.390 |

($k_{sc}$ inactive.) Food stock at the recovery equilibrium $R^* \approx 0.77$, under the pulse 0.97; food renewal time
$1/w \approx 24$ ticks, which is the old lag $1/\rho = 26$; habitat 0 cuts renewal by $b_h = 39\%$.

### Revision r2 (the one allowed)

In v1 the juveniles are counted in the prey total, so the stage adds to the count at once. r2 counts
adults only (switch letter `J` in the mechanism set, e.g. `--mech ABJ`), so the nursery can act as a
recruitment delay. Result: $m_J$ still runs to its upper bound (1.5/tick, mean nursery stay 0.7 tick)
in both AB r2 and AC r2. The data do not want a juvenile delay; A is used only as an instantaneous
crowding of recruitment, $m_J J/(1+J/(c_J s_i))$ at quasi-steady state, i.e. v7's $rP/(1+P/c)$.

## What each pair does with the data

- **AB**: B's food stock $R$ takes over the role of v7's lagged density $L$ (single overshoot after every
  release, settle at 123/100; observed 121/97) and habitat now acts where the brief says, on renewal.
  A collapses to instant nursery crowding. $d_0$ at its upper bound: the fit wants a high-turnover
  population (births 0.74 R, deaths 0.5), which is again the crowding shape.
- **AC**: no food stock, so **habitat alone has no effect** apart from hunting shelter: the 0.235 habitat
  block (observed $137 \to 76$) is missed entirely, and the overshoot is lost on recovery (max 127 on
  4,000-tick schedules vs 195 observed). Its LOO survives only because the habitat block is 45 ticks of one run.
- **BC**: without A there is no instantaneous crowding; the reset growth ($0.07$/head at $P = 86$) and the
  release growth ($0.3$/head at $P = 7$) cannot both be matched by $rR - d_0$ with $R \approx 1$ in both
  cases. Worst hold_rec fold and the compose fold falls below l0b_lin.
- **C in every pair**: $k_{sc}$ goes to a bound (20 = arrivals almost never settle, or 1,229-2,000 = always
  settle) and $\tau$ to 66-100 ticks. The transit pool is used as a slow sink/source, not as settlement
  competition. With one 45-tick corridor block C is not identified.

## Behaviour on 4,000-tick eval-shaped schedules (from `p1.hold_rec` $y_0$)

All pairs: finite, no sustained oscillation, 0 % outside the data range, one damped overshoot per
release then flat (last-quarter variation < 1 % on every hold except extinction tails).

| hold (400 ticks after 100 at recovery) | observed | v7 | AB | AB r2 | AC | BC |
|---|---|---|---|---|---|---|
| recovery settle N / S | 121 / 97 | 121 / 100 | 123 / 100 | 123 / 100 | 126 / 102 | 127 / 103 |
| pulse (7, 0.1, 1) N / S | **7.0 / 7.5** | 16.2 / 7.9 | 6.9 / 6.2 | 8.4 / 7.4 | 11.0 / 9.5 | 9.9 / 9.0 |
| peak after pulse release N | 206 | 163 | 169 | 173 | 128 | 160 |
| habitat 0 alone N / S | (0.235: 76 / 72) | 81 / 67 | 83 / 68 | 88 / 71 | 126 / 102 | 90 / 73 |
| quota 7, habitat 1 N / S | (5.95: 67 / 28, falling) | 53 / 13 | 48 / 7 | 50 / 9 | 64 / 8 | 49 / 11 |
| corridor 1 alone N / S | (0.85: 110 / 80) | 121 / 86 | 119 / 104 | 119 / 104 | 117 / 103 | 109 / 94 |
| max stress (8, 0, 1) N / S | unseen | 7.8 / 3.4 | 1.0 / 1.0 | 1.5 / 1.5 | 2.4 / 2.3 | 3.1 / 3.1 |

Transfer notes. (1) v7's worst systematic error is the pulse level: 16 in the north against 7 observed,
because its habitat-dependent lag target cannot combine with the north shelter at habitat 0.1. Every
mech pair fixes this (6.9-11), which matters for recovery-shaped tests where holds sit near the pulse.
(2) The new pairs drive prey to about 1 under quota 8 / habitat 0 / corridor 1, where v7 stays at 3-8;
this corner is unobserved, so it is a guess either way. (3) The corridor-alone response is worse in the
mech pairs (south barely moves, observed $-15\%$): symmetric emigration with slow transit cannot
reproduce v7's south-only drain. (4) After a corridor closes, AB releases its transit pool (peak 152 at +17
ticks); compose shows a 20-head rise over 20 ticks after the corridor closed, so some of this is real.

## What the data can and cannot distinguish

- B is supported: habitat-alone loss needs a slow state that habitat drives; only pairs with B get it.
- A as a delayed nursery is not supported ($m_J$ at its upper bound in all four A fits); A as instant
  recruitment crowding is needed (BC fails without it). We cannot tell "A active" from "crowding is
  part of the base physics".
- C is not identified: one 45-tick corridor block, settlement capacity always at a bound.
- AB vs v7 is a tie on LOO (0.844 vs 0.843). AB's pulse level is closer to the data, v7's corridor
  response is closer. A corridor-only run (on/off at two levels, ~150 ticks) is the purchase that would
  separate C from a plain drain.

## Verdict

`ode_lab` verdict for AB r2: ship (LOO 0.844 vs l0b_lin 0.793). Against v7 it is not a clear win on our
runs; its case for the public board is the pulse level (north 8.4 vs v7 16.2, observed 7.0) and a
mechanism structure the brief names. A 50/50 average of v7 and AB r2 is the cheap hedge.

