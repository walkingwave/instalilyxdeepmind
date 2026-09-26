# social_contagion: minimal grey-box ODE (`gtlab/ode/social_contagion_min.py`)

Date: 2026-09-25. Data: 2 runs, 520 ticks (`p1.hold_rec` 120 ticks at all-zero controls,
`p2.pulse200_200` 200 ticks at the pulse (seeding 9, incentive 2, bridge 0.6) then 200 at recovery).
`sigma_proxy` = (63.9, 36.8), so the 5-tick dips are almost free; the plateau, the post-stop
decay and the organic regrowth carry the score.

## What the data pinned down before any fitting

- Both runs start with a dip: ~30% of the initial members leave in ~15 ticks (per-capita loss
  ~0.028/tick at t=1 in both runs, independent of the incentive).
- Under the pulse, `a` rises 28 -> 199 and sits on a hard plateau of ~200; the approach is
  first order with $\tau \approx 23$ (residual pool 73, 51, 37, 26, 20, 15, 11, 8 at 8-tick steps).
  `b` rises 36 -> 131 more slowly ($\tau \approx 40$-50).
- After the stop, the excess over the floor decays as a clean exponential: log-excess
  $-0.42, -1.1, -1.8, -2.6$ at $t = 4, 12, 20, 28$, i.e. a constant $0.09$-$0.10$/tick, and
  78% of the pulse recruits leave (floor 43 for `a`, 33 for `b`). A constant rate on a fixed
  fraction rules out a decaying "expectation" churn (that would saturate the log-excess at
  $\ln(1/0.22) = 1.5$; we see 2.6). So there are two member classes: loyal and incentive-led.
- Regrowth from the floor is the same as the hold_rec growth (~0.25/tick at $A \approx 40$,
  decelerating), consistent with word of mouth $qA(1 - A/N)$ with $q \approx 0.05$ and no
  "disappointed" memory (the pool after the pulse behaves like the pool in hold_rec).

## Model (6 states, 12 parameters)

Per community $i \in \{a, b\}$: loyal members $L_i$, incentive-led members $M_i$, onboarding
queue $W_i$; pool $P_i = \max(N_i - L_i - M_i - W_i, 0)$ (smooth). Observed $A_i = L_i + M_i$.
Controls: seeding $s$, incentive $c$, bridge $\beta$.

$$
\text{seed}_a = s_a\, s\,(1 + k_{inc} c)(1 - \beta), \qquad
\text{seed}_b = s_b\, s\,(1 + k_{inc} c)\,\beta
$$
$$
r_i = \frac{\rho_i}{1 + \rho_i / 1.5}, \quad \rho_i = \text{seed}_i + q\,A_i / N_i
\qquad (\text{recruitment per pool member, saturating at } 1.5/\text{tick})
$$
$$
\dot W_i = r_i P_i - W_i/\tau_{on}, \qquad
\dot L_i = (1 - \phi)\,W_i/\tau_{on} - c_L L_i, \qquad
\dot M_i = \phi\,W_i/\tau_{on} - \frac{k_M}{1 + k_{ret} c}\,M_i
$$
$$
\phi = \phi_{max}\,\frac{c}{c + 0.7}
$$

Churned members return to the pool. Initial state: $L_i = (1 - m_0) y_{0,i}$, $M_i = m_0 y_{0,i}$,
$W_i = 0$ (queues begin empty, per the brief). RK4 with 2 substeps per tick; states clipped to
$[0, 10^4]$. Mechanism letters are accepted but every term is always on.

## Fitted parameters (full fit, both runs, budget 240 s, 10 starts)

| parameter | value | bounds | meaning |
|---|---:|---|---|
| $N_a$ | 213.9 | 60-3000 (log) | community a population |
| $N_b$ | 155.5 | 40-3000 (log) | community b population |
| $s_a$ | 0.00356 | 1e-4-1 (log) | seeding rate per pool member, community a |
| $s_b$ | 0.00049 | 1e-4-1 (log) | seeding rate per pool member, community b |
| $q$ | 0.0351 | 1e-3-1.5 (log) | word-of-mouth rate |
| $k_{inc}$ | 1.64 | 0-5 | incentive boost on seeding |
| $\tau_{on}$ | 12.6 | 1-100 (log) | onboarding lag, ticks |
| $\phi_{max}$ | 1.0 (AT BOUND) | 0-1 | incentive-led share of recruits at large incentive; $\phi(2) = 0.74$ |
| $c_L$ | 0.0165 | 1e-4-0.5 (log) | loyal churn per tick |
| $k_M$ | 0.0906 | 0.005-1.5 (log) | incentive-led churn per tick at zero incentive |
| $k_{ret}$ | 20.3 | 0-50 | churn suppression by incentive: churn at $c = 2$ is $0.0022$ |
| $m_0$ | 0.115 | 0-0.9 | incentive-led share of the initial members |

Fit cost (Cauchy, both runs) 10.7. $\phi_{max}$ at its bound is benign: it only says nearly every
recruit at incentive 2 is incentive-led (0.74 against the 0.78 read off the post-stop floor).

## Scores (own `sigma_proxy`)

| fold | ode (a, b) | mean | l0b_lin (a, b) | mean | persistence |
|---|---|---:|---|---:|---:|
| LOO, hold_rec held out | 0.876, 0.874 | 0.875 | 0.906, 0.854 | 0.880 | 0.847 |
| LOO, pulse200_200 held out | 0.643, 0.685 | 0.664 | 0.598, 0.599 | 0.598 | 0.554 |
| LOO mean | | 0.769 | | 0.739 | 0.700 |
| in-sample hold_rec | 0.906, 0.903 | 0.905 | | | |
| in-sample pulse200_200 | 0.919, 0.930 | 0.925 | | | |
| in-sample mean | | 0.915 | | | |

The pulse-held-out fold moves with the multistart seed (0.708 in a 150 s run, 0.664 in the 240 s
run) because hold_rec alone does not identify the seeding and incentive parameters.

Eval-shaped 4,000-tick rollouts (finite, seconds, fraction of ticks outside the observed range
$\pm 5\%$): sustained 0.3 s / 0.10 (b overshoots to 139 against 131 seen), order 0.3 s / 0.00,
recovery 0.3 s / 0.00, composition 0.3 s / 0.02. No collapse anywhere; a stays above 41, b above 19.

Verdict from the lab: ship (LOO 0.769 vs l0b_lin 0.739, +0.03). Honest reading: a tie on the
hold_rec fold (a is 0.03 below l0b_lin, b 0.02 above), a clear win on the pulse fold, and a
model that behaves at every control level.

## Structures tried

| version | change | LOO hold_rec | LOO pulse | in-sample | eval | result |
|---|---|---|---|---|---|---|
| v1 | mix $\phi = c/(c + \phi_{half})$, $\phi_{half}$ free | 0.862 | 0.667 | 0.919 | sustained collapses to (3, 2) | rejected |
| v2 | fixed mix $\phi = \phi_0$ (no control dependence) | 0.821 | 0.503 | 0.916 | fine | rejected |
| v3 | $\phi = \phi_{max}\, c/(c + 0.7)$, half-point fixed | 0.876 | 0.708 | 0.915 | fine | kept |

- v1: the fit drove $\phi_{half}$ to its lower bound (0.01) because only incentives 0 and 2 were
  observed. At an incentive of 0.03 that routes 77% of recruits into the incentive-led class,
  whose churn (0.09) exceeds word of mouth (0.03), and the community collapses. A cliff in the
  control, not a data feature; the eval schedules hold random intermediate incentives.
- v2: a control-independent mix has no cliff, but 35% of the pulse recruits become loyal, so the
  post-stop floor lands at ~62 instead of 43, and the model cannot separate the 30% initial
  churn from the 78% post-pulse churn. Worst LOO.
- v3: fixing the half-point at 0.7 keeps the mix smooth in the incentive (phi(0.03) = 0.04,
  phi(2) = 0.74 phi_max) and removes the cliff; stability at zero seeding needs
  $(1 - \phi) q > c_L$, which holds for every incentive with the fitted values.
- Not tried (out of state budget): a "disappointed" pool with a return delay; the regrowth
  rates after the pulse and in hold_rec match at equal $A$, so the data does not ask for it.

## Caveats

- Bridge outreach is only observed at 0.6; we split seeding as $(1 - \beta)$ to `a` and $\beta$
  to `b` (the brief: local recruitment vs introductions across communities). $s_a$, $s_b$ absorb
  the 0.6, so the model's response to other bridge values is a structural guess.
- Incentive is only observed at 0 and 2; $k_{inc}$, $k_{ret}$ and the mix half-point are set by
  form, not data. The half-point is fixed for that reason.
- The LOO fold that holds out the pulse run trains on hold_rec alone, which cannot identify the
  seeding or incentive parameters; that fold measures the parameter inits as much as the
  structure.
- `b`'s pulse response comes out mostly as word of mouth plus churn suppression ($s_b$ small);
  a sustained seeding with zero incentive would move `b` little in this model.

## 2026-09-26: v2 structure on three runs (p3.compose added)

Data: 3 runs, 811 ticks (`p1.hold_rec` 120, `p2.pulse200_200` 400, `p3.compose` 291: each
control alone at 85% of the pulse for 45 ticks with 22-tick recoveries between, then the joint
pulse 45, recovery 45). `sigma_proxy` = (59.6, 32.4). Lab: `scripts/ode_lab.py --budget 240
--starts 10 --nfev 60`, tags `p3` (old structure), `v2`, `v3`.

### What the compose run told us that the first two runs could not

- Seeding alone (7.65, incentive 0, bridge 0): `a` 37 -> 193 in 45 ticks with a 6-tick
  onboarding delay, and **`b` 30 -> 83 at bridge 0**. The old model routed all of `b`'s
  seeding through the bridge, so it gave `b` nothing here (in-sample compose 0.71 on `b`).
- After seeding stops the recruits **stay**: `a` drains its queue to 207 at t=54, then drifts
  down at 0.7%/tick (207 -> 190 by t=66). Seeding recruits at zero incentive are loyal.
- Incentive alone (1.7, t 67-112): `a` keeps the same slow drift (190 -> 144), `b` flat at
  93-95. Nothing joins, nothing visibly leaves. But when the incentive stops, `a` 140 -> 79 and
  `b` 93 -> 48 in 22 ticks: log-excess slope $0.07$-$0.09$/tick, the same rate as the
  post-pulse collapse in `pulse200_200`. So a paid incentive **converts** existing loyal
  members into incentive-expecting ones (about half in 45 ticks at 1.7, i.e.
  $k_{conv} c \approx \ln 2 / 45 = 0.015$/tick), and they leave when it ends. This is the
  brief's "incentive expectations retain history".
- Bridge alone (0.51, t 134-179): `a` 78 -> 64, `b` 47 -> 43, i.e. the tail of the collapse
  above; no bridge effect is visible. The per-seeding recruit rates at bridge 0 (compose),
  0.51 (joint) and 0.6 (pulse) agree within noise for both communities.
- Incentive does not boost seeding: the joint segment recruits at the same per-seeding rate
  as seeding alone (old $k_{inc}$ went to 0 as well).
- Every run starts with a dip (hold_rec 23% by t=15, compose and pulse 9% by t=5) **even under
  incentive 2**, so it is not the incentive-led class of the old model (whose churn the
  incentive suppresses). It is a class of initial members that leaves regardless.

### Model v2 (8 states, 12 parameters, `gtlab/ode/social_contagion_min.py`)

Per community $i$: loyal $L_i$, incentive-expecting $M_i$, onboarding queue $W_i$, initial
leavers $X_i$; pool $P_i = \max(N_i - L_i - M_i - W_i - X_i, 0)$ (smooth); observed
$A_i = L_i + M_i + X_i$. Controls: seeding $s$, incentive $c$; bridge $\beta$ only through
$k_{br}$.

$$
\text{seed}_a = s_a\, s\,(1 - k_{br}\beta), \quad \text{seed}_b = s_b\, s\,(1 + k_{br}\beta), \quad
r_i = \frac{\rho_i}{1 + \rho_i/1.5},\ \rho_i = \text{seed}_i + q A_i/N_i
$$
$$
\dot W_i = r_i P_i - W_i/\tau_{on}, \qquad \phi = \frac{c}{c + 0.7}, \qquad
\text{conv}_i = k_{conv}\, c\, L_i
$$
$$
\dot L_i = (1-\phi) W_i/\tau_{on} - c_L L_i - \text{conv}_i, \quad
\dot M_i = \phi W_i/\tau_{on} + \text{conv}_i - \frac{k_M}{1 + k_{ret} c} M_i, \quad
\dot X_i = -k_M X_i
$$

Initial state $L_i = (1 - m_0) y_{0,i}$, $X_i = m_0 y_{0,i}$, $M_i = W_i = 0$. RK4, 2 substeps,
states clipped to $[0, 10^4]$. Changes vs the old structure: `b` gets seeding at any bridge;
$\phi_{max}$ dropped (it sat at 1); $k_{inc}$ dropped (it sat at 0); conversion $L \to M$ under
incentive added; initial-leaver class $X$ added (churn shares $k_M$).

### Fits (same 3 runs, same lab settings)

| version | LOO hold_rec | LOO pulse | LOO compose | LOO mean | in-sample hold / pulse / compose | mean | at bound |
|---|---|---|---|---|---|---|---|
| old (`p3`) | 0.818 | 0.538 | 0.712 | 0.689 | 0.881 / 0.917 / 0.736 | 0.845 | $k_{inc}$, $\phi_{max}$, $m_0$ |
| v2 (kept) | 0.745 | 0.648 | 0.701 | 0.698 | 0.870 / 0.914 / 0.901 | 0.895 | $k_{br}$ = 0 |
| v3 (`min2`) | 0.751 | 0.622 | 0.695 | 0.689 | 0.871 / 0.887 / 0.900 | 0.886 | $\rho$ = 1 |
| l0b_lin | 0.712 | 0.535 | 0.707 | 0.651 | | | |

v2 theta: $N_a$ 249.9, $N_b$ 229.5, $s_a$ 0.00449, $s_b$ 0.00105, $q$ 0.0124, $\tau_{on}$ 8.41,
$c_L$ 0.00445, $k_M$ 0.0863, $k_{ret}$ 3.56, $k_{conv}$ 0.0177, $m_0$ 0.165, $k_{br}$ 0 (cost 30.7).
$k_{br}$ at 0 is the data speaking: no bridge effect between 0 and 0.6. Compose in-sample
0.888 / 0.914: the model tracks the seeding rise, the hold at ~190, the post-incentive collapse
(69 vs 78 at t=134) and the joint pulse (170 vs 170 at t=245); it misses the 6-tick onboarding
delay (first-order lag starts rising at once) and is ~10 high on `a` during the incentive-alone
drift.

v3 tried a cap on the incentive-led share, $\text{conv}_i = k_{conv} c L_i (1 - M_i/(\rho N_i))_+$,
with the bridge removed. The fit pushed $\rho$ to 1 and the cost rose to 38.8: the 200-tick
pulse run wants nearly every member convertible (78% left after the stop). Rejected; kept as
`social_contagion_min2.py` with its lab json (`_v3`).

Eval-shaped 4,000-tick rollouts, v2: finite, 0.3 s each; fraction outside the observed range
$\pm 5\%$: sustained 0.00, order 0.29, recovery 0.12, composition 0.26. The excursions are
extrapolations the data cannot check: 400 ticks of seeding at zero incentive send `b` to 180
(loyal recruits with $c_L = 0.0045$ fill $N_b = 229$; we only saw `b` at 131 under the pulse
and still rising), and 400 ticks of incentive then a stop empty $L$ through conversion, so
`a` bottoms at 18 before word of mouth regrows it (the data floor after 200 incentive ticks was
43). v3 gave 146 and 29 for the same schedules because its $N_b$ landed at 180, not because
of the cap.

### Verdict

Ship v2 as the ODE member for social_contagion: LOO 0.698 vs l0b_lin 0.651 (+0.05), in-sample
0.895 vs 0.845 for the old structure, compose 0.901 vs 0.736. The LOO gain over the old
structure is only +0.01 because the folds that hold out compose or pulse train on runs that
cannot identify $k_{conv}$, $s_b$ at bridge 0 or the plateau, so those folds score the initial
values as much as the structure. Limits: $N_b$ and the long-seeding plateau of `b` are
unobserved; onboarding in the data is a pure ~6-tick delay while the model uses a first-order lag; bridge has no
identified effect; the disappointed-pool delay in the brief is not modelled (regrowth after the
compose collapse was 0.12/tick vs 0.25/tick after the pulse at the same `a`, which a
slow-returning pool would explain, but it needs 2 more states).
