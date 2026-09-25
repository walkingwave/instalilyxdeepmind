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
