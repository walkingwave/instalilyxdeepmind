# Epidemic v8: structural fixes at the organizer's scale

Sat Sep 26. No credits. Data: 4 runs, 1,111 ticks (`p1.hold_rec` 120, `p2.pulse120_280` 400,
`p3.compose` 291, `p4.joint_hold` 300). Score scale: calibrated sigma
$\sigma = (13.0, 5.23)$ for (daily_cases, hospital_load) from `plans/sigma_calibrated.json`.
Every fit: `scripts/ode_lab.py --sigma-cal 1.5 --budget 300-450 --starts 8 --nfev 50`.
Every score below is at exactly $1.0\times\sigma$: $\frac{1}{4}\sum_{runs}$ `metric.score(Y, run.Y, sigma)`,
with Y from `rollout_from_blob` on the lab's `_doc.json` (so clipping is included).

## 1. Error budget before (epidemic_s1, pair AB, `plans/epidemic_epidemic_s1_cal4AB_doc.json`)

Calibrated in-sample 0.668. Loss per tick $= 1 - 1/(1+|e|/\sigma)$, summed; total 741.8 over 2,222 tick-observables.

| source | cases | beds | what the data show |
|---|---:|---:|---|
| p4 long joint hold, ticks 25-300 | 77 | 100 | cases rise slowly 43 -> 61 from t=130; we dip to 35 and overshoot; beds too high at the end (48 vs 40) |
| p2 release, ticks 145-400 | 89 | 71 | rebound after release faster in truth (48 -> 190 in 42 ticks), we are late and low, then high |
| first 25 ticks of every run | 60 | 70 | beds FALL first (54 -> 43 in p1) then rise; cases accelerate over ~5 ticks; we rise immediately |

The third item was the clearest structural miss: `x0` puts $I$ at the quasi-steady ratio
$I_i = E_i\sigma/\gamma_i$ and beds are fed instantly by $I$, so the model cannot produce the
initial bed dip or the slow start of the case curve.

## 2. Changes (each kept only if calibrated in-sample went up)

All versions keep the s1 skeleton: SEIR fractions in 3 age groups ($n = 0.2, 0.55, 0.25$), fixed
contact matrix, school closure removes child-child contacts, masks scale transmission, clinic
$= 1/(1 + k_{clinic} H/h_{cap})$, bed cap via smooth min, waiting list.

| version | change | in-sample AB | in-sample AC |
|---|---|---:|---:|
| s1 | baseline | 0.668 | - |
| v8 (v8a) | $I_i(0) = r_I\,E_i\sigma/\gamma_i$ with fitted $r_I$; referral pipeline $P$ (starts empty, mean delay $\tau_p$); `close_eff` folded into `school` (only the product is identified); `sev_c` = 0; bed-cap smoothing 0.2 % of $h_{cap}$ | 0.713 | 0.393 (bad fit) |
| v8b | waning $R\to S$ always on (every run settles on an endemic plateau, so it is base physics); B re-read as *developing* immunity ($S\to V\to R$, $V$ still susceptible, delay $\tau_{dev}$); `wl_leave` fixed 0.05 | 0.711 | 0.730 |
| v8c | referral pipeline is Erlang-2: $\dot P_1 = \text{ref} - k P_1$, $\dot P_2 = k P_1 - k P_2$, admissions demand $kP_2$, $k = 2/\tau_p$ | 0.713 | 0.735 |
| v8d | vaccination age-targeted, weights $w = (1, 1, 1+v_{eld})$; `imm0` = 0 (fits drove it to 0) | 0.734 | 0.735 |
| v8e/v8f | vaccine efficacy `vac_eff` on the fixed-count vaccination | 0.743 | 0.760 |
| **v8g** | vaccination per capita: $v_i = \text{vac\_eff}\cdot u_{vac}\cdot\text{clinic}\cdot w_i S_i$ | **0.749** | **0.759** |

v8e/v8f were rejected even though in-sample rose: the s1 vaccination is a fixed count,
$v_i = u_{vac}\,\text{clinic}\,S_i/(\sum_j n_j S_j + 0.02)$, so with `vac_eff` ~ 5 it removes
~1.5 %/tick of the population regardless of how few susceptibles remain, outruns waning
($1/\tau_{wane}$ ~ 1.2 %/tick) and drives the model extinct on long vaccination holds: 19 % of test
ticks had predicted cases < 5 (min 0) on 40 test-shaped 4,000-tick schedules. The data never show
this (p4: 300 ticks of vaccination, cases steady at 61). The per-capita form (v8g) gives the same
fit without the extinction: 0.6 % of test ticks < 5.

Pair BC stays weak on every structure (v8a 0.673, v8b 0.686, v8c 0.694, v8g 0.692): the long
restriction data need fatigue (A).

## 3. Final equations (v8g, 22 parameters)

Per age group $i$, with compliance $c = 1 - k_f F$ (A), restriction $r = (u_{clo} + u_{mask})/2$:

- $\lambda_i = \beta\,(1 - m_{eff} u_{mask} c)\,b_C \sum_j C_{ij}(u_{clo} c)\,I_j$, school closure removes a
  share $s\cdot u_{clo} c$ of child-child contacts and moves 30 % of it to child-adult.
- $\dot S_i = -\lambda_i S_i - v_i + R_i/\tau_{wane}$, $v_i = \text{vac\_eff}\,u_{vac}\,\text{clinic}\,w_i S_i$
- $\dot E_i = \lambda_i S_i - \sigma E_i$, $\dot I_i = \sigma E_i - \gamma_i I_i$, $\dot R_i = \gamma_i I_i + v_i - R_i/\tau_{wane}$
  (with B: $v_i$ goes to $V_i$, $\dot V_i = v_i - \lambda_i V_i - V_i/10$, $V_i/10$ enters $R_i$)
- referral $= N\sum_i n_i\,sev_i\,\gamma_i I_i$ into $P_1 \to P_2 \to$ demand (Erlang-2, mean $\tau_p$)
- beds: admissions $= \text{smin}(\text{demand} + WL, \text{room})$, $\dot H = \text{adm} - H/los$, overflow to $WL$
- A: $\dot F = (r - F)/\tau_{fat}$. C: $\dot D = r - (1-r)D/\tau_{debt} - 0.002D$, $b_C = 1 + k_D (1-r) D/\tau_{debt}$
- observation: cases $= N\sigma\sum_i n_i E_i$, hospital_load $= H$
- initial: $E_i$ from observed cases in the fixed age mix, $I_i = r_I E_i\sigma/\gamma_i$, $R = 0$,
  $H$ = observed beds, $P_1 = P_2 = WL = F = D = V = 0$. Both observables anchor the state.

Fitted (AB): $N$ 18,730, $\beta$ 0.273, $\sigma$ 0.158, $\gamma$ = (0.46, 0.58, 0.26), sev = (0, 0.10, 0.12),
los 9.2, $h_{cap}$ 154.6, school 1.0 (bound), mask 0.36, $k_{clinic}$ 20 (bound), $v_{eld}$ 5.3,
$r_I$ 0.40, $\tau_p$ 4.9, $\tau_{fat}$ 251, $k_f$ 0.85, $\tau_{wane}$ 92, vac_eff 4.2.
AC: same family of values, $\tau_{fat}$ 134, $k_f$ 0.78, vac_eff 4.4, $k_D$ 0.11, $\tau_{debt}$ 117; only
`school` at bound (its physical limit: closure removes all school contacts).

## 4. Results at calibrated sigma

In-sample, per run (cases, beds):

| run | s1 AB | v8g AB | v8g AC |
|---|---|---|---|
| p1.hold_rec | 0.652 / 0.704 | 0.704 / 0.828 | 0.743 / 0.790 |
| p2.pulse120_280 | 0.668 / 0.662 | 0.731 / 0.790 | 0.728 / 0.770 |
| p3.compose | 0.649 / 0.682 | 0.711 / 0.729 | 0.707 / 0.822 |
| p4.joint_hold | 0.723 / 0.606 | 0.781 / 0.718 | 0.797 / 0.712 |
| **mean** | **0.668** | **0.749** | **0.759** |

Leave-one-run-out (fit on 3 runs at $1.5\sigma$ from the module defaults, 250 s per fold, score the
held-out run at $1.0\sigma$):

| held out | s1 AB | v8c AC | v8e AC | v8g AB | v8g AC |
|---|---:|---:|---:|---:|---:|
| p1 | 0.424 | 0.701 | 0.727 | 0.722 | 0.684 |
| p2 | 0.444 | 0.422 | 0.542 | 0.511 | 0.475 |
| p3 | 0.566 | 0.442 | 0.390 | 0.494 | 0.413 |
| p4 | 0.477 | 0.654 | 0.584 | 0.690 | 0.670 |
| **mean** | **0.478** | **0.555** | **0.561** | **0.604** | **0.561** |

p3 is the only run with single-control blocks, so dropping it hurts every structure; s1 holds up
best there, v8g AB comes second.

Test-shaped sanity (40 schedules, 10 per category, 4,000 ticks): all finite; 0.6-0.9 s/episode;
v8g AB: 0.6 % of ticks with cases < 5, min (2.7, 1.9); v8g AC: 1.1 %, min (0.1, 0.1).

## 5. Error budget after (v8g AB, total loss 561 vs 742)

| source | cases | beds |
|---|---:|---:|
| p4 long joint hold, ticks 25-300 | 61 | 77 |
| p2 release, ticks 145-400 | 65 | 46 |
| p2 restriction, ticks 25-120 | 23 | 26 |
| first 25 ticks of all runs | 36 | 26 (was 70) |

The initial transient is fixed (beds now dip then rise). What's left is the slow dynamics under
long restriction and release: on p4 the case curve still dips and overshoots where the truth rises
smoothly, and the beds-to-cases ratio under restriction is too high (48/65 vs 40/61 at the end).

## 6. Decision

- Pick **v8g AB** (`plans/epidemic_epidemic_v8g_gAB_doc.json`): in-sample 0.749 (+0.081 over s1),
  LOO 0.604 (+0.126), fewest near-zero test ticks. v8g AC has the higher in-sample (0.759) but
  LOO only 0.561, so its extra fit does not carry over to held-out runs.
- Rejected: v8e/v8f (extinction on long vaccination holds), pair BC (0.69 at best), the s1-style
  fixed-count vaccination with a free efficacy.
- Two parameters sit at bounds in AB: `school` = 1 (its physical limit) and `k_clinic` = 20. The
  second means clinic availability is strongly tied to bed pressure, and it trades off against
  `vac_eff`. AC does not need it (14).
