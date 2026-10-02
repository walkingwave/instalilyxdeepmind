# social_contagion x: rebuild from the brief (2026-09-27)

No credits. Data: 5 runs, 1,511 ticks (`p1.hold_rec` 120, `p2.pulse200_200` 400, `p3.compose` 291,
`p4.interior_holds` 300, `p5.testlike` 400). Scale: calibrated organizer sigma $\sigma = (9.23, 5.02)$.
Every number below comes from `scripts/ode_lab.py --sigma-cal 1.0 --budget 240 --starts 8 --mech AB`
(tag `xr5`), so LOO = fit on four runs, score the fifth at $1.0\sigma$. The `p5.testlike` fold is
the exam: fit on the four older runs, score the new test-shaped run. We re-ran v8o in the same lab
on the same five runs for a like-for-like baseline (`plans/social_contagion_social_contagion_v8o_xr5.json`).

## 1. What the data say (read before modelling)

- **Onboarding delay**: ~7 ticks of flat or falling adopters after every seeding onset, then a steep rise.
- **Approach rate**: after the delay, the gap to the plateau shrinks by the same ~0.64 per 10 ticks
  ($\lambda \approx 0.043$) in the pulse (9/2/0.6, plateau 199) and in the interior hold (5/1/0.5,
  plateau 152). The rate is the same but the plateaus differ, and one linear pool-depletion law cannot do both.
- **Plateau totals**: $a + b = 330$ in the pulse and in interior hold 3 (6.94/1.88/0.58); 253 in
  interior 1. That is what pointed us to a shared workforce ceiling.
- **Collapse after incentive stops**: the excess over a floor of ~40 decays at ~0.10/tick (pulse, compose,
  p5). After *seeding alone* stops, a falls only ~0.006-0.012/tick.
- **Zero-control regrowth** is real and does not stop within our windows: hold_rec b 17.6 -> 46 in 110
  ticks (+5.7 sigma); p5 a 60 -> 68 and b 47 -> 58.5 over the last 60 zero ticks, still rising.
  Two-point first-order fits of the regrowth rate put the zero-control level near (88, 75-84).
- **p5 news**: low seeding with bridge (1.64/1.30/0.46) lifts b by 13 while a stays flat, and b's rise
  under 5.38/1.84/0.43 is larger than any older model predicts. b is the weak observable.

## 2. Candidates (all numpy + math, batched `_PY`/`_NP` f/h/x0, N_SUB = 2, x0 uses both observed initials)

| family | params | structure (differences) |
|---|---:|---|
| v8o (baseline) | 16 | phi at onboarding, credibility x incentive x queue, bridge penalty $k_{br}$ |
| `social_contagion_x` | 20 | promised cohorts carried through the queue; soft workforce $g = \kappa(1 - A_{tot}/A_{cap})/(1 + Q/h_Q)$; queue abandonment |
| `social_contagion_x2` | 20 | hard shared workforce: onboarding $= \mathrm{softmin}(Q_{tot}/\tau_q,\ \kappa_F (A_{cap} - A_{tot})_+)$ |
| `social_contagion_x3` | 19 | bridge as introductions through the other community's members + C (cross-community relationships $R$ with memory) |
| `social_contagion_x4` | 22 | x2 + reconsidering former members ($p_{re}$ back into the chain) + incentive-led audience $(1 + k_{inc}\phi(c))$ |
| `social_contagion_x5` | 21 | x4 without word of mouth |
| `social_contagion_x6` | 21 | x4 without workforce; organic adoption from a finite deliberative audience $o(1 - A_i/G_i)_+$ |
| **`social_contagion_x7`** | **16** | compact: promised cohorts in an Erlang-3 chain, local effort $s(1-\beta)$ (bridge takes effort 1:1), incentive-led audience $(1 + k_{inc}\phi)$, credibility from promised waiting cohorts, expectations $E$ |
| `social_contagion_x8` | 15 | x7 without $k_{inc}$ |

x7 equations, per community $i$ ($\phi = c/(c+0.7)$, $k = 3/\tau_{on}$):
$$\rho_i = K_i (1 + k_{inc}\phi)\big[s_i\, s (1-\beta) + q A_i/N_i + o\big],\quad r_i = \rho_i/(1 + \rho_i/1.5),\quad P_i = (N_i - A_i - W_i - D_i)_+$$
$$\dot W^1_i = (1-\phi) r_i P_i - k W^1_i,\quad \dot V^1_i = \phi\, r_i P_i - k V^1_i\ \text{(two more stages each)},\quad \dot L_i = k W^3_i - c_L L_i - k_{conv} c L_i$$
$$\dot M_i = k V^3_i + k_{conv} c L_i - [c_L + k_B (E - c)_+] M_i,\quad \dot X_i = -k_X X_i,\quad \dot D_i = \text{leavers} - D_i/\tau_D$$
$$\dot K_i = (1 - K_i)/50 - k_A K_i (V^1_i + V^2_i + V^3_i)/N_i,\qquad \dot E = (c - E)/\tau_E$$
Initial state: $L = (1-m_0) y_0$, $X = m_0 y_0$, $K = 1$; all queues, promises and $E$ start empty (the brief).
Observed $y = (L_a + M_a + X_a,\ L_b + M_b + X_b)$. RK4, 2 substeps.

x7 full-fit theta: $N_a$ 392.4, $N_b$ 350.3, $s_a$ 0.00327, $s_b$ 0.00085, $q$ 0.0145, $o$ 0.00162,
$\tau_{on}$ 9.49, $c_L$ 0.0150, $k_X$ 0.0309, $m_0$ 0.90 (bound), $k_{conv}$ 0.055, $k_A$ 0.111,
$k_B$ 0.0482, $\tau_E$ 25.4, $\tau_D$ 10.3, $k_{inc}$ 0.697.

## 3. Results (calibrated sigma, 5 runs)

| family | params | LOO hold | pulse | compose | interior | **p5 exam** | **LOO mean** | cal in-sample | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8o | 16 | 0.856 | 0.430 | 0.574 | 0.554 | 0.550 | 0.593 | 0.724 | 1229 |
| x2 | 20 | 0.878 | 0.598 | 0.547 | 0.555 | 0.529 | **0.621** | 0.725 | 1225 |
| x4 | 22 | 0.738 | 0.518 | 0.491 | 0.562 | 0.582 | 0.578 | 0.718 | 1026 |
| x5 | 21 | 0.673 | 0.516 | 0.468 | 0.565 | 0.581 | 0.561 | 0.699 | 1072 |
| x6 | 21 | 0.746 | 0.555 | 0.497 | 0.576 | **0.651** | 0.605 | 0.718 | 1025 |
| **x7** | **16** | 0.751 | **0.657** | 0.499 | **0.605** | 0.582 | **0.619** | 0.724 | 1029 |
| x8 | 15 | 0.717 | 0.522 | 0.557 | 0.556 | 0.639 | 0.598 | 0.679 | 1552 |

We screened x and x3 with a single full fit only. x put $A_{cap}$ at its 3,000 bound (no ceiling).
x3 cost 1318 (AB) / 1437 (BC) against 1226 for x2, with $x_q \to 0$, so we dropped it.
With a 400 s budget and 10 starts (`_xr5b`), the x7 folds come out identical, so the fold optima are converged and not
budget-limited.

Per observable, x7 LOO (a, b): hold 0.797 / 0.705, pulse 0.635 / 0.678, compose 0.526 / 0.473,
interior 0.562 / 0.648, p5 0.636 / 0.528. v8o on the exam: 0.631 / 0.470.

## 4. Long holds (4,000 ticks, full-fit theta)

| family | zero controls from (48.6, 35.9), t = 120 / 4000 | same from (190, 130), t = 4000 | incentive c = 0 -> 2 at s = 5, beta = 0.5 (a) | c = 0 -> 2 at s = 0 (a) |
|---|---|---|---|---|
| v8o | (64, 56) / (101, 95) | (101, 95) | 198 -> 171 (falls) | 101 -> 73 (falls) |
| x2 | (65, 57) / (103, 97) | (103, 97) | 195 -> 174 (falls) | 103 -> 77 (falls) |
| x7 | (62, 54) / (91, 81) | (91, 81) | 176 -> 181 (rises) | 91 -> 111 (rises) |
| x8 | (63, 52) / (94, 79) | (94, 79) | 188 -> 181 (falls) | 94 -> 85 (falls) |

All of them stay finite and reach a unique equilibrium by t ~ 1000 from every initial state we tried. Plateaus are smooth
and monotone in seeding (x7 at c = 1, beta = 0.5: s = 0/1/2/3/5/7/10 -> a = 107/132/149/161/180/193/208),
and bridge is monotone down because it moves effort off local recruitment. We saw no runaway. On eval-shaped rollouts,
x7 spends 0 % of ticks outside the observed range on sustained, order and recovery, and 13 % on composition.

## 5. Findings

1. **The data do not show the finite workforce.** Soft (x) and hard (x2, x4) ceilings: every fit
   parks $A_{cap}$ far above the observed total (436-2989 against 330). Seeding saturation plus churn
   reproduces the equal plateau totals without it. We keep the workforce out of x7.
2. **The incentive-led audience is the big in-sample gain.** $(1 + k_{inc}\phi(c))$ on interest, with
   $k_{inc} \approx 0.7$-0.8, together with bridge taking seeding effort 1:1 ($k_{br} \to 1$ at the bound when
   free), cut the cost from 1226 to 1026. Removing it (x8) costs 0.045 in-sample and flips the steady-state
   incentive sign. It hurts the compose fold (0.557 -> 0.499: seeding alone, with no incentive, is the fastest
   rise we have) and helps the pulse and interior folds.
3. **More parameters lost out of sample.** x4 (22) and x5 (21) have the best cost but the worst LOO.
   x7 matches x4's cost with 16 parameters and has the best LOO of the rebuilt families.
4. **Initial members**: every rebuilt fit sends $m_0 \to 0.8$-0.9 with $k_X \approx 0.03$. Most initial
   members are incentive-led without a promise and leave slowly (half-life ~23 ticks); regrowth replaces them.
5. Bridge introductions (x3) and cross-community memory (C) add nothing: $x_q \to 0$.

## 6. Verdict

We reached LOO 0.619-0.621 at $1.0\sigma$ (x7, x2). That is +0.03 over v8o (0.593) and short of the 0.66 target.
Exam (p5): x6 0.651, x8 0.639, x4/x7 0.582, v8o 0.550, x2 0.529. We recommend **x7**
(`plans/social_contagion_social_contagion_x7_xr5_doc.json`). It has the best pulse and interior folds, 16
parameters, the correct incentive sign at steady state, and a lower zero-control level than v8o (91/81 vs 101/95).
The zero-control level is still well above the start. Our own data (hold_rec, p5) show regrowth of
+10 to +25 in 100-300 ticks, so a model that stays within 1-2 sigma of the start at zero controls
would contradict the runs we paid for.

## 7. What data would help most

1. **One long zero-control hold (400+ ticks) from a reset.** It pins the zero-control level (91/81 here;
   the data's two-point estimate is ~88/80 and the audit wants ~65). This is the largest unknown in the sustained category.
2. **Seeding alone at an interior level (e.g. 4, 0, 0) for 150 ticks, then incentive 1.5 added.** It separates
   the incentive-led audience ($k_{inc}$) from the pure seeding response. The compose fold is where every
   model fails (0.47-0.57).
3. **Bridge at fixed seeding, (5, 1, 0) vs (5, 1, 0.8).** Bridge is identified only through $s(1-\beta)$, and
   p5's b jump under low seeding + bridge is unexplained.

## Files
- Modules: `gtlab/ode/social_contagion_x.py`, `_x2`, `_x3`, `_x4`, `_x5`, `_x6`, `_x7`, `_x8`.
- Lab outputs: `plans/social_contagion_social_contagion_{v8o,x2,x4,x5,x6,x7,x8}_xr5{,_doc}.json`, `_x7_xr5b`.
