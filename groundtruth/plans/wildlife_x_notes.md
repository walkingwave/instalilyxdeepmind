# wildlife x: rebuilding the structure from the brief (`gtlab/ode/wildlife_x*.py`)

Date: 2026-09-27. Data: the same 4 runs as v8 (`p1.hold_rec` 120, `p2.pulse200_200` 400,
`p3.compose` 291, `p4.corridor_habitat` 300; 1,111 ticks). No credits spent, nothing uploaded.
Judge: `scripts/ode_lab.py --sigma-cal 1.0` (Cauchy least squares and scoring at exactly the calibrated
organizer sigma $\sigma = (9.28, 0.187, 7.66, 0.253)$), budget 300 s (LOO folds 150 s each), 8 starts,
60 evaluations, every fold warm-started from the full-data theta (mildly optimistic for every model alike).
The folds are deterministic: rerunning v8h, x6 and x13 at budget 400 (folds 200 s) reproduced the first
three folds to the third decimal (the fold fits converge before the time limit); they still depend on the warm start.
"Calibrated in-sample" = the lab doc rolled through `gtlab.runtime.infer.rollout_from_blob`, scored
$\frac1{Tp}\sum 1/(1+|e|/\sigma)$ against the 4 runs.

## 1. Baseline at the organizer's scale

v8h refitted by the lab at $1.0\sigma$ (`plans/wildlife_wildlife_v8h_xbase10.json`):

| held out | prey_N | pred_N | prey_S | pred_S | mean |
|---|---|---|---|---|---|
| p1.hold_rec | .672 | .694 | .689 | .574 | 0.657 |
| p2.pulse200_200 | .525 | .666 | .546 | .717 | 0.614 |
| p3.compose | .470 | .598 | .521 | .771 | 0.590 |
| p4.corridor_habitat | .690 | .564 | .705 | .535 | 0.623 |
| **LOO mean** | | | | | **0.621** |

In-sample 0.715 (the shipped v8h doc: 0.709). The in-sample/LOO gap is 0.09: each run carries a regime
the others do not (the pulse floor, the low-habitat crash, corridor 0.5 and habitat 0.5), so LOO measures
how well the *structure* transfers, and that is what the public score rewards (0.709 in-sample -> 0.697 public).

## 2. What the brief says that v8h did not have

Line by line (`kit/briefs.md`, wildlife):

| brief | v8h | rebuilt |
|---|---|---|
| open pasture, mixed cover, sheltered browse differ in feeding, hunting exposure, predation | one prey pool per region | exposed pool (open + mixed) and sheltered pool; only exposed animals are hunted, die of exposure, or feed predators |
| habitat protection changes shelter ... especially in the north | exposure death $d_{h,i}(1-u_p)$ | shelter capacity $K_i = s_i c_h u_p$ (per region in x10) plus exposure death on the exposed pool only |
| food renewal shares a finite resource | births $\propto R$ | the food stock sets the nursery crowding scale ($c_J s_i R$), regrowth partly self-seeding ($r_l$) |
| young animals compete for nursery food | instantaneous $b/(1+P/c_J)$ with $c_J \approx 900$ (inactive) | explicit nursery pool $J$, maturation $m_J$, crowding against the food stock |
| arrivals compete for settlement space | constant transit survival $s_v$ | tested as mechanism C (x12) |
| regional totals alone do not identify patch occupancy, juvenile condition or animals in transit | | all three are hidden states (occupancy is algebraic, see §4) |

## 3. Models tried (all numpy + math, batched f/h/x0, N_SUB = 2, all four observed initials used)

| family | change vs previous | params | calibrated in-sample | LOO @1.0σ (p1, p2, p3, p4) | LOO mean |
|---|---|---|---|---|---|
| v8h (baseline) | | 18 | 0.715 | .657 .614 .590 .623 | 0.621 |
| x2 | v8h with adult death $d$ free | 19 | 0.715 | | (no gain in-sample: $d \to 0.64$) |
| x | exposed/sheltered pools, relocation rate $m$, harvest on exposed only | 22 | 0.718 | .670 .617 .612 .612 | 0.628 |
| x3 | x + food level set by habitat, predators feed on exposed prey ($\phi$), south habitat effect $x_S$ | 22 | 0.723 | .697 .646 .619 .557 | 0.630 |
| x4 | x + nursery crowding scale $\propto$ food stock ($c_J s R$), births not $\propto R$ ($\beta \to 0$) | 22 | 0.7255 | .698 .623 .596 .575 | 0.623 |
| x5 | x4 + predators feed on exposed prey | 22 | 0.740* | .706 .615 .650 .586 | 0.639* |
| **x6** | x5 + exposure death, shelter $\propto u_p$ (no base shelter) | 22 | **0.735** | .690 .644 .640 .611 | **0.646** |
| x7 | x6 + explicit nursery pool $J$ | 22 | 0.745* (0.728 converged) | stopped | |
| x8, x9 | x7 with per-region shelter / fixed relocation | 22 | not fitted (superseded by x10) | | |
| x10 | occupancy at equilibrium (no stiff relocation), nursery pool, per-region shelter and exposure death | 22 | 0.7345 | .682 .665 .614 .593 | 0.639 |
| x11 | x10 with one shelter slope, per-region habitat effect on renewal | 22 | 0.7349 | | (no gain) |
| x12 AC / BC | x10 with settlement competition C, pairs AC and BC | 20 / 21 active | 0.701 / 0.696 | | (AB stays best) |
| **x13** | x10 with one shelter slope and self-seeding regrowth $r_l$ | 22 | **0.7336** | **.683 .714 .614 .602** | **0.653** |
| x14 | x10 + predation on exposed prey $aQE$, one exposure death | 22 | 0.7354 | .687 .634 .568 .625 | 0.628 |
| x15 | x14 with instantaneous nursery + $r_l$ | 22 | 0.7356 | .686 .640 .563 .625 | 0.629 |
| x16 | x13 with per-region shelter, one exposure death | 22 | 0.734 | .688 .656 .612 .621 | 0.644 |

\* numerical artefact, see §4.

## 4. A trap we found: stiff relocation at N_SUB = 2

With an explicit relocation flux $m(S^* - S)$ the fits push $m$ to 5-6 per tick. With 2 RK4 substeps
($\Delta t = 0.5$) $m\Delta t$ reaches 2.8, the edge of RK4's stability region, and the integrator itself
starts to shape the trajectory. Rolling the same theta with 2, 4 and 8 substeps:

| fit | $m$ | N_SUB 2 | 4 | 8 |
|---|---|---|---|---|
| x7 (b) | 5.68 | 0.745 | 0.728 | 0.728 |
| x6 (b) | 5.69 | 0.731 | 0.724 | 0.724 |
| x5 (lab full fit) | ~5 | 0.740 | 0.722 | 0.722 |
| x6 (a, lab) | 3.80 | 0.735 | 0.735 | 0.735 |
| x10, x13, x14 (no $m$) | | = | = | = |

So the extra 0.01-0.02 of x5/x7 was the integrator, not the ecology. Fix (x10 on): animals relocate fast
compared with a tick, so occupancy sits at its equilibrium, $S = PK/(K+P)$, $E = P^2/(K+P)$; one prey state
per region, no stiffness, identical scores at 2/4/8 substeps. Every candidate we recommend passes this check.

## 5. The recommended model: x13 (`gtlab/ode/wildlife_x13.py`, 14 states, 22 parameters)

Per region $i$, scale $s_N = 1$, $s_S = k_s$; controls $u_h$ (hunting), $u_p$ (habitat), $u_c$ (corridor):

$$K_i = s_i\,c_h\,u_p,\qquad E_i = \frac{P_i^2}{K_i + P_i}\ \ (\text{exposed}),\qquad S_i = P_i - E_i\ \ (\text{sheltered browse})$$

$$\dot J_i = r P_i - m_J J_i,\qquad \text{rec}_i = \frac{m_J J_i}{1 + J_i/(c_J s_i R_i)}$$

$$\dot P_i = \text{rec}_i - d P_i - d_{h,i}(1-u_p) E_i - H u_h \frac{E_i^2}{E_i^2 + P_h^2} - e_i u_c P_i + \frac{s_v}{\tau} T_{\to i}$$

$$\dot R_i = w\big(1 - b_h(1-u_p)\big)\big(r_l + (1-r_l)R_i\big)(1-R_i) - k_R \frac{P_i}{s_i} R_i$$

$$\dot G_i = \frac1{t_z}\Big(\frac{E_i}{E_i + 100 s_i} - G_i\Big),\qquad
\dot Q_i = c_q (q_0 + q_1 G_i - Q_i)\frac{Q_i}{Q_i+4} - e_N u_c Q_i + \frac{s_v}{\tau} V_{\to i}$$

$$\dot T_{i\to j} = e_i u_c P_i - T_{i\to j}/\tau,\qquad \dot V_{i\to j} = e_N u_c Q_i - V_{i\to j}/\tau$$

Observed $y = (P_N, Q_N, P_S, Q_S)$. Reset: $P, Q$ from $y_0$; $J = rP/m_J$; $R = 1$; transit empty;
$G$ at its equilibrium for full habitat. Mechanisms A (nursery) and B (food renewal) active; AC and BC
lost 0.03-0.04 in-sample (x12).

Fitted theta (`plans/wildlife_wildlife_x13_a.json`, none at a bound):

| $r$ | $d$ | $k_s$ | $H$ | $P_h$ | $e_m$ | $e_{mS}$ | $\tau$ | $s_v$ | $c_J$ | $m_J$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.342 | 0.119 | 0.797 | 1.484 | 5.63 | 0.0298 | 0.0436 | 26.0 | 0.784 | 56.8 | 1.067 |

| $w$ | $k_R$ | $b_h$ | $r_l$ | $d_{hN}$ | $d_{hS}$ | $c_h$ | $q_0$ | $q_1$ | $c_q$ | $t_z$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0263 | 2.39e-4 | 0.446 | 0.414 | 0.0572 | 0.0349 | 253 | 1.727 | 2.271 | 0.0938 | 35.9 |

What the numbers say: adult death is low (0.12) and density regulation is strong nursery crowding
against the food stock (the v8h ridge $r - D_0$ with $D_0 = 0.7$ was an artefact of weak crowding);
at recovery the stock sits at $R \approx 0.37$ (0.87 under the pulse), so a long low-prey spell refills
it and gives the overshoot; almost all animals shelter when prey are rare ($K = 253 u_p$), which is what makes the
pulse floor; food regrowth is 59 % self-seeding, so a depleted stock at low habitat recovers slowly.

Calibrated in-sample per run (prey_N, pred_N, prey_S, pred_S):

| run | v8h | x13 |
|---|---|---|
| p1.hold_rec | .688 .702 .715 .626 = 0.683 | .707 .746 .745 .686 = 0.721 |
| p2.pulse200_200 | .760 .768 .801 .801 = 0.782 | .798 .786 .828 .825 = 0.809 |
| p3.compose | .604 .683 .594 .843 = 0.681 | .651 .726 .578 .846 = 0.700 |
| p4.corridor_habitat | .673 .643 .756 .694 = 0.691 | .673 .675 .723 .744 = 0.704 |
| **mean** | **0.709** | **0.734** |

## 6. Long holds (4,000 ticks from $y_0 = (85, 9, 80, 10)$)

All candidates settle to a fixed point: no oscillation (range over the last 1,000 ticks is 0), finite,
lab eval-shaped rollouts 0.3-0.7 s, 0 % of ticks outside the observed range $\pm 5\%$.

| hold (hunt, hab, cor) | v8h | x13 | observed |
|---|---|---|---|
| recovery (0, 1, 0) | 120.6 / 2.45 / 96.3 / 2.45 | 120.4 / **2.36** / 96.0 / **2.36** | 121.6 / 2.35 / 97.2 / 2.34 (p2 end) |
| pulse (7, .1, 1) | 10.0 / 1.37 / 9.1 / 1.37 | 9.4 / 1.41 / 7.9 / 1.41 | 7.0 / 1.62 / 7.6 / 1.65 (flat from t = 140) |
| habitat 0.5 | 95.5 / 2.36 / 80.1 / 2.38 | 98.5 / 2.41 / 81.1 / 2.44 | 91.0 / 2.44 / 78.7 / 2.34 (100 ticks) |
| corridor 1 | 116.5 / 1.99 / 88.0 / 1.98 | 116.9 / 1.92 / 87.9 / 1.90 | 111.7 / 1.72 / 77.5 / 1.67 (60 ticks) |
| hunting 6 at habitat 1 | 18.3 / 1.85 / 16.8 / 1.88 | 48.4 / 1.89 / 37.4 / 1.88 | north 66, south 28 after 45 ticks (still falling) |
| (8, 0, 1) | 8.4 / 1.35 / 7.7 / 1.35 | **0.45** / 1.37 / 0.45 / 1.37 | never observed |

The recovery predator level (half of the test ticks are recovery) is now right: 2.36 vs 2.35 observed
(v8h 2.45). The pulse predator level is still low (1.41 vs 1.63) and the pulse north floor high (9.4 vs 7.0).
**Risk**: with habitat 0 the model has no shelter at all, so heavy hunting drives prey to ~0.5 where v8h
keeps 8. We have no run below habitat 0.1. Recovery-category pulses cannot go below 0.1 (70-100 % of the
way to 0.1), but sustained holds may.

## 7. Error budget of x13 (largest residual segments)

1. p2 recovery overshoot: data peaks 206 / 164 at t = 240, x13 166 / 132 (v8h 167 / 132): the stock after
   200 ticks of pulse is still not full enough; worth ~0.01 of the run.
2. p3 low-habitat phase: data holds ~130 for 15 ticks and then crashes to 76; x13 declines smoothly to 87.
3. p3 last rebound: data rises fast to 134 and settles at 123; x13 rises later and overshoots to 141.
4. p4 habitat 0.5: north prey keep falling to 91 while north predators rise to 2.97; x13 stops at 98 with
   predators at 2.6. Only the north shows it, and only after corridor use: a north-specific predator
   exposure that 4 runs cannot separate from the rest (x14's predation term fitted it but lost 0.025 LOO).
5. Pulse north floor 9.4 vs 7.0 and pulse predators 1.41 vs 1.63.

## 8. Verdict

LOO at $1.0\sigma$: x13 **0.653** vs v8h **0.621** (+0.032; the best fold gain is the held-out pulse,
0.614 -> 0.714). Calibrated in-sample 0.734 vs 0.715 (+0.019). Below the 0.74 LOO we aimed for: after
x6 the in-sample plateau is 0.735 across five different structures, and the remaining residuals (§7) are
regimes each seen in one run only.

LOO per observable (mean over folds), v8h -> x13: prey_N 0.589 -> 0.657, pred_N 0.631 -> 0.651,
prey_S 0.615 -> 0.645, pred_S 0.649 -> 0.660. The gain is mostly prey, from the shelter floor and the
food-scaled crowding; the predators gain only through the recovery level.

What to expect public: v8h went 0.709 in-sample -> 0.697 public; x13 gains +0.019 in-sample and +0.032 on
held-out runs, so we expect roughly +0.02-0.03 public (about 0.72), not the +0.08 we wanted. Runner-up: x6
(`plans/wildlife_wildlife_x6_a_doc.json`, LOO 0.646, relocation $m = 3.8$ is numerically safe at 2 substeps).

## 9. What data would help most

1. **A long hold at moderate hunting with full habitat** (e.g. hunting 3-4, habitat 1, corridor 0, 300
   ticks): the models disagree most on the sustained-harvest level (x13 72 / 51, v8h 66 / 40 at hunting 4;
   x13 48 / 37, v8h 18 / 17 at hunting 6). The quota-versus-shelter structure decides it and no run
   has held it long enough.
2. **A hold at habitat 0-0.1 without hunting, then with hunting** (100 + 100 ticks): pins the shelter floor
   that §6 flags (prey 0.45 vs 8 at habitat 0).
3. **The pulse held longer or repeated from a different start** (pulse floor and pulse predators, §7.5).
4. **Habitat 0.5 without prior corridor use** (does the north predator rise come from habitat or from
   returning travellers, §7.4).

Files: `gtlab/ode/wildlife_x.py` ... `wildlife_x15.py` (families `wildlife_x`, `wildlife_x2` ... `wildlife_x15`);
lab reports `plans/wildlife_wildlife_x{,3,4,5,6,10,13,14,15,16}_a|b.json`, docs `..._doc.json`;
baseline `plans/wildlife_wildlife_v8h_xbase10.json`. Recommended doc: `plans/wildlife_wildlife_x13_a_doc.json`.
