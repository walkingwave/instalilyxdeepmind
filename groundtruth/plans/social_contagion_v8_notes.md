# social_contagion v8: structural refit at the organizer's scale

Date: 2026-09-26. No credits. Data: 4 runs, 1,111 ticks (`p1.hold_rec` 120, `p2.pulse200_200` 400,
`p3.compose` 291, `p4.interior_holds` 300). Score scale: calibrated organizer sigma
$\sigma = (9.23, 5.02)$ from `plans/sigma_calibrated.json`. At this scale in-sample reproduces public
within ~0.01, so "calibrated in-sample" below is our forecast.

Starting point: `social_contagion_s1` pair AB (`plans/social_contagion_social_contagion_s1_cal4AB_doc.json`),
calibrated in-sample **0.652**.

## 1. Error budget before (s1 cal4AB, calibrated sigma)

Loss per tick $= 1 - 1/(1 + |e|/\sigma)$, summed per constant-control segment (a, b), and mean bias
(model minus data).

| run | segment (s, c, beta) | ticks | loss a | loss b | bias a | bias b |
|---|---|---:|---:|---:|---:|---:|
| hold_rec | 0, 0, 0 | 120 | 25.3 | 40.6 | +1.0 | -2.9 |
| pulse | 9, 2, 0.6 | 200 | 66.0 | 71.3 | -2.0 | +3.3 |
| pulse | recovery | 200 | 42.6 | 81.7 | +0.3 | +0.2 |
| compose | 7.65, 0, 0 | 45 | 26.7 | 15.0 | -12.1 | -0.8 |
| compose | recovery after seeding | 22 | 18.2 | 10.4 | **-44.1** | -4.7 |
| compose | 0, 1.7, 0 | 45 | 20.2 | 10.3 | -9.2 | +1.7 |
| compose | 0, 0, 0.51 | 45 | 12.6 | 24.2 | -4.2 | +6.0 |
| compose | 7.65, 1.7, 0.51 | 45 | 26.4 | 23.8 | -17.7 | -2.2 |
| interior | 5, 1, 0.5 | 100 | 40.7 | 16.9 | +6.9 | -0.2 |
| interior | 4.84, 0.71, 0.59 | 100 | **57.1** | 15.2 | +12.7 | -0.9 |
| interior | 6.94, 1.88, 0.58 | 100 | 28.0 | **53.0** | +1.7 | -7.3 |

Per run (a, b): hold 0.789 / 0.662, pulse 0.729 / 0.617, compose 0.528 / 0.597, interior 0.581 / 0.716.

Three biggest sources, read off the trajectories:
1. **Onboarding shape.** The first-order queue ($\tau_{on} = 16$) starts rising at once; the data dip
   for ~7 ticks and then rise steeply (a pure delay). Every seeding onset pays for it (pulse on, compose
   seeding, compose joint), and the slow lag also leaves the compose seeding rise 40 short.
2. **Zero-control regrowth of b.** Data regrow b almost linearly then saturate near 65 (hold_rec 17 -> 47,
   pulse recovery 33 -> 64); the logistic word-of-mouth term gives an S-curve (too slow early, too high
   late). hold_rec b and pulse-recovery b together carry 122 loss units.
3. **Seeding response depends on context.** Seeding alone at 7.65 lifts a faster and higher than the
   model allows, while the 5/1/0.5 interior hold is over-predicted by 7-13. One seeding gain cannot fit both.

The old fit also parked $m_0$ at its 0.9 bound with $k_X = 0.036$: 90% of initial members as slow
leavers, a stand-in for missing structure.

## 2. What we changed, one step at a time

Every row is a full refit (least squares, Cauchy loss, 8-10 starts, 300 s). The fit column is the
sigma multiplier we fitted at. The last column is calibrated in-sample (sigma x 1.0).

| version | change | params | fit | cost | cal in-sample |
|---|---|---:|---|---:|---:|
| s1 AB (before) | | 15 | 1.5 | 685 | 0.652 |
| v8 | Erlang-3 onboarding chain; disappointed pool $D$ with return $1/\tau_D$; bridge takes $k_{br}$ of seeding effort; $k_C$ dropped | 16 | 1.5 | 564 | 0.660 |
| v8b | + separate word of mouth for b ($q_b$) | 17 | 1.5 | 547 | 0.686 |
| v8c | + onboarding capacity $(1 - A/\kappa N)_+$ | 18 | 1.5 | 549 | 0.685, $\kappa$ at bound (inactive): dropped |
| v8e | + incentive boost on seeding $(1 + k_{inc} c)$ | 18 | 1.5 | 516 | 0.676, $k_{br}$ at bound: dropped |
| v8g | + cross-community word of mouth | 18 | 1.5 | 512 | 0.696 |
| **v8f** | + **organic adoption** $o_a$, $o_b$ | 19 | 1.5 | 481 | **0.721** |
| v8f | same, fitted at sigma x 1.0 | 19 | 1.0 | 849 | 0.734 |
| v8m | shared $q$ and shared $o$ (the 1.0 fit put $q_a \approx q_b$, $o_a \approx o_b$) | 17 | 1.0 | 853 | 0.731 |
| v8m, $\tau_K = 50$ fixed | $\tau_K$ is flat: same optimum | 16 | 1.0 | 853 | 0.731 |
| v8k / v8l | credibility erosion driven by incentive x queue / $\phi$ x queue (v8f base) | 19 | 1.5 | 414 / 419 | 0.719 / 0.722 |
| v8n | v8m + credibility driven by $\phi$ x queue | 16 | 1.0 | 800 | 0.736 |
| **v8o** | v8m + **credibility driven by incentive x queue** ("paid promises") | 16 | 1.0 | 799 | **0.742** |
| v8p / v8q | + incentive-dependent baseline churn of $M$ | 17 | 1.0 | 800 / 799 | $k_M \to 0$: dropped |
| v8r | + seeding reach $N \cdot s/(s + h_R)$ | 17 | 1.0 | 795 | 0.734, $h_R \to 0$: dropped |
| v8s | + seeding exponent $\gamma$ | 17 | 1.0 | 792 | 0.743, $\gamma = 0.90$: not worth a parameter |
| v8t | v8o + separate organic rate for b | 17 | 1.0 | 775 | 0.740: dropped |
| probes on v8o | free $\tau_K$; separate churn $c_M$; churn when credibility is low | 17 | 1.0 | 799 / 779 / 799 | 0.742 / 0.739 / 0.742: dropped |
| v8o2 | v8o with the $k_{conv}$ bound opened to 5 | 16 | 1.0 | 798 | 0.742, $k_{conv} = 2.0$ |
| v8y | v8o + incentive gain on seeding per community $(1 + k_{i} c)$ | 18 | 1.0 | 784 | 0.730 ($k_a = 0.16$, $k_b = 0.15$, $k_{br} \to 0.52$): lower Cauchy cost, lower score: dropped |

Findings that carried:
- **Organic adoption is the big one** (+0.035): recruitment per pool member gets a constant $o$ next to
  word of mouth. With it, zero-control regrowth becomes a first-order approach (linear, then saturating)
  instead of an S-curve, which is what b does in hold_rec and in the pulse recovery. When $o$ is free the
  fit even drops a's word of mouth ($q_a \to 0.001$) at sigma x 1.5; at sigma x 1.0 the two communities
  share $q \approx 0.011$-$0.016$ and $o \approx 0.001$.
- **Erlang-3 onboarding** fixes the onset shape: the first 7 ticks keep dipping, then the rise is steep
  (mean delay $\tau_{on} = 7.3$).
- **Disappointed pool** ($\tau_D \approx 15$): leavers are not recruitable for a while. It slows regrowth
  right after a collapse, which the compose run showed (a flat near 63 for 50 ticks).
- **Credibility erodes with paid promises to waiting cohorts**, $\dot K = (1-K)/50 - k_A K c W/N$. The
  brief: "promises accompany waiting cohorts". Seeding with no incentive leaves credibility intact (compose
  seeding alone rises fastest), seeding with incentive erodes it over tens of ticks. It also takes over
  most of the bridge penalty: $k_{br}$ falls from 0.47-0.55 to 0.30.
- **Fitting at sigma x 1.0 beats x 1.5 on the calibrated score** for the same structure (v8o: 0.742
  vs 0.732; v8n: 0.736 vs 0.729). Both are reported below.

Rejected (the fit turned them off or they cost the calibrated score): onboarding capacity, seeding reach,
seeding exponent, incentive-dependent churn of $M$, churn from low credibility, incentive boost on
seeding (it only works by pushing $k_{br}$ to its bound, and per community it lowers the score), per-community organic rate.

## 3. Final model: `gtlab/ode/social_contagion_v8o.py` (pair AB, 16 parameters, 17 states)

Per community $i \in \{a, b\}$: loyal $L_i$, incentive-led $M_i$, onboarding chain
$W^1_i \to W^2_i \to W^3_i$, initial leavers $X_i$, disappointed $D_i$; credibility $K_i$; shared
incentive memory $E$. Controls: seeding $s$, incentive $c$, bridge $\beta$.

$$A_i = L_i + M_i + X_i, \quad W_i = W^1_i + W^2_i + W^3_i, \quad P_i = (N_i - A_i - W_i - D_i)_+$$
$$\rho_i = K_i\Big[s_i\, s\,(1 - k_{br}\beta)_+ + q\,\frac{A_i}{N_i} + o\Big], \qquad r_i = \frac{\rho_i}{1 + \rho_i/1.5}$$
$$\dot W^1_i = r_i P_i - k W^1_i, \quad \dot W^2_i = k(W^1_i - W^2_i), \quad \dot W^3_i = k(W^2_i - W^3_i), \quad k = 3/\tau_{on}$$
$$\phi = \frac{c}{c + 0.7}, \qquad \dot L_i = (1-\phi) k W^3_i - c_L L_i - k_{conv} c L_i$$
$$\dot M_i = \phi\, k W^3_i + k_{conv} c L_i - \big[c_L + k_B (E - c)_+\big] M_i, \qquad \dot X_i = -k_X X_i$$
$$\dot D_i = c_L L_i + \big[c_L + k_B (E - c)_+\big] M_i + k_X X_i - D_i/\tau_D$$
$$\dot K_i = (1 - K_i)/50 - k_A K_i\, c\, W_i/N_i, \qquad \dot E = (c - E)/\tau_E$$

Initial state: $L_i = (1 - m_0) y_{0,i}$, $X_i = m_0 y_{0,i}$, $K_i = 1$, everything else 0 (the brief:
no paid promises, queues empty). RK4, 2 substeps per tick. Observed: $y = (A_a, A_b)$.

Fitted at sigma x 1.0 (`plans/social_contagion_social_contagion_v8o_v8o_cal1_doc.json`):

| $N_a$ | $N_b$ | $s_a$ | $s_b$ | $q$ | $\tau_{on}$ | $c_L$ | $k_X$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 373.2 | 357.5 | 0.00284 | 0.00074 | 0.0161 | 7.35 | 0.0123 | 0.124 |

| $k_{conv}$ | $m_0$ | $k_A$ | $k_B$ | $\tau_E$ | $k_{br}$ | $\tau_D$ | $o$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 (bound) | 0.335 | 0.073 | 0.0457 | 30.3 | 0.304 | 14.9 | 0.00089 |

$k_{conv}$ at its bound is benign: with the bound opened to 5 (v8o2) it goes to 2.0 and the score does not
move (0.7422 vs 0.7421), i.e. conversion under incentive is just "fast". We keep 0.7 so the RK4 step stays
well inside its stability region ($k_{conv} c\, \Delta t \le 0.7$). $m_0 = 0.33$ and $k_X = 0.12$: a third
of the initial members leave with a 8-tick time constant: the dip in every run. Four random multistarts
(spread 0.7) and two local ones all land on cost 798.9: this optimum is not a seed accident.

## 4. Scores

Calibrated in-sample, `rollout_from_blob` on each lab doc, $\sigma = (9.23, 5.02)$:

| doc | mean | a | b | hold (a, b) | pulse (a, b) | compose (a, b) | interior (a, b) |
|---|---:|---:|---:|---|---|---|---|
| s1 cal4AB (before) | 0.652 | 0.657 | 0.648 | 0.789, 0.662 | 0.729, 0.617 | 0.528, 0.597 | 0.581, 0.716 |
| **v8o, fit x 1.0** (`_v8o_cal1_doc`) | **0.742** | 0.723 | 0.761 | 0.875, 0.889 | 0.846, 0.757 | 0.631, 0.728 | 0.540, 0.670 |
| v8o, fit x 1.5 (`_v8o_doc`) | 0.732 | 0.720 | 0.744 | 0.877, 0.868 | 0.819, 0.731 | 0.621, 0.706 | 0.561, 0.672 |
| v8n, fit x 1.0 (`_v8n_cal1_doc`) | 0.736 | 0.730 | 0.741 | 0.890, 0.909 | 0.821, 0.709 | 0.636, 0.718 | 0.575, 0.629 |
| v8n, fit x 1.5 (`_v8n_doc`) | 0.729 | 0.725 | 0.732 | 0.901, 0.895 | 0.785, 0.686 | 0.627, 0.699 | 0.589, 0.647 |

Leave-one-run-out (`scripts/ode_lab.py`, each fold refits on the other three runs, scored at the fit scale):

| model, scale | hold | pulse | compose | interior | LOO mean |
|---|---:|---:|---:|---:|---:|
| s1 AB, x 1.5 (cal4AB) | 0.473 | 0.597 | 0.522 | 0.537 | 0.532 |
| **v8o, x 1.5** | 0.776 | 0.645 | 0.606 | 0.602 | **0.658** |
| v8n, x 1.5 | 0.788 | 0.522 | 0.531 | 0.595 | 0.609 |
| s1 AB, x 1.0 (`_s1_v8cmp_cal1AB`) | 0.376 | 0.524 | 0.416 | 0.439 | 0.439 |
| **v8o, x 1.0** | 0.754 | 0.562 | 0.484 | 0.532 | **0.583** |
| v8n, x 1.0 | 0.799 | 0.400 | 0.456 | 0.518 | 0.543 |
| l0b_lin, x 1.0 | 0.466 | 0.322 | 0.247 | 0.186 | 0.305 |

v8o beats s1 on every fold at both scales. Its LOO gain (+0.13 / +0.14) is larger than its in-sample gain
(+0.09), so the new terms generalize across runs rather than memorize them.

Eval-shaped 4,000-tick rollouts (lab): all finite, 0.6 s each. Fraction of ticks outside the observed
range +-5% (sustained / order / recovery / composition): v8o 0.00 / 0.28 / 0.00 / 0.09, against
0.39 / 0.51 / 0.09 / 0.25 for s1 at the same scale. Range a 38-231, b 17-171.

## 5. Error budget after (v8o, fit x 1.0)

| run | segment | loss a | loss b | bias a | bias b |
|---|---|---:|---:|---:|---:|
| hold_rec | 0, 0, 0 | 15.0 | 13.3 | +0.3 | -0.5 |
| pulse | 9, 2, 0.6 | 32.6 | 24.3 | -0.8 | +0.6 |
| pulse | recovery | 28.9 | 72.7 | -0.9 | +1.0 |
| compose | 7.65, 0, 0 | 15.6 | 8.5 | -1.8 | +1.0 |
| compose | recovery after seeding | 11.6 | 7.0 | -10.5 | +2.4 |
| compose | 0, 1.7, 0 | 10.4 | 8.6 | +0.9 | +1.4 |
| compose | recovery after incentive | 12.3 | 8.8 | -12.7 | -3.9 |
| compose | 0, 0, 0.51 | 24.3 | 12.3 | -11.0 | +2.0 |
| compose | recovery | 9.5 | 8.9 | -7.0 | +3.4 |
| compose | 7.65, 1.7, 0.51 | 16.1 | 12.5 | -5.9 | +0.9 |
| compose | recovery | 7.7 | 12.5 | +0.2 | -0.2 |
| interior | 5, 1, 0.5 | 43.7 | 24.0 | +10.8 | -1.7 |
| interior | 4.84, 0.71, 0.59 | **70.5** | 11.9 | **+22.1** | +0.5 |
| interior | 6.94, 1.88, 0.58 | 23.8 | **63.0** | +4.7 | -9.4 |

Total loss fell from 793 to 610 tick-units (a 397 -> 322, b 396 -> 288). What is left, largest first:
1. **Interior holds, a** (114 units): the data flatten at 150 under 5/1/0.5 within ~50 ticks and dip to 132
   when the incentive drops to 0.71; the model keeps rising to 175 and dips to 154. Early rise rates per
   unit seeding are the same as in the pulse (0.6-0.65/tick), so the missing piece builds up over time and
   is specific to a. None of seeding reach, seeding exponent, onboarding capacity, incentive-dependent
   churn or credibility-driven churn found it.
2. **Pulse recovery, b** (73 units): b collapses to 33 (model 38) and levels off at ~64 (model still
   rising, 70 at t = 390).
3. **Interior 6.94/1.88/0.58, b** (63 units): b jumps 110 -> 139 in 50 ticks; the model reaches 131.
   Items 1 and 3 together say the interior holds move a and b differently from what one shared set of
   rates gives. A per-community incentive gain on seeding (v8y) did not separate them ($k_a \approx k_b$)
   and lowered the score. The fourth interior-type run we do not have (an interior hold followed by a
   stop) is what would pin this down.

## Files

- `gtlab/ode/social_contagion_v8o.py`: final family (16 parameters). `social_contagion_v8n.py`: same with
  credibility driven by $\phi$ x queue (no parameter at bound, slightly lower).
- Steps and rejected probes: `social_contagion_v8.py`, `_v8b`, `_v8c`, `_v8e`, `_v8f`, `_v8g`, `_v8j`,
  `_v8k`, `_v8l`, `_v8m`, `_v8o2`, `_v8p`, `_v8q`, `_v8r`, `_v8s`, `_v8t`, `_v8y`.
- Lab outputs: `plans/social_contagion_social_contagion_v8o_v8o_cal1{,_doc}.json`,
  `_v8o_v8o`, `_v8n_v8n_cal1`, `_v8n_v8n`, `_s1_v8cmp_cal1AB` (s1 refitted at x 1.0 for a like-for-like LOO).
