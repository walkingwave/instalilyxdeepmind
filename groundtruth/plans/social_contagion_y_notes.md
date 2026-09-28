# social_contagion y: mechanism identification on six runs (2026-09-27, evening)

No credits. Data: 6 runs, 1,811 ticks (`p1.hold_rec` 120, `p2.pulse200_200` 400, `p3.compose` 291,
`p4.interior_holds` 300, `p5.testlike` 400, `p6.voi` 300). `p6.voi` is new: a 300-tick joint hold at
seeding 9.03, incentive 0.135, bridge 0.673. Scale: calibrated sigma $\sigma = (9.23, 5.02)$.
Every number is from `scripts/ode_lab.py --sigma-cal 1.0 --budget 300 --starts 8 --nfev 60`
(tag `yr6<pair>`): LOO = fit on five runs, score the sixth at $1.0\sigma$; `p5.testlike` is the exam.
Because `p6` changes every fold, we re-ran x7 on the same six runs (`_x7_yr6`) as the baseline.
Old five-run numbers (x7 0.619) are not comparable with the six-run ones below.

## 1. What p6 says

- p6 plateau: a 181, b 141 (total 322). The pulse (9, 2, 0.6) gave 199 / 131, interior hold 3
  (6.94, 1.88, 0.58) gave 192 / 138. Almost no incentive (0.135 vs 2) costs a 18 and *raises* b by 10.
- So b's audience barely needs the offer while a's does, and bridge moves recruitment toward b.
  x7 cannot fit this even in-sample: its p6 b score is 0.32 with p6 in the training set.
- Totals near 330 again, but a shared workforce ceiling still does not help (Section 3).

## 2. Structures tried (all numpy + math, batched `_PY`/`_NP` f/h/x0, N_SUB = 2)

Base kept from x7 in every family: two communities; interest per pool member
$\rho_i = K_i\,(1 + k_{inc,i}\phi(c))\,[\,e_i + q A_i/N_i + o\,]$, $\phi = c/(c+0.7)$, saturating
$r = \rho/(1+\rho/1.5)$; a share $\phi$ of new interest carries a promise; Erlang-3 onboarding (mean
$\tau_{on}$) for unpromised ($W$) and promised ($V$) cohorts; offer converts $L \to M$ at $k_{conv} c$;
churn $c_L$ ($L$), $c_L + k_B(E-c)_+$ ($M$), $k_X$ (initial incentive-led $X$, no promise); leavers
$D$ return after $\tau_D$. Mechanisms: A $\dot K = (1-K)/50 - k_A K\,V/N$; B $\dot E = (c-E)/\tau_E$;
C see each row.

| family | params | base formulation | C form |
|---|---:|---|---|
| x7 | 16 | single pool, $e_i = s_i s(1-\beta)$, $k_{inc}$ shared | none |
| y | 21 | **audience-segmented**: separate incentive-led pool $f_{I,i} N_i$ (promised) and relationship/deliberative pool; offer-dependent churn $c_{off}(1-\phi/\phi(2))$ | $\dot R = (s\beta/10 - R)/\tau_R$, cross word of mouth $q_x R A_j/N_j$ |
| y2 | 19 | single pool (x7) + $c_{off}$ | same |
| y3 / y4 | 23 / 21 | y / y2 + **workforce**: stage rate $\times g$, $g = F_{on}/(F_{on} + h_Q Q_{tot})$, $F_{on} = (F_w - A_a - A_b)_+$ | same |
| y5 / y6 | 23 / 21 | y / y2 + bridge introductions $e_i = s(s_i(1-\beta) + s_{x,i}\beta)$ | same |
| y7 | 23 | y6 + workforce | same |
| y8 | 24 | y5 + workforce, $\tau_R$ pinned 30 | same |
| y9 | 19 | x7 + workforce $g = (1 - A_{tot}/F_w)_+$ + introductions into b ($s_{xb}$) | $\tau_R = 30$, $q_x$ |
| **y10** | **19** | x7 + **per-community incentive audience** $k_{inc,b}$ + introductions into b ($s_{xb}$) | $\tau_R = 30$, $q_x$ |
| y11 | 18 | y10 without introductions | same |
| y12 | 20 | y10 + overlapping populations: pool$_i$ minus $\kappa A_j$ | same |
| y13 | 19 | y10 | C = introductions act through retained relationships: $\dot R = (s\beta/10 - R)/\tau_R$, b effort $10\,s_{xb} R$ (without C: $s_{xb} s\beta$ at once) |

y, y2-y6 and y8 were built and smoke-tested; a CPU limit on the shared machine (one, later three,
fits at a time) meant we fitted only the most promising ones (Section 3). y5 was fitted once (pair AB) to
test the segmented formulation.

## 3. Results (six runs, calibrated sigma, LOO at 1.0 sigma)

| family | pair | params | hold | pulse | compose | interior | p6 | **p5 exam** (a/b) | **LOO** | in-sample | cost | at bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| x7 (baseline) | AB | 16 | 0.691 | 0.535 | 0.393 | 0.625 | 0.452 | 0.567 (0.61/0.53) | **0.544** | 0.647 | 1892 | q, m0 |
| y7 | AB | 23 | 0.677 | 0.502 | 0.425 | 0.561 | 0.498 | 0.605 (0.63/0.58) | 0.545 | 0.668 | 1506 | m0, s_xa, F_w |
| y9 | AB | 19 | 0.707 | 0.544 | 0.513 | 0.560 | 0.449 | 0.522 (0.56/0.49) | 0.549 | 0.672 | 1526 | m0, F_w |
| y9 | AC | 19 | 0.613 | 0.303 | 0.385 | 0.232 | 0.121 | 0.453 (0.44/0.46) | 0.351 | 0.546 | 3743 | m0, F_w, s_xb |
| y9 | BC | 19 | 0.706 | 0.531 | 0.528 | 0.451 | 0.457 | 0.606 (0.67/0.55) | 0.546 | 0.681 | 1573 | m0, F_w |
| y11 | AB | 18 | 0.677 | 0.536 | 0.503 | 0.565 | 0.424 | 0.570 (0.59/0.55) | 0.546 | 0.687 | 1743 | m0 |
| **y10** | **AB** | 19 | 0.735 | 0.571 | **0.635** | 0.557 | 0.424 | 0.564 (0.63/0.50) | **0.581** | 0.722 | 1284 | m0 |
| y10 | AC | 19 | 0.540 | 0.300 | 0.409 | 0.240 | 0.112 | 0.487 (0.44/0.53) | 0.348 | 0.568 | 3481 | s_xb |
| **y10** | **BC** | 19 | 0.683 | 0.550 | 0.560 | 0.570 | 0.482 | **0.630** (0.70/0.56) | **0.579** | 0.709 | 1410 | q_x (= 0) |
| y12 | AB | 20 | 0.736 | 0.571 | 0.635 | 0.557 | 0.424 | 0.556 (0.61/0.50) | 0.580 | 0.722 | 1284 | m0 ($\kappa \to 0$) |
| y12 | BC | 20 | 0.684 | 0.551 | 0.562 | 0.552 | 0.472 | 0.630 (0.70/0.56) | 0.575 | 0.708 | 1410 | q_x ($\kappa$ = 0.03) |
| y10 | B only | 19 | 0.683 | 0.565 | 0.576 | 0.570 | 0.482 | 0.630 (0.70/0.56) | 0.584 | 0.709 | 1410 | |
| y5 (segmented) | AB | 23 | 0.767 | 0.537 | 0.500 | 0.498 | 0.406 | 0.613 (0.64/0.59) | 0.554 | 0.712 | 1447 | s_xa |
| **y13** | **BC** | 19 | **0.787** | 0.476 | 0.631 | 0.570 | 0.471 | **0.636** (0.71/0.56) | **0.595** | 0.712 | 1304 | q, m0 |

Fold noise is about 0.02 on the LOO mean.

## 4. Mechanism identification

1. **B (incentive expectations) is active.** Every pair without B collapses: AC LOO 0.35 (y9 and y10) against
   0.55-0.58 with B; the interior fold falls to 0.23-0.24. The interior-hold dip (incentive 1 -> 0.71 drops a by
   18, then a recovers over ~60 ticks) is a transient against a memory, which only B produces.
2. **The second mechanism is not identified by these runs.** In every BC fit $q_x \to 0$ (cross-community
   word of mouth unused), so BC is B alone. In AB, credibility is used ($k_A$ 0.14) and the fits tie with BC
   on LOO (0.581 vs 0.579). A buys nothing out of sample; C in the word-of-mouth form buys nothing at all.
   y10 with B alone scores 0.584 (exam 0.630), the same as BC. With C rebuilt as *retained introductions*
   (y13: bridge effort reaches b through relationships $R$ that build and fade with $	au_R pprox 30$), BC reaches
   0.595 (exam 0.636), +0.011 over B alone: inside fold noise, so C is favoured over A only weakly. y13 AB is the
   same model as y10 AB (C switched off), 0.581.
3. **Audiences mix differently per community.** The gain over x7 (+0.035 LOO, compose fold +0.24) comes from
   $k_{inc,b}$ together with bridge introductions into b. $k_{inc,a} \approx 0.66$-0.89, $k_{inc,b} \approx 0.05$-0.26:
   a's recruitment depends on the offer, b's barely. Either term alone does nothing (y9: introductions only, 0.549;
   y11: $k_{inc,b}$ only, 0.546). Together they fit p6 b in-sample (0.81 against x7's 0.32) and the compose fold
   (seeding without bridge or incentive raises b less than a).
4. **No shared workforce ceiling.** With p6 the plateau total is ~330 in three different holds, yet both workforce
   forms (y7 soft, y9 hard) park $F_w$ at the upper bound (5000 / 3000), as last round. Overlapping populations
   (y12) put $\kappa \to 0$. The equal totals come out of pool saturation plus churn.
5. **The p6 fold stays at ~0.24 on b in every structure.** Without p6 in training nothing we own has high bridge
   with low incentive, so $k_{inc,b}$ and $s_{xb}$ are not learnt; with p6 in training they are. This is an
   identification limit of the older runs, not a model defect, and it is why the six-run LOO sits below the
   five-run numbers.

## 5. Long holds (4,000 ticks, full-fit theta)

| model | zero controls from (48.6, 35.9): t = 120 / 4000 | from (190, 130) | incentive c = 0/1/2 at s 5, beta 0.5 | c = 0/1/2 at s 0 (a, b) | bridge 0/0.5/1 at s 5, c 1 |
|---|---|---|---|---|---|
| x7 (6 runs) | (63, 53) / (127, 107) | same | a 189/182/183 | (127,107)/(128,108)/(132,111) | a 191/182/130, b 146/135/109 |
| y9 BC | (61, 53) / (90, 78) | same | a 176/174/175 | (90,78)/(91,79)/(92,81) | a 196/174/111, b 129/124/113 |
| y10 AB | (61, 55) / (97, 89) | same | a 173/180/181, b 132/120/119 | (97,89)/(119,82)/(123,82) | a 204/180/120, b 128/120/111 |
| y10 BC | (63, 55) / (95, 85) | same | a 173/179/182, b 131/122/122 | (95,85)/(110,76)/(114,76) | a 199/179/111, b 130/122/112 |
| y13 BC | (65, 50) / (118, 90) | same | a 183/179/180, b 134/125/125 | (118,90)/(122,82)/(125,83) | a 191/179/123, b 126/125/124 |

Every fit is finite, has one equilibrium reached from every start we tried (20/15, 48.6/35.9, 190/130), and is
monotone in seeding (y10 BC at c = 1, beta = 0.5: s = 0/1/3/5/7/10 -> a = 110/136/164/179/189/199). Incentive
raises a and lowers b a little at steady state (b's incentive-led share is small and the offer converts and then
churns its members). The zero-control level drops from x7's 127/107 (six-run refit) to 95/85, still above the
start: our own zero-control segments keep rising, so we did not force it lower.

## 6. Verdict

By the two tests we set (LOO at 1.0 sigma, p5 exam) the best structure is **y13 pair BC**
(`plans/social_contagion_social_contagion_y13_yr6BC_doc.json`): LOO 0.595 (x7 on the same six runs 0.544, +0.051),
exam 0.636 (x7 0.567), 19 parameters, finite and in range on all four eval-shaped rollouts (0 % outside), 0.5 s per
4,000 ticks. Two weak points: the pulse fold drops to 0.476 (b 0.33; x7 0.535) and its zero-control level for a is
higher (118 against 95 for y10). The runner-up **y10 BC** (`..._y10_yr6BC_doc.json`, LOO 0.579, exam 0.630, zero level
95/85, pulse fold 0.550) is the safer choice if the sustained category matters more than the mean.

The segmented-pool formulation (y5, 23 parameters) scored 0.554 and the workforce forms 0.545-0.549: the single pool with
a per-community incentive audience is the better base.

Success target (LOO >= 0.66 on the old five-run scale, exam >= 0.582) is not reached. On the six-run scale the
gain over x7 is +0.051 LOO for y13 BC (0.544 -> 0.595), which at the 0.6 transfer ratio forecasts **+0.03 public**
(0.626 -> ~0.657); y10 BC: +0.035 -> +0.02 (~0.647).

## 7. What would help most

1. A hold with high bridge and middle incentive (e.g. 6, 1, 0.9) for 150 ticks: pins $s_{xb}$ against $k_{inc,b}$.
2. The zero-control level (400+ ticks from a reset) remains the largest open question for the sustained category.
3. A-vs-C: a long queue of promised cohorts (high incentive with low onboarding) against a bridge-then-stop
   sequence. The current runs cannot tell A from C, and neither changes LOO.

## Files
- Modules: `gtlab/ode/social_contagion_y.py`, `_y2` ... `_y13`.
- Lab outputs: `plans/social_contagion_social_contagion_{x7_yr6, y7_yr6AB, y9_yr6{AB,AC,BC}, y10_yr6{AB,AC,BC,B},
  y11_yr6AB, y12_yr6{AB,BC}, y13_yr6BC, y5_yr6AB}{,_doc}.json`.
- Five-run-equivalent check (mean of folds p1-p5 only): x7 0.562, y10 AB 0.612, y10 BC 0.599, y13 BC 0.620.
