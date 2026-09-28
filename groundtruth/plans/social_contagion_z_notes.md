# social_contagion z: keep the compose/exam gains without losing the pulse fold (2026-09-27/28, night)

No credits. Data: the same 6 runs, 1,811 ticks as the y round (`p1.hold_rec` 120, `p2.pulse200_200` 400,
`p3.compose` 291, `p4.interior_holds` 300, `p5.testlike` 400, `p6.voi` 300). Scale: calibrated sigma
$\sigma = (9.23, 5.02)$. Every number is from `scripts/ode_lab.py --system social_contagion --mech BC --budget 300
--starts 8 --nfev 60 --sigma-cal 1.0 --tag zr6BC`, no polishing. LOO = fit on five runs, score the sixth at
$1.0\sigma$; `p5.testlike` is the exam. Acceptance rule (MATH_LOG §32): the pulse/recovery fold `p2` must not drop
below x7's 0.535; our target was LOO $\ge 0.60$, p2 $\ge 0.55$, exam $\ge 0.62$.

## 1. What the pulse fold needs

p2 is the recovery category in miniature: 200 ticks at the pulse action (9, 2, 0.6), then 200 ticks at zero.

| tick | 0 | 20 | 40 | 80 | 120 | 199 | 210 | 220 | 240 | 300 | 399 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 30 | 93 | 157 | 191 | 198 | 199 | 101 | 64 | 44 | 55 | 76 |
| b | 41 | 51 | 78 | 110 | 125 | 131 | 61 | 38 | 34 | 53 | 65 |

a saturates by t = 120; b is still creeping up at t = 200. After release both lose about 75 % within 40 ticks
(b ends below its start), sit flat for ~20 ticks, then regrow at about 0.2/tick.

Held-out p2 (fit on the other five runs), a / b:

| model | pulse plateau (a, b) | b at t = 220 | end (a, b) | p2 score a / b |
|---|---|---|---|---|
| data | 199, 131 | 38 | 76, 65 | |
| x7 | 181, 137 | 43 | 93, 76 | 0.50 / 0.57 |
| y13 BC | 193, 145 | 51 | 90, 86 | 0.63 / **0.33** |
| z16 BC | 190, 143 | 49 | 89, 76 | 0.61 / 0.52 |

y13 loses p2 on b alone. Its free relationship memory settles at $\tau_R$ = 28-53 ticks in the fold fits, so bridge
introductions keep recruiting in b for 30-50 ticks after the campaign ends: b's floor comes out at 46-51
(data 33-38) and b regrows to 86 (data 65). The a side of y13 is actually better than x7 (0.63 vs 0.50).

## 2. Structures tried (all numpy + math, batched `_PY`/`_NP`, N_SUB = 2, base = y10/y13)

| family | params | change |
|---|---:|---|
| z | 19 | C as relationships that **gate** introductions: $\dot R = (\beta - R)/\tau_R$, b effort $s_{xb}\,s\,R$ (nothing recruits once seeding stops) |
| z2 | 19 | z with $k_{inc,b} \in [-0.9, 3]$ (offer may put off b's relationship-led interest); fit kept it positive (0.16) |
| z3 | 20 | immediate introductions + relationships that amplify later outreach $s_{xr}\,s\,10R$ |
| z4 | 20 | z2 + organic interest per community ($o_b$) |
| z5 | 21 | z2 + finite deliberative audience $o\,(G_i - A_i - Q_i)_+$ ignoring the offer |
| z6 | 20 | z2 + a lingering introduction stock (y13 form, 30-tick memory), fit decides the share |
| z7 | 17 | z2 with $m_0 = 0.9$ and $k_A$ fixed |
| z8 | 19 | promises only on the incentive-led part of interest, $1 - 1/(1 + k_{inc,i}\phi)$ |
| z9 / z10 | 21 / 20 | z6 / z2 + expectations hold back new interest, $\div(1 + k_E (E - c)_+)$ |
| z11 | 20 | z2 + word of mouth from a into b, $q_{ab} A_a/N_a$ |
| z12 | 19 | local outreach in a only; b by introductions, $q_{ab}$ and organic interest |
| **z14 / z17 / z16 / z15** | 18 | **y13 with the relationship memory fixed at $\tau_R$ = 10 / 13 / 15 / 20 ticks** |
| **z20** | 19 | **z16 + onboarding time per community ($\tau_{on,b}$)** |
| z21 | 19 | z16 + base churn per community ($c_{L,b}$) |
| z24 | 20 | z20 + introductions recruiting in a too ($s_{xa}$) |

z16 equations (per community $i$, $\phi = c/(c + 0.7)$, $k = 3/\tau_{on}$; x7 notes give the full base):
$$\rho_a = (1 + k_{inc}\phi)\,[\,s_a\,s(1-\beta) + qA_a/N_a + o\,],\qquad
\rho_b = (1 + k_{inc,b}\phi)\,[\,s_b\,s(1-\beta) + 10\,s_{xb}R + qA_b/N_b + o\,]$$
$$\dot R = (s\beta/10 - R)/15,\qquad \dot E = (c - E)/\tau_E,\qquad \text{churn}_M = c_L + k_B (E - c)_+$$
z20 adds $k_b = 3/\tau_{on,b}$ for b's onboarding chain.

## 3. Results (six runs, calibrated sigma, LOO at 1.0 sigma; fold noise about 0.02)

| family | params | hold | **pulse p2** | compose | interior | **p5 exam** (a/b) | p6 | **LOO** | in-sample | long holds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| x7 (baseline) | 16 | 0.691 | 0.535 | 0.393 | 0.625 | 0.567 (0.61/0.53) | 0.452 | 0.544 | 0.647 | zero (127, 107) |
| y10 BC (tonight) | 19 | 0.683 | 0.550 | 0.560 | 0.570 | 0.630 (0.70/0.56) | 0.482 | 0.579 | 0.709 | zero (95, 85) |
| y13 BC | 19 | 0.787 | 0.476 | 0.631 | 0.570 | 0.636 (0.71/0.56) | 0.471 | 0.595 | 0.712 | zero (118, 90) |
| z | 19 | 0.703 | 0.567 | 0.579 | 0.570 | 0.595 (0.71/0.48) | 0.482 | 0.583 | 0.696 | |
| z2 | 19 | 0.703 | 0.567 | 0.579 | 0.562 | 0.631 (0.70/0.56) | 0.482 | 0.587 | 0.696 | zero (94, 80) |
| z3 | 20 | 0.683 | 0.524 | 0.575 | 0.569 | 0.630 | 0.482 | 0.577 | 0.708 | $s_{xr} \to 0$ |
| z4 | 20 | 0.658 | 0.564 | 0.557 | 0.535 | 0.590 | 0.502 | 0.568 | 0.688 | |
| z5 | 21 | 0.666 | 0.551 | 0.552 | 0.529 | 0.579 | 0.499 | 0.563 | 0.693 | |
| z6 | 20 | 0.788 | 0.500 | 0.560 | 0.571 | 0.612 | 0.482 | 0.585 | 0.720 | picks lingering |
| z7 | 17 | 0.703 | 0.567 | 0.579 | 0.554 | 0.595 | 0.482 | 0.580 | 0.696 | |
| z8 | 19 | 0.735 | 0.477 | 0.512 | 0.539 | 0.575 | 0.480 | 0.553 | 0.702 | |
| z9 | 21 | 0.726 | 0.500 | 0.471 | 0.555 | 0.472 | 0.482 | 0.534 | 0.728 | |
| z10 | 20 | 0.692 | 0.567 | 0.579 | 0.558 | 0.441 | 0.478 | 0.552 | 0.710 | |
| z11 | 20 | 0.701 | 0.568 | 0.574 | 0.526 | 0.614 | 0.506 | 0.581 | 0.692 | |
| z12 | 19 | 0.577 | 0.593 | 0.540 | 0.531 | 0.540 | 0.430 | 0.535 | 0.664 | |
| z14 ($\tau_R$ 10) | 18 | 0.715 | 0.566 | 0.601 | 0.570 | 0.635 (0.71/0.56) | 0.482 | 0.595 | 0.699 | zero (101, 85) |
| z17 ($\tau_R$ 13) | 18 | 0.726 | 0.564 | 0.607 | 0.571 | 0.636 (0.71/0.56) | 0.482 | 0.598 | 0.700 | zero (102, 84) |
| z16 ($\tau_R$ 15) | 18 | 0.735 | 0.563 | 0.611 | 0.572 | 0.636 (0.71/0.56) | 0.482 | 0.600 | 0.702 | zero (102, 83) |
| z15 ($\tau_R$ 20) | 18 | 0.756 | **0.545** | 0.619 | 0.577 | 0.637 (0.71/0.57) | 0.482 | 0.603 | 0.708 | zero (105, 84) |
| **z20** | 19 | 0.770 | **0.560** | 0.594 | 0.569 | **0.647** (0.72/0.57) | 0.479 | **0.603** | 0.718 | zero (97, 79) |
| z21 | 19 | 0.853 | 0.592 | 0.558 | 0.568 | 0.653* (0.76/0.55) | 0.471 | 0.616 | 0.731 | **zero (81, 113): fails** |
| z24 | 20 | — | — | — | — | — | — | — | — | not finished |

\* z21's p5 fold ran across a machine suspend (its 150 s wall-clock budget expired during the suspend), so that fold
was fitted with fewer evaluations than the others. z24 was still running when we stopped.

p2 per observable (a / b): x7 0.50 / 0.57, y10 BC 0.60 / 0.50, y13 BC 0.63 / 0.32, z16 0.61 / 0.52, z20 0.60 / 0.52.

## 4. Findings

1. **The relationship memory is the whole trade-off.** With y13's structure, fixing $\tau_R$ traces a smooth curve
   (LOO / p2): 10 ticks 0.595 / 0.566, 13 ticks 0.598 / 0.564, 15 ticks 0.600 / 0.563, 20 ticks 0.603 / 0.545,
   free (28-53) 0.595 / 0.476. The hold and compose folds want a longer memory, the pulse fold a shorter one. Left
   free, the fit always takes the long memory because five of six runs have short campaigns where introductions
   that linger fit b's hold-up after a campaign; only the 200-tick pulse shows that b does not keep being recruited.
   We fixed $\tau_R$ = 15, the longest value that keeps p2 above 0.55. This is one hyperparameter chosen with the
   folds in view (four values tried), so the held-out numbers carry a small selection bias.
2. **Gating introductions by current seeding (z, z2) fixes p2 but gives back the compose and hold gains**
   (0.583-0.587). When both a lingering and a gated channel are offered (z6), the fit chooses lingering and p2
   falls to 0.500. Amplifying later outreach through relationships (z3) is unused ($s_{xr} \to 0$).
3. **b onboards more slowly than a** (z20: $\tau_{on}$ 8.1 in a, 11.6 in b). It lifts the hold and exam folds
   (0.770, 0.647) at no cost to p2 (0.560) and lowers the zero-control level to (97, 79).
4. **Rejected, with reasons.** Expectations holding back new interest (z9, z10) wrecks the exam (0.44-0.47): the
   short p5 pulses need recruitment to restart at once. Promises only on incentive-led interest (z8) loses p2
   (0.477). A deliberative audience (z5), organic interest per community (z4), word of mouth from a into b (z11,
   z12) and extra parameters in general lost LOO. **z21 has the best LOO (0.616) and p2 (0.592) but fails the
   long-hold rule:** b's churn goes to 0.0014 (a 0.0086), so with outreach stopped b climbs from 36 to 113 and ends
   above a; the long-hold audit wants b to stay near 40-50. Its p5 fold was also cut short by the suspend.
5. **Fold luck is real.** z and z2 converge to the same optimum on five folds, but the exam fold lands on 0.595 in
   one and 0.631 in the other because the start design differs. Differences under 0.02 between families are not
   evidence.
6. Averaging y13 and z2 predictions (per fold) gives LOO 0.595 but p2 0.509: the pulse-fold loss survives
   averaging, so we did not pursue ensembles.
7. p6 b stays at 0.24 in every structure (b under high bridge with low incentive is not learnable without p6), as
   in the y round.

## 5. Long holds (4,000 ticks, full-fit theta)

| model | zero controls from (48.6, 35.9): t = 120 / 4000 | from (190, 130) | incentive c = 0/1/2 at s 5, beta 0.5 | bridge 0/0.5/1 at s 5, c 1 | seeding 0/1/3/5/7/10 at c 1, beta 0.5 (a) |
|---|---|---|---|---|---|
| y13 BC | (65, 50) / (118, 90) | same | a 183/179/180, b 134/125/125 | a 191/179/123, b 126/125/124 | 122 146 168 179 185 191 |
| z16 | (63, 51) / (102, 83) | same | a 179/179/181, b 131/123/124 | a 195/179/112, b 128/123/118 | 110 138 165 179 188 195 |
| **z20** | (64, 51) / (97, 79) | same | a 177/179/181, b 129/123/124 | a 199/179/107, b 128/123/117 | 105 133 163 179 189 199 |
| z21 | (64, 54) / (81, **113**) | same | a 170/178/181, b 140/128/128 | a 206/178/93, b 130/128/126 | 91 121 157 178 192 206 |

z16 and z20 are finite, reach one equilibrium from every start, are monotone in seeding, and raise a with the
incentive (b dips slightly, as in y10/y13). Eval-shaped rollouts: 0 % outside the observed range on sustained,
order and recovery, 4-5 % on composition; about 0.5 s per 4,000 ticks.

## 6. Verdict

**z20, pair BC** (`plans/social_contagion_social_contagion_z20_zr6BC_doc.json`) meets all three targets:
LOO 0.603 (x7 0.544, y10 BC 0.579, y13 0.595), p2 0.560 (x7 0.535), exam 0.647 (x7 0.567), plausible long holds
with the lowest zero-control level of the candidates (97, 79). Runner-up **z16** (18 parameters,
`..._z16_zr6BC_doc.json`): LOO 0.600, p2 0.563, exam 0.636, zero level (102, 83).

Forecast at 0.6 x held-out gain: over x7 +0.059 LOO -> **+0.035 public (0.626 -> about 0.66)**; over tonight's
y10 BC +0.024 -> +0.014. On the recovery-shaped fold alone, z20 is +0.025 over x7 and +0.010 over y10 BC, so the
§32 rule is met with margin 0.025.

## 7. What would help most

1. A second long pulse then release with a different bridge share (e.g. 9, 2, 0.2 for 150 ticks, then 150 at zero):
   pins how long introductions keep recruiting in b after a campaign, the one quantity the folds disagree on.
2. A long zero-control hold from a reset (400+ ticks) remains the open question for the sustained level.

## Files
- Modules: `gtlab/ode/social_contagion_z.py`, `_z2` ... `_z12`, `_z14` ... `_z17`, `_z20`, `_z21`, `_z24`.
- Lab outputs: `plans/social_contagion_social_contagion_z{,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,20,21}_zr6BC{,_doc}.json`.
