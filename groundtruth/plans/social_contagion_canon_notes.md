# social_contagion canon: literal textbook readings of the brief (Mon Sep 28, no credits)

Question: if the hidden simulator was written straight from the brief with textbook forms, which literal code
fits, and does it fix the known misses (p6 b ~0.24, p2 b ~0.5)? Pick to beat: med(z20, z21), LOO 0.622.

## 1. The brief, word by word, and the most likely literal code

Brief phrases: "seeding funds outreach", "incentive changes the offer", "bridge outreach allocates effort between
local recruitment and introductions across communities", "relationship-led, incentive-led and deliberative
audiences mix differently", "onboarding through a workforce also needed by existing members", "promises
accompany waiting cohorts", "disappointed former members need time before reconsidering", mechanisms
"credibility, incentive expectations and cross-community relationships can retain history", "initial members
have a fixed disclosed-style community mix and no paid promises; other queues begin empty".

Three literal readings, per tick, per community $i \in \{a, b\}$, $j$ = the other community:

**Version WF (shared workforce, zero-sum).** Motivated by $A_a + A_b \approx 330$ at the end of p2, p4 and p6.
```
local = seeding * (1 - bridge); intro = seeding * bridge
interest_i = P_i * (s_i*local + x_i*intro + q*A_i/N_i + o) * (1 + k_inc_i*phi(incentive))
free_staff = kap * max(A_cap - (A_a + A_b), 0)          # existing members use the workforce
served = min(ready_a + ready_b, free_staff)              # one queue, pro rata per community
queue_i -> members at served * ready_i / (ready_a + ready_b); queue_i -> disappointed at 1/tau_w
members -> disappointed at churn (+ k_B*max(E - incentive, 0) on incentive-led); disappointed -> pool at 1/tau_D
```
**Version SYM (symmetric two-community Bass).** Introductions are made by members of the other community:
`intro_to_i = s_x * seeding * bridge * A_j / N_j`, both directions, one coefficient.

**Version AUD (three audiences, fixed mix per community).** Interest of community $i$ =
`w_rel_i*(q*A_i/N_i + intro_i) + (s_i*local + o)*((1 - w_rel_i - w_inc_i') + w_inc_i'*phi*const)`:
relationship-led people only through word of mouth and introductions, incentive-led through outreach times the
offer, deliberative through outreach alone.

Implementation: each reading replaces one piece of the z20 base (onboarding chain with plain / promised cohorts,
initial leavers, disappointed pool, expectation memory E for B, relationship stock R for C), so the comparison
isolates the literal piece. Generator `scripts/lab_canon_social_mk.py` writes standalone families:

| family | pieces | params |
|---|---|---:|
| canon1 | WF: $\text{gate} = \text{cap}/(\text{dem}^4 + \text{cap}^4)^{1/4}$ on the last onboarding stage, $\text{cap} = \kappa (A_{cap} - A_a - A_b)_+$, give-up $1/\tau_w$ | 22 |
| canon2 | SYM: $x_a = 2 s_x (10R) A_b/N_b$, $x_b = 2 s_x (10R) A_a/N_a$ | 19 |
| canon3 | AUD: weights $w_{rel,i}$, $w_{inc,i}$ per community | 21 |
| canon4 | WF + SYM | 22 |
| canon5 | AUD + SYM | 21 |

## 2. Protocol

`scripts/lab_canon_social_loo.py`: exactly the ens9 protocol (cold init from the family defaults, 6 LHS starts,
spread 0.5, 50 nfev, 60 s per fold, calibrated $\sigma = (9.23, 5.02)$, score at $1.0\sigma$, every run a fold).
z20 / z21 fold predictions are the ens9 cache (seed 0), so every column is like for like. "x+y+z" = median of
members per tick. Summary: `plans/social_contagion_canon_loo.json`. Long holds: `scripts/lab_canon_social_hold.py`.

## 3. Fold tables (held-out mean, a/b in brackets)

| fold | z20 | z21 | **pick med(z20,z21)** | canon1 WF | canon2 SYM | canon3 AUD | canon4 | canon5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p1 hold_rec | 0.774 | 0.888 | 0.830 (0.91/0.75) | 0.664 | 0.731 | 0.823 (0.88/0.77) | 0.661 | 0.737 |
| p2 pulse | 0.560 | 0.592 | 0.572 (0.63/0.52) | 0.562 | 0.523 | 0.580 (0.64/0.52) | 0.529 | 0.563 |
| p3 compose | 0.599 | 0.575 | 0.620 (0.61/0.63) | 0.587 | 0.532 | 0.587 (0.62/0.55) | 0.524 | 0.459 |
| p4 interior | 0.576 | 0.568 | 0.565 (0.58/0.55) | 0.460 | 0.564 | 0.551 (0.56/0.55) | 0.454 | 0.538 |
| p5 exam | 0.589 | 0.674 | 0.645 (0.74/0.55) | 0.568 | 0.564 | 0.659 (0.74/0.58) | 0.507 | 0.578 |
| p6 voi | 0.486 | 0.471 | 0.498 (0.75/**0.24**) | 0.440 (b 0.25) | 0.499 (b 0.24) | 0.447 (0.65/**0.25**) | 0.451 | 0.466 |
| **LOO** | 0.598 | 0.628 | **0.622** | 0.547 | 0.569 | 0.608 | 0.521 | 0.557 |
| in-sample (full fit) | 0.718 | | | 0.724 | 0.694 | 0.722 | 0.701 | 0.713 |

Checks on the best literal form (canon3):

| variant | p1 | p2 | p3 | p4 | p5 | p6 | LOO |
|---|---:|---:|---:|---:|---:|---:|---:|
| canon3 seed 0 | 0.823 | 0.580 | 0.587 | 0.551 | 0.659 | 0.447 | 0.608 |
| canon3 seed 1 | 0.802 | 0.579 | 0.587 | 0.554 | 0.659 | 0.454 | 0.606 |
| canon3 pair B only | 0.819 | 0.586 | 0.524 | 0.554 | 0.661 | 0.452 | 0.600 |
| med(canon3, z20, z21) | 0.843 | 0.574 | 0.605 | 0.566 | 0.669 | 0.481 | 0.623 |
| med(canon3 s1, z20, z21) | 0.832 | 0.573 | 0.605 | 0.567 | 0.669 | 0.481 | 0.621 |
| med(canon2, z20, z21) | 0.774 | 0.561 | 0.581 | 0.575 | 0.624 | 0.499 | 0.602 |

## 4. Findings

1. **The shared workforce is rejected again, now on held-out folds.** canon1 (0.547) and canon4 (0.521) lose
   0.075-0.10 to the pick; the interior-holds fold collapses (0.46) because a binding ceiling cannot give the
   slow drift of p4's holds. The zero-sum totals (~330) are pool saturation plus churn, as found in the x/y rounds.
2. **Symmetric introductions (SYM) lose** (0.569; with AUD 0.557). Letting a's recruitment depend on b's members
   costs the compose and exam folds; p6 b does not move.
3. **The three-audience reading (canon3) is the only literal form that competes:** 0.608 alone, stable across
   seeds (0.606), better than z20 (0.598) and below z21 (0.628). Its full fit is plausible: a is served by
   deliberative + incentive-led audiences ($w_{rel,a} \to 0$), b is 14 % relationship-led; zero-control level
   (95, 88), monotone in seeding, bridge lowers a (201 / 181 / 105 at 0 / 0.5 / 1), finite, 0-5 % outside the data
   range on eval shapes, 0.8 s per 4,000 ticks. Its p2 (0.580) and p5 (0.659) beat the pick by +0.008 / +0.014,
   but p3 loses 0.033 and p4 0.014.
4. **As a third ensemble member it is a tie:** med(canon3, z20, z21) 0.623 vs 0.622 (seed 1: 0.621), inside the
   ±0.01 noise; p2 +0.002, p5 +0.024 (0.669 vs 0.645), p3 -0.015, p6 -0.017. Not a clear gain.
5. **p6 b stays at 0.24-0.25 in every literal form.** In the p6 fold canon3 drives the introduction coefficient
   to $s_x = 0$ (p2 fold 0.0015, full fit 0.0027) and predicts b's plateau at 118 (data 141). The full fit with p6
   reaches 0.82 on p6 b in-sample, so the structure can express it: the other five runs never have high bridge with
   low incentive, so the bridge-to-b gain is not identified from them. No reading of the brief fixes that; only
   data (or accepting p6 in training, which the shipped full fit already does) does.
6. p2 b stays at 0.50-0.52 in every family: the b plateau in the pulse (131) and the post-release floor (34) are
   missed by all, the same as z20.

## 5. Verdict

No literal form clears the bar (clear LOO gain, no loss on p2 or p5). Keep med(z20, z21). The simulator is not
a plain textbook Bass/SIS: the literal zero-sum workforce and symmetric introductions are the two readings the
data reject most clearly; the audience-mix reading is as good as our grey-box but no better. If a spare slot or
a fourth member is wanted, med(canon3, z20, z21) is the only candidate (exam +0.024, compose -0.015, LOO tie).

## Files
- Families: `gtlab/ode/social_contagion_canon1.py` ... `canon5.py` (generated by `scripts/lab_canon_social_mk.py`).
- LOO: `scripts/lab_canon_social_loo.py`; holds: `scripts/lab_canon_social_hold.py`; results:
  `plans/social_contagion_canon_loo.json` (per-fold per-observable scores, full-fit thetas).
- Fold predictions: scratch `canon_social/<family>[_tag]__<fold>.json`.
