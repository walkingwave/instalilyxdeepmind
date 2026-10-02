# wildlife w: refit with the long hold p7 (`gtlab/ode/wildlife_w*.py`)

Date: 2026-09-28. Data: 5 runs (p1 120, p2 400, p3 291, p4 300, **p7.longhold 700**; 1,811 ticks).
No credits spent, nothing uploaded. Scoring at the calibrated sigma (9.28, 0.187, 7.66, 0.253).
**Status: incomplete.** The machine was suspended from about 01:35 to 12:43, so the leave-one-run-out
lab runs (z9, v8h and the w candidates) did not finish and were stopped. Everything below is
**in-sample only** (one full fit, 150 s, 8 starts, 60 evaluations); no LOO number exists yet.

## 1. What p7 shows

Hold (hunt 5.9, hab 0.27, cor 0.9) for 566 ticks, then (8, 1, 0).

| t | data prey N / pred N / prey S / pred S | z9 (u015 cand.) | x13 | v8h |
|---|---|---|---|---|
| 100 | 11.1 / 1.8 / 11.1 / 1.8 | 18.2 / 1.9 / 14.4 / 1.9 | 19.6 / 1.9 / 15.9 / 1.9 | 12.5 / 1.9 / 11.3 / 1.9 |
| 565 | 11.1 / 1.66 / 11.0 / 1.67 | 17.7 / 1.4 / 14.0 / 1.4 | 19.5 / 1.5 / 15.8 / 1.5 | 12.4 / 1.5 / 11.1 / 1.5 |
| 699 | 20.1 / 1.9 / 8.7 / 1.8 | 26.9 / 1.8 / 23.0 / 1.8 | 39.1 / 1.9 / 31.8 / 1.9 | 11.4 / 1.9 / 11.0 / 1.9 |

Per observable on p7 (old thetas): z9 .564 .612 .649 .668; x13 .493 .630 .565 .677; v8h .822 .608 .937 .658.
Three misses:
1. The floor under hunting at low habitat: shelter $K = s\,c_h u_p$ scales with habitat, so the floor
   $P^* \approx gK/(Hu)$ goes as $u_p/u_h$. Data floors: 7.3 at (7, .1), 11 at (5.9, .27), i.e. $P^* u_h$ grows
   only 1.3x from habitat .1 to .27, and ~2x from .1 to 1. The shelter needs a large habitat-free part.
2. Predators sit at 1.66 during the hold (and 1.62 in the pulse) while every model decays to 1.4-1.5:
   travel losses at corridor 0.9 pull them below $q_0$. The data say predators barely travel when prey are scarce.
3. After the switch to (8, 1, 0) the north climbs slowly to 20 while the south falls to 8.7. Shelter models jump
   to 27-40 in both (full habitat hides them from hunting); v8h stays at 11 in both. No model has the split.

## 2. Candidates (one template, switches in `FLAGS`)

$K_i = s_i(c_0 + c_{h,i} u_p^{k_p})$; switches: `c0` base shelter, `chs` own south slope $c_{hS}$, `kp` shelter
power, `pt` predator journeys $\propto e_m P/(P+P_t)$, `hs` scarcity effort $P/(P+P_e)$ (z9), `xh` south take factor.

| family | switches | p1 | p2 | p3 | p4 | p7 | mean (5 runs) | notes |
|---|---|---|---|---|---|---|---|---|
| z9 (old theta, not refit) | hs | .732 | .818 | .709 | .703 | .623 | 0.717 | |
| w | hs c0 chs | .703 | .800 | .708 | .698 | .800 | 0.742 | $c_0$ = 109, $c_{hS} \approx c_h$ |
| w2 | hs c0 chs pt | .720 | .779 | .706 | .708 | .823 | 0.747 | $c_0$ = 140, $P_t$ = 6.3 |
| w3 | c0 chs pt (no hs) | .718 | .780 | .685 | .708 | .821 | 0.743 | $c_0$ = 321 |
| w4 | hs pt | .693 | .781 | .698 | .700 | **.855** | 0.745 | $c_h \to$ 4.4, $P_h$ = 12: shelter abandoned, v8h-like Hill floor |
| w5-w9 | kp / xh variants | | | | | | | written, not fitted |

Readings (in-sample only, so provisional):
- A habitat-free shelter ($c_0$) or dropping the shelter for a Hill floor both fix the p7 floor; p7 goes
  0.62 -> 0.80-0.86. The cost is on the pulse run: p2 falls 0.818 -> 0.78-0.80 in every variant, which is the
  fold the rule protects. That has to be checked by LOO before anything ships.
- `pt` fits $P_t \approx 5$-9 and lifts both predator scores on p7 (pred S .776 -> .832).
- The per-region shelter slope does not take ($c_{hS} \ge c_h$); the (8, 1, 0) north/south split stays unexplained.
- Scarcity effort survives: $P_e$ falls from 56 to 21-31, but w3 without it is only 0.004 lower in-sample.

## 3. Next

Run the lab with LOO on z9, v8h, w2, w4 and w3 (`scripts/ode_lab.py --system wildlife --family <fam> --mech AB
--budget 300 --starts 8 --nfev 60 --sigma-cal 1.0 --tag p7`), ~18 min each, 3 in parallel. Pick by LOO with p7 as a
fold and the p2 fold not below z9's. Scratch thetas: `%TEMP%\claude\scratch\wlw\wildlife_w*_screen.json`.
