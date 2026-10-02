# wildlife tm: settlement competition (mechanism C), harvest coefficient, p3 hunting floor (Mon Sep 28, no credits)

Data: the same 5 runs (p1 120, p2 400, p3 291, p4 300, p7 700). Score at calibrated sigma (1.0x).
LOO = fit on 4 runs, score the 5th. Fit = two-stage (as wild9p): stage 1 prey parameters on prey observables
(predator sigma x1e4, b1 100 s), stage 2 predator parameters with prey held (b2 50 s); 8 starts, 60 evals.
Parameters of switched-off terms are held, never fitted. Start = wild9p full-fit theta.
Control: `wildlife_tm` under AB reproduces wild9p to 0.0 max abs difference and its LOO exactly (0.6897).

## Family (`gtlab/ode/wildlife_tm.py`, variants from `scripts/lab_tm_wild_make.py`)

wild9p (v8h, predators on their own transit $e_{mq}, \tau_q, s_{vq}$) plus:

- **C, settlement competition**: arrivals settle with probability $1/(1 + P_{dest}/(K_c s_{dest}))$, the rest are lost.
  Smooth Beverton-Holt form, never negative. Flag `cq`: predator arrivals too, $1/(1 + Q_{dest}/(K_q s_{dest}))$.
- AC: B off ($F = 1$), crowding only through $c_J$. BC: A off, regulation only through $R$.
- Diagnostics (not in the brief's three): `pd` predation $-a_p Q P/(P + 100 s_i)$;
  `sh` shelter occupancy lags protection, $\dot S_i = (u_{hab} - S_i)/t_s$, harvest $\times (1 - x_h S_i)$, $S_i(0) = s_0$
  (the brief: "harvest before versus after protection", "patch occupancy" hidden).

## Fold table (LOO, folds p1 p2 p3 p4 p7)

| model | pair | LOO | p1 | p2 pulse | p3 | p4 | p7 long hold | in-sample |
|---|---|---|---|---|---|---|---|---|
| **current median v8h/wild9p/wild9m** | AB | **0.6846** | .634 | .748 | .607 | .677 | .757 | .733 |
| wild9p (= tm AB, control) | AB | 0.6897 | .632 | .761 | .639 | .680 | .737 | .738 |
| tm AC (C on prey) | AC | 0.6677 | .589 | .752 | .647 | .615 | .736 | .713 |
| tm BC | BC | 0.6368 | .600 | .694 | .582 | .619 | .688 | .709 |
| tm AB, $H = 1$ | AB | 0.6721 | .628 | .741 | .600 | .649 | .743 | .733 |
| tm AC, $H = 1$ | AC | 0.6712 | .589 | .750 | .665 | .615 | .736 | .713 |
| tmq AC (C on prey + predators) | AC | 0.6767 | .608 | .769 | .653 | .608 | .746 | .722 |
| tms AC (+ shelter lag) | AC | 0.6663 | .589 | .749 | .645 | .613 | .736 | .713 |
| tmqs AC | AC | 0.6746 | .608 | .765 | .647 | .606 | .746 | .721 |
| tmp AB (+ predation) | AB | 0.6889 | .646 | .765 | .638 | .689 | .707 | .741 |
| tms AB (+ shelter lag) | AB | 0.6950 | .639 | .765 | .645 | .688 | .737 | .747 |
| tmqs AB + predator settlement ($K_c$ off) | AB + C on predators | 0.7028 | .658 | .778 | .648 | .674 | .756 | .756 |
| **median(v8h, wild9m, tmq AC, tmp AB, tms AB)** | | **0.6973** | .639 | .764 | .637 | .688 | .758 | .739 |
| median(v8h, tmqs AB, tmq AC) | | 0.7020 | .651 | .776 | .650 | .671 | .763 | .751 |
| median(tms AB, tmq AC, tmp AB) | | 0.7089 | .654 | .778 | .669 | .697 | .747 | .755 |
| median(tmqs AB, tmq AC, tmp AB) | | 0.7131 | .664 | .791 | .673 | .683 | .755 | .761 |

Per observable (mean over folds; prey N, pred N, prey S, pred S): current median .684 .642 .735 .678;
median-5 .710 .653 .747 .680; median(v8h, tmqs, tmq) .708 .645 .748 .707.
About 20 blends were scored on the same folds: gains under 0.005 are noise.

## Findings

1. **Mechanism C does not win as a pair.** AC −0.022 and BC −0.053 LOO vs AB (wild9p). AC loses p1 (hold_rec north prey)
   and p4 (corridor + habitat); BC loses everything. C on predators (tmq) helps the predator observables on every fold
   (p2 predator S .748 -> .800, p7 .662 -> .680), but the AC prey structure still loses p1/p4. The prey side of C is
   nearly inert: in the AC full fit, $K_c = 374$ -> $\infty$ moves the p3 floor by 1 animal.
   **AB stays the pair.** The predator-settlement term is the one piece of C with support (tmqs AB, next item).
2. **$H = 1$ (text).** Under AC and BC the free fit lands at $H$ = 0.96 / 1.02, i.e. the text value, with no pin. Under AB
   it wants 1.37 (2.4-2.6 with the shelter lag, whose $(1 - S)$ factor takes it back to ~1.5 at reset). Pinning $H = 1$
   in AB costs −0.018 LOO (p2 −.020, p3 −.039); in AC it is +0.004 (p3 +.018). Not adopted in AB.
3. **The p3 vs p7 hunting floor is not explained by C.** Data at hunting quota ~5.9, cover ~0.25, corridor ~0.9:
   p3 (from 119, predators 2.3) +20 ticks 30.9 / 20.8, end 20.5 / 15.1 and still falling 0.3/tick; p7 (from 88,
   predators 11) +20 29.3 / 19.2, +45 13.7 / 14.6, floor 11 / 11. Every AB model puts p3 +20 at 13-17 (held out).
   AC is the only family that slows p3 (held out: +20 42.7, end 29.2) but it does so through the **transit pools**
   (sv = 0 removes the whole effect; $K_c = \infty$ removes 1 animal): p3 enters the hunt with ~52 / 60 animals in
   transit from the corridor-open block t134-179 ($\tau$ = 45, arrivals ~1.3/tick). It overshoots p3 and is too fast
   on p7 +20 (20.5 vs 29.3). The shelter lag and predation do not reproduce it either (p3 end 10.5 and 8.0).
   Part of the "floor" is simply that p3's segment is 45 ticks long and has not settled. Still open.
4. **Side results.** Shelter lag (tms AB) is the best single two-mechanism model: 0.6950, beats wild9p on four folds,
   ties p7 (.737); fit $t_s$ = 266, $x_h$ = 1.0, $s_0$ = 0.36. Predation (tmp AB) helps p1/p4, loses p7 (.707).

## Recommendation

- **Safe pick (acceptance met: every fold >= current, p2 +.016, p7 +.002): median of five**
  v8h p7, wild9m, tmq AC, tmp AB, tms AB. LOO 0.6973 vs 0.6846 (+0.013). Doc: `plans/wildlife_tm_ens5_doc.json`.
  In-sample 0.739; finite on 4,000-tick eval schedules, 0 % of ticks outside the data range; ~3.2 s per episode here
  (current median 2.0 s on the same machine).
- Higher, not fully safe: median(v8h, tmqs AB, tmq AC) 0.7020, p2 +.027, p7 +.006, but p4 −.006; tmqs puts C on
  the predators on top of AB, i.e. three mechanisms nominally. Doc: `plans/wildlife_tm_ens3q_doc.json`
  (in-sample 0.751, 1.9 s per episode, 0 % outside).
- Not recommended: median(tmqs, tmq, tmp) 0.7131 loses p7 by .002; median(tms, tmq, tmp) 0.7089 loses p7 by .010.

## Files

- Families: `gtlab/ode/wildlife_tm.py` (base; flags cq, pd, sh), `wildlife_tmq.py`, `wildlife_tmp.py`, `wildlife_tmqp.py`,
  `wildlife_tms.py`, `wildlife_tmsp.py`, `wildlife_tmqs.py` (generated by `scripts/lab_tm_wild_make.py`).
- Fits: `scripts/lab_tm_wild_fit.py <family> <tag> --mech AC [--pin H=1] [--set ...]` -> `plans/wildlife_tm_<tag>.json`
  (tags ab ac bc abh abp abs acs ach acq acqs abqs). Fits saved before the `sh` switch existed (ab ac bc abh abp acq ach)
  have no $t_s, x_h, s_0$ entries; `scripts/lab_tm_wild_doc.py` fills them with the defaults (inactive).
- Table: `scripts/lab_tm_wild_table.py <tags> --ens=a+b+c` (reads held-out predictions from `%TEMP%/wild9`, `%TEMP%/wildtm`).
- Docs: `scripts/lab_tm_wild_doc.py` -> `plans/wildlife_tm_ens5_doc.json`, `plans/wildlife_tm_ens3q_doc.json`, `plans/wildlife_tm_abs_doc.json`.
