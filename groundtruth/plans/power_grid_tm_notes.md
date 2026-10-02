# power_grid tm: text-implied terms (textmine items 7-8)

Mon Sep 28 evening, no credits. Fit runs hold_rec, pulse60_120, multilevel200, p8.longhold; p6.exam
never fitted. Calibrated sigma [9.65, 0.308, 0.066] at 1.0x. Lab: `scripts/lab_tm_pg_fit.py` (same fit
call as `lab_pg9_fit.py`: cauchy, 8 starts, 60 nfev, spread 0.5, mech AB, full fit 600 s, folds 300 s),
table: `scripts/lab_tm_pg_table.py`. The machine was at 100% CPU from other fits, so budgets were
raised from 300 to 600 s and w5 was refitted under the same conditions (`w5ref`) as the control.
It reproduced the w notes to three decimals on every fold and the exam, so the rows below compare
like with like.

## 1. Families

| family | change vs w5 | params |
|---|---|---:|
| `power_grid_tm1` | line temperature $\dot T_{ic}=a_{ic}(F-T_{ic})$, $F=g_i\,ic\,\text{der}+R_{eff}\frac{P_{rem}}{p_l+P_{rem}}+P_{ch}\,icf$; $\text{der}=\max(1-k_{ic}\,\text{hp}(T_{ic}-t_{ic})/100,0.1)$ scales remote capacity $P_{rem}=p_r\,ic\,\text{der}$ and remote delivery $g_i\,ic\,\text{der}$; share $\times(1-k_{cu}\,\text{hp}/100)$; $T_{ic}(0)=tic_0$ | 27 |
| `power_grid_w5`, $c_r=1$ pinned (tag `tm2p`) | reserve in power units | 21 |
| `power_grid_tm2` | $c_r=1$; two units: thermal $Q$ (cap $P_t$, lag $a_t$) first, fast unit $B=\text{smin}(R-Q,P_b\,s(E))$ fills the gap; total $\text{smin}(Q+B,p_l+p_r\,ic)$; stock drains by delivered $B$ | 23 |
| `power_grid_tm2b` | tm2 with the fast unit first, thermal takes the rest | 23 |
| `power_grid_tm2c` | tm2 with thermal available power $=p_l+p_r\,ic$ (no $P_t$, no cap on the total) | 22 |

hp = smooth hinge (width 2 units), $s(E)=E(1+0.05)/(E+0.05)$. tm1 run twice: all free (`tm1`),
and $k_{cu}=0$ pinned (`tm1d`). `tm3p` = tm1 with $c_r=1$, $k_{cu}=0$ (items 7 + 8a together), each fold
started from the matching tm1d fold. `tm1d_full8` = tm1d full fit restarted from its p8-fold theta.

## 2. Fold table (LOO = held-out run fitted on the other three; mean over observables)

| fit | hold_rec | pulse | multilevel | p8 | LOO mean | in-sample | exam | at bound |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **w5ref (control)** | 0.770 | 0.688 | 0.733 | 0.650 | 0.710 | 0.789 | 0.685 | none |
| tm1 (all free) | 0.786 | 0.701 | **0.666** | 0.637 | 0.697 | 0.792 | 0.687 | a_g, p_r |
| tm1d ($k_{cu}=0$) | 0.769 | 0.693 | 0.738 | 0.665 | **0.716** | 0.789 | 0.685 | k_ic (=0) |
| tm1d_full8 | - | - | - | - | - | 0.796 | 0.683 | a_g, p_ch, k_ic |
| tm2p ($c_r=1$) | 0.782 | 0.687 | 0.734 | 0.650 | 0.713 | 0.790 | 0.684 | none |
| tm2 (thermal first) | 0.782 | 0.683 | **0.698** | 0.693 | 0.714 | 0.794 | 0.679 | i0 |
| tm2b (fast first) | 0.769 | 0.685 | **0.709** | 0.658 | 0.705 | 0.795 | 0.681 | p_ch, a_t |
| tm2c (thermal = link limit) | 0.745 | **0.637** | 0.726 | 0.721 | 0.707 | 0.783 | 0.687 | p_ch, a_t |
| tm3p (tm1d + $c_r=1$) | 0.751 | 0.693 | 0.734 | 0.684 | 0.716 | 0.793 | **0.671** | a_g |

Per observable [load, frequency, share], selected folds:
w5ref p8 [0.703, 0.510, 0.738]; tm1d p8 [0.705, 0.552, 0.739]; tm3p p8 [0.705, 0.609, 0.738];
tm2c p8 [0.688, **0.737**, 0.738] but pulse [0.629, 0.472, 0.810]; tm2 multilevel [0.797, 0.444, 0.853].

## 3. Reading

1. **Line temperature (item 7).** With $k_{cu}$ free the curtailment term goes negative in three folds
   and takes the multilevel share fold from 0.853 to 0.662: rejected. With $k_{cu}=0$ the folds that
   hold out p8 or multilevel use the derating ($k_{ic}$ 0.9-2.3 per 100 units above $t_{ic}$ 25-165)
   and gain p8 frequency 0.510 -> 0.552, but **the full fit sets $k_{ic}=0$ and returns w5 exactly**
   (cost 496.37 vs 496.37, same exam). A restart from the p8-fold theta finds a lower cost (472.7) with
   the derating on ($a_{ic}=0.0066$, i.e. a 150-tick line time constant, $k_{ic}$ at its bound 5) but the
   exam drops 0.685 -> 0.683. So on the four fit runs the line temperature is weakly identified: it
   helps where p8 is held out and costs the exam when p8 is in. No deployable gain.
2. **$c_r=1$ (item 8, first part).** Pinning costs nothing: every fold within 0.001 except hold_rec
   +0.012; exam 0.684 vs 0.685; the fit moves $\beta$ 0.50 -> 0.46 and $p_r$ 82 -> 79 to absorb it.
   The text's unit statement is consistent with the data but does not add information.
3. **Two reserve units (item 8, second part).** Every merit order loses a fold. Thermal-first (tm2)
   gains p8 (+0.043) and hold (+0.012) but the multilevel frequency collapses (0.583 -> 0.444): the
   fitted energy stock is huge (cap 2,000 units) so the fast unit never runs out and the thermal lag
   cannot reproduce the 51.44 Hz plateau at reserve 126 / ic 0.58. Fast-first (tm2b) loses multilevel
   too. tm2c (thermal ramp limited by the link, $p_l+p_r\,ic$) is the only structure that explains the
   p8 frequency (0.737 held out, the first model above 0.61 on that fold) but loses the pulse frequency
   (0.472): with ic 0.2 the link limit is 30 units and the pulse needs more than that within ~10 ticks.
   That points to a fast unit whose power is NOT limited by the link and a slow unit that is; tm2c's
   fast unit has $P_b\,s(E)$ but the fit drives $P_b$ to 11-31 units on the full fit and the multilevel
   fold. Not pursued further within the time box.
4. **pg9 load population combined (item 3).** Not run: nothing in (1) or (2) produced a
   frequency/share model that beats w5 without a fold loss, so a perobs map with pg9_L0 load would be
   identical to the existing pg9 perobs doc.

## 4. Decision

- **No winner. Keep w5** (`plans/power_grid_power_grid_w5_w5_doc.json`). The acceptance rule (no loss
  on pulse, multilevel or exam) fails for every family except tm2p, which is a tie (pulse -0.001,
  exam -0.001, LOO +0.003) and adds nothing to ship.
- tm1d is the best LOO (0.716, ties the pg9 perobs doc) with no fold loss beyond 0.001 and the same
  exam, but its full fit is w5, so shipping it changes nothing.
- Open lead for a later buy or refit: p8 frequency is explained by a reserve unit whose ramp is
  limited by the link (tm2c p8 fold frequency 0.737 vs 0.510), provided a fast unit that bypasses the
  link covers the first ~10 ticks of a pulse. A run with reserve 150 at ic 0 and ic 1 would separate
  the two limits.
