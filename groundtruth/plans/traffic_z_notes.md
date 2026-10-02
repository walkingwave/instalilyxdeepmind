# traffic z: history mechanisms, toll demand and the long-hold leak (Sun Sep 27 night – Mon Sep 28)

No credits spent. Every fit: `scripts/ode_lab.py --system traffic --family <fam> --mech <pair> --budget 300
--starts 8 --nfev 60 --sigma-cal 1.0 --free <base + active pair>`, no polishing, run through a wrapper that
drops **p5.testlike** (the exam) from the loaded runs. Scores at the calibrated scale
$\sigma$ = (4.01, 3.53, 5.21, 4.60) for (flow_a, flow_b, speed_a, speed_b). LOO = refit on all runs but one,
score the held-out run. Baselines were refit the same way (v8d from its shipped theta), so every row
below is comparable within its block.

## 1. Structures (`gtlab/ode/traffic_z*.py`)

| family | what it adds to the v8d pipeline |
|---|---|
| traffic_z | the brief's three history mechanisms as switches. A route learning: admitted split $q$, $\dot q = (\sigma(\beta_L (s_a - s_b)/10) - q)/\tau_L$, $q(0) = 0.5$. B crew fatigue / switching cost: crew position $\dot c = (u_{clr} - c)/\tau_{sw}$, fatigue $\dot F = (c - F)/\tau_F$, exit boost $1 + k_c c(1 - k_F F)$. C persistent spillback fronts: finite exit buffer $K_3$ throttles junction→exit $r p_2(1 - p_3/K_3)^+$; front $G_i$ follows $(p_{3i}/K_3)^2$, rising at $r_{up}$, dissolving at $r_{dn}$; fronts block the shared junction $J/(1 + c_J(G_a + G_b))$ and add $w_G n_{ref} G_i$ to the speed occupancy |
| traffic_z2 | z + toll-elastic demand $a_{tot} = d_0 u_{ramp} e^{-k_{toll}(u_{toll} - 2.5)}$, crew taken from junction $J(1 - j_c c)$, approach diversion $-d_{div} p_1^2/n_{max}$ |
| traffic_z3 | z2 with a logistic toll response (fit went back to the exponential regime) |
| traffic_z4 / z7 | toll sets demand, ramp caps it: $a_{tot} = \mathrm{smin}(d_0 e^{-k_{toll} u_{toll}}, c_{ramp} u_{ramp})$ (z7 + minimum junction share) |
| traffic_z5 | z4 + speed lag equal to the journey time $n_i/\mathrm{out}_i$ |
| traffic_z6 | z2 + minimum junction share $s_{min}$ (fit put it at 0.08–0.10: inactive) |
| traffic_z8 | z + toll-elastic demand + lane closure per exit section $\mathrm{cap}(1 - (u_{lane}/L_i)^{n_l})^+$ (fit: $n_l = 1$, i.e. linear, $1/L_b = 0.85$) |

## 2. Mechanism identification (5 runs, exam held out)

LOO folds (hold_rec, pulse40, mid40, multilevel200, hold_mid), exam = p5.testlike scored with the full fit,
band = RMSE between the public sustained bands of u001/u003/u004/u005/u008/u010b/u012 and the scores each
of them would get if this candidate were the truth on 20 sustained-style 4,000-tick schedules.

| structure | pair | LOO folds | LOO | exam | band |
|---|---|---|---:|---:|---:|
| v8d (lab refit) | – | .988 .513 .855 .633 .829 | **.764** | .673 | .175 |
| z | AB | .975 .525 .856 .631 .779 | .753 | .672 | .155 |
| z | AC | .960 .461 .852 .598 .843 | .743 | .672 | .158 |
| z | BC | .977 .525 .856 .630 .769 | .751 | .674 | .177 |
| z2 | – | .992 .461 .865 .678 .767 | .753 | **.725** | .127 |
| z2 | AB | .988 .416 .867 .673 .632 | .715 | .733 | .114 |
| z2 | AC | .971 .436 .856 .662 .766 | .738 | .732 | .121 |
| z2 | BC | .989 .399 .870 .676 .689 | .725 | .730 | .108 |
| z2, toll only ($j_c = d_{div} = 0$) | – | .992 .458 .866 .678 .770 | .753 | – | – |
| z3 | – | .991 .472 .866 .623 .764 | .743 | .724 | .127 |
| z4 / z7 | – | .991 .496 .860 .60 .749 | .740 | .694 | .155 |
| z5 | – | .980 .451 .852 .579 .774 | .727 | .681 | .162 |
| z6 | – | .992 .461 .866 .658 .767 | .749 | .728 | .127 |
| shipped v8d doc (u012) | – | – | – | .682 | – |

Fold noise, from refits whose extra mechanisms stayed inert: pulse40 ±0.012, multilevel ±0.003,
hold_mid ±0.05. **No history mechanism is identified**: every pair fit either leaves the mechanism at its
inert values (fatigue $k_F \to 0$, switching lag at its 0.2 lower bound) or uses it as a fast term (route
choice with $\tau_L \to 1$, spillback fronts dissolving in 1–3 ticks: not persistent). Pairs never beat
their base beyond fold noise; the one pair that turned fatigue fully on (z2 AB on six runs, $k_F = 1$,
$\tau_F = 29$) makes full clearance worthless on long holds, a claim no run supports.

**Toll is the missing base term.** Every fit that may use it puts $k_{toll} \approx 0.21$–$0.25$ (demand at
toll 0 is 1.7× demand at 2.5, at toll 5 0.55×). Direct evidence in the exam's interior hold (ticks 140–180,
toll 4.75, ramp 0.87): truth flows 11.3 / 10.7, speeds 32.6 / 29.0; v8d 16.5 / 10.2, 28.1 / 19.2; z2
10.8 / 10.8, 33.6 / 24.9. That window scores 0.54 → 0.71.

It loses the pulse40 fold (0.513 → 0.46): pulse40 is the only run with toll 0 and full ramp, so without it
the toll-0 demand and the approach buffer are extrapolated. Check: the same LOO on the five runs **plus the
exam** (whose pulse is at toll 0.06):

| structure | pair | hold_rec | pulse40 | mid40 | multilevel | hold_mid | testlike | LOO | band |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| z (= v8d) | – | .961 | .519 | .815 | .650 | .681 | .673 | .717 | .103 |
| z2 | – | .966 | **.545** | .854 | **.672** | .838 | .725 | **.767** | .095 |
| z2 toll only | – | .966 | .539 | .854 | .672 | .838 | .727 | .766 | – |
| z2 | AB | .966 | .548 | .868 | .677 | .836 | .733 | .771 | .093 |
| z2 | AC | .979 | .551 | .844 | .656 | .859 | .732 | .770 | .097 |
| z2 | BC | .972 | .543 | .849 | .661 | .777 | .730 | .755 | .090 |

With one more toll-0 overload in the data, the toll structure wins every fold, pulse40 (+0.026) and
multilevel (+0.022) included. The pulse loss is coverage, not structure. $j_c$ and $d_{div}$ go to 0 in
every six-run fit (toll only ties z2), so the toll term alone carries the gain.

## 3. p7.longhold (600 ticks, 504 at signal .2, lane .5, toll 1.2, ramp .9, freight .9, clearance .1)

Truth settles by t ≈ 150 at flows 10.0 / 12.7–13.3, speeds **7.4 / 15.9**: the starved route is the slow
one (the long-hold audit's "symmetric routes" hint is wrong here). The shipped v8d predicts 6.6 / 6.7 /
10.9 / 12.3 (score 0.552); flow_b is 6 low because lane closure 0.5 cuts route B's exits to half in v8d.

Six-run fits (five runs + p7, exam held out):

| structure | hold_rec | pulse40 | mid40 | multilevel | hold_mid | p7 | LOO | exam | p7 full fit, t 300–500 | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| v8d (refit) | .941 | **.538** | .863 | .637 | .806 | .559 | .724 | .675 | 9.6 / 11.7 / 7.1 / 15.2 (0.782) | .105 |
| z8 (toll, lane shape) | .983 | .489 | .856 | **.662** | .846 | **.591** | **.738** | **.728** | 9.9 / 11.7 / 7.4 / 15.4 (0.798) | .096 |

Readings: (i) most of the p7 leak is closed by refitting v8d with p7 in the data (0.552 → 0.782 in-sample;
$\ell_b$ 0.99 → 0.78, $w_{sig}$ 1.0 → 0.76, clearance boost → 0); (ii) z8 adds the toll term: p7 fold +0.032,
multilevel +0.025, exam +0.053, but pulse40 −0.049, the same toll-0 coverage loss as in §2 (the lane-shape
exponent went to 1, so z8 is toll-only v8d); (iii) the older candidates without p7 score 0.61–0.65 on it.

## 4. Long-hold levels (end of 4,000-tick holds; flow_a / flow_b / speed_a / speed_b)

| model | pulse action | mid (0.5, .38, 2.5, .5, .5, .5) | full throughput (ramp 1, toll 0, lane 0) | h0 (toll 3.4) | h9 (toll .95, ramp .9) |
|---|---|---|---|---|---|
| u008 (sustained .647) | 4.8 / 4.0 / 9.6 / 9.2 | 9.7 / 9.6 / 31.4 / 30.2 | 15.9 / 15.9 / 20.4 / 20.4 | 13.6 / 12.6 / 26.7 / 22.8 | 14.7 / 9.9 / 21.4 / 13.5 |
| u010b (.633) | 4.9 / 3.9 / 8.2 / 7.8 | 10.4 / 10.3 / 33.5 / 32.2 | 16.4 / 16.4 / 18.1 / 18.1 | 14.8 / 14.1 / 28.5 / 24.5 | 15.5 / 10.3 / 19.8 / 11.9 |
| u012 v8d (.635) | 5.2 / 4.7 / 8.3 / 10.8 | 10.8 / 10.6 / 33.8 / 32.2 | 17.1 / 17.1 / 25.0 / 25.0 | 14.8 / 13.4 / 29.3 / 24.7 | 15.6 / 10.9 / 23.7 / 15.9 |
| v8d refit + p7 | 8.5 / 9.8 / 5.7 / 12.4 | 10.5 / 10.5 / 32.7 / 32.5 | 17.8 / 17.8 / 20.4 / 20.4 | 14.8 / 13.7 / 27.3 / 23.6 | 17.1 / 14.6 / 24.9 / 18.5 |
| z8 + p7 | 8.9 / 9.3 / 6.0 / 12.5 | 10.5 / 10.4 / 32.7 / 32.6 | 17.7 / 17.7 / 11.8 / 11.8 | 13.2 / 13.1 / 29.3 / 28.7 | 19.0 / 15.3 / 20.6 / 16.3 |

u008's lower speeds are partly just its miss on the one long hold we own (hold_mid: u008 9.6 / 9.6 / 30.5 /
30.8, data 10.2 / 10.8 / 31.9 / 33.5), so we do not steer toward u008's levels. The p7-trained models move
the pulse-action hold from symmetric low speeds to the p7 pattern (starved route A near 6, route B near 12)
and raise pulse flows (≈ 9 each, p7 shows 10 / 13). The two p7-trained models differ mainly where toll is
low and ramp high (full throughput: speeds 20 vs 12), a regime no run holds for long.

## 5. Decision

- **No candidate passes the rule on the exam-held-out folds**: every toll structure loses pulse40
  (0.538 → 0.489 with p7). With the exam's toll-0 pulse in the folds, the toll structure wins pulse40 and
  multilevel.
- **Safe pick (passes the rule): `plans/traffic_traffic_v8d_p7base_doc.json`**, v8d refit on the five runs
  + p7. Against the pre-p7 lab refit it keeps pulse40 (.513 → .538) and multilevel (.633 → .637) and fixes
  the long hold (p7 in-sample 0.552 → 0.782); exam 0.675 vs the shipped 0.682 (−0.007, within noise).
- **Higher-expected pick, rule exception: `plans/traffic_traffic_z8_z8p7_doc.json`**: LOO +0.014, exam
  +0.053, p7 fold +0.032, multilevel +0.025 over the safe pick; pulse40 −0.049 on exam-held-out folds,
  explained by toll-0 coverage (§2).
- Forecast (0.6 × held-out gain, public traffic 0.776): safe pick ≈ 0.776 + 0.6 × (p7 is the only new
  held-out evidence; exam −0.007) ≈ **0.77–0.79** (the p7 fix is in-sample and targets the sustained band,
  so the realised gain is uncertain); z8 ≈ 0.776 + 0.6 × 0.014 (LOO) to 0.6 × 0.046 (exam vs the shipped
  model) ≈ **0.785–0.80**.
- Next data that would settle it: a long toll-0, full-ramp hold (or the exam refit into the folds) to pin
  toll-0 demand; then refit z8 on all seven runs.
