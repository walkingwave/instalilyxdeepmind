# market y: mechanism identification on a brief-faithful base (Sun Sep 27, no credits)

Goal: public 0.627 (x4p: x4 with price from median(x4, l0b_lin)) toward 0.72 by finding the right mechanism
structure. Judged only by the lab: `scripts/ode_lab.py --system market --family <fam> --mech <pair> --budget 300
--starts 8 --nfev 60 --sigma-cal 1.0`, leave-one-run-out at the organizer's scale
($\sigma$ = 0.816, 0.643, 2.077; fold noise ≈ 0.02), the p5.testlike fold as the exam, and 4,000-tick holds.
Data: **8 runs, 1,540 ticks** (the 7 of the x round plus `p6.voi`, the joint (0.05, 0.025) hold, 200 ticks).
No score polishing anywhere.

**Result: `market_y3` pair AB, doc `plans/market_market_y3_AB_doc.json`. LOO 0.615 (x4 on the same 8 runs 0.578,
x4p 0.565), exam 0.591 (x4 0.541, x4p 0.512), multilevel fold 0.560 (x4 0.473). Short of LOO 0.64; exam goal
(≥ 0.56) met.** Mechanism reading: B (risk-capacity loss) is active; A vs C is not resolved by the data.

## 1. What p6.voi adds

| ticks | price | depth | reading |
|---|---|---|---|
| 0-12 | 102.7 → 103.3 | 109 → 72 | price holds or rises right after a reset (as in hold_rec's hump) |
| 12-95 | 103 → 87.5, fall speeds up −0.15 → −0.25/tick | 72 → 43 | S-shaped fall under a mid rate |
| 95-200 | 87.5 → 83.2, flattening (−0.016/tick at the end) | 43 → 53.4, rising ~0.13/tick, slightly accelerating | **an interior level ≈ 83 at (0.05, 0.025)**; depth recovers once the price stops falling |

- The price level at steady rate is close to linear: ≈ 92 (zero), ≈ 83 (0.05), ≈ 74 (0.1). x4 collapsed every rate
  ≥ 0.05 to its floor 74.5, so x4's voi fold is 0.53 and its in-sample price there 0.36.
- The depth recovery at a constant tax after the fall stops is the risk-capacity signature (mechanism B): the 43-45
  "tax plateau" read earlier from falling-price segments is partly capacity lost to adverse moves.

## 2. Structures

All: numpy + math, batched f/h/x0 (the `_PY`/`_NP` pattern of market_x4), N_SUB = 2, every observed initial used,
the three mechanisms as real switches (`"A" in mech` etc.). Shared base (market_x4's): producers' working cash $C$,
common fundamental $Q \to p_{full}$ ($\tau_Q$), cash-supported level $A_1 = p_{lo} + (Q - p_{lo})C$ with a lagged
anchor $A$ (fast down, slow up), committed orders $\dot P = v$, $\dot v = (g_{eff}\,\mathrm{push} - v)/\tau_v$,
$\mathrm{push} = s_v\tanh(\kappa(A-P)/s_v)$, $g_{eff} = g_l + (1-g_l)g$ (orders keep executing in a freeze), tax gate $g$
at $x_c$, depth $D \to D^*$ (τ 8) plus reset-burst inventory $G \to H$ settling slower under tax, volume as x4.

Mechanisms:
- **A settlement tie-up of funding**: stranded inventory $S \to (1-g)\,r/0.1$ costs capacity at the rate,
  $D^* = d_0(1 - m_1 s(x))(1 - m_2 S)$. Off: a freeze thins quotes by $m_2$ regardless of the rate,
  $D^* = d_0(1 - m_1 s(x))(1 - m_2(1-g))$.
- **B risk-capacity loss**: $\dot R = k_R\,\mathrm{fall}\,(0.3 + 0.7\,\tfrac{r'}{1+r'})\,s_R(x)/20 - R/\tau_R$, depth $= D(1-R) + H$;
  in market_y also a price channel: lost capacity amplifies a fall, $\mathrm{push} \leftarrow \mathrm{push} - k_{BP} R \max(-\mathrm{push}, 0)$.
- **C investor momentum**: $\dot M = (v - M)/12$, $\mathrm{push} \leftarrow \mathrm{push} + k_M M$.

Bases:

| family | params | price law | idea |
|---|---:|---|---|
| market_y | 27 | cash $\dot C = \rho g (C + c_0)(1-C) - a_r (r/0.1)^{n_r}(1 - C + b_0) - c_f (1-g) C$ | x4 plus a freeze cash drain (production without revenue), A/B/C switches, B price channel, momentum |
| market_y2 | 25 | reduced order flow: debt $L$ in price units, $\dot L = a_r r' + i_r r' L + c_f(1-g) - \rho g L$, $A_1 = p_{lo} + 1.5\,\mathrm{softplus}((Q - L - p_{lo})/1.5)$ | linear-in-rate steady state with interest on the debt, soft floor |
| **market_y3** | 26 | market_y cash **plus consumers' warehouses** $I_w$ (0.5 at reset): $\dot I_w = g_{eff} k_w (I^* - I_w)$, $I^* = i_0 - i_r r/0.1$; $A_1 = p_{lo} + (Q-p_{lo})C + k_I (I^* - I_w)$ | storage demand: half-full warehouses restock after a reset (hump), a rate lowers the desired fill and destocking pushes the price down until the warehouses reach it; pinned $c_p$ 1.145, $w_d$ 0.0014, $x_R$ 0.037, $\tau_Q$ 204; B on depth only |
| market_y4 | 26 | market_y with a two-stage order pipeline $\dot O = (g_{eff}\,\mathrm{push} - O)/\tau_v$, $\dot v = (O - v)/\tau_v$; $c_p$ pinned; $p_{full} \in [89, 99]$ | orders prepared before a change keep executing after it |
| market_y5 | 26 | market_y3 + the two-stage pipeline | |
| market_y6 | 26 | market_y3 with $p_{full} \in [89, 99]$ | zero-control level check |

## 3. Results (lab, 1.0σ, 8 runs; fold = held-out run mean of price, volume, depth)

| family | pair | hold_rec | pulse40 | mid40 | multilevel | compose | rate_hold | **exam** | voi | **LOO** | LOO P/V/D | exam P/V/D | in-sample | at bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| x4 (ref) | – | 0.693 | 0.597 | 0.689 | 0.473 | 0.559 | 0.543 | 0.541 | 0.527 | 0.578 | .41/.76/.55 | .37/.76/.49 | 0.659 | p_full, c0, tau_s |
| y | AB | 0.697 | 0.558 | 0.654 | 0.476 | 0.605 | 0.666 | 0.553 | 0.531 | 0.592 | .42/.76/.59 | .39/.77/.49 | 0.652 | – |
| y | AC | 0.657 | 0.576 | 0.665 | 0.448 | 0.628 | 0.527 | 0.558 | 0.560 | 0.577 | .42/.77/.54 | .38/.76/.53 | 0.637 | w_d, p_full, c0 |
| y | BC | 0.689 | 0.655 | 0.684 | 0.468 | 0.602 | 0.580 | 0.552 | 0.579 | 0.601 | .44/.76/.60 | .41/.77/.47 | 0.664 | p_full, c0 |
| y2 | AB | 0.686 | 0.509 | 0.645 | 0.446 | 0.558 | 0.527 | 0.495 | 0.508 | 0.547 | .38/.74/.52 | .28/.73/.47 | 0.639 | i_r, c_f |
| y2 | AC | 0.623 | 0.501 | 0.646 | 0.426 | 0.566 | 0.482 | 0.509 | 0.517 | 0.534 | .32/.74/.53 | .26/.73/.53 | 0.625 | w_d |
| y2 | BC | 0.685 | 0.509 | 0.648 | 0.502 | 0.555 | 0.491 | 0.488 | 0.508 | 0.548 | .36/.74/.55 | .28/.73/.46 | 0.643 | i_r |
| **y3** | **AB** | 0.712 | 0.627 | 0.662 | **0.560** | 0.644 | 0.605 | **0.591** | 0.518 | **0.615** | .46/.78/.61 | .41/.79/.58 | 0.667 | p_full, c0, c_f |
| y3 | AC | 0.670 | 0.547 | 0.670 | 0.445 | 0.641 | 0.499 | 0.573 | 0.557 | 0.575 | .43/.76/.53 | .40/.79/.53 | 0.646 | p_full, c0, a_r, b0, c_f |
| y3 | BC | 0.673 | 0.629 | 0.662 | 0.466 | 0.643 | 0.576 | 0.586 | 0.591 | 0.603 | .43/.76/.62 | .42/.79/.56 | 0.673 | p_full, c0, c_f |
| y4 | AB | 0.696 | 0.611 | 0.673 | 0.485 | 0.624 | 0.542 | 0.533 | 0.521 | 0.586 | .40/.76/.59 | .36/.76/.48 | 0.660 | p_full, c0 |
| y4 | BC | 0.692 | 0.626 | 0.680 | 0.509 | 0.578 | 0.461 | 0.541 | 0.522 | 0.576 | .40/.75/.58 | .38/.77/.48 | 0.664 | x_c, p_full, c0, a_r |
| y5 | AB | 0.714 | 0.559 | 0.674 | 0.580 | 0.648 | 0.599 | 0.587 | 0.530 | 0.611 | .47/.77/.59 | .40/.80/.57 | 0.668 | p_full, c0, c_f |
| y6 | AB | 0.702 | 0.616 | 0.687 | 0.454 | 0.646 | 0.499 | 0.584 | 0.585 | 0.597 | .45/.77/.57 | .39/.79/.58 | 0.667 | p_full, c0, b0, c_f |

The two-stage order pipeline adds nothing (y5 AB 0.611 vs y3 AB 0.615; y4 below y). Bounding $p_{full} \ge 89$
costs 0.018 (y6).

x4 was re-run by the lab on the same 8 runs (`plans/market_market_x4_y8ref.json`); its 7-run numbers (LOO 0.591,
exam 0.525) are not comparable. The first nine fits ran while the machine was overloaded (the lab's budget is wall
clock, so those fits got fewer evaluations); x4's reference ran under the same load.

### Mechanism pair

| base | AB | AC | BC |
|---|---:|---:|---:|
| y | 0.592 | 0.577 | **0.601** |
| y2 | 0.547 | 0.534 | **0.548** |
| y3 | **0.615** | 0.575 | 0.603 |

- **AC is last in all three bases** (by 0.013-0.040; in y3 by 0.040 = 2 fold-noise units, with the multilevel,
  rate_hold and pulse40 folds all worse). Every pair with B beats the pair without it: **B (risk-capacity loss after
  adverse moves) is active.** Physically: depth dips while the price falls (rate_hold 90 → 72, compose 98 → 78) and
  rebuilds once the fall stops (voi 43 → 53 at a constant tax; rate_hold 72 → 90 under a constant rate), which the
  depth fold scores pick up (AC depth 0.53-0.54 vs 0.60-0.62 with B).
- **A vs C is not identified**: AB wins in y3 (+0.012), BC wins in y (+0.009) and y2 (+0.001), all below fold noise.
  With C on, the momentum gain is used ($k_M$ 0.36 in y BC, 0.26 in y3 BC); with C off, y's AB fit moves the job to
  B's price channel ($k_{BP}$ 2.26). In these models A only acts on depth (the freeze loss priced at the rate), so the
  A/C split rests on a few frozen segments. The data that would separate them: a pure-tax freeze (0, 0.05) held
  40-60 ticks (A on: depth settles ≈ 45, the freeze loss needs the rate; A off: ≈ 23 as under (0.1, 0.05)).

### Price blend with l0b_lin (held-out, paired: same fold fits, `blend` = price $(1-w)\,$ODE $+ w\,$l0b_lin)

| model | w = 0 | w = 0.25 | w = 0.5 (= median of two) | exam w = 0 / 0.25 / 0.5 |
|---|---:|---:|---:|---|
| x4 | 0.578 | 0.600 | 0.565 (x4p) | 0.541 / 0.542 / 0.512 |
| y3 AB | **0.618** | 0.599 | 0.575 | 0.591 / 0.596 / 0.556 |
| y3 BC | 0.607 | 0.613 | 0.579 | 0.586 / 0.589 / 0.547 |
| mean(y3 AB, y3 BC) | 0.613 | 0.605 | 0.580 | 0.588 / 0.592 / 0.551 |

(Paired re-fits reproduce the lab folds to ±0.01; y3 AB 0.618 vs lab 0.615.) **The price blend no longer helps the
winner**: w = 0.5 costs y3 AB 0.043 held-out (price 0.467 → 0.336), w = 0.25 costs 0.019. For x4 the median (w = 0.5)
also loses held-out (0.578 → 0.565, exam 0.541 → 0.512); only w = 0.25 helps x4 (0.600). The AB/BC committee does
not beat AB alone.

## 4. Long holds (4,000 ticks from initial price 92 / 100 / 108; y3 AB)

| hold (r, x) | t450 | t4000 | spread over $y_0$ | depth t4000 |
|---|---|---:|---:|---:|
| (0, 0) | 86.8 / 87.9 / 89.0 | 86.0 | 0 | 91.2 |
| (0.025, 0) | 86.8 / 87.9 / 88.9 | 86.0 | 0 | 91.2 |
| (0.05, 0) | 79.8 / 80.2 / 80.6 | 79.4 | 0 | 91.1 |
| (0.05, 0.025) | 79.8 / 80.2 / 80.6 | 79.4 | 0 | 44.6 |
| (0.07, 0) / (0.1, 0) / (0.085, 0.0425) | 75.1 | 75.1 | 0 | 91.0 / 89.8 / 44.3 |
| (0, 0.025) | 86.8-89.0 | 86.0 | 0 | 44.6 |
| (0, 0.05) frozen | 82.6 / 85.4 / 88.9 | 75.1 | 0 | 44.7 |
| (0.1, 0.05) frozen | 75.2 / 79.0 / 83.7 | 75.1 | 0 | 22.2 |

Volume 1.78-1.79 at t4000, all finite, 0.6 s per 4,000 ticks, lab eval schedules ≤ 0.4 % of ticks outside the
observed range. Every hold settles to a control-set level independent of the start (spread 0 by t4000; trading
holds by t450). Three known issues:
1. **Zero-control level 86** ($p_{full}$ at its lower bound in every y-family fit, as x4 hit 89): the runs return to
   91.5-94.5 within 300 ticks. The fit uses $p_{full}$ to buy the slow recoveries after deep drains (testlike's end,
   compose's last 60 ticks). y4 with $p_{full} \ge 89$ lost 0.006-0.025 LOO; market_y6 (y3 with $p_{full} \ge 89$):
   LOO 0.597 (−0.018; multilevel 0.560 → 0.454, rate_hold 0.605 → 0.499), exam 0.584, and $p_{full}$ sits at 89,
   zero-control hold → 89. The data prefer the low level; we keep the unconstrained fit and flag it.
2. **(0.05, 0.025) at 79.4**, below voi's 83.2 at t200 (the data are still falling slowly, −0.016/tick).
3. **Frozen at zero rate drifts to the floor 75** (freeze cash drain $c_f$ at its upper bound 0.1). Only 12 frozen
   zero-rate ticks exist (testlike 330-342, flat price); this is an extrapolation, not a measurement.

## 5. Fitted parameters (y3 AB, full 8-run fit)

$d_0$ 92.33, $m_1$ 0.516, $x_d$ 0.0185, $m_2$ 0.503, $k_R$ 0.853, $\tau_R$ 19.9, ($k_M$ 0.012: C off), $g_l$ 0.109,
$x_c$ 0.04465, $p_{full}$ 86.0 (bound), $p_{lo}$ 75.13, $\rho$ 0.0165, $c_0$ 2.0 (bound), $a_r$ 1.103, $n_r$ 7.09,
$b_0$ 2.36, $c_f$ 0.1 (bound), $i_0$ 0.630, $i_r$ 0.437, $k_w$ 0.0260, $k_I$ 54.2, $\tau_A$ 11.8, $\tau_{Au}$ 3.08,
$\kappa$ 0.0273, $s_v$ 0.628, $\tau_v$ 9.81. In-sample per run (P/V/D): hold_rec .55/.89/.79 (price 0.40 in x4: the
reset hump is now the warehouses restocking), pulse40 .54/.75/.68, mid40 .67/.75/.57, multilevel .65/.68/.57,
compose .60/.79/.68, rate_hold .49/.83/.66, testlike .48/.79/.58, voi .61/.76/.63.

The warehouse channel carries the gain: $k_I (I^* - I_w)$ with $k_I$ ≈ 54 and $i_r$ ≈ 0.44 gives a destocking
push of up to ≈ −24 at rate 0.1 that fades as the warehouses reach the lower fill ($k_w$ 0.026, τ ≈ 40), and a
restocking push of ≈ +7 after a reset at zero controls. With $n_r$ 7 the cash drain is a sharp threshold near
r ≈ 0.06; below it the price level is set by the warehouses and the cash equilibrium together.

## 6. Recommendation

- **Doc: `plans/market_market_y3_AB_doc.json`** (market_y3, pair AB, lab fit on all 8 runs), alone, no price blend.
- Held-out gain over x4p: LOO 0.618 − 0.565 = **+0.053** (exam +0.079). **Forecast public gain = 0.6 × 0.053 ≈
  +0.032, i.e. ≈ 0.66** (from 0.627). Against the best x4 variant we found (x4 with a w = 0.25 price blend, 0.600)
  the held-out gain is only +0.018, below fold noise, so the forecast rests on x4p being the shipped form.
- Risk: the zero-control level (86 vs 92-94 observed) and the frozen zero-rate drift are extrapolations in the
  sustained band; see §4.
- Next data, if any credits go to market: the pure-tax freeze (0, 0.05) for 40-60 ticks (separates A from C, and
  measures the frozen zero-rate drift), then a 300-tick zero-control hold after a deep drain (pins $p_{full}$ vs
  slow cash regrowth).

Files: `gtlab/ode/market_y.py`, `market_y2.py`, `market_y3.py`, `market_y4.py`, `market_y5.py`, `market_y6.py`;
lab reports `plans/market_market_{y,y2,y3}_{AB,AC,BC}.json`, `plans/market_market_y4_{AB,BC}.json`,
`plans/market_market_y5_AB.json`, `plans/market_market_y6_AB.json`, `plans/market_market_x4_y8ref.json`
(+ `_doc.json` each).
