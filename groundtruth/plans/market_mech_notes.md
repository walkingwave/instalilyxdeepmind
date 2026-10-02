# market: mechanism-pair search (`gtlab/ode/market_mech.py`)

Data: 5 real runs, 740 ticks (p1.hold_rec 120, p2.pulse40 80, p2.mid40 40, p2.multilevel200 200, p3.compose 300).
$\sigma_{proxy}$ = (9.59, 7.56, 24.4) for (price, volume, depth). Every fit:
`scripts/ode_lab.py --system market --family market_mech --mech XY --budget 300 --starts 10 --nfev 60 --tag XY`
(Cauchy least squares, 10 starts, RK4 with 2 substeps). Reference: `market_min` on the same 5 runs
(`plans/market_market_min_p3.json`): LOO 0.839, in-sample 0.893, l0b_lin LOO 0.817.

The question: the brief names three market mechanisms, exactly two active. A settlement tie-up (inventory ties
funding until settlement), B risk capacity loss after adverse price moves, C momentum (exposure shifts toward
recently successful strategies). Can the five runs tell which pair, and does a mechanism model beat the minimal
one (which uses a hard price floor we guessed)?

## What the price does (the weak observable)

| segment | controls $(r, x)$ | price |
|---|---|---|
| hold_rec | (0, 0) | 93.4 -> 96.4 -> 94.4, flat |
| pulse40 0-40 | (0.1, 0.05) | 108.8 -> 106.3, frozen |
| pulse40 40-80 | (0, 0) | 106.3 -> 94.7, starts slow, accelerates to -0.5/tick |
| mid40 | (0.05, 0.025) | 91.6 -> 90.4 |
| multilevel 0-37 | (0.023, 0.034) | 106.4 -> 104.5 |
| multilevel 37-95 | (0.09, 0.022) | 104.5 -> 84, accelerating |
| multilevel 95-180 | tax 0.046-0.05 | 84 -> 79.3 in 20 ticks, then flat at 79 for 65 ticks |
| multilevel 180-200 | (0, 0) | 78.8 -> 80.9, slow start |
| compose 0-60 | (0.085, 0) | 94 -> 76.5, S-shaped (slow 15 ticks, -0.5/tick, then -0.15/tick) |
| compose 60-90 | (0, 0) | 76.5 -> 81.8, accelerating |
| compose 90-150 | (0, 0.0425) | 81.8 -> 91.7 at 0.2-0.3/tick: the tax does NOT freeze the price |
| compose 180-240 | (0.085, 0.0425) | 91.7 -> 73.6, same shape and size as the rate-alone block |
| compose 240-300 | (0, 0) | 73.6 -> 85.4, S-shaped, still rising at the end |

Readings: the rate drives the price down with an S-shaped (second-order) response; at zero controls it returns
toward ~92-96, also S-shaped and slowly (tau ~ 60 ticks); a tax of 0.0425 leaves the price free but 0.046-0.05
freezes it (pulse40, multilevel 95-180). None of the rate blocks lasted long enough to show where the fall stops.

## Model (7 states, 15 fitted parameters)

States $B$ backlog, $P$ price, $D$ depth, $T$ tied funding, $E$ price EMA, $R$ lost risk capacity, $m$ momentum.
Fixed: $\tau_b = 2.7$, $\tau_E = \tau_m = 10$, $C_A = 20$, $W_X = 0.002$.

Base (always on):
$$\dot B = -B/\tau_b, \qquad \text{volume} = v_0 + B$$
$$D^* = \frac{d_0\, e^{-a_t x - a_r r}}{\rho}, \quad e = D^* - D, \quad w = \tfrac12\big(1 + e/\sqrt{e^2+4}\big), \quad \dot D = \big(k_{dn} + (k_{up} f_A - k_{dn})\, w\big)\, e$$
$$c = g(x)\, f_A \Big(\frac{k_p (p_a - P)}{\rho} - k_r\, r\, \rho\Big), \qquad \dot P = c + [C]\, m$$

No floor: without mechanisms the equilibrium is $P = p_a - k_r r / k_p$.

Switchable terms ($f_A = 1$, $\rho = 1$ when the letter is off):

- A tie-up: $\dot T = g_A\,(v_0 + B + C_A |\dot P|)(1 - T) - T/\tau_A$, $T \in [0,1]$, $f_A = 1 - T$.
  Trading (the reset order burst, then price moves that dealers absorb) ties funding; tied funding slows the
  price move and the depth refill.
- B risk capacity: $\dot E = (P - E)/\tau_E$, $\dot R = g_R \max(E - P, 0) - R/\tau_R$, $R \in [0, 5]$,
  $\rho = 1 + R$. A fall below the recent average cuts capacity: weaker pull back to the anchor, bigger impact
  of the rate pressure, lower depth target.
- C momentum: $\dot m = (k_m c - m)/\tau_m$, $|m| \le 5$. With $k_m > 1$ a started move accelerates.

Tax gate on the price, two versions:

- v1: $g(x) = e^{-a_x x}$ (as in market_min).
- v2 (the one structure revision, kept in the module): $g(x) = 1/\big(1 + e^{(x - x_c)/W_X}\big)$, a threshold:
  the price trades freely below $x_c$ and freezes above it.

Reset: $B_0 = \max(\text{volume}_0 - v_0, 0)$, $P_0 = E_0 = \text{price}_0$, $D_0 = \text{depth}_0$, $T_0 = R_0 = m_0 = 0$.
All rates <= 1.5/tick at the bounds; states clipped; batched and scalar $f$ agree to 0.

## Results (score per observable = mean of $1/(1 + |err|/\sigma_{proxy})$)

LOO = fit on 4 runs, score the 5th, mean over folds. P/V/D = price, volume, depth.

| model | LOO P | LOO V | LOO D | LOO mean | in-sample P | V | D | in-sample mean | cost | at bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| market_min (floor) | 0.721 | 0.947 | 0.851 | 0.839 | 0.828 | 0.950 | 0.903 | 0.893 | | none |
| v1 AB | 0.716 | 0.946 | 0.823 | 0.828 | 0.846 | 0.950 | 0.912 | 0.902 | 80.3 | g_A, tau_R |
| v1 AC | **0.742** | 0.946 | 0.853 | **0.847** | 0.846 | 0.950 | 0.904 | 0.900 | 82.5 | none |
| v1 BC | 0.704 | 0.947 | 0.785 | 0.812 | 0.824 | 0.950 | 0.908 | 0.894 | 92.9 | none ($k_m \to 0$) |
| v2 AB | 0.680 | 0.946 | 0.822 | 0.816 | 0.867 | 0.950 | 0.915 | 0.910 | | g_A, tau_R |
| v2 AC | 0.726 | 0.946 | 0.857 | 0.843 | **0.877** | 0.950 | 0.905 | **0.910** | 61.8 | g_A |
| v2 BC | 0.685 | 0.947 | 0.831 | 0.821 | 0.862 | 0.950 | 0.912 | 0.908 | | tau_R |
| l0b_lin | | | | 0.817 | | | | | | |

Price LOO by held-out run:

| model | hold_rec | pulse40 | mid40 | multilevel | compose |
|---|---:|---:|---:|---:|---:|
| market_min | 0.903 | 0.624 | 0.886 | 0.635 | 0.557 |
| v1 AB | 0.893 | 0.706 | 0.897 | 0.768 | 0.317 |
| v1 AC | 0.864 | 0.756 | 0.889 | 0.766 | 0.437 |
| v1 BC | 0.837 | 0.674 | 0.883 | 0.815 | 0.313 |
| v2 AB | 0.745 | 0.856 | 0.869 | 0.660 | 0.268 |
| v2 AC | 0.746 | 0.884 | 0.879 | 0.625 | 0.498 |
| v2 BC | 0.732 | 0.827 | 0.871 | 0.597 | 0.399 |

Fitted parameters, v2 AC (`plans/market_market_mech_AC2.json`): $v_0$ 2.31, $d_0$ 89.7, $a_t$ 23.7, $a_r$ 1.80,
$k_{dn}$ 0.0525, $k_{up}$ 0.271, $p_a$ 97.1, $k_p$ 0.00707, $k_r$ 4.31, $x_c$ 0.0472, $g_A$ 0.02 (upper bound),
$\tau_A$ 11.5, $k_m$ 1.56 ($g_R$, $\tau_R$ unused). v1 AC: $p_a$ 96.3, $k_p$ 0.0167, $k_r$ 11.07, $a_x$ 13.3,
$g_A$ 0.017, $\tau_A$ 23.4, $k_m$ 0.448, depth as above within 10%.

The threshold is the one number every fit agrees on: $x_c$ = 0.0481 (AB), 0.0472 (AC), 0.0484 (BC).

Eval sanity (ode_lab eval-shaped 4,000-tick rollouts): all finite, 0.3-0.5 s each; fraction outside the data
range +-5%: sustained 0.08, order 0.20-0.23, recovery 0, composition 0.06-0.11 (market_min: 0 everywhere).
Constant 4,000-tick holds from $y_0 = (94, 95, 95)$, final price:

| model | (0,0) | (0.1, 0.05) | (0.085, 0.0425) | (0.1, 0) | (0, 0.05) | (0.05, 0.025) |
|---|---:|---:|---:|---:|---:|---:|
| market_min | 94.0 | 79.0 | 79.0 | 79.0 | 94.0 | 79.0 |
| v1 AC | 96.3 | 30.2 | 40.1 | 30.2 | 96.3 | 63.3 |
| v2 AC | 97.1 | 36.2 | 45.3 | 36.2 | 97.1 | 66.6 |
| v2 BC | 100.2 | 43.2 | 51.7 | 43.2 | 100.2 | 71.7 |
| v2 AB | 99.1 | 39.3 | 48.3 | 39.3 | 99.1 | 69.2 |

Without a floor, every mechanism model puts the long-hold price at 30-50 under a high rate; the lowest price we
have ever seen is 73.6, and it was still falling. The frozen 79 in multilevel is explained by the tax threshold
in v2, so the data no longer needs a floor, but nor does it rule one out.

## Verdict

- Pair ranking by LOO: AC (0.847 v1 / 0.843 v2) > AB (0.828 / 0.816) > BC (0.812 / 0.821). AC is first in both
  versions; A (tie-up after trading) + C (momentum, $k_m$ = 1.56 in v2: a started move accelerates) is our pick.
  B's fits hit $\tau_R$ = 3 (its lower bound): the risk state collapses to an instant function of the last fall, i.e.
  a momentum surrogate. In v1 BC the momentum gain went to 0. So B is doing C's job; we cannot separate B from C.
- Against market_min: +0.008 (v1 AC) / +0.004 (v2 AC) LOO, inside fold-to-fold noise. v2 AC is better in sample
  (0.910 vs 0.893, price 0.877 vs 0.828) and much better on the pulse40 fold (0.884 vs 0.624), but worse on the
  hold_rec and multilevel folds. Lab verdict "maybe" for v2 AC, "ship" for v1 AC (only because the lab threshold
  is +0.03 over l0b_lin: 0.847 vs 0.817).
- We do not replace market_min with this yet. The two disagree by up to 45 price points on long high-rate holds
  (79 vs 30-45), and nothing in 740 ticks decides it. The cheap test: one long rate-only hold (r = 0.1, x = 0,
  150+ ticks) shows whether the price stops near 75-79 (floor) or keeps going.
- Tax threshold near 0.047 is the robust new finding. The pulse rule puts test taxes at 0.035-0.05 (70-100%),
  straddling it: about 20% of pulses would freeze the price. Worth porting into market_min as its tax gate.

Files: `plans/market_market_mech_{AB,AC,BC}.json` (+ `_doc.json`) are v1 fits; their theta vector has $a_x$ at
position 10, which the current module reads as $x_c$. They are records only: do not package them with the
current module. `plans/market_market_mech_{AB2,AC2,BC2}.json` (+ `_doc.json`) match the current module.
