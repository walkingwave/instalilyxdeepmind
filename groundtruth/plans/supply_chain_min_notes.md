# supply_chain: minimal grey-box ODE (`gtlab/ode/supply_chain_min.py`)

Data: 2 real runs, 520 ticks. `hold_rec` (120 ticks at the recovery action) and
`pulse200_200` (200 ticks at the pulse action, then 200 at recovery). Sigma proxy
$[9.1, 92.0, 44.8]$ for (shipments, inventory_supplier, inventory_retail).

## What the runs show

- Recovery (order 0, production 1, receiving 1.5, maintenance 1): shipments 0; retail drains
  105 -> 0 in 4 ticks (~27.5/tick); supplier climbs 98 -> 362 at ~11.3/tick and stays at
  361.6 +- 1.2 in both runs. We treat 362 as a hard stock cap.
- Pulse (order 80, production 1.5, receiving 0.35, maintenance 0, lead 0.2): supplier empties
  in 2 ticks, shipments 19-21/tick from tick 3 (everything produced ships), supplier stays at 0
  until tick 30, then refills at 40-50/tick and sits at the cap from tick 40. Production
  therefore jumps from ~19 to ~70 about 30 ticks after orders start: adaptive commitment.
  At tick 101 shipments halve (18.4 -> 9.2) with no control change: 100 ticks without
  maintenance. Retail rises 5 -> 134 (ticks 3-42, drain ~17/tick) then falls to 0 by tick 70
  (drain ~24/tick).
- Back at recovery (tick 200): arrivals keep coming for 60 ticks (5 -> 27 -> 55 -> 15 -> 12 -> 0,
  ~1,000 units total) with orders 0. A conveyor backlog built during the pulse at
  dispatch - arrivals ~ 6-8/tick over 160 ticks, released slowly under lead_time 1.0.

## Equations (7 states, 12 parameters, RK4, N_SUB = 2)

States $S$ supplier stock, $C_1, C_2$ commitment cascade, $Q_1, Q_2$ conveyor, $R$ retail,
$W$ machine heat. Controls $o$ order, $\ell$ lead_time_buy, $e$ production_effort,
$r$ receiving_effort, $m$ maintenance (product_mix unused). $\min_\epsilon$ is the smooth min
$\tfrac12(a+b-\sqrt{(a-b)^2+\epsilon^2})$ with $\epsilon = 1$.

$$P = e\,(p_0 + C_2),\qquad D = \min_\epsilon(\min_\epsilon(o, S),\ d_{cap})$$
$$k_q = \frac{k_{q0}}{1 + k_\ell \ell},\qquad g = 1 - \frac{w_{loss}}{1 + e^{-(W-1)/0.05}},\qquad
A = \min_\epsilon\big(k_q Q_2,\ g\,a_0\big),\qquad \text{sale} = \min_\epsilon(dem + k_d R,\ R)$$
$$\dot S = P - D \ (S \le 362),\quad \dot C_1 = \frac{k_c o - C_1}{\tau_c},\quad
\dot C_2 = \frac{C_1 - C_2}{\tau_c},\quad \dot Q_1 = D - k_q Q_1,\quad \dot Q_2 = k_q Q_1 - A$$
$$\dot R = A - \text{sale}\ (R \le 362),\qquad \dot W = \frac{1-m}{\tau_{heat}} - k_{cool}\, m\, W \ (W \le 1.5)$$

Observation: shipments $= A$, inventory_supplier $= S$, inventory_retail $= R$.
Receiving effort does not enter: the terminal cap is 19.5 at receiving 0.35 and 20-27 at 1.5,
so its slope was unidentified (3.7 per unit in v3) and we dropped it. The retail cap 362 is
a backstop equal to the observed supplier cap, so long holds saturate; sales grow with stock
($k_d$), which is what the data show (drain 17/tick below 90 units, 24-30/tick at 130-210).
Initial state uses every $y_0$ entry: $S = y_{0,1}$ (clipped to 362), $R = y_{0,2}$,
$Q_1 = y_{0,0}$ (initial in-transit content), $C_1 = C_2 = Q_2 = W = 0$.

## Fitted parameters (full fit, both runs, cost 125.9; tau_heat held at 96.2)

| name | value | bounds | role |
|---|---:|---|---|
| p0 | 7.80 | 1-100 (log) | base production per unit effort |
| kc | 3.47 | 0-5 | commitment gain on order_quantity |
| tau_c | 83.5 | 2-100 (log) | commitment stage time constant |
| dcap | 19.96 | 5-300 (log) | dispatch (forward transport) cap |
| kq0 | 2.58 | 0.1-3 (log) | conveyor rate at lead_time 0 |
| kl | 45.9 | 0-60 | conveyor slow-down per unit lead_time |
| a0 | 32.1 | 1-100 (log) | terminal throughput cap (unworn) |
| dem | 19.5 | 1-100 (log) | retail demand per tick at zero stock |
| kd | 0.0253 | 0.02-0.5 (log) | extra retail sales per unit stock |
| tau_heat | 96.2 (held) | 20-1000 (log) | ticks to wear threshold without maintenance |
| wloss | 0.734 | 0-0.9 | throughput lost past the threshold |
| kcool | 0.0355 | 0.01-1 (log) | cooling rate under maintenance |

No parameter at a bound. tau_heat is pinned by the data (shipments halve 101 ticks into
maintenance 0, and $W$ reaches the threshold 1 at $t = \tau_{heat}$); left free, the fitter
moves it to 45 to fake the retail fall at tick 43 with the wear gate and loses the shipments
halving (v4/v5 below).

## Scores (per observable: shipments, supplier, retail)

| fold | ode | l0b_lin | persistence |
|---|---|---|---|
| LOO hold out hold_rec | 0.979 0.888 0.984 (0.950) | 0.774 0.416 0.787 (0.659) | 0.292 |
| LOO hold out pulse200_200 | 0.611 0.886 0.804 (0.767) | 0.607 0.885 0.828 (0.774) | 0.331 |
| in-sample hold_rec | 0.979 0.933 0.985 (0.966) | | |
| in-sample pulse200_200 | 0.872 0.923 0.812 (0.869) | | |

LOO mean 0.859 vs l0b_lin 0.716; in-sample 0.917. Lab verdict: ship.
The second fold is fitted on hold_rec alone, where orders are zero, so dcap, kq0, kl, a0, a1,
kc, tau_c and the wear terms are unidentified; that fold measures our starting values, not the
fit, and it lands 0.007 below l0b_lin.

Eval sanity (4,000-tick rollouts, 0.3 s each, all finite): sustained max [20, 362, 95],
order max [32, 362, 362], recovery max [20, 362, 83], composition max [32, 362, 303]. Retail
reaches its cap 362 only under long order holds; observed retail max is 226 with range 226, so
every excursion stays within one range of the data (the gate we use for shipping).

## Versions tried

1. v1 (11 params, 6 states, no wear, m_loss on production, q0 * shipments as Q1): LOO
   0.914 / 0.785, in-sample 0.891. q0 hit 1.0, kl hit 10, m_loss went to 0. One arrival cap
   cannot be 19 (ticks 3-100) and 9.5 (ticks 101-200), so the fit settled at 14.7 and lost
   the shipments halving and the retail hump.
2. v2: heat/wear gate on the terminal, m_loss dropped, Q1 = shipments fixed, kl bound 30:
   LOO 0.938 / 0.767, in-sample 0.918. Shipments in-sample on the pulse run 0.75 -> 0.885.
3. v3: a1 bound 100 -> 6 to limit arrivals at receiving 1.5. Scores unchanged (a0 rose
   instead). Eval order-category retail ran to 6,570 (28x the range): arrivals a0 + 1.5 a1 = 32
   above a fixed demand of 20 for thousands of ticks. Rejected on the excursion gate.
4. v4: sales dem + kd R, retail hard cap 362, a1 dropped (12 params). Gate passes (retail
   max 351) but the fitter moved tau_heat to 45 and used the wear gate for the retail fall:
   pulse shipments 0.793, in-sample 0.910, kc and kl at bounds.
5. v5: kc bound 5, kl bound 60, warm start from v3. Same optimum as v4 (0.910).
6. v6 (kept): as v5 with tau_heat held at 96.2. In-sample 0.917, pulse shipments 0.872,
   retail 0.812, LOO 0.950 / 0.767, no bound hit, gate passes.

Rejected: production maintenance loss (fit 0); explicit dispatch buffer with return-space
overflow (would need withdrawals of 80/tick, contradicted by the ~1,000-unit backlog);
two-class retail demand (explains the 27.5/tick initial drain and the tick-3..70 retail hump,
but needs 2 more states than the 7-state budget).

## What still limits it

- The retail hump (rise to 134 then fall) needs class-dependent demand; with one demand rate
  the model holds retail flat at ~15 during the pulse (retail 0.81 on that run).
- The first two pulse ticks withdraw 75 then 27 units (class shares); the dispatch cap of 20
  smooths that. Cheap in score (sigma 92).
- The 30-tick production jump is a step in the data, a two-stage cascade in the model.
- Wear is a threshold identified from one event; tau_heat and kcool rest on one drop and one
  recovery.
