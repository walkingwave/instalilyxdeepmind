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
A = \min_\epsilon\big(k_q Q_2,\ g\,(a_0 + a_1 r)\big),\qquad \text{sale} = \min_\epsilon(dem, R)$$
$$\dot S = P - D \ (S \le 362),\quad \dot C_1 = \frac{k_c o - C_1}{\tau_c},\quad
\dot C_2 = \frac{C_1 - C_2}{\tau_c},\quad \dot Q_1 = D - k_q Q_1,\quad \dot Q_2 = k_q Q_1 - A$$
$$\dot R = A - \text{sale},\qquad \dot W = \frac{1-m}{\tau_{heat}} - k_{cool}\, m\, W \ (W \le 1.5)$$

Observation: shipments $= A$, inventory_supplier $= S$, inventory_retail $= R$.
Initial state uses every $y_0$ entry: $S = y_{0,1}$ (clipped to 362), $R = y_{0,2}$,
$Q_1 = y_{0,0}$ (initial in-transit content), $C_1 = C_2 = Q_2 = W = 0$.

## Fitted parameters (full fit, both runs, cost 125.9)

| name | value | bounds | role |
|---|---:|---|---|
| p0 | 7.64 | 1-100 (log) | base production per unit effort |
| kc | 0.474 | 0-2 | commitment gain on order_quantity |
| tau_c | 24.2 | 2-100 (log) | commitment stage time constant |
| dcap | 19.8 | 5-300 (log) | dispatch (forward transport) cap |
| kq0 | 0.907 | 0.1-3 (log) | conveyor rate at lead_time 0 |
| kl | 17.3 | 0-30 | conveyor slow-down per unit lead_time |
| a0 | 26.0 | 1-100 (log) | terminal throughput intercept |
| a1 | 3.71 | 0-6 | terminal throughput per unit receiving_effort |
| dem | 20.1 | 1-100 (log) | retail demand per tick |
| tau_heat | 96.2 | 20-1000 (log) | ticks to wear threshold without maintenance |
| wloss | 0.650 | 0-0.9 | throughput lost past the threshold |
| kcool | 0.0331 | 0.01-1 (log) | cooling rate under maintenance |

No parameter at a bound.

## Scores (per observable: shipments, supplier, retail)

| fold | ode | l0b_lin | persistence |
|---|---|---|---|
| LOO hold out hold_rec | 0.979 0.857 0.981 (0.939) | 0.774 0.416 0.787 (0.659) | 0.292 |
| LOO hold out pulse200_200 | 0.613 0.886 0.803 (0.767) | 0.607 0.885 0.828 (0.774) | 0.331 |
| in-sample hold_rec | 0.979 0.929 0.984 (0.964) | | |
| in-sample pulse200_200 | 0.883 0.918 0.811 (0.871) | | |

LOO mean 0.853 vs l0b_lin 0.716; in-sample 0.917. Lab verdict: ship.
The second fold is fitted on hold_rec alone, where orders are zero, so dcap, kq0, kl, a0, a1,
kc, tau_c and the wear terms are unidentified; that fold measures our starting values, not the
fit, and it lands 0.007 below l0b_lin.

Eval sanity (4,000-tick rollouts, 0.3 s each, all finite): sustained/recovery stay inside the
observed range; composition retail max 391; order retail max 6,570 (9% of ticks outside the
range) because $a_0 + 1.5 a_1 = 31.6$ arrivals exceed $dem$ under long high-receiving holds
with orders on. The packaged runtime clip (3x observed range) bounds that; the true demand at
high retail stock is unobserved.

## Versions tried

1. v1 (11 params, 6 states, no wear, m_loss on production, q0 * shipments as Q1): LOO
   0.914 / 0.785, in-sample 0.891. q0 hit 1.0, kl hit 10, m_loss went to 0. One arrival cap
   cannot be 19 (ticks 3-100) and 9.5 (ticks 101-200), so the fit settled at 14.7 and lost
   the shipments halving and the retail hump.
2. v2: heat/wear gate on the terminal, m_loss dropped, Q1 = shipments fixed, kl bound 30:
   LOO 0.938 / 0.767, in-sample 0.918. Shipments in-sample on the pulse run 0.75 -> 0.885.
3. v3 (kept): a1 bound 100 -> 6 to limit arrivals at receiving 1.5. Scores unchanged
   (a0 rose instead); kept as the final for the tighter bound.

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
