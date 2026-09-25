# hospital_queue: minimal grey-box ODE (`gtlab/ode/hospital_queue_min.py`)

Lab notebook, Fri Sep 25. Data: 4 runs, 440 ticks (hold_rec 120, pulse40 80, mid40 40,
multilevel200 200). `sigma_proxy` = [73.2, 127.0, 5.48] for [wait_time, queue, discharges].
Everything below is free (no simulator credits). Harness: `scripts/ode_lab.py --system hospital_queue
--family hospital_queue_min --budget 240 --starts 10 --nfev 60` (4 LOO folds + full fit).

## What the data say (before any fit)

- Arrivals per tick = base + elective: queue jumps from y0 by +10 (elective 0), +20 (elective 10),
  +31 (elective 20) on the very first tick. Base $\lambda_0 \approx 10$–$11.5$.
- Queue is capped at 333 (overflow referred). Under recovery (staffing 20) it drains to 23 and
  wait goes to 0; discharges settle at 11.5 = arrivals after a burst (0 for 5 ticks, then ~26).
- Under the pulse (staffing 5, elective 20) the queue climbs 30/tick until ~230, then only ~10/tick
  while discharges are ~2/tick: about 20 patients/tick disappear without discharge (leave or referral).
- wait_time behaves like a first-order filter (rate ~0.05–0.1/tick) on a target that grows with the
  waiting count and shrinks with staffing. Its long slow decline after the pulse (73 -> 44 in 40 ticks
  at staffing 20) points at a lag in staff effectiveness (handover) and/or a long filter.

## Structure v1 (kept as the reference fit, LOO 0.770)

States $W$ (waiting), $A$ (assessment), $T$ (treatment), $w$ (reported wait), $F$ (fatigue), $R$
(return pool). $q = W + A + T$.

$$\text{eff} = S\,(1 + g_{ot} O)(1 - k_{fat} F),\quad r_a = c_a\,\text{eff}\,D,\quad
r_t = \frac{c_t\,\text{eff}\,(1-D)}{1 + k_{urg} U}$$
$$\text{smin}(r, s) = \frac{r s}{(r^4 + s^4)^{1/4}},\quad f_1 = \text{smin}(r_a, W),\quad
f_2 = A/\tau_a,\quad f_3 = \text{smin}(r_t, T)$$
$$\dot W = (\lambda_0 + E + R/20)\,\sigma\!\left(\tfrac{q_{cap}-q}{8}\right) - f_1 - k_l W \tfrac{w}{w_{sat}+w},\quad
\dot A = f_1 - f_2,\quad \dot T = f_2 - f_3$$
$$\dot w = (W/r_a - w)/\tau_w,\quad \dot F = (O - F)/30,\quad \dot R = r_{ret}(1-F_u) f_3 - R/20$$
Observables: wait $= w$, queue $= q$, discharges $= f_3$. Reset: $W = q_0$, $w = w_0$, $A = T = F = 0$,
$R = 20\, r_{ret}\, d_0$.

Fitted (full data, cost 184.9):

| param | value | note |
|---|---|---|
| lam0 | 10.92 | base arrivals/tick |
| c_a | 1.820 | assessment work per staff |
| c_t | 1.729 | treatment work per staff |
| tau_a | 1.692 | assessment duration |
| ot_gain | 0.0 | AT BOUND (lo) |
| k_fat | 0.9 | AT BOUND (hi) |
| r_ret | 0.0 | AT BOUND (lo) |
| k_l | 0.001 | AT BOUND (lo): no leaving needed |
| w_sat | 300 | AT BOUND (hi) |
| q_cap | 305.9 | referral gate centre |
| tau_w | 32.8 | wait filter |
| k_urg | 0.0 | AT BOUND (lo) |

| fold (held out) | ode [wait, queue, disch] | mean | l0b_lin | mean | persistence |
|---|---|---|---|---|---|
| hold_rec | 0.985 0.705 0.651 | 0.780 | 0.853 0.372 0.856 | 0.694 | 0.771 |
| pulse40 | 0.759 0.836 0.663 | 0.753 | 0.732 0.640 0.663 | 0.678 | 0.488 |
| mid40 | 0.966 0.878 0.599 | 0.814 | 0.889 0.546 0.511 | 0.649 | 0.599 |
| multilevel200 | 0.576 0.917 0.708 | 0.734 | 0.514 0.692 0.589 | 0.598 | 0.430 |
| **LOO mean** | | **0.770** | | 0.655 | 0.572 |

In-sample: hold_rec 0.903, pulse40 0.804, mid40 0.816, multilevel200 0.806 (mean 0.832).
Eval sanity (4,000 ticks, all four categories): finite, <0.5 s each; wait_time reaches 915 in the
"order" schedule (7% of ticks outside the observed range), queue and discharges stay in range.
Verdict from the lab: ship (LOO +0.115 over l0b_lin, above it on every fold).

What v1 got wrong: the fit switched off leaving ($k_l$ at floor) and overtime, and pushed fatigue to
its cap; overtime and fatigue are then unidentifiable (only the pulse and the multilevel run carry
overtime). The pulse-phase queue plateau came only from the referral gate. The queue at
$q=333$ empties $W$ into $T$ (no bed limit), so wait_time collapses at full queue where the data show
wait 200–320 at staffing 1.

## Structure v3 (finite chairs and beds, handover lag)

States $W$, $A_1, A_2$ (two assessment stages, finite chairs), $T$ (finite beds), $w$, $F$, $S_e$
(oriented staff). $q = W + A_1 + A_2 + T$, $K = 3$ (max drain rate per tick).

$$\text{eff} = S_e (1 + g_{ot} O)(1 - k_{fat} F),\quad r_a = c_a\,\text{eff}\,\frac{D}{D + 0.05},\quad
r_t = c_t\,\text{eff}\,(1 - D)$$
$$f_1 = \min(r_a,\, K W,\, K(\text{chairs} - A_1 - A_2)),\quad f_{2a} = \tfrac{2}{\tau_a} A_1,\quad
f_{2b} = \min(\tfrac{2}{\tau_a} A_2,\, K(\text{beds} - T)),\quad f_3 = \min(r_t, K T)$$
$$\dot W = (\lambda_0 + E)\,\sigma\!\left(\tfrac{330 - q}{6}\right) - f_1 - k_l W \tfrac{w}{w_{sat}+w},\quad
\dot A_1 = f_1 - f_{2a},\quad \dot A_2 = f_{2a} - f_{2b},\quad \dot T = f_{2b} - f_3$$
$$\dot w = \left(\tfrac{W}{f_1 + 1} - w\right)/\tau_w,\quad \dot F = (O - F)/30,\quad \dot S_e = (S - S_e)/\tau_h$$
Reset: $W = q_0$, $w = w_0$, $A_1 = A_2 = T = F = 0$, $S_e = d_0 / (0.6\, c_t)$ clipped to $[1, 20]$.

Fitted (full data, cost 150.2):

| param | value | note |
|---|---|---|
| lam0 | 11.26 | |
| c_a | 0.751 | |
| c_t | 5.861 | |
| tau_a | 1.293 | |
| chairs | 61.9 | |
| beds | 19.0 | |
| ot_gain | 0.344 | |
| k_fat | 0.846 | |
| k_l | 0.0133 | |
| w_sat | 1.0 | AT BOUND (lo): leaving is simply $k_l W$ |
| tau_w | 10.80 | |
| tau_h | 1.0 | AT BOUND (lo): no handover lag wanted |

| fold (held out) | ode [wait, queue, disch] | mean | l0b_lin mean | persistence |
|---|---|---|---|---|
| hold_rec | 0.883 0.410 0.454 | 0.582 | 0.694 | 0.771 |
| pulse40 | 0.779 0.887 0.668 | 0.778 | 0.678 | 0.488 |
| mid40 | 0.909 0.861 0.595 | 0.789 | 0.649 | 0.599 |
| multilevel200 | 0.786 0.907 0.728 | 0.807 | 0.598 | 0.430 |
| **LOO mean** | | **0.739** | 0.655 | 0.572 |

In-sample: hold_rec 0.942, pulse40 0.832, mid40 0.796, multilevel200 0.828 (mean 0.849).
Eval sanity: finite, 0.4 s per 4,000-tick category, 0% of ticks outside the observed range;
wait_time stays below 193 on every schedule. Lab verdict: ship.

Reading: the bed limit fixes the full-queue wait (multilevel fold 0.786 vs 0.576 in v1) and the
in-sample fit, but when hold_rec is held out the three remaining runs (all near the 333 cap) do not
pin down the drain from a half-full queue, and the fold falls below persistence. Handover ($\tau_h$)
and the wait dependence of leaving ($w_{sat}$) both went to their floors.

## Structure v4 (v3 minus handover, linear leaving, free $q_{cap}$ and $d_h$)

This is the version kept in `gtlab/ode/hospital_queue_min.py` and in
`plans/hospital_queue_hospital_queue_min.json` / `_doc.json`. Six states $W, A_1, A_2, T, w, F$,
twelve parameters, no mechanism switches (letters accepted, everything active). Same equations as
v3 with $S_e \to S$ (staffing used directly), leaving $= k_l W$, and $q_{cap}$, $d_h$ fitted:

$$\text{eff} = S (1 + g_{ot} O)(1 - k_{fat} F),\quad r_a = c_a\,\text{eff}\,\frac{D}{D + d_h},\quad
r_t = c_t\,\text{eff}\,(1 - D)$$
$$f_1 = \min(r_a,\, 3W,\, 3(\text{chairs} - A_1 - A_2)),\quad f_{2a} = \tfrac{2}{\tau_a} A_1,\quad
f_{2b} = \min(\tfrac{2}{\tau_a} A_2,\, 3(\text{beds} - T)),\quad f_3 = \min(r_t, 3T)$$
$$\dot W = (\lambda_0 + E)\,\sigma\!\left(\tfrac{q_{cap} - q}{6}\right) - f_1 - k_l W,\quad
\dot A_1 = f_1 - f_{2a},\quad \dot A_2 = f_{2a} - f_{2b},\quad \dot T = f_{2b} - f_3$$
$$\dot w = \left(\tfrac{W}{f_1 + 1} - w\right)/\tau_w,\quad \dot F = (O - F)/30$$
Observables: wait $= w$, queue $= q = W + A_1 + A_2 + T$, discharges $= f_3$.
Reset: $W = q_0$, $w = w_0$, $A_1 = A_2 = T = F = 0$ (services start empty; the reported $d_0$ is not
used, discharges are 0 for the first ticks of every run). RK4, 2 substeps per tick; every rate is at
most 3 per tick so the integrator is stable for any control in the brief's bounds. Urgent priority
and follow-up capacity are read but have no effect (see rejected list).

Fitted (full data, cost 144.1, 93 s), nothing at a bound:

| param | value | meaning |
|---|---|---|
| lam0 | 11.32 | base arrivals per tick |
| c_a | 0.710 | assessment throughput per effective staff at full diagnostic share |
| c_t | 1.892 | treatment throughput per effective staff at $D = 0$ |
| tau_a | 1.769 | assessment duration (two stages of $\tau_a/2$) |
| chairs | 26.9 | assessment capacity |
| beds | 12.7 | treatment capacity |
| ot_gain | 0.447 | work gain at overtime 1 |
| k_fat | 0.878 | work loss at full fatigue |
| k_l | 0.0120 | leaving rate of waiting patients per tick |
| tau_w | 10.63 | wait_time filter |
| q_cap | 322.4 | referral gate centre (width 6) |
| d_h | 0.0407 | diagnostic share half-saturation |

| fold (held out) | ode [wait, queue, disch] | mean | l0b_lin [wait, queue, disch] | mean | persistence |
|---|---|---|---|---|---|
| hold_rec | 0.970 0.825 0.739 | 0.844 | 0.853 0.372 0.856 | 0.694 | 0.771 |
| pulse40 | 0.845 0.941 0.722 | 0.836 | 0.732 0.640 0.663 | 0.678 | 0.488 |
| mid40 | 0.918 0.870 0.597 | 0.795 | 0.889 0.546 0.511 | 0.649 | 0.599 |
| multilevel200 | 0.794 0.961 0.740 | 0.832 | 0.514 0.692 0.589 | 0.598 | 0.430 |
| **LOO mean** | | **0.827** | | 0.655 | 0.572 |

| run | in-sample [wait, queue, disch] | mean |
|---|---|---|
| hold_rec | 0.989 0.935 0.880 | 0.935 |
| pulse40 | 0.859 0.946 0.726 | 0.844 |
| mid40 | 0.933 0.881 0.597 | 0.804 |
| multilevel200 | 0.826 0.937 0.754 | 0.839 |
| **mean** | | **0.855** |

Eval sanity (4,000 ticks from the hold_rec $y_0$, seed 0): all four categories finite, 0.3 s each,
0% of ticks outside the observed range $\pm 5\%$; wait_time max 184 / 171 / 72 / 208 (sustained /
order / recovery / composition), queue 27–334, discharges 0.3–18.

Comparison of the three fitted structures (LOO mean, in-sample mean, at-bound count):
v1 0.770 / 0.832 / 6, v3 0.739 / 0.849 / 2, v4 0.827 / 0.855 / 0.

## Rejected on the way

- v2 (never run through the lab): v1 with a two-stage assessment chain, hard $\min(r, 2W)$ flows and
  an oriented-staff state, but still no chair/bed limits. With the queue at 333 all of $W$ drains into
  treatment and the modelled wait falls to ~5 while the data sit at 220–320, so it was replaced by v3
  before fitting.
- Returns after discharge scaled by $(1 - F_u)$: $r_{ret}$ fitted to 0 in v1; follow-up is perfectly
  confounded with staffing and elective in these four runs (always 1 in recovery, 0 in the pulse),
  so it was dropped for the chair/bed parameters.
- Urgent priority as extra treatment work ($k_{urg}$): fitted to 0 in v1, dropped.
- Leaving proportional to $W \cdot w/(w_{sat}+w)$: fitted to zero in v1 (the referral gate alone
  explains the plateau); kept in v3 with the same bounds to let the data decide again.

- Handover lag on staffing ($S_e$, $\tau_h$, v3): fitted to the 1-tick floor and made the hold_rec
  fold worse than persistence (0.582). Removing it (v4) raised that fold to 0.844. The slow wait
  decline after the pulse is explained by the bed limit plus the 10-tick wait filter instead.
- Wait-dependent leaving $k_l W w/(w_{sat}+w)$: $w_{sat}$ went to its floor in v3, so leaving is
  linear in $W$ in v4.
- Fixed $q_{cap} = 330$ and $d_h = 0.05$ (v3): freed in v4, they moved to 322 and 0.041.

## Verdict

Ship as a candidate for hospital_queue: LOO 0.827 vs 0.655 for l0b_lin fitted on the same folds,
above l0b_lin and persistence on every fold and every observable except discharges on hold_rec
(0.739 vs 0.856 for l0b_lin: the model smooths the 26/tick burst at $t = 5$ into a ramp).
In-sample 0.855. Nothing at a bound, every 4,000-tick schedule stays inside the observed range.

What limits it:
- Discharges are batchy (0 then 20 then 5 in consecutive ticks) and the model gives a smooth mean;
  in-sample discharge scores stay at 0.60–0.88 whatever the structure.
- Overtime, fatigue and urgent priority are only exercised together in the pulse and the multilevel
  run; $g_{ot}$ and $k_{fat}$ are a compromise, not a measurement.
- Follow-up capacity and returns are perfectly confounded with staffing and elective in these runs;
  the model ignores follow-up, so a test that adds follow-up after a discharge burst is unmodelled.
- The referral gate and the leaving rate share the job of holding the queue at 322–333; only the
  sustained-pulse behaviour is checked, not a long hold at a mid level.
- Scores use `sigma_proxy` (73 / 127 / 5.5); the organizer's sigma for wait_time is probably much
  smaller, so the wait scores above are optimistic.

Next checks before it goes into an upload: run it through `scripts/screen.py` next to the l0b_lin
and l1 candidates, and try it as a blend member behind the current pick.
