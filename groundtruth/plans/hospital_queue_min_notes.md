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

## Sat Sep 26: the compose run breaks the wait model (v5, stranded cohort)

Data now: 5 runs, 770 ticks (the four above plus `p3.compose`, 330 ticks). `sigma_proxy` moved to
[100.0, 120.1, 5.27]. Harness: `scripts/ode_lab.py --system hospital_queue --family hospital_queue_min
--budget 300 --starts 10 --nfev 60 --tag v2` (5 LOO folds + full fit, 18 min). The old v4 structure was
refitted on the same five runs for the comparison (`plans/hospital_queue_hospital_queue_min_p3b.json`; the
committed `_p3.json` is the same full fit, identical theta, LOO 0.822 from another multistart draw).

### What the compose run showed

Recovery baseline (staffing 20, elective 0, diag 0.4, urgent 0.6, overtime 0, followup 1); each control
alone at 85% of its pulse level for 30 ticks, 15 ticks of recovery in between, then the joint pulse.

| block | ticks | queue | wait | discharges | reading |
|---|---|---|---|---|---|
| staffing 7.25 | 0-30 | 49 -> 193 | 3 -> 11 | 0 -> ~10 (batchy) | arrivals ~11.5 minus a slow drain |
| recovery | 30-45 | 190 -> 154 | 11 -> 16 | ~12 | wait keeps rising while the queue falls: lag |
| elective 17 | 45-75 | 170 -> 261, plateau | 17 -> 29 | ~10 | plateau at 260, far below the 322-333 cap |
| diagnostic 0.7 | 90-120 | 250 -> 306 | 34 -> 41 | 6-7 | assessment share up, treatment down |
| urgent 0.94 | 135-165 | 283 -> 199 | 41 -> 33 | 11-12 | the queue drains 1.5/tick faster with the SAME discharges |
| overtime 0.85 | 180-210 | 161 -> 23 | 26 -> 20, then 32, 56, 78, ... 254 | 20-28, then 11.5 | wait explodes once the queue nears empty |
| recovery, followup 0.15, recovery | 210-270 | 23 | 259 -> 396 | 11.5 | wait keeps climbing ~2-3/tick with nothing moved |
| joint pulse | 270-300 | 51 -> 319 | 370 -> 60 | 0-8 | the refill resets the estimate, time constant ~8 |
| recovery | 300-330 | 320 -> 289 | 59 -> 46 | ~9 | ordinary Little behaviour again |

Reading the wait as a first-order filter $\dot w = (w^* - w)/\tau_w$ with $\tau_w \approx 10$ gives the target
$w^* = w + \tau_w \dot w$: about 20 up to $t = 187$, then $\approx 270$ at $t = 188$ (one tick), then
$+2$ to $+3$ per tick for 80 ticks (300 at $t = 210$, 330 at 225, 380 at 250, 410 at 265), and back to
$\approx 60$ within two ticks of the refill. So the reported wait is NOT a function of the current queue:
after the drain the estimate points at something that has been ageing for a long time (270 ticks at 2/tick
puts its origin near $t = 50$, the start of the long high-queue period), and it is only hidden while the
ordinary waiting pool is large. The plateau at 260 under elective 17 and the faster drain under urgent
priority (at unchanged discharges) are two more things v4 could not do: it has one referral gate at 322 and
no urgent term.

### Structure v5 (cohort model; ends up as the alternate `gtlab/ode/hospital_queue_min2.py`, see verdict)

Seven states $W, A_1, A_2, T, w, C, a$: waiting pool, two assessment stages (chairs), treatment (beds),
reported wait, stranded cohort size and its mean age. $q = W + A_1 + A_2 + T$, $K = 3$. Fatigue and
$k_{fat}$ are dropped (no throughput loss is visible after the overtime block: discharges 11.5 = arrivals).

$$\text{eff} = S(1 + g_{ot} O),\quad r_a = c_a\,\text{eff}\,\frac{D}{D + 0.04},\quad r_t = c_t\,\text{eff}\,(1 - D)$$
$$f_1 = \min(r_a, 3W, 3(\text{chairs} - A_1 - A_2)),\quad f_{2a} = \tfrac{2}{1.77}A_1,\quad
f_{2b} = \min(\tfrac{2}{1.77}A_2, 3(\text{beds} - T)),\quad f_3 = \min(r_t, 3T)$$
$$\dot W = \lambda_0\,\sigma\!\left(\tfrac{325 - q}{6}\right) + E\,\sigma\!\left(\tfrac{q_e - q}{20}\right)
- f_1 - k_l U W - s,\qquad s = k_s W \frac{w}{w + 30}$$
$$\dot A_1 = f_1 - f_{2a},\quad \dot A_2 = f_{2a} - f_{2b},\quad \dot T = f_{2b} - f_3$$
$$\dot C = s - C/\tau_z,\quad \dot a = g_z - a\,\frac{s}{C + 1} - a/\tau_z,\quad \tau_z = 150$$
$$w^* = \frac{W}{f_1 + 1} + a\,\frac{C}{C + 5}\,\sigma\!\left(\tfrac{w_{th} - W}{5}\right),\quad
\dot w = (w^* - w)/\tau_w$$
Observables: wait $= w$, queue $= q$, discharges $= f_3$. Reset: $W = q_0$, $w = w_0$, everything else 0.
Elective cases are cancelled by their own gate at $q_e$ (the 260 plateau); urgent priority pushes routine
cases out of the pending list ($k_l U W$, so the drain speeds up with $U$ at unchanged discharges);
waiting patients deteriorate into the cohort at a rate that needs a long reported wait; the cohort ages at
$g_z$ per tick, is diluted by newcomers and cleared over $\tau_z$ (so $a \le g_z \tau_z$); the estimate
shows the cohort age only when the ordinary pool is nearly empty ($W < w_{th}$), and a refill hides it
again through the same $\tau_w$ filter that the data show. Twelve parameters; $\tau_a = 1.77$, $d_h = 0.04$,
$q_{cap} = 325$ are frozen at their v4 values.

### First fit of v5 (tag v2: $\tau_z = 600$, $s = k_s W$, $w_{th} \le 80$)

Full fit cost 262.0 (old structure on the same runs: 438.6), nothing at a bound:
lam0 10.37, c_a 0.816, c_t 0.926, chairs 30.5, beds 12.2, ot_gain 0.378, k_l 0.0121, tau_w 6.36,
q_e 231.9, k_s 1.9e-4, g_z 4.98, w_th 75.1.

| fold (held out) | v4 refit [wait, queue, disch] | mean | v5/v2 [wait, queue, disch] | mean | l0b_lin | persistence |
|---|---|---|---|---|---|---|
| hold_rec | 0.989 0.919 0.768 | 0.892 | 0.879 0.818 0.798 | 0.832 | 0.710 | 0.768 |
| pulse40 | 0.790 0.932 0.739 | 0.820 | 0.771 0.853 0.708 | 0.777 | 0.733 | 0.503 |
| mid40 | 0.981 0.888 0.587 | 0.819 | 0.951 0.936 0.599 | 0.829 | 0.614 | 0.603 |
| multilevel200 | 0.759 0.942 0.764 | 0.822 | 0.710 0.896 0.734 | 0.780 | 0.549 | 0.444 |
| compose | 0.689 0.732 0.695 | 0.705 | 0.706 0.772 0.726 | 0.735 | 0.607 | 0.615 |
| **LOO mean** | | **0.812** | | **0.790** | 0.643 | 0.587 |

| run | v4 refit in-sample | mean | v5/v2 in-sample | mean |
|---|---|---|---|---|
| hold_rec | 0.989 0.916 0.850 | 0.918 | 0.904 0.910 0.824 | 0.879 |
| pulse40 | 0.849 0.897 0.735 | 0.827 | 0.787 0.865 0.712 | 0.788 |
| mid40 | 0.980 0.881 0.588 | 0.816 | 0.959 0.953 0.597 | 0.836 |
| multilevel200 | 0.824 0.952 0.762 | 0.846 | 0.768 0.950 0.765 | 0.828 |
| compose | 0.738 0.873 0.746 | 0.786 | 0.896 0.914 0.738 | 0.849 |
| **mean** | | **0.839** | | **0.836** |

Eval sanity (4,000 ticks): v4 refit 0% outside on all four categories, wait max 186; v5/v2 sustained and
order 0%, but recovery 17% and composition 18% outside with wait reaching 1,095 and 1,466.

Reading: the cohort does what it was built for (compose wait 0.896 vs 0.738 in-sample, compose fold
+0.03) but it leaks into the other runs: with $s = k_s W$ the age $a$ grows from reset at 5/tick even when
the cohort is a fraction of a patient, and $C/(C+5)$ is linear in $C$, so hold_rec picks up ~10 ticks of
phantom wait; and $w_{th} = 75$ lets the cohort show whenever the pool is merely low. On long eval
schedules the cohort formed during a pulse is exposed by every later drain and the age is unbounded
($g_z \tau_z = 3{,}000$).

### Second fit (tag v3: $\tau_z = 150$, deterioration needs a long wait $s = k_s W\,w/(w+30)$, $w_{th} \le 40$)

Full fit cost 266.1, nothing at a bound: lam0 10.40, c_a 0.829, c_t 0.924, chairs 33.4, beds 11.2,
ot_gain 0.507, k_l 0.0136, tau_w 7.97, q_e 258.8, k_s 1.9e-3, g_z 5.83, w_th 39.6.

| fold (held out) | v4 refit mean | v5/v3 [wait, queue, disch] | mean |
|---|---|---|---|
| hold_rec | 0.892 | 0.803 0.808 0.797 | 0.803 |
| pulse40 | 0.820 | 0.797 0.835 0.696 | 0.776 |
| mid40 | 0.819 | 0.979 0.957 0.588 | 0.841 |
| multilevel200 | 0.822 | 0.714 0.897 0.732 | 0.781 |
| compose | 0.705 | 0.704 0.770 0.724 | 0.733 |
| **LOO mean** | **0.812** | | **0.787** |

In-sample: hold_rec 0.882, pulse40 0.789, mid40 0.834, multilevel200 0.823, compose 0.844 (wait 0.894);
mean 0.834 vs 0.839. Eval sanity: sustained / order / recovery 0% outside, composition 7%, wait max
165 / 86 / 435 / 538 (the eval schedules do drain the queue after overtime, so the cohort shows there by
design; the bound $g_z \tau_z = 875$ now holds it).

Reading: the elective gate ($q_e = 259$, plateau reproduced) and the urgent drain both fit; the cohort
reproduces the compose explosion; but hold_rec (held out) still loses 0.09: a cohort of 0.1 patient
still carries $0.1/5.1 = 2\%$ of an age that has grown to ~500, i.e. 10 ticks of phantom wait at queue 23,
and the fold also gives up some queue accuracy because $\tau_a$, $d_h$ and $q_{cap}$ are frozen (the v4
refit moved them to 0.80, 0.087 and 316.6 on these five runs).

### Third fit (tag v4: cohort weight $C^2/(C^2 + 25)$ instead of $C/(C + 5)$)

A cohort of a fraction of a patient now carries nothing, so the age state cannot leak into runs that
never strand anyone. Full fit cost 263.9; $w_{th}$ at its upper bound (40):
lam0 10.51, c_a 0.834, c_t 0.930, chairs 33.2, beds 11.2, ot_gain 0.486, k_l 0.0131, tau_w 7.67,
q_e 242.0, k_s 2.4e-3, g_z 4.71, w_th 40.0.

| fold (held out) | v4 refit [wait, queue, disch] | mean | cohort v4 [wait, queue, disch] | mean | l0b_lin | persistence |
|---|---|---|---|---|---|---|
| hold_rec | 0.989 0.919 0.768 | 0.892 | 0.982 0.839 0.810 | 0.877 | 0.710 | 0.768 |
| pulse40 | 0.790 0.932 0.739 | 0.820 | 0.791 0.836 0.701 | 0.776 | 0.733 | 0.503 |
| mid40 | 0.981 0.888 0.587 | 0.819 | 0.973 0.956 0.593 | 0.841 | 0.614 | 0.603 |
| multilevel200 | 0.759 0.942 0.764 | 0.822 | 0.709 0.894 0.733 | 0.778 | 0.549 | 0.444 |
| compose | 0.689 0.732 0.695 | 0.705 | 0.703 0.769 0.723 | 0.732 | 0.607 | 0.615 |
| **LOO mean** | | **0.812** | | **0.801** | 0.643 | 0.587 |

| run | v4 refit in-sample | mean | cohort v4 in-sample | mean |
|---|---|---|---|---|
| hold_rec | 0.989 0.916 0.850 | 0.918 | 0.990 0.900 0.841 | 0.910 |
| pulse40 | 0.849 0.897 0.735 | 0.827 | 0.809 0.850 0.706 | 0.789 |
| mid40 | 0.980 0.881 0.588 | 0.816 | 0.974 0.956 0.592 | 0.841 |
| multilevel200 | 0.824 0.952 0.762 | 0.846 | 0.761 0.932 0.767 | 0.820 |
| compose | 0.738 0.873 0.746 | 0.786 | 0.894 0.906 0.745 | 0.848 |
| **mean** | | **0.839** | | **0.842** |

Eval sanity (4,000 ticks): finite, 0.4 s per category, outside 0 / 0 / 0 / 7% (sustained / order /
recovery / composition), wait max 165 / 86 / 444 / 538, queue 25-326, discharges 0.3-21.

### Verdict

The old pipeline (v4 structure, refitted on the five runs) stays in `gtlab/ode/hospital_queue_min.py`
and `plans/hospital_queue_hospital_queue_min_p3.json` / `_doc.json` (0.822 there, 0.812 in our repeat `_p3b`):
LOO 0.812-0.822 vs 0.801 for the cohort model, the difference sitting in the pulse40 and multilevel folds (queue 0.93 vs 0.84, wait 0.76
vs 0.71). The cohort model is kept as the alternate `gtlab/ode/hospital_queue_min2.py` with
`plans/hospital_queue_hospital_queue_min2_v4.json` / `_doc.json` (v2 and v3 there are the two earlier
variants, thetas not interchangeable with the module). It is the only one of the two that does what the
compose run shows: wait 0.894 vs 0.738 in-sample on that run, and a held-out compose fold of 0.732 vs
0.705. The test schedules that move overtime alone and then drain the queue are exactly where the old
model predicts a wait near zero and the data say 300-400, so the cohort model is the candidate to try as
a blend member or as the wait_time source behind the old model's queue and discharges.

What the old structure cannot do on this data: the elective plateau at 260 (it only has the gate at
322-333), the faster drain under urgent priority at unchanged discharges, and the stranded wait. What
the cohort model loses: a queue fit on the two pulse-shaped runs, partly because $\tau_a$, $d_h$ and
$q_{cap}$ had to be frozen to stay within twelve parameters (the old refit moved them to 0.80, 0.087 and
316.6), and partly because the elective cancellation gate at 242 slows the climb to 333 in pulse40, where
the data climb through 242-321 at ~9/tick: elective cases are cancelled by wait or by staffing shortfall,
not by the queue alone. Neither model gives discharges better than ~0.75 (batchy in the data).
Structures rejected on the way: the age growing regardless of the cohort (v2, hold_rec leak of 10-25
ticks of phantom wait, unbounded age on 4,000-tick schedules), the linear cohort weight (v3, same leak at
a smaller size). Not tried for lack of time: a wait-driven elective cancellation, unfreezing $\tau_a$ by
dropping $w_{th}$ (fix it at 40, where the fit put it), and a warm start of $C$ from a large $w_0$ at reset
(an episode that starts after a stranding is unmodelled by both).
