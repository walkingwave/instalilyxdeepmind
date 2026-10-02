# Control audit: which control effects do our models get wrong or ignore? (Mon Sep 28, afternoon, no credits)

Question: the traffic gain of u016 (0.776 → 0.839) came from a control our model treated as inert
(toll changes the arriving demand). Do other shipped models have the same blind spot, and does
adding the missing effect pay on held-out runs?

Models audited: `submissions/20260928-1440-final1/<system>/predict.py` exactly as shipped (for
ad_auction and power_grid the per-observable documents as shipped). Scale: calibrated organizer σ
(`plans/sigma_calibrated.json`). Data: every ledger run at 15:25, including the p8 long holds that
arrived during the afternoon (epidemic, power_grid, reservoir, ad_auction). Script:
`scripts/control_audit.py`; numbers: `plans/control_audit.json` (keys `audit`, `fixes`; a re-run of
the script rewrites the file with the audit part only).

## 1. Method

For system $s$, control $j$, observables $k$ with calibrated $\sigma_k$:

- **Authority** (is the control used?). From the mid-range action $\bar u$, hold 300 ticks with
  $u_j$ at its lower and at its upper bound;
  $\mathrm{auth}_{jk} = \frac{1}{200}\sum_{t=100}^{299} |\hat y^{hi}_{tk} - \hat y^{lo}_{tk}| / \sigma_k$.
  Zero = the model ignores $u_j$. (A first pass from the recovery action gave false zeros: traffic's
  recovery has ramp = 0, which switches demand off and hides toll entirely.)
- **Events.** Every tick where some control changes (tick 0 counts as a change from the recovery
  action). Window $W = [t, \min(t_{next}, t+60))$. Per-tick loss
  $\ell_t = \frac1p\sum_k \big(1 - \frac{1}{1+|y_{tk}-\hat y_{tk}|/\sigma_k}\big)$ (one minus the score).
  Excess loss of a window $= \sum_{t\in W} (\ell_t - \bar\ell_{pre})$, $\bar\ell_{pre}$ = mean loss over
  the 20 ticks before the change (run median at tick 0). The excess is split over the controls that
  moved in proportion to $|\Delta u_j|/(\mathrm{hi}_j-\mathrm{lo}_j)$; negative excess counts as 0.
  Ranking key: attributed excess per data tick of the system.
- **Isolated moves** (only $u_j$ moved): observed step $\Delta y^{obs}$ = window mean minus the
  20-tick pre-change mean, predicted step $\Delta\hat y$ the same on the model, both in σ.
- **Linear fix, held-out.** Leave-one-run-out ridge of the residual $(y-\hat y)/\sigma$ on
  $[1, u_j, \mathrm{EMA}_8(u_j), \mathrm{EMA}_{40}(u_j)]$ (normalised $u_j$); score of the corrected
  prediction on the held-out run minus the same with the intercept alone. Positive = a
  control-driven residual that transfers across runs.

## 2. Ranked audit table

In-sample on every run (the shipped fits saw most of these runs, so levels are optimistic; the
ranking is what matters). "used?": no < 0.05 σ ≤ weak < 1 σ ≤ yes.

| # | system | control | used? (max authority, σ) | moves / isolated | largest isolated miss: observed vs predicted step (σ) | excess lost score / tick (×1000) | linear fix, held-out (×1000 / tick) |
|---|---|---|---|---|---|---:|---:|
| 1 | market | interest_rate | yes (24.23) | 32 / 5 | price -7.6 vs -2.0 (p5.testlike t360) | 15.9 | +7.3 |
| 2 | wildlife | corridor_access | yes (1.39) | 12 / 6 | predator_north -5.0 vs -3.8 (p4.corridor_habitat t90) | 15.5 | -14.3 |
| 3 | epidemic | school_closure | yes (1.66) | 9 / 2 | daily_cases -16.4 vs -14.2 (p3.compose t45) | 14.2 | -14.1 |
| 4 | ad_auction | budget_cap | yes (4.48) | 23 / 2 | conversions -0.1 vs -0.5 (p6.exam t355) | 14.0 | -14.2 |
| 5 | epidemic | mask_mandate | yes (4.34) | 11 / 3 | hospital_load +13.0 vs +13.9 (p6.voi t0) | 13.8 | -11.3 |
| 6 | epidemic | vaccination_rate | yes (5.80) | 10 / 2 | hospital_load +4.4 vs +5.1 (p3.compose t179) | 13.2 | -10.4 |
| 7 | power_grid | charging_allowance | no (0.00) | 24 / 2 | frequency +0.3 vs +4.1 (p6.exam t324) | 13.0 | -15.8 |
| 8 | market | transaction_tax | yes (26.76) | 31 / 4 | depth -1.7 vs +2.7 (p5.testlike t330) | 11.8 | -6.7 |
| 9 | power_grid | interconnector | yes (3.20) | 23 / 2 | frequency -0.6 vs -1.9 (p6.exam t344) | 11.4 | -14.5 |
| 10 | ad_auction | targeting_breadth | yes (5.64) | 22 / 2 | spend -1.1 vs -0.9 (p6.exam t340) | 10.7 | -16.2 |
| 11 | social_contagion | seeding | yes (15.32) | 25 / 3 | adopters_a +2.4 vs +4.1 (p5.testlike t310) | 9.6 | -23.1 |
| 12 | social_contagion | incentive | yes (1.10) | 26 / 4 | adopters_a +2.9 vs +0.9 (p5.testlike t325) | 9.4 | -29.1 |
| 13 | wildlife | habitat_protection | yes (4.72) | 10 / 4 | predator_north +2.6 vs +0.2 (p4.corridor_habitat t180) | 9.3 | -2.8 |
| 14 | ad_auction | bid | yes (6.60) | 23 / 2 | conversions +0.3 vs -0.3 (p6.exam t360) | 9.0 | -1.2 |
| 15 | traffic | ramp_metering | yes (6.07) | 29 / 2 | flow_a -2.9 vs -0.6 (p5.testlike t351) | 8.5 | -18.6 |
| 16 | traffic | toll | yes (2.58) | 31 / 2 | flow_a +1.8 vs -0.2 (p5.testlike t360) | 7.8 | -10.7 |
| 17 | wildlife | hunting_quota | yes (9.14) | 8 / 2 | prey_south +5.4 vs +4.2 (p3.compose t45) | 7.6 | +6.8 |
| 18 | hospital_queue | overtime | no (0.03) | 44 / 4 | wait_time +4.6 vs -0.3 (p3.compose t180) | 7.2 | +4.0 |
| 19 | power_grid | reserve_dispatch | yes (4.25) | 23 / 2 | frequency +3.3 vs +6.7 (p6.exam t320) | 7.1 | -15.1 |
| 20 | social_contagion | bridge_outreach | yes (11.24) | 26 / 4 | adopters_b -4.2 vs -2.4 (p3.compose t134) | 6.5 | -18.6 |
| 21 | traffic | clearance_effort | no (0.03) | 31 / 2 | flow_b -2.7 vs +0.6 (p5.testlike t327) | 6.4 | -15.7 |
| 22 | traffic | lane_closure | weak (0.71) | 29 / 2 | speed_a -0.6 vs +0.6 (p5.testlike t363) | 6.4 | -18.1 |
| 23 | traffic | signal_timing | yes (3.18) | 30 / 2 | speed_a +1.5 vs +1.0 (p5.testlike t372) | 6.3 | -10.6 |
| 24 | power_grid | price_signal | yes (4.30) | 22 / 1 | load -6.9 vs -2.0 (p6.exam t308) | 5.2 | -33.3 |
| 25 | traffic | freight_priority | no (0.00) | 31 / 2 | flow_b +2.0 vs -2.3 (p5.testlike t342) | 4.1 | -11.7 |
| 26 | hospital_queue | followup_capacity | no (0.00) | 44 / 4 | wait_time +2.6 vs -0.1 (p3.compose t225) | 3.8 | -4.2 |
| 27 | hospital_queue | elective_scheduling | weak (0.47) | 43 / 4 | queue +2.3 vs +3.7 (p3.compose t45) | 3.8 | +4.3 |
| 28 | hospital_queue | staffing | yes (7.34) | 44 / 4 | discharges -1.9 vs -2.6 (p3.compose t0) | 3.7 | -16.4 |
| 29 | supply_chain | order_quantity | yes (3.69) | 4 / 0 | no isolated move | 3.4 | -42.3 |
| 30 | supply_chain | maintenance | weak (0.07) | 4 / 0 | no isolated move | 3.4 | -42.3 |
| 31 | hospital_queue | urgent_priority | no (0.00) | 45 / 4 | discharges +0.9 vs -0.8 (p6.exam t345) | 3.0 | -6.5 |
| 32 | supply_chain | lead_time_buy | no (0.00) | 4 / 0 | no isolated move | 3.0 | -40.4 |
| 33 | supply_chain | receiving_effort | yes (3.69) | 4 / 0 | no isolated move | 2.9 | -40.0 |
| 34 | hospital_queue | diagnostic_allocation | yes (2.21) | 45 / 4 | discharges -1.5 vs -3.5 (p6.exam t354) | 2.9 | +0.6 |
| 35 | supply_chain | production_effort | yes (4.44) | 4 / 0 | no isolated move | 1.1 | -27.0 |
| 36 | supply_chain | product_mix | yes (2.73) | 3 / 0 | no isolated move | 0.6 | -22.4 |
| 37 | reservoir | withdrawal_depth | weak (0.69) | 4 / 0 | no isolated move | 0.4 | -8.0 |
| 38 | reservoir | irrigation_allocation | yes (5.77) | 2 / 0 | no isolated move | 0.4 | +1.1 |
| 39 | reservoir | aeration | weak (0.69) | 4 / 0 | no isolated move | 0.4 | -8.0 |
| 40 | reservoir | release_rate | yes (5.89) | 4 / 0 | no isolated move | 0.3 | -7.9 |

- epidemic: 1811 ticks, in-sample score 0.760
- market: 1840 ticks, in-sample score 0.599
- traffic: 1840 ticks, in-sample score 0.801
- power_grid: 1600 ticks, in-sample score 0.685
- supply_chain: 1770 ticks, in-sample score 0.937
- wildlife: 1811 ticks, in-sample score 0.758
- reservoir: 1220 ticks, in-sample score 0.858
- ad_auction: 1600 ticks, in-sample score 0.873
- social_contagion: 1811 ticks, in-sample score 0.704
- hospital_queue: 1770 ticks, in-sample score 0.666

## 3. What the audit shows

1. **Controls the shipped models ignore (authority < 0.05 σ):** traffic `freight_priority` (0.00)
   and `clearance_effort` (0.03: z8's exit boost is cancelled by crew fatigue after ~100 ticks);
   power_grid `charging_allowance` (0.00: the refill term only acts while the store is depleted);
   hospital `urgent_priority` and `followup_capacity` (0.00: p3 has no term for either) and
   `overtime` (0.03 at mid staffing); supply_chain `lead_time_buy` (0.00) and `maintenance` (0.07).
   reservoir `withdrawal_depth` and `aeration` only move quality (0.69 σ).
2. **Where the score goes:** the ignored controls sit low in the ranking. The largest attributed
   excess losses are on controls the models do use: market `interest_rate` (after the rate pulse
   train in p5 the price slides 83 → 73.6, −11.5 σ at t396 with the model flat; the p7 long hold
   averages price +13.9 σ and depth −11.4 σ: the dealer-inventory leak known from §35), wildlife
   `corridor_access`, and the three epidemic controls (all three at 0.013–0.014 per tick: a generic
   post-switch error, not one control).
3. **Linear corrections do not transfer.** The held-out linear fix is negative for 34 of 40
   controls: with 3–9 runs a control-history correction fitted on some runs hurts the others. Only
   market `interest_rate` (+0.007 per tick, driven by p7), wildlife `hunting_quota` (+0.007) and
   hospital `overtime` / `elective_scheduling` (+0.004) carry a residual that transfers.
4. **Largest single residual pocket in any system:** hospital p3.compose t186–270. Overtime at full
   staffing drains the queue from 160 to 23 in 15 ticks; then wait_time climbs 21 → 383 (+2 per
   tick) while queue stays at 23 and discharges at 11.5, and falls back when electives return at
   t270. p3 predicts wait ≈ 1 (z up to +16.7). It reads as a drained cohort whose mean age keeps
   growing (reported wait behaves like an age average, not a Little estimate): a state effect of the
   drain, not a control law. Second pocket: after the standard pulse (p4, p6.voi) the recovery queue
   plateaus at 92 where p3 predicts 15 (z +2.8 for 150 ticks): capacity at recovery is only just
   above arrivals. Both known since §31 (orientation and fatigue confounded).
5. **power_grid p8 (new long hold):** at all-zero controls frequency sits at 48.90 Hz for 300 ticks,
   v9c at 49.35 (−1.5 σ); the step that caused it closed interconnector and charging together
   (1 → 0) with the reserve at zero, where v9c's interconnector is inert.

## 4. Fixes tried (lab at 1.0 σ, leave-one-run-out, defaults: budget 240 s, 10 starts, nfev 60)

Each new family nests the shipped one: with the new parameters at their neutral values it
reproduces the shipped rollout exactly (checked: max |ΔY| = 0 on a real run). Baselines were re-run
under the same machine load where the run set changed or no same-protocol report existed (traffic,
epidemic, hospital, power_grid); wildlife and social use their §36 reports
(`plans/wildlife_wildlife_v8h_p7.json`, `plans/social_contagion_social_contagion_z20_zr6BC.json`,
same runs, same σ). The lab is time-bounded (every fold stops at 120 s), so a nested family can
score below its parent even in-sample (social 0.718 → 0.712): extra parameters cost iterations.
Columns: held-out run, baseline fold mean, new fold mean, difference, new per observable.

### 4.1 traffic: freight priority and clearance at the junction (`gtlab/ode/traffic_c.py`, `traffic_c2.py`)

Brief: clearance "shifts a shared crew from intersection operation toward downstream exits";
freight priority "changes which waiting class gets service". z8 has an exit boost only and no
freight term. Isolated moves in p5: freight 0.5 → 1 lifts flow_a by +2.9 σ (model −0.8);
clearance 1 → 0 drops both flows by ≈ 2.8 σ (model +0.1 to +0.6).
traffic_c: junction capacity $J\,(1-k_{cj}\,c)\,(1-k_{fr}(f-0.5))$, $c$ = crew state.
traffic_c2: freight term only. Full fits: c: $k_{cj}$ = 0.20, $k_{fr}$ = 0.15, exit boost clr → 0;
c2: $k_{fr}$ = 0.13, clr 0.61.

traffic_c (`plans/traffic_traffic_c_ca.json`) vs z8 re-run (`plans/traffic_traffic_z8_cabase.json`,
reproduces §36's 0.742):

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.975 | 0.963 | -0.011 | [1.0, 1.0, 0.93, 0.924] |
| p2.pulse40 | 0.525 | 0.517 | -0.008 | [0.441, 0.378, 0.668, 0.581] |
| p2.mid40 | 0.850 | 0.853 | +0.003 | [0.818, 0.784, 0.897, 0.914] |
| p2.multilevel200 | 0.683 | 0.679 | -0.004 | [0.66, 0.615, 0.738, 0.703] |
| p3.hold_mid | 0.819 | 0.794 | -0.025 | [0.843, 0.776, 0.806, 0.75] |
| p5.testlike | 0.729 | 0.708 | -0.021 | [0.716, 0.654, 0.829, 0.634] |
| p7.longhold | 0.613 | 0.558 | -0.055 | [0.613, 0.325, 0.537, 0.756] |
| **mean** | 0.742 | 0.725 | -0.017 | in-sample 0.787 -> 0.792 |

traffic_c2 (`plans/traffic_traffic_c2_ca.json`):

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.975 | 0.973 | -0.001 | [1.0, 1.0, 0.95, 0.944] |
| p2.pulse40 | 0.525 | 0.518 | -0.007 | [0.441, 0.377, 0.659, 0.594] |
| p2.mid40 | 0.850 | 0.852 | +0.002 | [0.817, 0.785, 0.892, 0.914] |
| p2.multilevel200 | 0.683 | 0.685 | +0.003 | [0.656, 0.614, 0.752, 0.717] |
| p3.hold_mid | 0.819 | 0.808 | -0.011 | [0.859, 0.792, 0.81, 0.771] |
| p5.testlike | 0.729 | 0.711 | -0.019 | [0.718, 0.653, 0.83, 0.641] |
| p7.longhold | 0.613 | 0.551 | -0.062 | [0.537, 0.331, 0.564, 0.771] |
| **mean** | 0.742 | 0.728 | -0.014 | in-sample 0.787 -> 0.792 |

Verdict: **no**. In-sample +0.005, held-out −0.014 to −0.017; the long-hold fold loses 0.055–0.062.
The isolated freight / clearance moves in p5 last 3–6 ticks; the fit buys them with the long hold.

### 4.2 epidemic: school closure (`gtlab/ode/epidemic_c.py`)

y2's closure strength `school` sits at its bound 1.0, and the isolated closure in p3.compose falls
2.2 σ further than predicted (−16.4 vs −14.2). epidemic_c: closure also removes a share $s_{ca}$ of
child–adult contacts, and the share of closed child–child contacts displaced into the child–adult
block is fitted: $c_{01} = C_{01}(1-s_{ca}\,cl) + d\,C_{00}\,\mathrm{school}\,cl$ ($d$ = 0.3 fixed in y2).
Full fit: $s_{ca}$ = 0.003, $d$ = 0.56, school = 1.0: the data do not want a stronger closure; the
larger displacement even weakens it.

epidemic_c (`plans/epidemic_epidemic_c_ca.json`) vs y2 re-run on the six runs
(`plans/epidemic_epidemic_y2_cabase.json`):

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.742 | 0.704 | -0.038 | [0.66, 0.749] |
| p2.pulse120_280 | 0.734 | 0.734 | +0.000 | [0.694, 0.775] |
| p3.compose | 0.652 | 0.673 | +0.021 | [0.63, 0.717] |
| p4.joint_hold | 0.732 | 0.708 | -0.024 | [0.788, 0.628] |
| p6.voi | 0.567 | 0.590 | +0.023 | [0.612, 0.568] |
| p8.longhold | 0.768 | 0.781 | +0.013 | [0.804, 0.757] |
| **mean** | 0.699 | 0.698 | -0.001 | in-sample 0.762 -> 0.762 |

Verdict: **no** (−0.001: compose +0.021 and voi +0.023 against hold −0.038 and joint hold −0.024).

### 4.3 wildlife: predators' corridor use (`gtlab/ode/wildlife_c.py`)

v8h ties the predators' journey rate to the north prey rate; wildlife_c frees the ratio $r_Q$
(predators leave at $r_Q\,e_m\,\mathrm{cor}\,Q$). Full fit $r_Q$ = 0.92: the tie holds.

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.635 | 0.632 | -0.003 | [0.676, 0.672, 0.694, 0.485] |
| p2.pulse200_200 | 0.724 | 0.725 | +0.001 | [0.744, 0.645, 0.806, 0.706] |
| p3.compose | 0.594 | 0.595 | +0.001 | [0.464, 0.675, 0.498, 0.741] |
| p4.corridor_habitat | 0.666 | 0.670 | +0.004 | [0.647, 0.566, 0.699, 0.77] |
| p7.longhold | 0.756 | 0.746 | -0.010 | [0.8, 0.601, 0.934, 0.651] |
| **mean** | 0.675 | 0.674 | -0.001 | in-sample 0.727 -> 0.727 |

Verdict: **no** (−0.001). The corridor residual is not a predator-rate problem.

### 4.4 social_contagion: incentive retains members (`gtlab/ode/social_contagion_c.py`)

Incentive-only block in p3.compose (seeding off): b holds 93 → 96 while z20 churns to 89.5.
social_contagion_c: member churn $c_L(1-k_{ret,i}\,\varphi(c))$ per community. Full fit:
$k_{ret,a}$ = 0.34, $k_{ret,b}$ = 0 (the fit puts retention in a, not in b where we saw it).

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.770 | 0.774 | +0.004 | [0.886, 0.661] |
| p2.pulse200_200 | 0.560 | 0.560 | -0.000 | [0.605, 0.516] |
| p3.compose | 0.594 | 0.609 | +0.016 | [0.615, 0.604] |
| p4.interior_holds | 0.569 | 0.502 | -0.066 | [0.473, 0.531] |
| p5.testlike | 0.647 | 0.637 | -0.010 | [0.71, 0.564] |
| p6.voi | 0.479 | 0.415 | -0.064 | [0.595, 0.236] |
| **mean** | 0.603 | 0.583 | -0.020 | in-sample 0.718 -> 0.712 |

Verdict: **no** (−0.020; interior holds −0.066, voi −0.064).

### 4.5 hospital_queue: follow-up takes shared staff, urgent case mix (`gtlab/ode/hospital_queue_c.py`, `hospital_queue_c2.py`)

Brief: follow-up "diverts shared staff"; urgent priority "changes new service admissions". p3 has
no term for either. hospital_queue_c: effective work
$\mathrm{eff}\,(1-k_{fu}F_u)(1+k_u(U-0.6))$; c2: follow-up only. Full fits: c $k_{fu}$ = 0.19,
$k_u$ = −0.15; c2 $k_{fu}$ = 0.10. The recovery action has follow-up 1, so the term lowers recovery
capacity (hold_rec queue: p3 settles at 15, the data at 23).

hospital_queue_c (`plans/hospital_queue_hospital_queue_c_ca.json`) vs p3 re-run on the eight runs
(`plans/hospital_queue_hospital_queue_p3_cabase.json`):

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.614 | 0.692 | +0.077 | [0.879, 0.388, 0.807] |
| p2.pulse40 | 0.716 | 0.726 | +0.011 | [0.834, 0.774, 0.571] |
| p2.mid40 | 0.617 | 0.649 | +0.032 | [0.843, 0.715, 0.388] |
| p2.multilevel200 | 0.677 | 0.681 | +0.004 | [0.627, 0.828, 0.587] |
| p3.compose | 0.481 | 0.490 | +0.009 | [0.515, 0.415, 0.54] |
| p4.pulse_long_recovery | 0.743 | 0.749 | +0.006 | [0.859, 0.606, 0.782] |
| p6.voi | 0.732 | 0.717 | -0.015 | [0.869, 0.607, 0.674] |
| p6.exam | 0.678 | 0.649 | -0.029 | [0.644, 0.727, 0.577] |
| **mean** | 0.657 | 0.669 | +0.012 | in-sample 0.696 -> 0.691 |

hospital_queue_c2 (`plans/hospital_queue_hospital_queue_c2_ca.json`):

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.614 | 0.682 | +0.068 | [0.87, 0.398, 0.778] |
| p2.pulse40 | 0.716 | 0.729 | +0.013 | [0.82, 0.799, 0.567] |
| p2.mid40 | 0.617 | 0.643 | +0.026 | [0.85, 0.688, 0.391] |
| p2.multilevel200 | 0.677 | 0.658 | -0.019 | [0.577, 0.798, 0.598] |
| p3.compose | 0.481 | 0.542 | +0.061 | [0.543, 0.483, 0.598] |
| p4.pulse_long_recovery | 0.743 | 0.752 | +0.009 | [0.856, 0.617, 0.783] |
| p6.voi | 0.732 | 0.714 | -0.018 | [0.856, 0.613, 0.674] |
| p6.exam | 0.678 | 0.639 | -0.040 | [0.625, 0.722, 0.569] |
| **mean** | 0.657 | 0.670 | +0.013 | in-sample 0.696 -> 0.686 |

Verdict: **hold, do not ship**. Mean +0.012 / +0.013 (below the 0.02 fold noise), carried by
hold_rec (+0.07) and, for c2, compose (+0.06); the exam fold loses 0.029 / 0.040 and the voi fold
(recovery shape) loses 0.015 / 0.018: breaks the exam and recovery-fold rules. The best candidate
of the round; c2 is the one to re-test if new hospital data arrive.

### 4.6 power_grid: interconnector imports power (`gtlab/ode/power_grid_c.py`)

power_grid_c: frequency balance $+\,p_{imp}\,ic$. Full fit $p_{imp}$ = 11.1 (positive: closing the
interconnector lowers frequency, as in p8 and the exam).

power_grid_c (`plans/power_grid_power_grid_c_ca.json`) vs v9c re-run on the five runs
(`plans/power_grid_power_grid_v9c_cabase.json`):

| held-out run | base | new | diff | new per obs |
| p1.hold_rec | 0.784 | 0.766 | -0.018 | [0.82, 0.73, 0.747] |
| p2.pulse60_120 | 0.711 | 0.697 | -0.014 | [0.643, 0.609, 0.839] |
| p2.multilevel200 | 0.756 | 0.718 | -0.038 | [0.799, 0.516, 0.839] |
| p6.exam | 0.634 | 0.642 | +0.008 | [0.609, 0.562, 0.757] |
| p8.longhold | 0.618 | 0.632 | +0.015 | [0.669, 0.506, 0.723] |
| **mean** | 0.701 | 0.691 | -0.009 | in-sample 0.740 -> 0.739 |

Verdict: **no**. The long-hold fold gains (frequency 0.46 → 0.51) but multilevel frequency loses 0.10
and the pulse fold loses 0.014. power_grid ships load and frequency from v9c, so these folds decide.

## 5. Decision and forecast

| system | family | held-out gain | pulse / recovery fold | exam | decision | forecast public gain (0.6 × held-out) |
|---|---|---:|---|---|---|---:|
| traffic | traffic_c / c2 | −0.017 / −0.014 | −0.008 / −0.007 | p5 −0.021 / −0.019 | keep z8 | 0 |
| epidemic | epidemic_c | −0.001 | 0.000 | — | keep y2 | 0 |
| wildlife | wildlife_c | −0.001 | +0.001 | — | keep v8h | 0 |
| social_contagion | social_contagion_c | −0.020 | 0.000 | p5 −0.010 | keep z20 | 0 |
| hospital_queue | hospital_queue_c / c2 | +0.012 / +0.013 | pulse40 +0.011 / +0.013, voi −0.015 / −0.018 | p6 −0.029 / −0.040 | keep p3 | 0 (c2 if shipped: ≈ +0.008 by the rule, the exam says less) |
| power_grid | power_grid_c | −0.009 | −0.014 | p6 +0.008 | keep v9c | 0 |
| market, ad_auction, supply_chain, reservoir | — | not tried | | | keep | 0 |

No change to the Final picks from this round. What we learned:

- Every shipped model uses every control that moves the data a lot. The controls with zero
  authority (freight, clearance, charging, follow-up, urgent priority, lead time) have small,
  short-window footprints in our runs (3–10 ticks in the test-shaped runs), so the lab cannot buy
  them without losing a long fold. The toll lesson does not repeat: toll had a sustained footprint
  (the traffic p7 long hold) that these controls lack.
- The big residual pockets are state effects (hospital's drained cohort, market's dealer inventory,
  power_grid's frequency level at zero reserve), not missing control laws.
- A term that nests the shipped family can still lose held-out score in the time-bounded lab; a
  comparison is only fair against a baseline re-run on the same runs under the same load.
