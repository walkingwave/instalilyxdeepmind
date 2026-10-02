# Round-constant snap test (Mon Sep 28, no credits)

Question: the simulators were written by hand, so their true constants may be round (0.1, 0.25, 1/3,
10, 20, 50, 100 ...). If our shipped grey-box parameters sit on round numbers, that is evidence the
structure is right, and pinning them should free the fit to explain something else.

Scripts: `scripts/lab_snap_scan.py` (scan, `plans/snap_scan.json`), `scripts/lab_snap_loo.py`
(LOO test, `plans/snap_loo_<system>_<family>[_ctl|_los|_tau|_rates|_w].json`),
`scripts/lab_snap_table.py` (tables, `plans/snap_tables.md`). Models = final2
(`submissions/20260928-1810-final2`): ad_auction v8b (+ min member), epidemic y2, hospital hosp9,
market y3, power_grid w5, reservoir str9, social z20, supply_chain v8b, traffic z8, wildlife v8h.

## Bottom line

1. **Roundness is at chance level.** Over the 181 fitted parameters of the ten primary families, the
   share within 3% of a round value (value, 1/value, or 1 - value) is 73 vs 72.2 expected for random
   log-uniform numbers; within 1%: 32 vs 24.1 (p = 0.057); within 0.5%: 16 vs 12.0 (p = 0.15). About
   half of the < 1% hits are parameters that barely moved from a round starting value in PARAMS (market
   tau_R 20 -> 19.91, m2 0.5 -> 0.503, supply kd 0.01 -> 0.00998, wildlife ks 0.8 -> 0.799, cq 0.1 ->
   0.1009). That is where the fit started, not what the simulator uses.
2. **Pinning at the round value is no better than pinning at a non-round value the same distance
   away.** Control "mirror" = same parameters pinned at $2\hat\theta - \theta_{round}$ (same distance
   from the fit, other side, not round). Snap minus mirror, LOO mean: hospital −0.006, power_grid
   −0.005, reservoir −0.000, social +0.002, supply +0.005, wildlife −0.001 (mean −0.001). The round
   value carries no information the fit does not already have.
3. **Where pinning "gains" (reservoir +0.053, social +0.061, supply +0.043, hospital +0.017, wildlife
   +0.017, power_grid +0.008), the gain is regularization, and it is leaky.** Each gain is one fold
   where the all-free polish drifts (reservoir pulse +0.135, supply hold_mid +0.17, social voi +0.25);
   fewer free parameters stop the drift. The pins sit within 3% of the full-data fit, which saw the
   held-out run, so part of the gain is held-out information leaking in. Not an acceptance pass.
4. **Where pinning hurts, the parameter is well identified and not round**: traffic v_free 48.7 (fold
   range 48.63–48.83 excludes 50): snap3 −0.029, snap1 −0.016, long-hold fold −0.05 to −0.08; market
   snap1 −0.018 (k_M 1/80, rho 1/60 pins); epidemic snap1 −0.012 (tau_fat 60 pin; joint_hold −0.05).
5. **Single round pins that tie (so round is consistent with the data but not proven):** epidemic
   los = 10 (−0.001, fold range 9.51–10.52), power_grid w = 0.1 (+0.001, 0.0993–0.1014), market
   tau_R = 20, tau_v = 10, p_lo = 75 (−0.002), traffic rho = 0.3, r_pipe = 0.25, r_m = 0.25 (−0.001;
   but the long-hold fold −0.068 against +0.054 on hold_mid).
6. **Nothing to ship.** No snap passes the acceptance rule (LOO at 1.0 σ on every run, no loss on the
   pulse/recovery-shaped or exam fold) against its own control. The hypothesis is not supported for
   our reduced models: they are effective descriptions of larger simulators, so their constants are
   compound quantities (rate × share × unit scale), which need not be round even if the simulator's are.

## Method

**Round grid.** Mantissas per decade: tier 1 {1, 2, 5}; tier 2 {1.5, 2.5, 3, 4, 6, 7.5, 8}; tier 3
{10/3, 20/3, 10/6}. Decade invariance covers percentages and per-sub-step rates (dt = 0.1). Forms
tested: $v$, $1/v$ (rate vs time constant), $1 - v$ for $0.5 < v < 1$. Parameters at a bound or exactly
at their PARAMS init are excluded. Chance that a log-uniform number passes (all forms), Monte Carlo:
6.7% at 0.5%, 13.3% at 1%, 26.5% at 2%, 39.9% at 3%. So a single 3% flag is worth almost nothing.

**LOO test.** Calibrated organizer σ (`plans/sigma_calibrated.json`, 1.0×), every run a fold,
including p6.exam, p5.testlike, p7/p8 long holds. Every fold is one warm-started polish from the
shipped θ (`least_squares`, Cauchy, nfev 40, budget 90 s), identical for every variant, so the
comparison isolates the pin. Variants: base (all free), snap3 (every flag within 3% pinned at the round
value), snap1 (flags within 1%), mirror3/mirror1 (same parameters pinned at $2\hat\theta-\theta_r$), and
single-parameter pins. Absolute LOO here is not comparable with the lab LOO of the shipped fits
(multistart; e.g. reservoir 0.8745, hospital 0.699, social 0.607): the base polish is weaker on a few
folds, which is exactly where the pins "gain".

## Results: LOO mean (Δ vs base)

| system / family | base | snap3 | snap1 | mirror3 | mirror1 | single pins |
|---|---:|---:|---:|---:|---:|---|
| ad_auction v8b | 0.8636 | −0.007 (2) | −0.003 (1) | | | |
| ad_auction min | 0.8481 | +0.001 (5) | −0.002 (3) | | | |
| epidemic y2 | 0.6996 | −0.003 (5) | −0.012 (2) | | | los = 10: −0.001 |
| hospital hosp9 | 0.6753 | +0.017 (8) | −0.008 (3) | **+0.023** | | |
| market y3 | 0.5965 | −0.004 (8) | −0.018 (5) | | | tau_R 20, tau_v 10, p_lo 75: −0.002 |
| power_grid w5 | 0.6979 | +0.008 (8) | +0.003 (3) | **+0.013** | | w = 0.1: +0.001 |
| reservoir str9 | 0.8336 | +0.053 (9) | +0.044 (2) | **+0.053** | **+0.045** | |
| social z20 | 0.6194 | +0.061 (8) | +0.020 (4) | **+0.060** | | |
| supply v8b | 0.8744 | +0.043 (5) | +0.041 (2) | +0.038 | **+0.041** | |
| traffic z8 | 0.7562 | −0.029 (13) | −0.016 (4) | | | rho .3, r_pipe .25, r_m .25: −0.001 |
| wildlife v8h | 0.6750 | +0.017 (7) | +0.014 (5) | **+0.018** | +0.013 | |

(n) = number of pinned parameters. Per-fold deltas: `plans/snap_tables.md`.

## Per system: parameters, units, round or not

Units: ticks (t), observable units (obs), 1/t for rates. "fold range" = spread over the base LOO fits.

**ad_auction (v8b; min and s1 members agree).** N 267 people/opportunity pool; b0 3.83 rival bid;
gam 0.44 price exponent; kd 0.138 exposure removal/imp; tau_e 64 t return; c 0.268 conversions/imp;
tau1 5.4, tau2 5.8 t fulfilment stages; F 5.39 completions/t (fold 5.47–5.70; min/s1 give 6.00);
kw 0.37, kr 0.38, g0 0.506, kp 0.333 (= 1/3 at 0.05%), Kmax 28.6. Round looking: kp 1/3, g0 1/2,
F 6 in min/s1 (not in v8b). Pinning kp and g0: −0.007; F = 6 in min: tie. Not round and tight:
v8b F 5.56 and min c 0.226. Verdict: no evidence.

**epidemic (y2, AC).** pop 18,687 (at bound); beta 0.230/t; sigma 0.154/t (latent 6.5 t); recovery
0.44 / 0.37 / 0.36 per t (children/adults/elderly, 2.3–2.8 t); severity 0.082 / 0.153; **los 9.93 t
(fold 9.51–10.52; pin 10 ties)**; hcap 154 beds; mask_eff 0.42; k_clinic 5.0 (at bound); tau_fat 59.5 t
(fold 59.5–113: not identified; pinning 60 costs the joint_hold fold 0.05); tau_wane 87 t; debt_k 0.22,
tau_debt 143 t. Round candidates: los = 10 days of stay is the one physically plausible hit; data
cannot tell 10 from 9.9. Not round: the three recovery rates and sigma.

**hospital_queue (hosp9).** lam0 10.6 arrivals/t; c_a 0.996 and c_t 0.950 service multipliers
(start 1.5 / 2); tau_a 0.80 t; chairs 17.9, beds 10.5; ot_gain 0.55; k_fat 0.35; k_l 0.0078/t
reneging; tau_w 51 t, tau_dn 19.8 t wait rise/fall; q_cap 317 (fold 317.5–323: tight, NOT 333 = 1000/3,
though the shipped post rule caps at 333); eps_w 1.498 and c_w 1.70 wait law. Every flagged parameter
except c_t is poorly identified across folds (eps_w 0.05–3.19, c_w 0.82–2.47, tau_dn 6.5–31.5, tau_w
20–60). Pinning helps (+0.017) but the non-round control helps more (+0.023): regularization of the
wait law, not a round constant.

**market (y3, AB).** d0 92.3 depth (tight, not 100); m1 0.52, m2 0.503 depth cuts; x_d 0.0185 and
x_c 0.0446 tax thresholds (tight, not round; tax range 0–0.05); k_R 0.853, tau_R 19.9 t; k_M 0.0125;
g_l 0.109 (tight); p_lo 75.1–75.7 (≈ 75); rho 0.0166; n_r 7.05 (tight); b0 2.36; i0 0.62, i_r 0.44
(tight); k_w 0.026; k_I 54.0 (tight); tau_A 11.8, tau_Au 3.08 t; kap 0.027; s_v 0.63; tau_v 9.8 t.
Pinning tau_R 20, tau_v 10, p_lo 75 ties; adding k_M 1/80 and rho 1/60 loses 0.018 (multilevel
−0.058, compose −0.038, rate_hold −0.041). Many tight non-round constants (d0, x_d, x_c, g_l, n_r, i_r,
k_I): market is the system where the reduced model is most clearly an effective description.

**power_grid (w5).** d0 127.5 load at price 0 via $D(p)$; d1 19.2 per price unit; d2 −7.7; **w 0.0993
rad/t (fold 0.0993–0.1014; pin 0.1 ties)**, zeta 0.32; kick 0.63; kf 0.038 Hz per unit; gb −9.2; a_g
0.055/t; kg 4.6; c_r 1.09; cap 304 storage; p_ch 86; s0 0.365 (tight, not round); k_th 0.2501;
f_hi 1.89 Hz; beta 0.503; g_i 13.7 (fold 13.7–21.9); p_l 21.9, p_r 82 (fold 82–154). The oscillation
frequency 0.1 rad/t (period 63 t) is the best round candidate in the whole study; it is consistent but
unproven. Group snap +0.008 < mirror +0.013.

**reservoir (str9).** q_m 11.28, q_a 2.21 inflow mean/amplitude (tight, not round; period 67.75 t,
phase locked, not round either); seep 0.00055/t; c_out 13.4, p_out 0.34 outlet law; q_base 0.981–0.993;
tau_q 4.2 t (tight); tau_ret 32.7 t; q_d 0.030–0.035; tau_d 54 t; k_b 0.009, tau_b 15.2 t bank storage;
f_in 0.046; g_gw 0.0025, L_gw 481 (tight, not 500); q_n 0.041, tau_n 118 t; n0 0.394; q_aer 0.0036.
Snap +0.053 = mirror +0.053; all of it in the pulse fold (+0.135), where the all-free polish drifts.
The inflow constants are the cleanest evidence against the hypothesis: inflow is observed directly,
so q_m, q_a and the period are measured, and none is round in ticks.

**social_contagion (z20, BC).** N_a 293 (fold 293–309; 300 plausible), N_b 241; s_a 0.0040, s_b
0.0015 recruitment; o 0.0030 outside interest; tau_on 8.1 t (fold 8.0–8.6); c_L 0.0060 churn; k_X
0.022; k_conv 0.021; k_A 2.14; k_B 0.054; tau_E 21 t; tau_D 27 t; k_inc 0.69, k_inc_b 0.20. Snap
+0.061 = mirror +0.060, nearly all in the voi fold (+0.25) where the base polish collapses (the lab
LOO had the same fold at 0.48). Regularization, not roundness.

**supply_chain (v8b).** p0 12.1 production/t; kc 0.39, tau_c 1.0 (at bound); qm 1,887 conveyor limit;
dq 0.083; kq0 1.16, kl 7.0; a0 53 terminal; dem 41 demand/t; fb 0.352 class-B share (tight, not 1/3);
kd 0.00998 (start 0.01: anchored); kcool 0.013; ws 0.598 (start 0.5). Snap +0.043, mirror3 +0.038,
mirror1 +0.041: the gain is the hold_mid fold (+0.16–0.17) under any pin of kd and ws.

**traffic (z8).** dem0 46.9 veh/t; w_sig 0.72; r_pipe 0.243/t (fold 0.240–0.249: below 0.25); junc 35;
cap 16.7; L_a 1.68, L_b 1.23 lane thresholds; clr 0.38; **v_free 48.7 (fold 48.63–48.83: not 50)**;
n_ref 319; rho 0.297/t; n_max 1,070; gam 1.06; w1 1.75; r_m 0.248; kap 0.66; k_toll 0.196/toll unit;
n_l 1.36; beta_L 0.26, tau_L 6.5 t; tau_sw 0.72 t; k_F 0.59, tau_F 76 t; ce0 0.072; K3 5.0 (bound).
Pinning v_free = 50 and the lane law is the largest loss in the study (−0.029; long hold −0.083). The
reported speed mixes stopped and moving vehicles (brief), so the fitted free speed is an effective
quantity; 48.7 against 50 is a structure hint for the speed law, not a mis-fit.

**wildlife (v8h).** r 0.88/t births (tight, not round); ks 0.799 south scale (start 0.8); H 1.36
harvest per quota unit; Ph 24.9 half-saturation prey (start 20; fold 24–28.6); em 0.026, emS 0.040
corridor rates; tau 26 t transit; sv 0.78 survival; cJ 900 nursery; w 0.026; kR 2.4e-5; bh 0.35;
dhN 0.044, dhS 0.025; q0 1.75, q1 1.16; cq 0.101 (start 0.1); tz 27 t. Snap +0.017 = mirror +0.018.

## Parameters clearly not round (well identified, > 3% from any round value)

market d0 92.3, x_d 0.0185, x_c 0.0446, g_l 0.109, n_r 7.05, i_r 0.437, k_I 54.0; reservoir q_m 11.28,
q_a 2.21, tau_q 4.24, period 67.75; traffic v_free 48.7, r_pipe 0.243; power_grid s0 0.367; hospital
q_cap 320; supply fb 0.352; wildlife r 0.888; ad_auction v8b F 5.56, min c 0.226. These are the
places where a reduced model is standing in for a compound mechanism (market above all). The
reservoir inflow terms are observed directly, so for those the non-round value is the simulator's own.

## What was rejected and why

- Shipping any snap: gains equal the non-round control and are partly held-out leakage.
- Reading the 3% flags as evidence: 40% of random numbers pass.
- A pin-to-regularize lever (pin weakly identified parameters near the fit) is visible here, but the
  shipped fits come from multistart fits whose lab LOO already sits at or near the pinned numbers
  (reservoir 0.8745 lab vs 0.886 pinned-and-leaky). Not pursued.
