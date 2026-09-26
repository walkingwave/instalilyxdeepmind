# hospital_queue: mechanism-pair search (`gtlab/ode/hospital_queue_mech.py`)

Lab notebook, Sat Sep 26. Data: 5 runs, 770 ticks (hold_rec 120, pulse40 80, mid40 40, multilevel200 200,
compose 330). `sigma_proxy` = [100.0, 120.1, 5.27] for [wait_time, queue, discharges]. No credits spent.
Harness, one run per pair, in parallel:

    PYTHONPATH=. python scripts/ode_lab.py --system hospital_queue --family hospital_queue_mech \
        --mech AB --budget 300 --starts 10 --nfev 60 --tag AB --free <base + AB params>

`--free` = `free_for(mech)` in the module (base parameters plus the active pair's), so an inactive
mechanism's parameters stay at their init and do not enter the fit.

## Question

The brief says exactly two of fatigue (A), handover/orientation (B) and returning case mix (C) are
active. Our shipped pipeline (`hospital_queue_p3`) has fatigue always on and nothing else. We ask which
pair the five runs prefer when all three are written as switchable terms on one common base.

## Structure (v1, the one kept)

Base = the p3 pipeline plus the cohort wait of `hospital_queue_min2` v4. States
$W, A_1, A_2, T, w, C, a, F, S_e, R$; $q = W + A_1 + A_2 + T$; $K = 3$.

$$\text{eff} = S^\*(1 + g_{ot} O)\,(1 - k_{fat} F)^{[A]},\quad S^\* = S_e^{[B]}\ \text{else}\ S$$
$$r_a = c_a\,\text{eff}\,\frac{D}{D + 0.087},\quad r_t = c_t\,\text{eff}\,(1 - D)$$
$$f_1 = \min(r_a, KW, K(\text{chairs} - A_1 - A_2)),\ f_{2a} = \tfrac{2}{0.8}A_1,\
f_{2b} = \min(\tfrac{2}{0.8}A_2, K(\text{beds} - T)),\ f_3 = \min(r_t, KT)$$
$$\dot W = (\lambda_0 + E + R/\tau_{ret}^{[C]})\,\sigma\!\left(\tfrac{q_{cap} - q}{6}\right) - f_1 - k_l U W - s,
\quad s = k_s W \tfrac{w}{w + 30}$$
$$\dot C = s - C/150,\quad \dot a = g_z - a\,\tfrac{s}{C + 1} - a/150$$
$$w^\* = \tfrac{W}{f_1 + 1} + a\,\tfrac{C^2}{C^2 + 25}\,\sigma\!\left(\tfrac{40 - W}{5}\right),\quad \dot w = (w^\* - w)/\tau_w$$

Mechanisms:

- A fatigue: $\dot F = O(1 - F)/20 - F/\tau_{fat}$ (builds in ~20 ticks of full overtime, recovers
  over $\tau_{fat}$); 2 params $k_{fat}, \tau_{fat}$.
- B orientation: $\dot S_e = \max(S - S_e, 0)/\tau_h + \min(S - S_e, 0)/\max(0.25\,\tau_h, 0.5)$,
  $S_e(0) = 20$; new staff ramp in over $\tau_h$, cuts act 4x faster; 1 param $\tau_h$.
- C returns: $\dot R = r_{ret} f_3 (1 - F_u) - R/\tau_{ret}$, outflow re-enters arrivals; 2 params.

Frozen: $\tau_a = 0.80$, $d_h = 0.087$ (p3 refit values), cohort constants as in min2 v4.
16 parameters in the module, 14-15 fitted per pair. Observables $w$, $q$, $f_3$. Reset: $W = q_0$,
$w = w_0$, everything else 0 except $S_e = 20$. RK4, 2 substeps; every rate at most 3 per tick.

## Results

LOO = mean of 5 held-out folds; per observable [wait, queue, discharges].

| model | LOO | LOO per obs | in-sample | in-sample per obs | compose fold | cost |
|---|---|---|---|---|---|---|
| AB | 0.801 | 0.834 0.870 0.700 | **0.843** | 0.891 0.907 0.730 | 0.687 0.700 0.657 (0.681) | 242.3 |
| AC | 0.804 | 0.832 0.874 0.705 | 0.841 | 0.888 0.905 0.731 | 0.702 0.725 0.680 (0.702) | 247.5 |
| **BC** | **0.811** | 0.843 0.873 0.718 | 0.836 | 0.887 0.892 0.730 | **0.742** 0.770 0.760 (0.757) | 255.9 |
| p3 (shipped, `_min_p3`) | **0.822** | 0.854 0.893 0.719 | 0.839 | 0.876 0.904 0.736 | 0.730 0.783 0.742 (0.752) | 438.6 |
| cohort min2 v4 | 0.801 | 0.832 0.859 0.712 | 0.842 | 0.886 0.909 0.730 | 0.703 0.769 0.723 (0.732) | 263.9 |

Per fold (LOO mean): hold_rec 0.894 / 0.896 / 0.893, pulse40 0.808 / 0.804 / 0.786,
mid40 0.807 / 0.805 / 0.803, multilevel200 0.816 / 0.812 / 0.817, compose 0.681 / 0.702 / 0.757
(AB / AC / BC). l0b_lin on the same folds 0.643, persistence 0.587.

Fitted theta (full data), nothing at a bound in any pair:

| param | AB | AC | BC |
|---|---|---|---|
| lam0 | 10.55 | 10.50 | 10.51 |
| c_a | 1.007 | 1.014 | 0.980 |
| c_t | 0.971 | 0.966 | 0.922 |
| chairs | 20.8 | 20.1 | 19.5 |
| beds | 12.6 | 12.3 | 11.5 |
| ot_gain | 0.796 | 0.794 | 0.653 |
| k_l | 0.0091 | 0.0100 | 0.0140 |
| tau_w | 7.86 | 8.42 | 8.02 |
| q_cap | 317.4 | 316.6 | 318.2 |
| k_s | 3.27e-3 | 1.95e-3 | 2.26e-3 |
| g_z | 4.31 | 4.75 | 4.51 |
| k_fat (A) | 0.295 | 0.440 | - |
| tau_fat (A) | 71.6 | 36.3 | - |
| tau_h (B) | 3.30 | - | 5.03 |
| r_ret (C) | - | 0.083 | 0.799 (hi 0.8) |
| tau_ret (C) | - | 234 | 148 |

BC's $r_{ret}$ sits 0.2% under its upper bound (the lab's at-bound test uses 0.1%, so it is not
flagged); with $F_u = 0$ in pulses, 80% of discharges come back after ~150 ticks, i.e. mostly after
the end of every run. It acts as a slow extra arrival stream, not as a measured return rate.

Eval sanity (4,000 ticks, all four categories): finite, 0.5-0.7 s each; outside the observed range
0 / 0 / 0.02 / 0.07 (AB), 0 / 0 / 0.01 / 0.06 (AC), 0 / 0 / 0 / 0.05 (BC); wait max 443-528 in
recovery / composition (the cohort showing after a drain, bounded by $g_z \cdot 150$).

## Revision tried and dropped (tags AC2, BC2)

Reading: in hold_rec and in compose 210-270 the steady discharges are 11.5 while every fit puts
$\lambda_0 = 10.5$ (the model settles at 10.5 and queue 15 vs 23). One patient per tick unexplained
with follow-up at 1 looked like returns leaking past a finite program. v2: $\dot R = r_{ret} f_3 -
\min(r_{ret} f_3, p_{cap} F_u) - R/\tau_{ret}$, with $g_z$ frozen at 4.5 to stay at 16 parameters.

| model | LOO | LOO per obs | in-sample | compose fold |
|---|---|---|---|---|
| AC2 | 0.805 | 0.835 0.872 0.708 | 0.841 | 0.708 |
| BC2 | 0.799 | 0.842 0.846 0.708 | 0.836 | 0.706 |

AC2: $r_{ret}$ 0.048, $p_{cap}$ 6.9 (the program never binds: same as no returns). BC2: $r_{ret}$ 0.74,
$p_{cap}$ 8.7, same slow-arrival use as BC. No gain; the module was put back to v1 so the AB/AC/BC
thetas and docs load. AC2/BC2 JSONs use the v2 layout and do not load into the module as it stands.
BC's compose fold dropped 0.757 -> 0.706 in BC2 with a nearly identical in-sample fit, so a single
fold moves by ~0.05 between multistart draws; LOO differences below ~0.01-0.02 are noise.

## Also considered: fatigue as the source of the post-overtime wait spike

Rejected without a fit. In compose the filtered wait implies a target of ~270 at $t = 188$, eight
ticks after overtime starts, and it then grows 2-3 per tick. An age that is 270 when it shows needs
a cohort that has been ageing since ~$t = 100$-$130$ (the long high-queue period), well before any
overtime. Fatigue starting at $t = 180$ cannot build that age at $g_z \le 6$. A fatigue-driven
Little estimate (dividing by the fatigued capacity) would also rise under the joint pulse at 270 (more
overtime, fewer staff), where the data fall from 390 to 60 as the pool refills. The wait spike
comes from the waiting history and the pool size, not from overtime.

## Verdict

- Ranking by LOO: BC 0.811 > AC 0.804 > AB 0.801; by in-sample AB 0.843 > AC 0.841 > BC 0.836.
  The order flips between the two and every gap is within the fold noise we measured (~0.05 on one fold).
  **The five runs do not identify the pair.**
- None beats the shipped p3 structure on LOO (0.822); all match or beat it in-sample (p3 0.839) because
  they carry the cohort wait (compose wait in-sample 0.90 vs 0.74). Same trade as the cohort model
  (0.801 / 0.842): the cohort buys compose and costs pulse40 / multilevel queue and wait.
- Best compose fold: BC 0.757 (wait 0.742) vs p3 0.752 (0.730), cohort 0.732 (0.703), AB 0.681 (0.687),
  AC 0.702 (0.702).
- Kept as candidates, not shipped: `plans/hospital_queue_hospital_queue_mech_{AB,AC,BC}.json` / `_doc.json`.

What the data can and cannot tell apart:
- Fatigue and orientation both explain the slow discharge ramp after staffing goes back to 20 at the
  end of a pulse (pulse40 at $t = 40$-70, compose at $t = 300$): every staffing rise came right after
  overtime. We need a staffing step without overtime, or equal overtime hours spaced differently at fixed staffing.
- Returns are never seen: follow-up is 1 in every recovery and drops only together with a pulse, or
  for 30 ticks in compose (225-255) with the queue at 23, where no extra discharges appear in the next 15
  ticks. C's fitted delays (150-230 ticks) are longer than what follows in any run, so C only reshapes slow
  arrivals. The brief's own test (add follow-up after the same discharge burst, with >150 ticks of tail) is the one that would pin it.
- The unexplained ~1/tick gap between the fitted arrivals (10.5) and steady discharges (11.5) is common
  to every pair and to p3; it is not evidence for C (v2 did not use it).
