# hospital_queue y: mechanism identification on a brief-faithful base (Sun Sep 27, evening)

No credits spent. Training data: 7 runs, 1,370 ticks (hold_rec 120, pulse40 80, mid40 40,
multilevel200 200, compose 330, pulse_long_recovery 300, p6.voi 300). **p6.exam (400 ticks,
test-shaped) is never fitted**: every lab call runs `scripts/ode_lab.py` through a wrapper that drops
that run from the loaded list, so it is a clean held-out exam for every model below (p3 and v9g
included). Fits: `ode_lab.py --system hospital_queue --family <fam> --mech <pair> --budget 300 --starts 8
--nfev 60 --sigma-cal 1.0 --free <base + active pair>`; no polishing. Scores at the calibrated organizer
scale $\sigma$ = [22.94, 27.54, 2.149] (wait, queue, discharges). LOO = mean of the 7 held-out folds
(fold noise ≈ 0.02). Queue cap 333 kept (`le_const` post rule in every doc).

Baselines recomputed on the same 7 runs and scale: **p3 structure LOO 0.670, v9g structure 0.700**
(the old 6-run numbers were 0.634 / 0.639; the p6.voi run is an easy fold for everyone).

## 1. What the data say before any fit (flow balance per constant-control segment)

| evidence | numbers | reading |
|---|---|---|
| fresh start at recovery (hold_rec) | queue 76 → 23 in 12 ticks, ≈ 18 discharges/tick after the 3-tick fill | fresh capacity at staffing 20 ≥ 18/tick |
| after staffing 7.25 → 20, **no overtime** (compose 30-180) | 9.3-12.5 discharges/tick with the queue at 150-300 | capacity after a staffing rise is well below fresh: orientation (B) |
| after the pulse (p4 and p6.voi, same schedule) | 10.6 (ticks 40-126), 11.4 (126-213), 11.1 (213-300); plateau queue 92, wait 4.5 | capacity ≈ 11.1 < arrivals 11.5 for ≥ 260 ticks after 40 ticks at staffing 5, overtime 1 |
| overtime 0.85 at staffing 20 (compose 180-210) | 16.2/tick, queue 161 → 23 | overtime gain ≈ ×1.4 while it lasts |
| follow-up 0.15 for 30 ticks at queue 23 (compose 225-255) | discharges 11.49-11.50 through tick 270 | no returns within 45 ticks: C, if active, is slow (> 45 ticks) |
| arrivals at recovery | 11.50 in hold_rec; plateau balance 11.03 + leaving | $\lambda_0 \approx 11.5$, leaving ≈ 0.01 W |

## 2. Structures

All keep p3's pipeline (waiting → two assessment stages in finite chairs → treatment in finite beds →
discharges; a finished assessment holds its chair when beds are full; arrivals gated at $q_{cap}$;
$K = 3$, N_SUB = 2, RK4, batched f/h/x0, every observed initial used: wait and queue set the state,
services and the program start empty). Mechanisms are switches; only the active pair's parameters are fitted.

**Base 1 (`hospital_queue_y`, 16 base params).** Case types: non-elective waiting $W_n$ (arrivals
$\lambda_0$) and elective waiting $W_e$ (arrivals = elective_scheduling); admissions shared by weight
$W_n(1+U) : e_w W_e$ (urgent priority favours urgent/routine admissions); electives cancelled at $k_e W_e$,
non-electives leave at $k_l W_n\,(1 + b_U(U - 0.6))^+$. Follow-up diverts shared staff:
$S_h = S^*(1 - \phi_f F_u)$. $\text{eff} = S_h(1 + gO)[\cdot]_A$, $r_a = c_a\,\text{eff}\,D/(D+d_h)$,
$r_t = c_t\,\text{eff}\,(1-D)$. Reported wait rises to $W/(f_1+1)$ over $\tau_{up}$, falls over $\tau_w$.

- A fatigue: $\dot F = (O - F)/\tau_f$, eff × $(1 - k_{fat}F)$ (base 1) or $(1 - \tfrac{g}{1+g}F)$ (y3-y6 and
  base 2: overtime exactly neutral once $F = O$, since $(1+g)(1 - \tfrac{g}{1+g}) = 1$).
- B orientation: $S^* = S_e + \rho(S - S_e)^+$, $\dot S_e = (S-S_e)^+/\tau_o + (S-S_e)^-/0.5$, $S_e(0) = 20$.
- C returns: $\dot R = r_{ret}\,(f_3 - \min(f_3, p_{cap}F_u)) - R/\tau_{ret}$, returns re-enter $W_n$.

**y3** = base 1 + neutral fatigue law, $b_U \ge -2.5$. **y4** = y3 + class-aware reported wait
$w^* = \tfrac{N}{f_1+1}\cdot\tfrac{W_n/(1+U) + W_e/e_w}{W}$, $N = W_n(1+U) + e_w W_e$ (each class served at
its admission share). **y5** = y3 + a handover cost inside B: eff × $(1 - \eta|S - S_l|/20)$,
$\dot S_l = (S - S_l)/\tau_h$. **y6** = y3 with $\tau_o \le 200$.

**Base 2 (`hospital_queue_y2`, 19 base params).** Three explicit case types with different treatment
work: routine $W_r$ ($\lambda_0(1-p_u)$), urgent $W_u$ ($\lambda_0 p_u$), elective $W_e$; waiting routine
patients deteriorate into urgent ones at $k_d W_r$; urgent work $m_u$ (routine and elective 1);
admission weights $1 : e^{a_U U} : e_w$. Work is carried through assessment ($Z_A$) into beds ($X$);
completions by processor sharing, $w_d = \min(r_t, KX)$, $f_3 = w_d\,T/X$. Returns re-enter as urgent
cases (the "returning case mix"). A uses the neutral law.

## 3. Results

| model | pair | free | LOO | hold_rec | pulse40 | mid40 | multilevel | compose | p4 | voi | in-sample | exam [w q d] | ot0 | ot1 | diag 0.8 | pulse | agree holds / sus |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| p3 (lab refit) | as shipped | 12 | **0.670** | 0.679 | 0.713 | 0.655 | 0.654 | 0.517 | 0.757 | 0.715 | 0.692 | 0.644 [0.62 0.75 0.57] | 53/309/4.61 | 74/314/2.93 | 102/318/1.54 | 151/327/0.76 | 0.812 / 0.755 |
| v9g (lab refit) | as shipped | 15 | **0.700** | 0.781 | 0.708 | 0.677 | 0.669 | 0.537 | 0.778 | 0.752 | 0.727 | 0.668 [0.70 0.73 0.57] | 40/312/4.53 | 40/312/4.53 | 89/319/1.51 | 107/328/1.18 | 0.855 / 0.743 |
| y | AB | 20 | **0.717** | 0.748 | 0.729 | 0.671 | 0.661 | 0.587 | 0.827 | 0.795 | 0.741 | 0.666 [0.67 0.75 0.58] | 39/303/4.56 | 26/283/6.44 | 88/312/1.52 | 86/328/1.77 | 0.695 / 0.669 |
| y | AC | 21 | **0.679** | 0.722 | 0.640 | 0.645 | 0.635 | 0.555 | 0.791 | 0.766 | 0.721 | 0.653 [0.63 0.76 0.57] | 40/308/4.54 | 37/307/4.93 | 90/315/1.51 | 94/330/1.57 | 0.742 / 0.686 |
| y | BC | 21 | **0.716** | 0.794 | 0.723 | 0.649 | 0.659 | 0.571 | 0.821 | 0.793 | 0.743 | 0.663 [0.65 0.76 0.57] | 40/307/4.57 | 22/285/8.08 | 90/314/1.52 | 78/328/2.10 | 0.650 / 0.630 |
| **y3** | **AB** | 19 | **0.721** | 0.789 | 0.723 | 0.664 | 0.654 | 0.591 | 0.826 | 0.797 | 0.712 | **0.674** [0.69 0.77 0.57] | 55/311/4.53 | 55/311/4.53 | 104/319/1.58 | 122/330/1.27 | **0.872** / 0.706 |
| y3 (12 starts, other optimum) | AB | 19 | **0.723** | 0.789 | 0.739 | 0.664 | 0.654 | 0.591 | 0.826 | 0.797 | 0.732 | 0.610 [0.60 0.68 0.56] | 35/300/5.37 | 35/300/5.37 | 83/312/1.79 | 93/331/1.69 | 0.692 / 0.652 |
| y3 | AC | 20 | **0.685** | 0.750 | 0.667 | 0.637 | 0.623 | 0.557 | 0.772 | 0.787 | 0.721 | 0.653 [0.66 0.73 0.58] | 35/294/4.52 | 35/294/4.52 | 82/308/1.51 | 93/331/1.44 | 0.742 / 0.782 |
| y4 | AB | 19 | **0.718** | 0.829 | 0.657 | 0.687 | 0.676 | 0.587 | 0.802 | 0.785 | 0.743 | 0.680 [0.69 0.78 0.57] | 40/305/4.65 | 40/305/4.65 | 92/313/1.55 | 112/333/1.26 | 0.780 / 0.711 |
| y4 | BC | 21 | **0.717** | 0.839 | 0.693 | 0.664 | 0.670 | 0.538 | 0.824 | 0.790 | 0.747 | 0.684 [0.72 0.76 0.57] | 45/309/4.54 | 26/297/8.29 | 101/315/1.51 | 90/327/2.26 | 0.671 / 0.636 |
| y5 | AB | 21 | **0.726** | 0.806 | 0.745 | 0.654 | 0.672 | 0.577 | 0.827 | 0.802 | 0.744 | 0.682 [0.68 0.80 0.57] | 37/296/4.69 | 37/296/4.69 | 87/309/1.56 | 107/331/1.28 | 0.729 / 0.689 |
| y5 | BC | 23 | **0.725** | 0.843 | 0.732 | 0.671 | 0.654 | 0.549 | 0.824 | 0.806 | 0.746 | 0.663 [0.67 0.76 0.57] | 43/318/4.59 | 28/310/7.33 | 97/325/1.53 | 75/329/2.30 | 0.696 / 0.635 |
| y6 | AB | 19 | **0.724** | 0.811 | 0.723 | 0.673 | 0.676 | 0.560 | 0.826 | 0.797 | 0.744 | 0.674 [0.66 0.79 0.58] | 38/299/4.66 | 38/299/4.66 | 86/310/1.55 | 109/332/1.21 | 0.739 / 0.702 |
| y2 | AB | 22 | **0.652** | 0.541 | 0.725 | 0.590 | 0.679 | 0.511 | 0.775 | 0.742 | 0.714 | 0.566 [0.57 0.56 0.56] | 43/322/4.80 | 43/322/4.80 | 106/333/1.41 | 117/333/1.19 | 0.793 / 0.672 |
| y2 | AC | 23 | **0.658** | 0.585 | 0.677 | 0.640 | 0.649 | 0.546 | 0.752 | 0.755 | 0.725 | 0.671 [0.69 0.75 0.57] | 54/311/4.28 | 54/311/4.28 | 135/321/1.18 | 122/332/1.50 | 0.797 / 0.701 |
| y2 | BC | 24 | **0.717** | 0.795 | 0.718 | 0.666 | 0.677 | 0.536 | 0.828 | 0.799 | 0.746 | 0.667 [0.65 0.77 0.57] | 48/311/3.95 | 28/299/7.17 | 110/321/1.25 | 77/327/2.31 | 0.709 / 0.623 |
| p3 public doc (u010b) | | | | | | | | | | | 0.678 | 0.654 [0.69 0.70 0.56] | 52/313/4.56 | 51/313/4.60 | 116/320/1.52 | 137/329/1.20 | 1 / 1 |
| v9g public doc (u013) | | | | | | | | | | | 0.711 | 0.664 [0.68 0.75 0.57] | 38/312/4.60 | 38/312/4.60 | 87/319/1.53 | 104/328/1.20 | 0.837 / 0.845 |

Map = end of 4,000-tick holds at staffing 8, other controls at recovery (wait / queue / discharges),
same holds as `plans/longhold_audit.md`; "agree" = calibrated score of the model against p3's public
predictor on those holds (+ an overtime-then-recovery hold) and on 8 eval-shaped sustained schedules.
In-sample = the lab's full fit on the 7 runs. y5 AC is y3 AC exactly (the handover terms sit inside B), so
it was not refitted.

Lab jsons / docs: `plans/hospital_queue_hospital_queue_<family>_<tag>[_doc].json`, tags `yAB yAC yBC
y3AB y3AB12 y3AC y4AB y4BC y5AB y5BC y6AB y2AB y2AC y2BC`; baselines `p3_ybase`, `v9g_ybase`. Every
doc: 4,000-tick eval rollouts finite, 0 % outside the observed range, ≤ 0.7 s per episode (y3 AB and
y5 AB also run through `gtlab.runtime.infer.predict_episode`: 0.6 s per 4,000-tick episode, finite).

## 4. Mechanism identification

- **B (orientation) is active.** Dropping it costs LOO on both bases: base 1 AC 0.679 vs AB 0.717 /
  BC 0.716; y3 AC 0.685 (six parameters at bounds) vs AB 0.721; base 2 AC 0.658 vs BC 0.717. The
  compose run (staffing 7.25 → 20 with no overtime at all, capacity stuck at 9-12.5 against ≥ 18 fresh)
  is the direct evidence: fatigue cannot act there.
- **A vs C is not identified by the owned runs or the exam.** AB and BC tie on LOO on every base where
  both fitted well (y 0.717 / 0.716, y4 0.718 / 0.717, y5 0.726 / 0.725; base 2 AB found a poor optimum),
  and the exam splits them both ways (y5 AB 0.682 vs BC 0.663; y4 AB 0.680 vs BC 0.684). It is decided by
  the long-hold evidence: without A, overtime is a permanent gain in every BC fit (wait 22-28, discharges
  7.2-8.3 at overtime 1 vs 40-48 / 4.0-4.6 at 0), the property that cost u012 its sustained band, and the
  neutral law the public ranks first needs fatigue. Several BC fits also keep a congested state after an
  overtime-then-recovery hold (queue 125-187 at tick 4,000), where compose shows the queue back at 23
  after the same sequence. C's fitted return delays (128-400 ticks) sit beyond every follow-up change we
  own: it acts as a slow extra arrival stream, not as a measured return.
- **Pick: A + B** (fatigue + orientation), returns inactive.
- Two optima inside y3 AB with near-equal LOO show what the exam adds: $\tau_o$ = 16 with slow neutral
  fatigue ($\tau_f$ = 227) scores 0.674 on the exam; $\tau_o$ = 1,996 (new staff never fully effective,
  $\rho$ 0.76) fits the 7 runs better in-sample (0.732 vs 0.712) and scores 0.610. So the post-pulse
  plateau is fatigue that clears over a few hundred ticks, not staff that never learn. y6 bounds
  $\tau_o \le 200$ and lands on the fast-orientation side ($\tau_o$ 15, $\tau_f$ 342).

## 5. Decision

**Recommended doc: `plans/hospital_queue_hospital_queue_y3_y3AB_doc.json`** (family `hospital_queue_y3`,
pair AB, 19 fitted parameters, nothing at a bound).

| | p3 public | v9g public | y3 AB |
|---|---:|---:|---:|
| lab LOO at 1.0σ (structure refit on the 7 runs) | 0.670 | 0.700 | **0.721** |
| exam (held out, never fitted) | 0.654 | 0.664 | **0.674** |
| long-hold map ot 0 / ot 1 (w/q/d) | 52/313/4.56 · 51/313/4.60 | 38/312/4.60 · 38/312/4.60 | **55/311/4.53 · 55/311/4.53** |
| diag 0.8 / pulse hold | 116/320/1.52 · 137/329/1.20 | 87/319/1.53 · 104/328/1.20 | 104/319/1.58 · 122/330/1.27 |
| agreement with p3 on the holds | 1 | 0.837 | **0.872** |

Theta: $\lambda_0$ 11.49, $c_a$ 1.114, $c_t$ 1.019, $\tau_a$ 0.786, chairs 40.4, beds 12.9, $g$ 0.811,
$k_l$ 0.0134, $\tau_w$ 36.0, $q_{cap}$ 318.1, $d_h$ 0.360, $\tau_{up}$ 13.9, $e_w$ 0.375, $k_e$ 0.014,
$b_U$ −1.87, $\phi_f$ 0.034, $\tau_f$ 227, $\tau_o$ 16.4, $\rho$ 0.296 (C parameters unused).
Reading: overtime adds 81 % work while it lasts and is exactly neutral once fatigue has caught up
(over ~227 ticks); new staff work at 30 % and are oriented over ~16 ticks; urgent priority lowers
routine leaving (factor 2.1 at U = 0 down to 0.25 at U = 1); electives are admitted at 0.375 the
weight of non-electives.

Why y3 AB over the higher-LOO y5 AB / y6 AB (0.726 / 0.724, exam 0.682 / 0.674): their long-hold wait sits
at 37-38 under the staffing-8 holds, the level v9g shipped with (sustained 0.649 vs p3's 0.677 at 52).
y3 AB is the only candidate with LOO ≥ 0.70 that keeps p3's steady-state map (wait 55 vs 52, overtime
neutral, stress discharges ≥ 1.2). The LOO differences among y3 / y5 / y6 AB (≤ 0.005) are inside fold
noise. Alternate for a sequence-heavy bet: y5 AB (`plans/hospital_queue_hospital_queue_y5_y5AB_doc.json`).

Forecast (public gain = 0.6 × LOO gain over the p3 structure): y3 AB 0.6 × (0.721 − 0.670) = **+0.031**,
i.e. ≈ 0.72 from p3's 0.690; the exam gain over p3's shipped doc (+0.020) agrees in sign and size.
Caveat: v9g also had +0.030 LOO over the p3 structure and lost 0.018 on the board through its wait level;
y3 AB's map is p3's, so that failure mode is controlled, not excluded.

## 6. Open risks

- Saturated wait is still low on the exam: under interior pulse holds the data sit at 42-70, y3 at 27-48,
  p3 at 32-63. No wait definition we tried offline on y3's states (W/f1, W/f3, (W+A)/f3, q/f3, each with
  its best filter on the training runs) beat the current one on the exam.
- Discharges stay at ≈ 0.57 everywhere (batchy completions; two realizations of the mid40 schedule differ
  by 10-20 per tick in the first 20 ticks).
- compose wait (0.51-0.59 in every model): the wait climbing 2-3/tick at queue 23 after the overtime
  block is not explained by any structure here.
- $\tau_f$ ≈ 230 means fatigue lingers ~700 ticks after a pulse; the owned runs stop 260 ticks after the
  last pulse. The full fit can also land on the never-oriented optimum (see §4); y6's bound removes it
  if a refit is needed.
- The recommended doc's in-sample (0.712) is below its own LOO folds: the full fit ended in a slightly
  worse local optimum within the 300 s budget. We keep it because its map and exam are the best of the
  AB fits; a refit is not guaranteed to land in the same place.
