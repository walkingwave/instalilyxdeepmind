# hospital_queue v8: structural revision at the organizer's scale (Sat Sep 26)

No credits spent. Data: 6 runs, 1,070 ticks (hold_rec 120, pulse40 80, mid40 40, multilevel200 200,
compose 330, pulse_long_recovery 300). Score = mean over runs of `metric.score(Y, run.Y, sigma)` with
$\sigma$ = calibrated organizer scale [22.94, 27.54, 2.149] (wait, queue, discharges), queue capped at 333
(doc `post` rule). Starting point: `hospital_queue_p3` fitted at 1.5x that scale
(`plans/hospital_queue_hospital_queue_p3_cal4AB_doc.json`), calibrated in-sample **0.6927**.

## 1. Error budget of p3 (calibrated scale)

| run | wait | queue | disch | mean |
|---|---:|---:|---:|---:|
| hold_rec | 0.930 | 0.667 | 0.754 | 0.784 |
| pulse40 | 0.800 | 0.707 | 0.558 | 0.688 |
| mid40 | 0.892 | 0.669 | 0.412 | 0.658 |
| multilevel200 | 0.634 | 0.838 | 0.605 | 0.692 |
| compose | 0.553 | 0.619 | 0.570 | 0.581 |
| pulse_long_recovery | 0.854 | 0.610 | 0.797 | 0.754 |
| **mean** | 0.777 | 0.685 | 0.616 | **0.6927** |

Loss share (tick weighted): wait 0.29, queue 0.35, discharges 0.37. The three largest sources:

1. **One service capacity for every history.** p3 serves 11.0/tick under the recovery action in every run
   ($r_t = c_t\cdot 20\cdot 0.6 = 11.03$, binding). That matches the post-pulse plateau (discharges
   11.04 for 170 ticks) but drains hold_rec over 120 ticks where the data drain 100 -> 23 in 10 ticks at
   ~19/tick, and it underpredicts the steady 11.47 of hold_rec and compose 210-270 (disch 0.82 instead of
   0.98 on those ticks).
2. **Discharges under a pulse are waves, not a flow.** At staffing 5 the data are 0 on 70-75% of ticks
   with bursts of 5-17. The best constant for $1/(1+|e|/\sigma)$ there is 0 (0.79-0.80 per segment) while the
   smooth mean flow (1.6-1.9) scores 0.51.
3. **Compose wait spike** (wait 300-400 with the queue at its floor, ticks 181-270): ~0.014 of the overall
   score. No structure we have explains it without breaking the plain-pulse runs; left alone.

Also: p4 plateau at queue 92 (p3 drains to 36): with arrivals 11.47 and capacity 11.04, the balance
$\lambda - \mu = k_l W$ gives $W \approx 0.43/0.0065 \approx 66$ and queue $\approx 66 + 25 = 92$, wait
$W/\mu \approx 5$ (data 4.55). So the plateau is a capacity just below arrivals plus abandonment; it needs a
capacity loss that survives >= 260 ticks after the pulse and does not exist in a fresh start.

## 2. Structures tried

All keep the p3 pipeline (waiting $W$ -> assessment $A_1, A_2$ in finite chairs -> treatment $T$ in finite
beds -> discharges $f_3$; arrivals gated at $q_{cap}$; abandonment $k_l W$). $K = 3$, N_SUB = 2, RK4.

**v8** (`hospital_queue_v8`, 14 params): orientation + slow fatigue.
$$\text{eff} = S_e(1 + g_{ot}O)(1 - k_{fat}F),\quad
\dot S_e = \frac{\max(S - S_e, 0)}{\tau_o} + \frac{\min(S - S_e, 0)}{0.5},\quad S_e(0) = 20$$
$$\dot F = \frac{O(1 - F)}{20} - \frac{F}{\tau_f}$$

**v8b** (15): v8 + wave-completion discount on the *reported* discharges:
$$\hat y_D = f_3\,\frac{r_t^4}{r_t^4 + r_b^4},\qquad r_t = c_t\,\text{eff}\,(1 - D)$$
With little treatment work per tick the typical tick sees no completion; the discount moves the
prediction to the mode there and is ~1 at normal staffing ($r_t \approx 12$ in recovery).

**v8c** (16): v8 + deterioration while waiting. $\dot D_w = k_{det}(W - D_w) - (f_1 + k_l W)\frac{D_w}{W+1}$,
service rates divided by $1 + k_d D_w/(W+1)$. Bistable (a congested hospital stays congested under the
action that drains a fresh one), which is what p4 vs hold_rec looks like. **v8d** (16): v8c + discount,
no orientation.

**v8e** (16): v8b with partial orientation, effective staff $S_e + \rho\max(S - S_e, 0)$ (new staff work
at $\rho$ until trained over $\tau_o$).

**v8f** (16): v8b with an asymmetric wait estimate:
$$\dot w = \frac{\max(w^* - w, 0)}{\tau_{up}} + \frac{\min(w^* - w, 0)}{\tau_w},\qquad w^* = \frac{W}{f_1 + 1}$$

Fitting: `scripts/ode_lab.py --sigma-cal 1.5` (cauchy least squares) gives a first theta; we then polish the
same theta directly on the calibrated score (Powell on the bounded unit cube, mean over runs of the
calibrated score, queue capped at 333) from several starts, and write the lab doc from the polished theta
(`--theta0 <polished> --free tau_f --starts 1 --nfev 1`, which leaves theta unchanged; checked, max rel.
difference 0).

## 3. Results (calibrated in-sample, mean over runs)

| model | lab fit | best polish | notes |
|---|---:|---:|---|
| p3 (current) | 0.6927 | 0.7052 | polish alone buys little |
| v8 | 0.7114 | | $\tau_o$ 10.6, $k_{fat}$ 0.08, $\tau_f$ 4,640 |
| v8b | 0.7172 | 0.7392 | $r_b$ 3.1 |
| v8c | 0.7144 | 0.7244 | bistable start; drains p4 (queue 0.55) |
| v8d | 0.7029 | 0.7271 | same trade: hold_rec 0.95, p4 queue 0.51-0.66 |
| v8e | | 0.7231 / 0.7411 | $\rho \to 0.003$ when started from v8b: partial orientation rejected |
| **v8f** | | **0.7461** | two optima: 0.7445 ($\tau_{up}$ 25.6) and 0.7461 ($\tau_{up}$ 14.5) |

Best: **v8f**, `plans/hospital_queue_hospital_queue_v8f_best_doc.json` (lab json `_v8f_best.json`).

| run | wait | queue | disch | mean | vs p3 |
|---|---:|---:|---:|---:|---:|
| hold_rec | 0.937 | 0.890 | 0.829 | 0.885 | +0.101 |
| pulse40 | 0.943 | 0.768 | 0.692 | 0.801 | +0.113 |
| mid40 | 0.879 | 0.740 | 0.393 | 0.671 | +0.013 |
| multilevel200 | 0.602 | 0.837 | 0.627 | 0.689 | -0.004 |
| compose | 0.501 | 0.647 | 0.622 | 0.590 | +0.009 |
| pulse_long_recovery | 0.915 | 0.783 | 0.825 | 0.841 | +0.087 |
| **mean** | 0.796 | 0.778 | 0.665 | **0.7461** | **+0.053** |

Theta: lam0 11.61, c_a 1.12, c_t 1.063, tau_a 1.32, chairs 53.9, beds 22.6, ot_gain 0.62, k_fat 0.133,
k_l 0.0179, tau_w 42.1, q_cap 319.1, d_h 0.303, tau_o 15.6, tau_f 4,988, r_b 3.76, tau_up 14.5.
Nothing at a bound by the lab's 0.1% test, but $\tau_f$ sits at 99.8% of its upper bound (5,000): fatigue
does not recover within any run, i.e. the data only say "no recovery within 260 ticks".

Eval sanity (lab, 4,000-tick eval-shaped schedules): finite, 0.6 s each, 0% outside the observed range in
all four categories; wait max 140 / 77 / 56 / 88, queue 23-328, discharges 0-20.5. The sustained schedule
of that draw is a pulse-like hold, where discharges sit at ~0 by design (the wave discount).

## 4. Leave-one-run-out

Two versions. (i) Polish-LOO at the calibrated scale: the direct-score polish refitted on 5 runs
(1,000 evaluations), started from the lab's cauchy fit on all runs (the same start for every family, a
mild leak shared by all rows). (ii) The lab's own LOO: cauchy fit from the module defaults at 1.5x the
calibrated scale, scored at 1.5x (comparable with the p3 lab json's 0.701).

| held out | p3 (i) | v8b (i) | v8f (i) | p3 (ii) | v8b (ii) | v8f (ii) |
|---|---:|---:|---:|---:|---:|---:|
| hold_rec | 0.791 | 0.846 | 0.843 | 0.750 | 0.745 | 0.735 |
| pulse40 | 0.648 | 0.685 | 0.699 | 0.738 | 0.764 | 0.754 |
| mid40 | 0.627 | 0.631 | 0.629 | 0.717 | 0.711 | 0.726 |
| multilevel200 | 0.655 | 0.679 | 0.675 | 0.716 | 0.752 | 0.740 |
| compose | 0.545 | 0.602 | 0.605 | 0.542 | 0.601 | 0.636 |
| pulse_long_recovery | 0.690 | 0.771 | 0.772 | 0.744 | 0.744 | 0.687 |
| **mean** | **0.659** | **0.702** | **0.704** | **0.701** | **0.719** | **0.713** |

Both LOO means improve on p3 (+0.045 calibrated, +0.012 at 1.5x). The one fold that drops is
pulse_long_recovery under the lab fit (ii): without that run the cauchy fit has no evidence that capacity
stays low after a pulse, so it drains the queue (queue 0.50 vs p3 0.60). p3 gets that fold by serving
11/tick everywhere, the same choice that costs it hold_rec. Under the direct-score polish (i) the fold is
+0.08 over p3. v8b and v8f are within fold noise of each other on LOO; v8f is kept for its in-sample gain
(+0.007).

## 5. What worked, what did not

- **Worked:** long-memory fatigue (capacity after any overtime stays ~13% lower; gives the p4 plateau and
  the post-pulse discharges 11.04) together with a fresh start at full effectiveness (hold_rec drains
  fast); the wave discount on discharges (pulse40 disch 0.56 -> 0.69, p4 0.80 -> 0.83); a wait estimate
  that rises faster than it forgets (multilevel and pulse rises vs the slow post-pulse decay).
- **Did not:** deterioration-driven bistability (it can hold p4 congested, but the fit never finds a point
  that also keeps pulse40 and compose; best 0.727); partial orientation ($\rho \to 0$).
- **Biggest remaining loss:** compose, 0.590. Two parts: (a) the queue in 40-200 at staffing 20 (the model
  drains faster than the data: capacity after the staffing step 7.2 -> 20 is ~13-14/tick in the data vs ~19
  fresh), and (b) the wait spike 181-270 (0.07 per tick where it shows). Next: mid40 discharges (0.39,
  waves at staffing 10.5 that are not zero-mode) and multilevel wait at staffing 1 (predicted 105 vs 195).
