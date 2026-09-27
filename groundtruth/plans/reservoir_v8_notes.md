# reservoir v8: structural pass at the organizer's scale (`gtlab/ode/reservoir_v8.py`)

No credits. Same 2 runs (`p1.hold_rec` 120 ticks, `p2.pulse200_200` 400 ticks). Every score below
is $\frac{1}{1+|e|/\sigma}$ at the calibrated $\sigma$ = [110.4, 0.672, 1.849, 0.0113]
(`plans/sigma_calibrated.json`), mean over ticks, then over observables, then over the two runs.
Columns: level, inflow, outflow, quality.

Ship doc: `plans/reservoir_reservoir_v8_v8_doc.json` (theta polished directly on the calibrated
score, written with `ode_lab --theta0 ... --starts 1 --nfev 1 --no-loo --tag v8`). **Its `post`
list is empty on purpose**: the inflow comes from the model (season + return flow), not from the
`exo_harmonic` rule (see leak 1). Checked through `rollout_from_blob` (runtime path, clip + post).

## Error budget of the shipped model (reservoir_min doc, 0.829)

| run | level | inflow | outflow | quality | mean |
|---|---:|---:|---:|---:|---:|
| p1.hold_rec | 0.935 | 0.833 | 0.874 | 0.683 | 0.831 |
| p2.pulse200_200 | 0.910 | 0.812 | 0.895 | 0.692 | 0.827 |

Noise ceilings (perfect model scored against noisy readings, noise s.d. 4.3 / 0.07-0.14 / 0.058 /
0.0055): level ~0.97, inflow ~0.93, outflow ~0.975, quality ~0.74-0.77. The three leaks:

1. **Inflow (−0.10 vs ceiling).** The doc's `exo_harmonic` post rule overwrites the model's inflow
   with a harmonic fitted on the 400-tick run: mean 11.43, which is the irrigation-inflated mean.
   The recovery mean is 11.28 (run 1: 11.279; run 2 after tick 240: 11.277) and under irrigation it
   is 11.62. The rule is biased +0.15 at recovery and −0.2 under irrigation, and it throws away the
   return-flow state. Dropping it alone: 0.848 → 0.866 (both polished, same model).
2. **Outflow at full pool (−0.08).** At full pool the model had outflow = inflow − 1.74 (seepage
   $0.00185 \times 941$). The data: inflow − outflow falls from 1.9 to ~1.0 over ~40 ticks after the
   pool fills (both runs), and a regression of outflow on inflow at full pool gives a gain of
   0.91-0.93, not 1 (best first-order spill lag 0.5 tick; longer lags fit worse).
3. **Level while filling/draining (−0.04 to −0.06).** The fill slows at high stage: loss ~1.6/tick at
   L ≈ 590, ~2.0 at 760, ~2.4 at 880 (from $\Delta L$ − inflow + 2), while the loss at full pool is
   only ~1.2. The model overshot the fill by 15-23 and undershot the drawdown by ~25.

Quality: run 2 also has a slow part the model lacked (window means below).

| run 2 ticks | 25-50 | 75-100 | 125-150 | 175-200 | 225-250 | 300-325 | 375-400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| quality | 0.955 | 0.940 | 0.935 | 0.933 | 0.945 | 0.944 | 0.941 |

Run 1 (recovery all along) sits at 0.950-0.953. Window s.e. ~0.001.

## Model v8

States $L$ level, $Q$ quality, $R$ return flow, $c$ clock, $P$ surcharge, $D$ stratification
deficit, $B$ bank storage.

$$\text{inflow}(c) = q_m + q_a \sin(2\pi c / 67.75)$$
$$\text{del} = \operatorname{softmin}_{0.5}\big(r + i,\ c_{out}(L/500)^{p_{out}}\big)$$
$$\text{net} = \text{inflow}(c) + R - \text{del} - \text{seep}\,L - f_{in}\,\text{inflow}(c) - k_b (L - B)$$
$$\dot L = \text{net}, \quad L \le 941.3 \text{ (clip)}, \qquad \dot B = (L - B)/\tau_b$$
$$\dot P = g(L)\,\operatorname{softplus}_{0.5}(\text{net}) - P/\tau_s, \quad \tau_s = 0.5 \text{ (fixed)}$$
$$\dot Q = (q_{base} - q_d D - Q)/\tau_q, \qquad \dot D = \big(\tfrac12(1-a) + \tfrac12 d - D\big)/\tau_d$$
$$\dot R = (k_{ret}\, i - R)/\tau_{ret}$$
$$y = [L,\ \text{inflow}(c) + R,\ \text{del} + P/\tau_s,\ Q]$$

$g(L) = \sigma((L - 941.3 + 2)/0.5)$. Reset: $L_0, Q_0$ from the readings, $B_0 = L_0$ (banks in
equilibrium), $R_0 = D_0 = 0$, $c_0 = 0$, $P_0 = \tau_s\, g(L_0)\,\operatorname{softplus}(\text{net at release 2})$.
RK4, 2 substeps. Constants: period 67.75 (both runs agree to ±0.1), phase 0 (fits gave 0.002-0.02),
$\tau_s = 0.5$, stage-area exponent 0.

What each term answers:
- **Bank storage** $k_b(L-B)$: water enters the banks while the stage rises and returns while it
  falls. It makes the loss large during a fast fill and lets it decay after the pool fills (leak 2
  and 3 with one mechanism, $\tau_b \approx 15$).
- **Inflow-proportional loss** $f_{in} = 0.044$: the full-pool gain of 0.92.
- **Deficit $D$**: quality falls slowly without aeration / with deep withdrawal and recovers slowly.
  Aeration and withdrawal depth moved together in our data (1,0 → 0,1), so the drive is split
  equally between them (min-norm choice); $q_{aer}$ is dropped, $D$ carries the aeration effect.

## Parameters (14 free; `plans/reservoir_reservoir_v8_v8.json`)

| name | value | bounds | role |
|---|---:|---|---|
| q_m | 11.270 | [8, 15] | season mean |
| q_a | 2.229 | [0.5, 5] | season amplitude |
| seep | 0.00067 | [1e-4, 0.02] log | loss per unit stage |
| c_out | 13.47 | [5, 30] log | discharge capacity at L = 500 |
| p_out | 0.347 | [0.05, 1] | head exponent |
| q_base | 0.9524 | [0.5, 1] | quality, fully mixed |
| tau_q | 3.53 | [0.67, 50] log | fast quality relaxation |
| k_ret | 0.0465 | [0, 0.3] | return per unit irrigation |
| tau_ret | 32.7 | [5, 50] log | return delay |
| q_d | 0.0198 | [0, 0.1] | quality loss at full deficit |
| tau_d | 115.6 | [5, 400] log | deficit time constant |
| k_b | 0.0082 | [0, 0.2] | bank exchange rate |
| tau_b | 15.1 | [3, 400] log | bank equilibration |
| f_in | 0.0443 | [0, 0.3] | inflow-proportional loss |

None at a bound.

## Scores

In-sample, calibrated $\sigma$ (runtime path):

| model | p1.hold_rec | p2.pulse200_200 | mean |
|---|---|---|---:|
| shipped reservoir_min (with post) | 0.935 0.833 0.874 0.683 | 0.910 0.812 0.895 0.692 | 0.829 |
| **v8** (no post) | 0.966 0.928 0.974 0.765 | 0.973 0.885 0.965 0.712 | **0.896** |

Step by step (each polished on the calibrated score with Powell, same protocol):

| step | in-sample | kept |
|---|---:|---|
| reservoir_min, post on, polished | 0.848 | |
| reservoir_min, post off, polished | 0.866 | yes (leak 1) |
| + stage-area exponent $\beta$ + surcharge lag $\tau_s$ | 0.882 | superseded |
| + deficit $D$ (drop $q_{aer}$, fix period) | 0.890 | yes |
| + bank storage, $\beta$ fixed 0 | 0.892 | yes |
| + $f_{in}$, phase and $\tau_s$ fixed (14 params) | **0.896** | yes |
| return flow from the *delivered* irrigation (head-limited) | 0.896 | no: no gain |

Leave-one-run-out, two protocols:

| protocol | fold | reservoir_min | v8 |
|---|---|---|---|
| lab fitter (Cauchy LSQ, calibrated $\sigma$, all free) | held-out hold_rec | 0.826 | **0.893** |
| | held-out pulse | 0.627 | **0.677** |
| | mean | 0.726 | **0.785** |
| Powell on the score; params the fold cannot see frozen at the full fit | held-out hold_rec | 0.822 | **0.893** |
| | held-out pulse | 0.838 | **0.875** |
| | mean | 0.830 | **0.884** |

The frozen set in the recovery-only fold: $c_{out}, p_{out}$ (the head limit never binds at release
2), $k_{ret}, \tau_{ret}$ (no irrigation), and $q_d, \tau_d$ for v8 / nothing extra for min. Left
free, those run to their bounds in that fold for both models (held-out pulse 0.70 min, 0.63 v8),
which says nothing about structure. Lab files: `plans/reservoir_reservoir_v8_v8lab*.json`,
`plans/reservoir_reservoir_min_cal_loo*.json` (those two docs still carry the lab's default post rule).

Eval sanity (4,000 ticks, all four categories): finite, 0.7 s each, level [241, 941.3], inflow
[9.0, 13.9], outflow [0.6, 18.0], quality ~0.9-1; 1-2% of ticks outside the observed range ±5%.
Outflow up to 18 is the head law at full pool with a large request (13.47 × 1.88^0.347 = 16.8 plus
spill), an extrapolation we have not observed.

## Remaining leaks

1. **Inflow under irrigation (run 2: 0.885 vs ceiling ~0.93).** In the pulse the season has
   amplitude 2.15 (2.25 at recovery) and a +0.05 rad phase shift; the residual oscillates ±0.2 with
   the season period. A seasonal component in the return is not something one constant irrigation
   level can pin down; the head-limited return (return ∝ delivered irrigation) did not help.
2. **Quality (run 2: 0.712).** After the pulse, quality recovers to 0.947 in ~40 ticks and then
   drifts to 0.941 at full pool; a single slow state cannot do both. Run 1 also reads ~0.003 lower
   at full pool than while filling. Effects are 0.003-0.006 against $\sigma$ 0.011 and noise 0.0055.
3. **Attribution.** The deficit drive is split 50/50 between aeration and withdrawal depth, and the
   return flow is charged to irrigation only: composition tests that move one of these alone will
   show whether the split is right. A run with aeration off at shallow withdrawal would settle it.
