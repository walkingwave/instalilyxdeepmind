# reservoir: minimal grey-box ODE (`gtlab/ode/reservoir_min.py`)

Data: 2 real runs, 520 ticks. `p1.hold_rec` (120 ticks at the recovery action) and
`p2.pulse200_200` (200 ticks at the pulse action, then 200 at recovery). Lab harness:
`scripts/ode_lab.py --system reservoir --family reservoir_min --budget 240 --starts 10 --nfev 60`
(leave-one-run-out, full fit, 4,000-tick eval-shaped rollouts). Sigma proxy (lab metric):
level 257.7, inflow 1.57, outflow 4.32, quality 0.017.

## What the data say (before any fitting)

- Level is a capped integrator. Under release 2 it fills at 7-10/tick and stops at 941 (tail
  means 941.2 and 941.5 in the two runs, noise s.d. 5). The level budget residual
  $\Delta L - (\text{inflow}-\text{outflow})$, binned by level below the cap, gives a loss of 0.6 at
  $L\approx300$, 1.0 at 500, 1.6-1.8 at 600-700: roughly proportional to $L$ with
  $k\approx 0.002$/tick (seepage). At this noise (bin s.e. 0.3-0.9) nothing separates a linear
  loss from a mild power law.
- Outflow at the recovery action equals the request exactly (2.00 +- 0.01) while the pool is
  below the cap. At the pulse (request 12+8 = 20) only 13.7 is delivered, decaying to 11.1 as the
  level falls 536 -> 283. The delivered amount tracks $1.69\,L^{1/3}$ to +-0.05 at every
  checkpoint (536: 13.7, 394: 12.37 vs 12.36, 338: 11.75 vs 11.71, 283: 11.07 vs 11.1). A
  head-limited discharge explains the pulse with no hidden state; a fouling state would need
  the same numbers plus two parameters the two control vectors cannot identify.
- When full, outflow = inflow - seepage (spill). With the sill height chosen so the plateau is
  941, any first-order spill rate $\geq 0.7$/tick gives the same outflow rmse (0.35): there is no
  measurable spill lag, so the spill can be algebraic.
- Inflow is a reset-locked season, $11.4 + 2.19\sin(2\pi t/67.9 + \phi)$, noise s.d. 0.07. The
  season-subtracted residual has a control-dependent part: +0.25 under the pulse (irrigation 8),
  building over ~80 ticks, back to -0.15 within ~30 ticks of the return to recovery. Amplitude
  0.4 against noise 0.07: the brief's delayed return from irrigated land. (Run 1 also shows a
  +1.0, +0.45 blip on its first two ticks that no run-2 tick shows; we leave it.)
- Quality relaxes from the initial reading (0.87, 0.79) to 0.95 in ~3 ticks in both runs, at
  both actions, and then sits at 0.94-0.96. The action moves it by ~0.01 at most.
- The initial inflow and outflow readings carry no state: both jump to their algebraic values at
  the first tick (outflow 6.44 -> 2.02, 4.63 -> 13.71).

## Model (v4, shipped)

States $L$ (level), $Q$ (quality), $R$ (return flow), $c$ (clock, ticks since reset; the
observation after tick $t$ sees $c = t+1$, the phase $\phi$ absorbs the offset). Controls $r$
release, $i$ irrigation, $d$ withdrawal depth, $a$ aeration.

$$\text{inflow}(c) = q_m + q_a \sin\!\left(\tfrac{2\pi c}{\text{per}} + \phi\right) + R$$
$$\text{cap}(L) = c_{out}\,(L/500)^{p_{out}}, \qquad
  \text{del} = \operatorname{softmin}_{0.5}(r + i,\ \text{cap}(L))$$
$$\dot L = \text{inflow}(c) - \text{del} - \text{seep}\,L, \qquad L \le L_{full} = 941.3 \text{ (clip)}$$
$$\dot Q = (q_{base} + q_{aer}\,a - Q)/\tau_q, \qquad
  \dot R = (k_{ret}\, i - R)/\tau_{ret}, \qquad \dot c = 1$$
$$\text{spill} = g(L)\,\operatorname{softplus}_{0.5}\!\big(\text{inflow}(c) - \text{del} - \text{seep}\,L\big),
  \quad g(L) = \sigma\!\left(\tfrac{L - L_{full} + 2}{0.5}\right)$$
$$y = [\,L,\ \text{inflow}(c),\ \text{del} + \text{spill},\ Q\,]$$

with $\operatorname{softmin}_e(a,b) = \tfrac12(a+b-\sqrt{(a-b)^2+e^2})$,
$\operatorname{softplus}_e(z) = \tfrac12(z+\sqrt{z^2+e^2})$, $\sigma$ the logistic. Reset:
$L_0, Q_0$ from the initial readings, $R_0 = 0$, $c_0 = 0$. RK4, 2 substeps per tick; the level
clip at $L_{full}$ is applied after every stage. Rates: $1/\tau_q \in [0.02, 1.5]$,
$1/\tau_{ret} \in [0.02, 0.2]$; the seepage rate is ~0.002 and only slows the integrator. State
clip: $L\in[0, 941.3]$, $Q\in[0,1]$, $R\in[0,5]$. Withdrawal depth enters nowhere: with two
control vectors its effect is not separable from irrigation and aeration. 12 parameters, 3 states
plus the clock.

The alternate `reservoir_min2.py` (v3) is the same model with a soft first-order spill over a
fitted sill instead of the hard clip, and without $q_{aer}$; it also passes the lab.

## Parameters (v4 full fit, cost 104.1)

| name | fitted | bounds | role |
|---|---:|---|---|
| q_m | 11.273 | [8, 15] | season mean inflow |
| q_a | 2.188 | [0.5, 5] | season amplitude |
| per | 67.87 | [40, 110] log | season period (ticks) |
| phi | 0.0454 | [-1.6, 1.6] | season phase at reset |
| seep | 0.00185 | [1e-4, 0.02] log | seepage rate per tick |
| c_out | 13.18 | [5, 30] log | discharge capacity at L = 500 |
| p_out | 0.217 | [0.05, 1] | head exponent |
| q_base | 0.940 | [0.5, 1] | quality target, no aeration |
| q_aer | 0.0070 | [0, 0.05] | quality gain from aeration |
| tau_q | 3.00 | [0.67, 50] log | quality relaxation (ticks) |
| k_ret | 0.0361 | [0, 0.3] | return flow per unit irrigation |
| tau_ret | 12.6 | [5, 50] log | return-flow delay (ticks) |

None at a bound. The full-pool level 941.3 and the capacity reference 500 are constants.

## Scores (lab metric, own sigma proxy; columns level, inflow, outflow, quality)

Leave-one-run-out (fit on the other run, score the held-out run):

| held out | ode | mean | l0b_lin | mean | persistence |
|---|---|---:|---|---:|---:|
| p1.hold_rec | 0.975 0.929 0.925 0.710 | 0.885 | 0.840 0.549 0.528 0.699 | 0.654 | 0.421 |
| p2.pulse200_200 | 0.759 0.829 0.834 0.516 | 0.735 | 0.525 0.564 0.457 0.648 | 0.548 | 0.418 |

In-sample (full fit on both runs):

| run | ode | mean |
|---|---|---:|
| p1.hold_rec | 0.970 0.952 0.930 0.757 | 0.902 |
| p2.pulse200_200 | 0.958 0.950 0.949 0.761 | 0.905 |

Eval sanity (4,000 ticks, all four categories): finite, 0.5 s each, level in [201, 941],
inflow [9.1, 13.8], outflow [0.6, 15.1], quality 0.9; 1-9% of ticks outside the observed range
+-5%, all of it level below 283 under long drawdowns (a legitimate extrapolation of the head law).

## Structures tried

| version | change | LOO mean (fold rec / fold pulse / avg) | in-sample | at bound | outcome |
|---|---|---|---:|---|---|
| v1 | soft spill over a free sill, free spill rate, 12 params | 0.881 / 0.702 / 0.791 | 0.894 | k_spill (hi) | ship, but plateau drifted to 965 |
| v2 | spill rate fixed at 1, sill bounded [915, 945], q_aer bound [0, 0.05] | 0.883 / 0.731 / 0.807 | 0.896 | l_cap (hi) | ship, plateau 952 |
| v3 (`reservoir_min2`) | v2 + return-flow state, q_aer dropped | 0.878 / 0.764 / 0.821 | 0.898 | l_cap (hi) | ship, plateau 952 |
| v4 (`reservoir_min`) | hard clip at 941.3, algebraic spill, q_aer back | 0.885 / 0.735 / 0.810 | 0.903 | none | ship, plateau 941.3 |

v3 beats v4 on the LOO average only through the quality column of the pulse fold (0.653 vs 0.516):
when the training fold is the recovery run alone (aeration always 1), $q_{base}$ and $q_{aer}$
are not separable and the split the fitter picks is arbitrary; in the full fit $q_{aer}$ is 0.007,
below the quality noise. On level v4 is better on both folds and in-sample, and it has no
parameter at a bound, so v4 ships and v3 is kept as the alternate.

Rejected along the way:
- Fouling state for the pulse discharge: the head law $c_{out}(L/500)^{p}$ reproduces the whole
  13.7 -> 11.1 decay with two parameters and no state.
- Free spill rate and sill (v1, v2): the rate is unidentifiable above 0.7/tick and the sill runs
  to its bound because the lab's level sigma (258) makes a 10-20 unit plateau error almost free
  in the Cauchy cost, while the outflow-at-full term prefers ~0.1 more seepage loss. The hard
  clip pins the plateau to the observation instead of to the fit.
- Outflow as a lagged state seeded from the initial reading: the first observation already
  equals the algebraic value, and the fastest admissible rate (1.5/tick) would leave 22% of the
  initial gap after one tick.
- Withdrawal-depth and irrigation-specific quality terms: not identifiable from two control
  vectors.

## Verdict

Ship (lab): LOO 0.810 vs l0b_lin 0.601 (+0.21), both folds above l0b_lin, in-sample 0.903 with
level/inflow/outflow all above 0.92. What limits it: (1) quality sits at the noise floor
(s.d. 0.01 against sigma 0.017: ceiling ~0.75) and the aeration effect is unidentified from one
control vector; (2) the level term barely counts in the lab cost, so any level parameter has to be
pinned by structure, not by the fit; (3) two control vectors leave withdrawal depth, and the split
of the return flow between irrigation and release, unidentified; (4) the seepage shape (linear vs
power law) and the head exponent (0.22 vs the 0.33 read off by hand) rest on one drawdown.
Composition and order runs would settle (1) and (3).
