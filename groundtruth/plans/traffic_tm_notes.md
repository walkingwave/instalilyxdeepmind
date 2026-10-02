# traffic tm: text-implied terms that z8 lacks (Mon Sep 28, no credits)

Question: do the brief's base-physics phrases that z8 does not model improve held-out fit?
Tested in order of the text-mining report (`plans/textmine_report.md`, traffic ranks 1, 2, 10):

1. finite exit and junction buffers as base physics, with blocked vehicles holding the shared junction;
2. clearance as a crew transfer (junction capacity down, exit capacity up);
3. two vehicle classes (toll moves the heavy share, freight priority orders service, heavy vehicles take
   more junction work and weigh more in the speed occupancy).

## Families (`gtlab/ode/traffic_tm*.py`)

| family | adds to z8 (A+B active, C off) | neutral values (= z8 exactly, checked max abs dY = 0 on all 7 runs) |
|---|---|---|
| traffic_tm1 | junction to exit transfer $r\,p_{2i}\,(1 - k_{x3}\,p_{3i})^+$; shared junction $J_{free} = J\,(1 - k_{jo}(p_{2a}+p_{2b}))^+\,(1 - k_j\,c)$, $c$ = crew state (lagged under B); exits keep z8's boost $1 + clr\cdot c(1-k_F F)$ | $k_{x3} = k_{jo} = k_j = 0$ |
| traffic_tm2 | tm1 + heavy approach queue $h_{1i}$; arrivals heavy share $h = \mathrm{clip}(h_0 e^{k_h(toll-2.5)}, 0, .95)$; junction work $W = \mathrm{smin}(r(p_1 + (w_h-1)h_1), J_{free}\,share)$, split by priority $\phi = e^{k_f(f-0.5)}$: $W_H = \min(W\phi D_H/(\phi D_H + D_L), D_H)$, $W_L = \min(W - W_H, D_L)$, leftover back to heavy; vehicles served $W_L + W_H/w_h$; speed occupancy $w_1(p_1 + (w_{hs}-1)h_1)$ | $w_h = w_{hs} = 1$, $k_f = 0$ |

$k_{x3} = 1/K_3$ (exit buffer), $k_{jo} = 1/J_{cap}$ (junction buffer): linear parameters so 0 is exactly the
z8 limit and the fit can reach it.

## Protocol (`scripts/lab_tm_traffic_loo.py`)

Deterministic, independent of machine load: one start from the family init, `max_nfev` 60, no time budget,
free = base + A + B parameters (for tm1/tm2 the new terms are in the base list), $\sigma$ = calibrated 1.0.
Leave one run out over all seven runs (exam p5.testlike and p7 long hold are folds), full fit on all seven.
The z8 baseline was re-run under the same protocol (`plans/traffic_tm_z8base.json`); it reproduces the
§36 z8all7 folds (0.739 vs 0.742; exam and p7 folds identical).

## Fold table (LOO mean score per held-out run)

| cand | hold_rec | pulse40 | mid40 | multilevel | hold_mid | exam | p7 long | LOO | vs z8 | in-sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z8 (baseline) | .965 | **.525** | **.853** | **.678** | **.809** | **.729** | .613 | **.739** | | .793 |
| tm1, neutral start | .964 | .518 | .850 | .677 | .809 | .724 | **.624** | .738 | −.001 | .793 |
| tm1, active start ($K_3$ 200, $J_{cap}$ 330, $k_j$ .2) | .965 | .518 | .843 | .670 | .804 | .722 | .539 | .723 | −.016 | .792 |
| tm2 (two classes, start $w_h$ 1.5, $k_f$ .3) | .973 | .510 | .842 | .653 | .745 | .666 | .606 | .714 | −.025 | .794 |

(`tm1a` and `tm1b` in plans/ are the same run: the new terms sit in the base list, so both freed all three.)

p7 per observable (flow_a, flow_b, speed_a, speed_b): z8 .763 / .326 / .548 / .813; tm1 neutral .793 / .435 /
.536 / .732: the buffer term lifts p7 flow_b (+0.11) but gives it back on speed_b (−0.08).

## Identification

In-sample profile (`scripts/lab_tm_traffic_profile.py`, `plans/traffic_tm_profile.json`): hold
$(k_{x3}, k_{jo}, k_j)$ fixed, refit the rest from the z8 fit on all seven runs.

| $k_{x3}$ | $k_{jo}$ | $k_j$ | cost | in-sample |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2123.3 | .7929 |
| .003 | 0 | 0 | 2123.3 | .7929 |
| .01 ($K_3$ = 100) | 0 | 0 | 2122.8 | .7934 |
| 0 | .003 | 0 | 2123.2 | .7929 |
| .003 | .003 | 0 | 2123.2 | .7937 |
| .003 | .003 | .3 | 2122.7 | .7936 |

The cost surface is flat to 0.03 % over exit buffers from infinite down to 100 vehicles, a shared junction
buffer of 330 and a 30 % crew transfer: the rest of the model re-arranges and the seven runs cannot tell
these structures apart. Fold fits agree: $k_{x3}$ 0 to 0.0014, $k_{jo}$ 0 to 0.0015, $k_j$ 0 to 0.29,
exit boost $clr$ 0 to 2.1 across folds; the full fit puts all three new terms at exactly 0 (z8 back).
Under B the fit parks the crew state (switching lag τ_sw ≈ 285, fatigue τ_F ≈ 5), so clearance has almost
no sustained authority either way: the transfer can not show up. Two classes: $k_f$ changes sign across
folds (−1.9 to +1.35), $w_{hs} \to 0.05$ (heavy vehicles weigh nothing in speed), $h_0 \approx 0.7$: the
class split is used as extra free shape, and it costs hold_mid −0.064 and the exam −0.063.

## Decision

- **No candidate passes** (rule: LOO at 1.0 σ on all runs, must not lose the pulse fold or the exam,
  noise ±0.01 on the mean). Best: tm1 neutral start, LOO −0.001 (tie), p7 +0.011, but pulse40 −0.007 and
  exam −0.005. Its full fit is z8 with the new terms at 0.
- Best candidate doc: `plans/traffic_tm_tm1a_doc.json` (z8-equivalent refit; nothing to ship over z8).
- **Keep z8** (`plans/traffic_traffic_z8_z8all7_doc.json`, public 0.8385).
- Forecast if shipped anyway: 0.8385 + 0.6 × (−0.001) ≈ **0.838** (tm1); tm1 active ≈ 0.829; tm2 ≈ 0.824.
- Why the text terms do not pay: their footprint in our runs is short (clearance and freight move for
  3–10 ticks; exits never stay blocked long except in p7, where lane closure on route b's exit already
  explains the level). The exit-buffer / shared-junction story needs a run where one route's exit is
  blocked while the other route's demand and signal share are held fixed (brief: "compare clearance with a
  signal reversal after stopping arrivals"): e.g. ramp 0 after a pulse, then signal 0.15 → 0.85 at
  clearance 0 vs clearance 1. About 120 steps; not bought.
