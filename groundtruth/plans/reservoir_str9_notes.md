# reservoir str9: groundwater inflow + slow quality stock (Mon, no credits)

Family `gtlab/ode/reservoir_str9.py`; lab `scripts/lab_str9_loo.py`; runtime check
`scripts/lab_str9_verify.py`; diagnostics `scripts/lab_str9_diag.py`.
Pick: **`plans/reservoir_str9_reservoir_str9_c_clean_doc.json`** (21-entry theta, post empty).
Scores: calibrated sigma [110.4, 0.672, 1.849, 0.0113], columns level / inflow / outflow / quality.

## Where the shipped model (v8, final1) loses on p8 (700 ticks, never fitted)

| p8 segment | level | inflow | outflow | quality |
|---|---:|---:|---:|---:|
| whole run | 0.925 | 0.866 | 0.917 | **0.628** |
| bias (sigma units) | -0.06 | -0.15 | -0.11 | **+0.36** |

1. **Quality at full pool drifts down**: 0.951 at t250 -> 0.935 at t675 (recovery hold, aeration on);
   v8 relaxes to q_base 0.952 and stays. p2 shows the same slide after its pool fills
   (0.948 -> 0.941), p1 too (0.953 -> 0.949). Seasonal component in the quality residual:
   amplitude <= 0.003 (0.25 sigma), not pursued.
2. **Inflow is high while the stage is low**: excess over the season +0.18 (t0-50, L ~ 420) rising
   to +0.41 (t100-150, L ~ 320), gone once L > 500. p8 has **no irrigation**, yet the excess has the
   same size as p2's (irrigation 8), which v8 had charged to irrigation return (k_ret). A level-driven
   groundwater inflow explains both runs; the missing inflow is also v8's level undershoot on p8
   (-17 at t150, -34 at t250).

## Model

v8 plus
$$G = g_{gw}\,\operatorname{softplus}_{20}(L_{gw} - L)\quad\text{(observed in inflow and added to the balance)}$$
$$\dot N = (L/L_{full} - N)/\tau_n,\quad N(0) = n_0\ \text{(reference profile)}$$
$$\dot Q = (q_{base} - q_d D - q_n N - q_{aer}\,a - Q)/\tau_q$$
Irrigation return removed ($k_{ret} = 0$ fixed): once $G$ is in, every fit drove $k_{ret}$ to
0.01-0.03, and left free in the pulse fold (no irrigation in training) it ran to its bound (0.29).
$w_L$ (level share of the N target) fitted 0.96-1.0 in every free fit: fixed at 1.
Full fit: $g_{gw}$ 0.00246, $L_{gw}$ 481, $q_n$ 0.041, $\tau_n$ 118, $n_0$ 0.39, $q_{aer}$ 0.0036,
$q_{base}$ 0.981, $q_d$ 0.030, $\tau_d$ 54. Fold values of $g_{gw}$ 0.0019-0.0028 and $L_{gw}$ 461-482:
the groundwater term is pinned by each pair of runs.

## Leave-one-run-out at 1.0 sigma (lab fitter, Cauchy LSQ, 300 s)

Start: v8 ship theta (fitted on p1+p2) with new parameters at defaults, for every family (the p8 fold
is clean; p1/p2 folds share the same start leak).

| fold | v8 refit | str9 a (k_ret free) | str9 b (k_ret 0) | **str9 c (k_ret 0, n0, q_aer)** | str9 d (c + D0) |
|---|---|---|---|---|---|
| p1.hold_rec (recovery hold) | 0.855 | 0.858 | 0.857 | **0.861** | 0.857 |
| p2.pulse200_200 (pulse/recovery) | 0.748 | 0.698 | 0.866 | **0.878** | 0.862 |
| p8.longhold (sustained) | 0.832 | 0.884 | 0.885 | **0.885** | 0.876 |
| mean | 0.812 | 0.813 | 0.869 | **0.8745** | 0.865 |

Pulse fold with the parameters it cannot see (c_out, p_out, k_ret, tau_ret) frozen at each family's
full fit (equal leak): v8 0.862 vs str9 a 0.889. Per observable, str9 c folds:
p1 [0.959 0.926 0.959 0.599], p2 [0.937 0.910 0.931 0.733], p8 [0.945 0.911 0.943 0.742];
v8: p1 [0.948 0.902 0.956 0.615], p2 [0.857 0.644 0.804 0.687], p8 [0.938 0.850 0.909 0.628].
Only loss anywhere: p1 quality -0.016 (the fill-phase quality sits 0.005 high, see below).

Shipped predictor on p8 (held out from it): 0.834; str9 c p8 fold 0.885 (+0.051).

## Runtime check (packaged predict.py, `lab_str9_verify.py`)

| run | ship | str9 c |
|---|---|---|
| p1.hold_rec | 0.908 | 0.893 |
| p2.pulse200_200 | 0.884 | 0.898 |
| p8.longhold | 0.834 (held out) | 0.895 |

Ship was polished on p1/p2 directly; the in-sample p1 difference is quality (0.765 vs 0.712).
Noise ceilings against the noisy readings: level ~0.97, inflow ~0.93, outflow ~0.975, quality
~0.74-0.77, so str9 c is at the ceiling on every observable of p2 and p8.
4,000-tick episodes of all four categories: finite, 0.8-1.0 s each.
Constant 4,000-tick holds (no drift after ~500 ticks): recovery quality 0.937 (ship 0.952; p8 reads
0.935 at t675), pulse level 269 (ship 258), zero action quality 0.925, release 12 shallow with
aeration quality 0.965 (unobserved corner), deep withdrawal without aeration at full pool 0.910
(lowest; unobserved corner, D and N effects stack).

## Rejected

- d (reset deficit D0 free): LOO 0.865, loses all three folds vs c.
- k_ret free (a): pulse fold 0.698, the irrigation return is unidentified without irrigation data.

## Forecast

0.6 x held-out gain (+0.05 on p8, +0.06 LOO mean vs v8): reservoir 0.831 -> ~0.86 (+0.03 on the
system, +0.003 on the mean). Remaining leak: fill-phase quality (p1 t10-60 model 0.958 vs 0.953);
aeration and withdrawal depth still move together in every run, so their split stays min-norm.
