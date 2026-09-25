# wildlife: minimal grey-box ODE (`gtlab/ode/wildlife_min.py`)

Date: 2026-09-25. Data: 2 runs, 520 ticks (`p1.hold_rec` 120 ticks at recovery; `p2.pulse200_200`
200 ticks at pulse then 200 at recovery). Fitter: `scripts/ode_lab.py --budget 240 --starts 10 --nfev 60`
(least squares, Cauchy loss, residuals in units of `sigma_proxy` = [62.7, 1.16, 50.6, 1.43]).

## What the data say

- Recovery (hunting 0, habitat 1, corridor 0): prey overshoot then settle, $86 \to 199$ (t=22) $\to 121$;
  south $96 \to 159 \to 97$. Per-capita prey growth is a function of $P$ alone during the rise:
  $0.28$ at $P=10$, $0.11$ at $50$, $0.075$ at $90$, $0.05$ at $150$, i.e. $\approx r/(1+P/c)$ with
  $r \approx 0.35$, $c \approx 25$ (nursery competition), not logistic. The same $0.075$ at $P \approx 90$
  appears at reset and 17 ticks after a release, so the crowding is instantaneous; the collapse
  after the peak is slow ($-0.02$/tick), so the density feedback that ends the rise is lagged.
- Predators fall monotonically to $\approx 2.35$ under recovery and $\approx 1.62$ under pulse,
  nearly independent of prey ($121$ vs $7$). Initial prey growth is the same at $Q = 8.7$ and $Q = 2.0$:
  predation on prey is invisible, so we do not model it.
- Pulse (hunting 7, habitat 0.1, corridor 1): prey per-capita decline $-0.075$ at $P = 90$,
  $-0.12$ at $P = 25$, $0$ at $P = 7$. Both regions land at $7.0 / 7.5$.

## Model (final)

Per region, north scale 1, south scale $k_s$ on the density scale ($c \to k_s c$, $d_l \to d_l / k_s$):

$$\dot P = \frac{r P}{1 + P/c} - d_l L P - h_q\, u_{hunt}\, P, \qquad
\dot L = \rho (P - L), \qquad
\dot Q = s\,(q_0 + q_1 P - Q)\,\frac{Q}{Q + q_h}$$

$L$ is a lagged density (depleted food / cover, renewing at rate $\rho$): a delayed logistic, which
is what produces one overshoot and a flat settle. Observed $y = (P_N, Q_N, P_S, Q_S)$. Reset:
$P, Q$ from $y_0$, $L = 0$. States 6, RK4 with 2 substeps per tick. Habitat protection and corridor
access carry no term (see rejected structures). Equilibria: recovery $P^*$ solves
$r/(1+P^*/c) = d_l P^*$; predators $Q^* = q_0 + q_1 P^*$.

## Structures tried

| version | prey growth / death | harvest | predators | params | LOO hold_rec | LOO pulse | in-sample |
|---|---|---|---|---|---|---|---|
| l0b_lin (baseline) | | | | | 0.807 | 0.574 | |
| v1 | $rRP/(1+P/c) - dP$, food $R$ eaten at $\rho P R/k_r$, renewal scaled by habitat | $h_q u P/(P+p_0)$ | logistic to $q_0+q_1P$ | 12 | 0.857 | 0.674 | 0.876 / 0.937 |
| v2 | $rRP/(1+P/c) - d(1-R)P$ (starvation), no habitat term | same | $s(K_q-Q)Q/(Q+q_h)$ | 12 | 0.840 | 0.694 | 0.892 / 0.943 |
| v3 | delayed logistic $rP/(1+P/c) - d_l L P$, reset $L = l_0 P$ | same | same | 12 | 0.812 | 0.676 | 0.899 / 0.943 |
| v4 | delayed logistic, reset $L = 0$ | $h_q u P$ | same | 10 | **0.840** | **0.705** | 0.892 / 0.943 |

Per-observable LOO (ode vs l0b_lin), best version by LOO mean:

| fold held out | prey_north | predator_north | prey_south | predator_south | mean |
|---|---|---|---|---|---|
| hold_rec: ode | 0.883 | 0.882 | 0.896 | 0.696 | 0.840 |
| hold_rec: l0b_lin | 0.864 | 0.722 | 0.838 | 0.805 | 0.807 |
| hold_rec: persistence | | | | | 0.441 |
| pulse200_200: ode | 0.665 | 0.722 | 0.668 | 0.766 | 0.705 |
| pulse200_200: l0b_lin | 0.523 | 0.645 | 0.508 | 0.621 | 0.574 |
| pulse200_200: persistence | | | | | 0.341 |
| in-sample hold_rec | 0.919 | 0.949 | 0.921 | 0.779 | 0.892 |
| in-sample pulse200_200 | 0.960 | 0.920 | 0.946 | 0.947 | 0.943 |

LOO mean 0.772 vs l0b_lin 0.691 (persistence 0.391); in-sample mean 0.918. Full-fit Cauchy cost 36.8.

Quick 90 s in-sample screens that did not make it to a full run: food eaten by births instead of by
heads (cost 48 vs 37, worse), death $d/(0.1+R)$ (0.880 / 0.941, no gain), habitat scaling the food
renewal (fitted to $0.000$ in every variant that had it: the pulse run is explained by hunting alone,
and once prey are scarce the food stock is full whatever the renewal rate).

Rejected and why:
- Habitat term on food renewal: fitted to zero three times. We cannot separate it from hunting with
  one joint pulse, so it stays out rather than ship an arbitrary value.
- Saturating take $h_q u P/(P+p_0)$: $p_0$ ran to its upper bound in v1, v2 (2000) and v3 (5000), i.e.
  the fitter wants a take proportional to $P$. The hand analysis suggested $p_0 \approx 100$ from the
  accelerating per-capita decline under the pulse, but the fit prefers to explain that with the lagged
  density ($L$ still high from the pre-pulse population).
- Reset lag fraction $l_0$: improved in-sample (0.899 vs 0.892 on hold_rec) but cost 0.03 on the
  hold_rec LOO fold, where a single run cannot pin it. Fixed at 0.
- Food stock $R$ with $d(1-R)$ death: $d$ ran to its bound (50) with $k_r \to 8 \cdot 10^4$; the product
  $d/k_r$ is the only identified quantity, which is exactly the delayed logistic $d_l L P$ of v3/v4.
- Corridor migration $\mu\,u_{cor}(P_{other} - P)$: not tried; the two regions differ only by the
  scale $k_s = 0.83$ and no run varies corridor alone, so $\mu$ would be unidentified.

## Fitted parameters (final)

| name | value | bounds | meaning |
|---|---|---|---|
| $r$ | 0.2868 | 0.05-1.5 | max per-capita prey birth per tick |
| $c$ | 44.0 | 3-500 | nursery-crowding scale (birth halves at $P=c$) |
| $d_l$ | 6.4e-4 | 1e-5-0.1 | death per head per unit lagged density |
| $ho$ | 0.0340 | 0.005-0.5 | lag rate of $L$ (time constant 29 ticks) |
| $h_q$ | 0.0347 | 0.001-1 | per-capita take per unit hunting quota (0.24 at quota 7) |
| $q_h$ | 10.79 | 0.05-50 | predator relaxation half-rate density |
| $s$ | 0.1913 | 0.003-0.5 | predator relaxation rate at large $Q$ |
| $q_0$ | 1.557 | 0.1-20 | predator capacity floor |
| $q_1$ | 0.00749 | 1e-4-0.5 | predator capacity per prey |
| $k_s$ | 0.8333 | 0.3-1.5 | south density scale |

None at a bound. Implied recovery equilibria: north $P^* = 119$, south $99$; $Q^* = 2.45 / 2.30$.
Under hunting 7 the take $0.243$ per head against a birth rate $r = 0.287$ leaves $P^* pprox 7$.

## Eval sanity (4,000-tick rollouts on eval-shaped schedules, from `p1.hold_rec` $y_0$)

| category | finite | seconds | frac outside data range | min | max |
|---|---|---|---|---|---|
| sustained | yes | 0.2 | 0.00 | 9.3, 1.7, 7.8, 1.6 | 172, 8.2, 153, 11.6 |
| order | yes | 0.3 | 0.00 | 1.4, 1.6, 1.2, 1.6 | 195, 8.2, 171, 11.6 |
| recovery | yes | 0.2 | 0.00 | 10.1, 1.9, 8.4, 1.8 | 195, 8.2, 171, 11.6 |
| composition | yes | 0.3 | 0.00 | 4.8, 1.6, 4.0, 1.6 | 177, 8.2, 147, 11.6 |

## Verdict

`ode_lab` verdict: **ship** (LOO 0.772 vs l0b_lin 0.691, both folds above the baseline; in-sample 0.918).
Files: `plans/wildlife_wildlife_min.json` (report + theta) and `plans/wildlife_wildlife_min_doc.json`
(model document for `build_cfg` / `package`); the per-version runs are `_v1` .. `_v4`.

What limits it: (1) only two runs, one of them the sole source of hunting information, so the
pulse-fold LOO ($\approx 0.69$) is really "fit on recovery only"; (2) south predators decay slower than
north early on (12.6 vs 8.7 start, same fractional decline), which neither a logistic nor a
linear relaxation reproduces (predator_south 0.70-0.78); (3) the overshoot after a release from
$P = 7$ (peak 206) is larger than after reset from $P = 86$ (peak 199), which a fixed reset state cannot
produce; (4) habitat and corridor have no effect in the model, so any test episode that moves them
alone is predicted as a recovery hold.
