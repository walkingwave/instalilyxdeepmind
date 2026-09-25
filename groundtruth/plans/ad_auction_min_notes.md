# ad_auction: minimal grey-box ODE (`gtlab/ode/ad_auction_min.py`)

Date: 2026-09-25. Data: 3 real runs, 500 ticks (hold_rec 120, pulse60_120 180, multilevel200 200).
Fitter: `scripts/ode_lab.py --budget 240 --starts 10 --nfev 60` (least_squares, cauchy loss, 10 LHS starts,
leave-one-run-out against persistence and l0b_lin on the same folds). No credits spent.

## What the data says (read off the runs before writing any equation)

- `win_rate` rises 0.14 to 0.25 during the capped phase of hold_rec and is flat once the cap is slack.
  That is not learning: while spend is pinned at the cap we can only afford $\text{cap}/p$ impressions, so
  $\text{win\_rate} = \text{cap}/(p A)$ rises as the opportunity pool $A$ shrinks. With cap slack, win_rate $= w(\text{bid})$.
- Opportunity pool at $t=0$: $A = 140$ at breadth 0.55 and $199$ at 0.775 (from win_rate under the cap), ratio 1.42
  $=$ breadth ratio, so $A \propto$ breadth, $N \approx 255$ (in impression units at bid 1.5 cost).
- Cost per impression grows sublinearly with the bid: $p(5)/p(1.5) \approx 1.5$, $p(0.87)/p(1.5) \approx 0.85$,
  i.e. $p \propto \text{bid}^{0.35..0.45}$ (second-price flavour).
- Pool depletion is about 0.1 person per impression (hold: 140 to 80 in 30 ticks at 20 imp/tick; pulse: 199 to 120 in
  14 ticks at 66 imp/tick), return time 30-70 ticks (steady-state balance gives 68, the post-pulse spend recovery 27).
- Conversions: $\approx 0.21..0.25$ per impression in steady state; a ramp of ~24 ticks after a step, an exponential drain
  with $\tau \approx 8.4$ when the bid drops to 0, and a plateau at 5.4 for ~16 ticks after the pulse release
  (a backlog draining at a capacity limit). Uncapped win rate at bid 5 is 0.54 at breadth 0.775 but 0.64 at breadth 0.1.
- Conversions are 0.0 on the first two ticks of every run whatever the initial report says: the reset state is clean
  (as the brief says), the initial report is a reading, not a state.

## Model (kept version)

States $X$ (fraction of the targeted pool temporarily removed by exposure), $Q_1, Q_2$ (pending purchases, two stages).
Controls bid $b$, cap $C$, breadth $\beta$.

$$w = \frac{b^{n}}{b^{n}+b_0^{n}} \cdot \frac{1}{1+k_w(\beta-0.1)}, \qquad p = (b/1.5)^{\gamma}, \qquad A = N\,\beta\,(1-X)$$
$$\text{imp} = \min\!\left(w A,\; C/p\right), \qquad \text{spend} = p\cdot\text{imp} \le C, \qquad \text{win\_rate} = \text{imp}/A$$
$$\dot X = \frac{k_d\,\text{imp}}{N\beta} - \frac{X}{\tau_e}, \qquad
\dot Q_1 = c\,\text{imp} - \frac{Q_1}{\tau_1 (1+k_b(\beta-0.1))}, \qquad
\dot Q_2 = \frac{Q_1}{\tau_1 (1+k_b(\beta-0.1))} - \text{conv}, \qquad
\text{conv} = \min\!\left(Q_2/\tau_2,\; F\right)$$

Both mins are smooth (`_smin`, widths 0.5 impressions / 0.1 conversions) followed by a hard min so spend never exceeds the
cap. $x_0 = (0,0,0)$ for every initial report. RK4, 2 substeps per tick. 12 parameters, 3 states. Mechanism letters are
accepted but every term is always on.

## Fitted parameters (full fit on all 3 runs)

| param | meaning | value | bounds |
|---|---|---:|---|
| $N$ | pool scale (impressions at breadth 1, cost units of bid 1.5) | 249.5 | 20-5000 (log) |
| $b_0$ | bid at which the won fraction is 1/2 | 2.824 | 0.2-30 (log) |
| $n$ | Hill exponent of the won fraction | 1.413 | 0.3-4 |
| $\gamma$ | cost-per-impression exponent, $p=(b/1.5)^\gamma$ | 0.460 | 0-1 |
| $k_d$ | people removed per impression | 0.1419 | 0.005-2 (log) |
| $	au_e$ | return time of removed people | 48.06 | 3-400 (log) |
| $c$ | purchases started per impression | 0.2154 | 0.01-2 (log) |
| $	au_1$ | first fulfilment stage at breadth 0.1 | 1.485 | 0.7-60 (log) |
| $	au_2$ | second fulfilment stage | 11.18 | 0.7-60 (log) |
| $F$ | completion capacity per tick | 6.017 | 0.5-100 (log) |
| $k_b$ | extra work per purchase per unit breadth | 3.752 | 0-10 |
| $k_w$ | rival pressure per unit breadth on the won fraction | 0.405 | 0-3 |

None at a bound. Full-fit cost 41.8 (cauchy, 500 ticks x 3 observables). Files: `plans/ad_auction_ad_auction_min.json`
(theta + report), `plans/ad_auction_ad_auction_min_doc.json` (model doc for the packager).

## Scores (per observable: win_rate, spend, conversions)

Leave-one-run-out (fit on the other two runs, score the held-out run; sigma_proxy = [0.174, 18.84, 1.69]):

| held out | ode | mean | l0b_lin | mean | persistence |
|---|---|---:|---|---:|---:|
| p1.hold_rec | 0.928 0.985 0.803 | 0.905 | 0.788 0.801 0.611 | 0.733 | 0.569 |
| p2.pulse60_120 | 0.770 0.846 0.728 | 0.781 | 0.568 0.766 0.687 | 0.674 | 0.511 |
| p2.multilevel200 | 0.891 0.955 0.737 | 0.861 | 0.417 0.440 0.435 | 0.430 | 0.578 |
| **mean** | | **0.849** | | **0.613** | 0.553 |

In-sample (full fit):

| run | ode | mean |
|---|---|---:|
| p1.hold_rec | 0.937 0.984 0.846 | 0.923 |
| p2.pulse60_120 | 0.915 0.938 0.841 | 0.898 |
| p2.multilevel200 | 0.946 0.958 0.827 | 0.910 |
| **mean** | | **0.910** |

Eval sanity (4,000-tick rollouts on eval-shaped schedules, seed 0): all four categories finite, 0.4 s each,
0.00 of ticks outside the observed range +-5%; ranges win_rate 0-0.6, spend 0-100, conversions 0-6.

## Structures tried

| version | change | LOO mean (ode / l0b_lin) | in-sample | at bound | kept? |
|---|---|---|---|---|---|
| v1 | pool $X$ + 2-stage queue + capacity $F$ + `seed` (initial report seeds $Q_2$) | 0.840 / 0.613 | 0.894 | none (`seed` fitted to 0.000) | no |
| v2 | v1 with two nested audience segments (narrow core 10% always targeted, broad 90% in proportion to breadth), each with its own $X$ | 0.840 / 0.613 | 0.894 | none | no: identical to v1 to 3 decimals. Serving impressions in proportion to each segment's available count makes both segments deplete at the same fractional rate, so the two $X$ only diverge at breadth exactly 0.1; the data cannot tell them apart |
| v3 | v1 with $k_w$ (breadth term on the won fraction), `seed` dropped | 0.849 / 0.613 | 0.910 | none | **yes** |
| v3b | v3 with a third queue stage | 0.856 / 0.613 | 0.910 | `tau1` at lower bound 0.7 | no: the fitter collapses the extra stage to a 1-tick delay, the +0.007 LOO is not structural |

## Verdict

**ship** by the lab rule (LOO 0.849 vs l0b_lin 0.613, margin 0.236; in-sample 0.910). The ODE beats l0b_lin on every fold
and every observable (9 of 9), and persistence on every fold. Next: run it through `scripts/build_cfg.py` as a candidate
for ad_auction (current public best 0.628 with l1) and let the local screen decide the blend.

## What limits it

- The conversion ramp is a linear ~24-tick ramp with a sharp knee in the data; a 2-stage linear chain gives a concave
  approach, so conversions stay the weakest observable (LOO 0.73-0.80).
- Conversions per impression drift with the bid (0.205 at bid 0.87, 0.245 at 1.5, 0.26 at 5); a single $c$ splits the difference.
- The pool return time is 27 from the post-pulse recovery but 68 from the steady-state balance: one first-order return
  state cannot do both; the fit sits at 48.
- Breadth switches are only observed at the very end of multilevel200 (27 ticks at breadth 0.1), so the nested-audience
  structure the brief describes is unidentified; composition-category behaviour on breadth is an extrapolation.
- Only 3 runs: the LOO folds are each a different regime, so the LOO mean is a rough guide, not a confidence interval.
