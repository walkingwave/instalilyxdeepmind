# ad_auction v8: structural pass at the organizer's scale (`gtlab/ode/ad_auction_v8b.py`)

Date: 2026-09-26. Data: the same 3 real runs (hold_rec 120, pulse60_120 180, multilevel200 200). No credits spent.
Scale: every score here is $\frac{1}{1+|e|/\sigma}$ with the calibrated organizer $\sigma$ from `plans/sigma_calibrated.json`
($\sigma$ = [win_rate 0.0907, spend 9.82, conversions 0.881]), rollouts through the runtime path including the post rule
spend $\le$ 1.04 budget_cap. Starting point: `ad_auction_min` doc, calibrated in-sample **0.851** (public 0.830).

## 1. Error budget of the current best (loss = $\sum_t (1-s_t)$ per segment, by observable)

| run / segment | controls (bid, cap, breadth) | win_rate | spend | conversions |
|---|---|---:|---:|---:|
| hold_rec 0-120 | 1.5, 20, 0.55 | 13.4 | 3.5 | **30.1** |
| pulse 0-60 | 5, 100, 0.775 | 6.6 | 14.5 | **18.7** |
| pulse 60-180 | 1.5, 20, 0.55 | **20.3** | 4.3 | **27.6** |
| multilevel 0-63 | 0.87, 91, 0.79 | 5.2 | 9.1 | 19.5 |
| multilevel 63-131 | 0, 100, 0.1 | 1.5 | 0 | 13.4 |
| multilevel 131-173 | 4.04, 15.3, 0.67 | 2.6 | 0.2 | 9.4 |
| multilevel 173-200 | 5, 100, 0.1 | 9.2 | 5.3 | 11.8 |

Conversions carry 130 of 225 loss units. Top 3 leaks and what the data shows:

1. **Pulse conversions**: truth rises to 7.48 at tick 13, falls to 5.45 by tick 45, then after the release stays flat at
   5.4 for 16 ticks and only then drains with $\tau\approx 8.5$. The old model caps completions at a fixed $F=6.0$, so it
   misses both the burst above 5.45 and the drop to it. A fixed cap cannot produce both a 7.48 peak and a 5.4 plateau.
2. **Win rate at the recovery bid**: truth 0.26-0.27 in both runs, 0.287 right after the pulse release, the model a flat
   0.245. The won fraction is higher when the audience has been depleted: rivals are pushed out of a depleted audience.
3. **Conversion ramp**: the ratio conversions / impressions climbs the same way in hold_rec and in the pulse
   (0.13 at tick 13 in both runs, 0.23 at tick 30 in both) although the pulse delivers ~2.9x more impressions per
   person. The ramp is a preparation of the audience that saturates quickly with exposure, not a fixed per-impression yield.

A plain refit of `ad_auction_min` at the calibrated scale (Powell on the score) already reaches 0.868: the old fit was
made at the proxy scale (2x the organizer's $\sigma$).

## 2. Model v8b (14 parameters, 5 states)

States: $X$ (share of the targeted pool removed by exposure), $Q_1, Q_2$ (pending purchases), $P$ (prepared share of the
audience), $K$ (idle fulfilment capacity). Controls: bid $b$, cap $C$, breadth $\beta$. $e = \text{imp}/(N\beta)$.

$$R = 1 - k_r X, \qquad w = \frac{b^{n}}{b^{n} + (b_0 R)^{n}}\cdot\frac{1}{1+k_w(\beta-0.1)}, \qquad p=(b/1.5)^{\gamma}, \qquad A = N\beta(1-X)$$
$$\text{imp}=\min(wA,\;C/p), \quad \text{spend}=p\,\text{imp}, \quad \text{win\_rate}=\text{imp}/A$$
$$\dot X = k_d e - X/\tau_e, \qquad \dot P = k_p\,\frac{e}{1+e/E_0}\,(1-P)$$
$$\dot Q_1 = c\,(g_0 + (1-g_0)P)\,\text{imp} - Q_1/\tau_1, \qquad \dot Q_2 = Q_1/\tau_1 - \text{conv}, \qquad
\text{conv} = \min\!\left(Q_2/\tau_2,\; F + K/T_K\right)$$
$$\dot K = (F-\text{conv})_+\,(1-K/K_{\max}) - (\text{conv}-F)_+$$

Reset: $X=Q_1=Q_2=P=0$ ("available and unprepared, no pending purchases"), $K=K_{\max}$ (fulfilment idle).
Fixed shape constants: $n=1.3$, $E_0=0.2$, $T_K=13$ (each was fitted in the 21-parameter superset: 1.27, 0.23, 13.5;
fixing them cost nothing, see section 3). RK4, 2 substeps. Rival capital is algebraic: its fitted relaxation time went to
the 3-tick lower bound, i.e. it moves within a tick.

Reading of the three new pieces against the brief:
- $R$: "rival campaigns may move a shared pool of capital between audiences". Rivals leave an audience whose reachable
  people are depleted; our won fraction rises with $X$ (hold_rec 0.245 to 0.27, post-pulse 0.287).
- $P$: "people are available and unprepared" at reset; exposure prepares them and a prepared audience starts about
  twice as many purchases per impression ($g_0 \approx 0.5$). The saturation $e/(1+e/E_0)$ makes the ramp nearly
  independent of the exposure rate, which is what the two runs show.
- $K$: "started purchases compete for limited fulfilment work". Idle capacity accumulated before the pulse is spent in
  the first ~15 ticks (burst to 7.5), then completions settle at the sustained rate $F=5.39$, and the backlog left at the
  release drains at $F$ (the 16-tick plateau).

## 3. What we tried (superset `gtlab/ode/ad_auction_v8.py`, every addition off at its default)

Calibrated in-sample, all runs, runtime path. Fits: least squares (cauchy) from the current theta, then Powell on the
score itself.

| variant | change vs refit of min (0.868) | in-sample | kept? |
|---|---|---:|---|
| A | refit of min at calibrated $\sigma$ | 0.868 | baseline |
| B | + rival capital $R$ (relax to $1-k_rX$, price $\propto R^{g_r}$) | 0.883 | yes (price term $g_r$ went to 0, relaxation to 3-tick bound: made algebraic) |
| C | + bid-dependent yield $c\,(b/1.5)^{c_b}$ | 0.867 | no, $c_b$ = 0.16 and no gain; $-0.02$ when combined |
| D | + preparation $P$ (linear in $e$) | 0.871 | only with $K$ |
| E | + exposure removal exponent $e^{m}$ | 0.867 | no, $m$ = 1.15, no gain |
| B+D | $R$ + $P$ | 0.882 | no gain alone |
| B+K | $R$ + idle capacity | 0.891 | partial |
| **B+D+K+E0** | $R$ + $P$ (saturating) + idle capacity | **0.903** | yes: pulse conversions 0.74 to 0.94 |
| + audience exponent $A\propto\beta^{a}$ | on top | 0.903 | no ($a$ stays 1.0) |

Ablations from the 18-free-parameter fit (0.9030), each refitted by Powell:

| fixed | in-sample | decision |
|---|---:|---|
| $k_b=0$ (breadth-dependent work) | 0.9024 | drop |
| $n=1.3$ | 0.9043 | fix |
| $E_0=0.2$ | 0.9035 | fix |
| $T_K=13$ | 0.9054 | fix |
| $\tau_1=0.7$ (one stage) | 0.896 | keep $\tau_1$ |
| $k_w=0$ | 0.882 | keep |
| $g_0=0$ (no unprepared yield) | 0.876 | keep |

v8c (`gtlab/ode/ad_auction_v8c.py`: v8b with a price floor $p=((b+p_b)/(1.5+p_b))^\gamma$ in place of a free $\tau_1$,
fixed at 5.4): in-sample 0.9014 (+0.001, multilevel spend 0.922 to 0.933) but lab LOO 0.807 vs 0.820 for v8b (the pulse
fold drops 0.773 to 0.730: with $p_b$ free, $\gamma$ runs to 0.98 and $p_b$ to 3.1; three bid levels do not identify the
price curve). **Rejected.**

## 4. Scores

Calibrated in-sample per observable (win_rate, spend, conversions), runtime path with the spend cap:

| run | min doc (current) | v8b |
|---|---|---|
| hold_rec | 0.888 0.971 0.750 = 0.869 | 0.945 0.942 0.797 = 0.895 |
| pulse60_120 | 0.851 0.895 0.743 = 0.830 | 0.926 0.911 0.936 = 0.924 |
| multilevel200 | 0.908 0.927 0.729 = 0.855 | 0.919 0.922 0.804 = 0.882 |
| **mean** | **0.851** | **0.900** |

Leave-one-run-out at the calibrated scale (`scripts/ode_lab.py --sigma-cal 1.0 --budget 300 --starts 10 --nfev 60`,
same protocol, lab rollouts without the post rule):

| held out | min (warm start) | v8b |
|---|---|---|
| hold_rec | 0.887 0.975 0.682 = 0.848 | 0.869 0.920 0.738 = 0.842 |
| pulse60_120 | 0.659 0.777 0.654 = 0.697 | 0.893 0.816 0.611 = 0.773 |
| multilevel200 | 0.883 0.920 0.654 = 0.819 | 0.917 0.933 0.680 = 0.843 |
| **mean** | **0.788** | **0.820** |

The hold_rec fold is 0.006 lower (spend 0.975 to 0.920: fitted on pulse + multilevel only, the removal/return balance
at the recovery action is extrapolated), the other two folds gain +0.076 and +0.024. A second protocol (Powell on the
score, warm start from the full fit, runtime path with the spend cap) gives every fold up:

| held out | min | v8b |
|---|---|---|
| hold_rec | 0.900 0.968 0.672 = 0.847 | 0.943 0.941 0.767 = 0.884 |
| pulse60_120 | 0.857 0.849 0.722 = 0.809 | 0.900 0.900 0.680 = 0.826 |
| multilevel200 | 0.891 0.933 0.677 = 0.834 | 0.921 0.926 0.802 = 0.883 |
| **mean** | **0.830** | **0.864** |

Eval sanity (4,000-tick eval-shaped rollouts, all four categories): finite, 0.5-0.6 s each, 0.00 of ticks outside the
observed range $\pm$5%; conversions max 7.4 (data max 7.5). No parameter at a bound.

## 5. Files

- `gtlab/ode/ad_auction_v8.py`: exploratory superset (21 parameters, not for shipping).
- `gtlab/ode/ad_auction_v8b.py`: shipped structure (14 parameters).
- `plans/ad_auction_ad_auction_v8b_v8.json` / `_doc.json`: final theta (Powell-polished on the calibrated score, written
  through the lab with `--theta0 --starts 1 --nfev 1`). `plans/ad_auction_ad_auction_v8b_loo.json`: LOO report.
- `plans/ad_auction_ad_auction_min_v8ref*.json`: LOO reference for the old structure at the calibrated scale.

## 6. What still limits it

- Hold_rec conversions (loss 24.3 of the remaining ~155): the ramp from tick 15 to 30 is still too slow (truth 3.50 vs
  2.84 at tick 18, 4.44 vs 3.67 at tick 30) and the level at the recovery action stays 0.1-0.25 below truth after tick 80.
  Hold_rec spend drifts low late (11.7 vs 13.0 at tick 119): the pool return at the recovery action is still too slow.
- Narrow breadth (0.1) at bid 5: the model depletes the narrow audience too fast (spend 5.4 vs 7.3 at tick 199, mean
  residual -2.9), and conversions there fall too early. Only 27 ticks of data at breadth 0.1 with bids on.
- Multilevel conversions under a tight cap at high bid (ticks 131-173, loss 12.2): the model rises too fast early
  (1.86 vs 1.05 at tick 147); the preparation built earlier in that run carries over too strongly.
- Three runs only; each LOO fold is a different regime, so LOO is a rough guide.
