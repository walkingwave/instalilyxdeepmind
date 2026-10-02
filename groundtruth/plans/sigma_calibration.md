# Organizer sigma calibration, all ten systems

Date: 2026-09-26. Script: `scripts/calibrate_sigma.py` (free, ~20 s; `--alt` adds the eval_like check).
Numbers: `plans/sigma_calibrated.json`. No credits spent, no gateway calls.

## 1. Question

The public score is $\frac{1}{T p}\sum_{t,j} 1/(1+|\hat y_{tj}-y_{tj}|/\sigma_j)$ with a frozen organizer
scale $\sigma_j$ per observable. Our fitting and screening use $\sigma^{proxy}_j$ (std of all our
observations, floored at 3x diff noise). The wildlife study found $\sigma \approx 0.15\,\sigma^{proxy}$.
We now estimate $\sigma_j = k\,\sigma^{proxy}_j$ for every system.

## 2. Method

1. **Anchors.** Every upload with a public score for the system (u001 persistence, u002-u005, u008,
   u009, u010b). We import the shipped `predict.py` and roll it on each of our ledger runs
   ($y_0$ = run.y0, $U$ = run.U). Uploads with identical predictions on every run (e.g. u008 = u009 =
   u010b for power_grid) are merged into one anchor (public scores averaged; they were identical).
   u001 has only a zip in its folder, so persistence is rolled directly ($\hat y_t = y_0$).
2. **Run-based score.** $\mathrm{ours}_i(k) = \frac1R\sum_r \frac{1}{T_r p}\sum_{t,j} 1/(1+|\hat y^i_{rtj}-y_{rtj}|/(k\,\sigma^{proxy}_j))$,
   equal weight per run (like the 40 equal-weight episodes).
3. **Fit.** $k^* = \arg\min_k \sum_i(\mathrm{ours}_i(k)-\mathrm{public}_i)^2$ (bounded search on $\log k$).
   Reported: $R^2$, RMSE, residual per anchor, the $k$ range where SSE stays within +50%,
   leave-one-anchor-out RMSE.
4. **l1 anchors excluded.** l1 fits its initial state on the noisy $y_0$ of the very runs we score it on;
   it is wildly optimistic in-sample (ours 0.65-0.93 vs public 0.34-0.63: market u004, wildlife u003,
   ad_auction u003, hospital u003). Kept in the json as `k_with_excluded`.
5. **Per-observable $k_j$.** L-BFGS on $\log k_j$ with a weak ridge ($10^{-3}$) towards $\log k^*$. Accepted
   only if $n \ge p+2$ anchors, Jacobian $\partial\,\mathrm{ours}_i/\partial\log k_j$ condition < 30,
   SSE drops by half, and leave-one-out RMSE drops by 20%. Only supply_chain passes.
6. **Checks.**
   - *Out-of-sample:* each anchor scored only on runs bought after it was uploaded ($k_{oos}$).
   - *Tick-pooled* instead of per-run average ($k_{pool}$).
   - *Bands:* hold-like runs (hold_rec, hold_mid, multilevel200, pulse200_200, pulse120_280) vs the
     sustained band; the rest vs the sequence band ($k_{sus}$, $k_{seq}$).
   - *eval_like:* 8 schedules (2 per category, $T$ = 1000), best-scored upload as truth, $k$ fitted on
     the other anchors ($k_{alt}$).
   - *Noise floor:* diff noise $\hat s_j = 1.4826\,\mathrm{MAD}(\Delta y_j)/\sqrt2$; flag $\sigma_j < 2\hat s_j$.

## 3. Results

| system | $k^*$ | 50% SSE range | $R^2$ | RMSE | LOO RMSE | $k_{oos}$ | $k_{pool}$ | $k_{alt}$ | $k_{sus}$ / $k_{seq}$ | n anchors |
|---|---|---|---|---|---|---|---|---|---|---|
| epidemic | 0.133 | 0.105-0.171 | 0.85 | 0.055 | 0.071 | 0.21 | 0.12 | 0.15 | 0.12 / 0.17 | 5 |
| market | 0.085 | 0.065-0.105 | 0.69 | 0.063 | 0.076 | 0.14 | 0.10 | 0.07 | 0.06 / 0.10 | 6 (u004 l1 out) |
| traffic | 0.570 | 0.483-0.683 | 0.93 | 0.043 | 0.053 | 1.12 | 0.64 | 0.72 | 0.16 / 1.04 | 6 |
| power_grid | 0.432 | 0.319-0.595 | 0.18 | 0.082 | 0.104 | 0.80 | 0.46 | 0.49 | 0.23 / 0.59 | 5 |
| supply_chain | 0.565 | 0.451-0.683 | 0.88 | 0.042 | 0.051 | 3.1 | 0.95 | 0.35 | n/a | 7 |
| wildlife | 0.171 | 0.149-0.196 | 0.96 | 0.037 | 0.044 | 0.27 | 0.17 | 0.18 | 0.14 / 0.23 | 5 (u003 l1 out) |
| reservoir | 0.428 | 0.366-0.483 | 0.92 | 0.041 | 0.051 | 0.68 | 0.45 | 0.49 | n/a | 5 |
| ad_auction | 0.521 | 0.392-0.683 | 0.73 | 0.068 | 0.091 | 0.78 | 0.55 | 0.53 | 0.38 / 0.68 | 4 (u003 l1 out) |
| social_contagion | 0.155 | 0.149-0.159 | 0.94 | 0.020 | 0.023 | 0.27 | 0.18 | 0.29 | 0.08 / 0.33 | 7 |
| hospital_queue | 0.229 | 0.171-0.297 | 0.37 | 0.067 | 0.079 | 0.50 | 0.29 | 0.31 | 0.08 / 0.38 | 6 (u003 l1 out) |

Wildlife reproduces the earlier study (0.15, inside our 0.149-0.196 range; eval_like check 0.18).

### Organizer sigma per observable ($k^*\sigma^{proxy}$; noise floor applied where flagged)

| system | observable: $\sigma$ (proxy) |
|---|---|
| epidemic | daily_cases 12.9 (97.1), hospital_load 5.23 (39.2) |
| market | price 0.82 (9.59), volume 0.64 (7.56), depth 2.08 (24.4) |
| traffic | flow_a 4.01 (7.04), flow_b 3.53 (6.20), speed_a 5.21 (9.15), speed_b 4.60 (8.07) |
| power_grid | load 9.65 (22.4), frequency 0.308 (0.713), renewable_share 0.0659 (0.153) |
| supply_chain | shipments 9.00 (15.9), inv_supplier 98.1 (174), inv_retail 235 (415); **per-obs fit: 13.7, 185, 49.0** |
| wildlife | prey_N 9.28 (54.3), pred_N 0.187 (1.10), prey_S 7.66 (44.8), pred_S 0.253 (1.48) |
| reservoir | level 110 (258), inflow 0.673 (1.57), outflow 1.85 (4.32), quality 0.0073 -> **0.0113 floored** (0.0170) |
| ad_auction | win_rate 0.0907 (0.174), spend 9.82 (18.8), conversions 0.881 (1.69) |
| social_contagion | adopters_a 9.23 (59.6), adopters_b 5.02 (32.4) |
| hospital_queue | wait_time 22.9 (100), queue 27.5 (120), discharges 1.21 -> **2.15 floored** (5.27) |

Noise flags ($\sigma < 2\hat s$): reservoir quality ($\sigma/\hat s$ = 1.3), hospital discharges (1.1).
The organizer compares with noiseless truth, but our "truth" is noisy: on these two observables our
residual is mostly our own measurement noise, so $k^*\sigma^{proxy}$ is not credible there; we floor at $2\hat s$.
Market price was flagged (1.7) when u004 was in the fit; at $k^* = 0.085$ it is 2.3, just clear.

### Per-observable $k_j$ (unconstrained fit; only supply_chain is identified)

| system | $k_j$ | note |
|---|---|---|
| supply_chain | 0.86, 1.07, **0.12** | accepted: LOO RMSE 0.039 vs 0.051 scalar, cond < 30 |
| market | 0.12, **0.02**, 0.20 | not identified (LOO gain < 20%) |
| epidemic | **0.03**, 0.45 | not identified (2 obs, LOO 0.069 vs 0.071) |
| hospital_queue | 0.87, **0.07**, 0.32 | not identified |
| power_grid | 0.21, 0.24, 3.8 | not identified (fewer than p+2 distinct anchors) |
| reservoir | 1.21, 0.47, 0.46, 0.17 | not identified |

These hints agree in one direction: where one observable is much smoother/steadier than the others
(market volume, hospital queue, supply retail inventory, epidemic cases), the organizer scale for it
looks much tighter than its share of our std.

## 4. What it implies

- **Scale.** Organizer sigma is 0.09-0.57 of $\sigma^{proxy}$, not 1. Three groups:
  tight ($k \approx 0.09$-0.23: market, epidemic, social_contagion, wildlife, hospital_queue) and loose
  ($k \approx 0.43$-0.57: power_grid, reservoir, ad_auction, traffic, supply_chain). Local screens at
  $\sigma^{proxy}$ overstate the tight group by 0.2-0.3 and compress differences between models.
- **Which observable dominates the loss** (share of $1-\text{score}$ of the best upload, at $\sigma^{proxy}$ -> calibrated):
  - supply_chain (per-obs fit): inventory_retail 0.23 -> **0.54**, shipments 0.48 -> 0.31. Retail inventory,
    not shipments, is the lever if the per-obs fit holds.
  - market: price 0.54 -> 0.42, volume 0.18 -> **0.26**, depth 0.29 -> 0.33. Volume matters more than we thought.
  - hospital_queue: discharges 0.55 -> 0.43, queue 0.20 -> **0.29**. Queue matters more; discharges less
    (partly noise floor).
  - reservoir: quality 0.57 -> 0.46, inflow 0.21 -> 0.26 (floor on quality).
  - All others move by < 0.06: a scalar $k$ barely re-weights observables; the per-observable story
    needs more anchors.
- **Model selection.** The worst in-sample-vs-public misses are all l1 (excluded) plus the market lam 0.8
  blend (u005: ours 0.24 vs public 0.34). Rank agreement (Spearman, ours at $k^*$ vs public, non-l1 anchors):
  epidemic, wildlife, reservoir 1.0; supply_chain 0.96; hospital 0.94; social 0.93; power_grid 0.90;
  ad_auction 0.80; traffic 0.71; market 0.66. Market and traffic local screens are the least trustworthy
  for picking between close models (e.g. market u008 is ranked above u009/u010b locally, below publicly).
- `sigma` in the json is the value to use for fitting/screening (noise floor applied); supply_chain also
  carries `sigma_per_obs_fit`.

## 5. Caveats

- Our runs are not the test distribution: fitted models are scored partly in-sample, which biases $k^*$
  low; the out-of-sample variant ($k_{oos}$, few runs) sits 1.5-2x higher. Truth is likely between; $k^*$ is
  the right factor to translate an in-sample local score into a public number (wildlife v7 check).
- Bands disagree ($k_{sus}$ < $k_{seq}$ in every system with bands): our hold runs are shorter and
  calmer than 4,000-tick sustained episodes, so hold-run scores are not a clean sustained proxy.
- Our $y$ is noisy, the organizer truth is not: for observables near the noise floor (reservoir quality,
  hospital discharges) the fit is noise-dominated.
- power_grid ($R^2$ 0.18), hospital_queue (0.37) and ad_auction (4 anchors) are weakly anchored.
- supply_chain $k_{oos}$ = 3.1 and $k_{alt}$ = 0.35 bracket $k^*$ widely: its long holds dominate.
