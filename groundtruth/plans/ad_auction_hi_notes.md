# ad_auction hi (Tue, no credits): audience split into breadth bins

Start: final1 perobs (win = v8b, spend = med(v8b,s1), conversions = med(v8b,min)), public 0.8656.

## Residuals of final1 on every run (1.0 sigma, `scripts/lab_hi_ad_resid.py [--regimes]`)

| run | win | spend | conv | mean | conv bias (sigma) |
|---|---:|---:|---:|---:|---:|
| p1.hold_rec | 0.945 | 0.957 | 0.789 | 0.897 | -0.29 |
| p2.pulse60_120 | 0.926 | 0.912 | 0.848 | 0.895 | +0.02 |
| p2.multilevel200 | 0.919 | 0.925 | 0.758 | 0.868 | +0.14 |
| p6.exam | 0.877 | 0.867 | 0.745 | 0.830 | -0.33 |
| p8.longhold | 0.892 | 0.929 | 0.848 | 0.890 | -0.25 |

The worst ticks are 10-50 after a change (conv 0.67-0.73). By regime, the big misses are all **widening
the targeting after a narrow phase**:
- exam t186 (4.22/83/0.75 after 0.17): spend 79 vs 39; exam t114 and t276 (after 0.55): 55 vs 47;
- p8 t590 (2.09/69/0.59 after 285 ticks at 0.19): spend 40 vs 22, win 0.27 rising to 0.36 (model flat 0.34),
  conversions peak 5.8 vs 3.6.

Diagnosis: v8b keeps the exposure removal $X$ and preparation $P$ as one fraction of "the targeted audience".
Widening then spreads the depleted narrow core over the new people, so the fresh audience never appears.

## Model: hi1 = v8b with the pool nested along breadth (`gtlab/ode/ad_auction_hi1.py`)

Bins $k = 0..9$ cover breadth $(0.1k, 0.1(k+1)]$, coverage $c_k = \mathrm{clip}((b - 0.1k)/0.1, 0, 1)$.
Each bin keeps its own $X_k$, $P_k$:

$$A = N \sum_k 0.1\,c_k (1 - X_k), \qquad imp_k = imp\,\frac{c_k(1-X_k)}{\sum_j c_j(1-X_j)}$$
$$e_k = \frac{imp}{A}(1 - X_k), \quad \dot X_k = k_d c_k e_k - X_k/\tau_e, \quad
\dot P_k = c_k k_p \frac{e_k}{1 + e_k/E_0}(1 - P_k)$$
$$\text{purchases} = c \sum_k imp_k (g_0 + (1-g_0)P_k), \qquad R = 1 - k_r\,(1 - A/(N b))$$

Auction, price, fulfilment ($Q_1, Q_2$, cap $F + K/T_K$) as v8b. With equal $X_k, P_k$ it is exactly v8b.
Same 14 parameters; 23 states. At the untouched final1 v8b theta (no refit) it already moves exam
0.811 → 0.834 and p8 0.879 → 0.898 (spend 0.835 → 0.913 on the exam). Batched and scalar paths agree
to 1e-14; the scalar path is plain `math`, 0.23 ms per tick.

Tried on top, no gain (fits stay at hi1): hi2 price $\times R^{k_{pr}}$ (second price follows rival
capital), hi3 preparation fades ($-P_k/\tau_p$), hi4 second-price payment shape $M(bid/B)^{g_s}$.
LOO means 0.8641 / 0.8640 / 0.8642 vs hi1 0.8641.

## Leave-one-run-out, 1.0 sigma (`scripts/lab_hi_ad_loo.py`, `scripts/lab_hi_ad_select.py`)

Protocol as str9/pair: Cauchy LSQ, 8 starts, 60 evals, 150 s per fold, start = final1 v8b theta.
`v8b`, `s1`, `min` = cached fold predictions of the shipped members (ens9); `v8r` = v8b refitted per fold
with our protocol (str9 lab thetas), the like-for-like control. Table in `plans/ad_auction_hi_select.json`.

| option (win; spend; conv) | p1 hold | p2 pulse | p2 multi | p6 exam | p8 long | mean | wins vs final1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| final1: v8b; med(v8b,s1); med(v8b,min) | 0.918 | 0.812 | 0.840 | 0.829 | 0.898 | 0.8592 | ref |
| final1 structure with v8r | 0.918 | 0.827 | 0.840 | 0.829 | 0.898 | 0.8623 | 2 |
| v8r alone | 0.907 | 0.834 | 0.848 | 0.823 | 0.899 | 0.8622 | 3 |
| hi1 alone | 0.891 | 0.851 | 0.842 | 0.838 | 0.899 | 0.8641 | 4 |
| **med(hi1,v8b); hi1; med(hi1,v8b)** | 0.898 | 0.853 | 0.842 | 0.842 | 0.902 | **0.8674** | **4** |
| med(hi1,v8r); hi1; med(hi1,v8r) | 0.898 | 0.866 | 0.843 | 0.842 | 0.902 | 0.8702 | 4 |
| hi1; hi1; med(hi1,v8r) | 0.897 | 0.864 | 0.840 | 0.842 | 0.910 | 0.8706 | 4 |

Per observable (pick row): win 0.912, spend 0.922 (was 0.907), conversions 0.769 (was 0.765).
Against the like-for-like control (final1 structure with v8r) the pick row gains on 4/5 folds:
pulse +0.026 to +0.040, exam +0.013, p8 +0.005, multilevel +0.002; it loses p1.hold_rec −0.020.
The gain is spread over four folds, not carried by one run. hi1 alone only ties v8r in mean
(0.8641 vs 0.8622-0.8642): the gain comes from spend on widening plus the median with v8b.

## Decision

Candidate doc: **`plans/ad_auction_hi_hi1m_doc.json`** = perobs, win and conversions =
med(hi1, v8b final1 theta), spend = hi1 (full fit on all 5 runs, `plans/ad_auction_hi_ad_auction_hi1_a.json`);
final1's clip and post rule (spend ≤ 1.04 cap, lag 1). Use it with `build_cfg.py` as
`"ad_auction": {"doc": "plans/ad_auction_hi_hi1m_doc.json"}`. Flat predict.py matches the dev
runtime exactly, finite on all four eval shapes, same run time as final1 (≈2.2 s per 4,000-tick episode
under load).

Acceptance: mean up (+0.008 vs final1, +0.005 vs like-for-like), exam up (+0.013), pulse up (+0.04),
4/5 folds. Risk: p1.hold_rec −0.020 (steady recovery hold from reset, the sustained shape).

Forecast: public ad_auction 0.8656 → about 0.870-0.875 (+0.005 to +0.01); 0.86 if the p1 loss is what the
sustained tests look like.

Not fixed: conversions stay the weak observable (0.77). Remaining pattern: fresh start peaks too low
(exam t20: 6.9 vs 5.4) but re-widening peaks too high (exam t198: 5.8 vs 7.0); conversions at bid 5 with a
binding cap rise 3.0 → 3.75 and the win rate is 0.34 vs 0.27 (price per impression at a high bid is lower
than $(bid/1.5)^{\gamma}$ when the budget binds). Neither the fading preparation nor the second-price
shape moved the fit within 60 evals; a longer fit with those two terms free is the next try.
