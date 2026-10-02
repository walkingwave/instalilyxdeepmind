# Mechanism-pair pruning: ad_auction and reservoir (Mon, no credits)

Question: the brief says exactly two of three named mechanisms are active. The text-mining pass
(`plans/textmine_report.md`, rank 6) flagged our ad_auction and reservoir families as running all three.
We map every family term to A/B/C, refit with each mechanism removed (AB, AC, BC), and keep a pair only
if it ties (within 0.005) or wins leave-one-run-out, without losing the pulse fold or the exam.
We also test the text-implied missing terms (ad_auction: converted customers leave the pool,
breadth-dependent fulfilment work; reservoir: screen fouling).

Protocol (identical for every variant): `scripts/lab_pair_loo.py`, Cauchy LSQ, 8 starts, 60 evals,
300 s full fit, 150 s per fold, same start per system; score $\frac{1}{1+|e|/\sigma}$ at the calibrated
$\sigma$ (1.0), lab rollouts (no post rule). Families `gtlab/ode/ad_auction_pair.py` (v8b with letters)
and `gtlab/ode/reservoir_pair.py` (str9 with letters); both reproduce their parent exactly with all
letters on / AC (max abs diff 1e-5 / 0). Fold ensembles: `scripts/lab_pair_ens.py`.
Reports: `plans/<system>_pair_<tag>.json`, docs `..._doc.json`.

## reservoir

Brief: "Groundwater, irrigated land and deposited material may return water or contaminants after a delay."

| letter | brief | family term |
|---|---|---|
| A | groundwater return | $G = g_{gw}\,\mathrm{softplus}_{20}(L_{gw} - L)$ into inflow and balance (bank storage $k_b$ tested with it) |
| B | irrigated-land return | $\dot R = (k_{ret}\,irr - R)/\tau_{ret}$ into inflow |
| C | deposited material | $\dot N = (L/L_{full} - N)/\tau_n$, quality target $-q_n N$, $N_0 = n_0$ |
| base | stratification, head law, season, seep, $f_{in}$ | $D$, $c_{out}(L/500)^{p_{out}}$, inflow(t) |

The shipped pick (str9 c) already has $k_{ret} = 0$ fixed: it **is** the AC pair. The textmine row
"all three present" refers to str9 a (irrigation return free).

LOO at 1.0 σ (start `plans/reservoir_str9_c_theta0_clean.json`):

| fold | **AC (str9 c)** | AB | BC | BC, no bank | ABC | AC + fouling |
|---|---:|---:|---:|---:|---:|---:|
| p1.hold_rec | **0.861** | 0.864 | 0.850 | 0.828 | 0.858 | 0.857 |
| p2.pulse200_200 | **0.878** | 0.875 | 0.660 | 0.652 | 0.865 | 0.872 |
| p2 with $k_{ret},\tau_{ret}$ frozen at full fit | | 0.866 | 0.873 | 0.819 | 0.878 | |
| p8.longhold | **0.886** | 0.859 | 0.863 | 0.834 | 0.885 | 0.884 |
| mean | **0.8750** | 0.8661 | 0.7907 | 0.7714 | 0.8692 | 0.8711 |

- AC reproduces the recorded str9 c (0.8745 → 0.8750). AB loses p8 quality (0.635 vs 0.744: without $N$
  nothing carries the slow quality slide at full pool). BC loses inflow badly (p2 fold inflow 0.47):
  irrigation return cannot stand in for the level-driven groundwater term, and p8 has no irrigation.
- Near-zero check: with A and B both on, $k_{ret}$ fits 0.011 (ABC) and 0.009 (AB, with $\tau_{ret}$ at its
  5-tick bound); every other return parameter is well inside its bounds. The data says B is the inactive one.
- Screen fouling $\dot F = g_f(1-a)(1-F) - F(r_{fl}\,rel/12 + r_a a)$, capacity $\times(1-F)$: $g_f$ fits
  0.00046 (equilibrium $F \approx 0.002$ on the aeration-off, release-12 blocks of p2 and p8), LOO −0.004.
  The head law already explains the discharge. Rejected.
- Mean of AC and ABC fold predictions: 0.8726. No gain.

**Decision: keep `plans/reservoir_str9_reservoir_str9_c_clean_doc.json` (= AC).** It is already a legal
pair and the best of every pair and extra tested. `plans/reservoir_pair_ac_ref_doc.json` is the same
model refitted (0.8750), not a new pick.

## ad_auction

Brief: "Rival campaigns may move a shared pool of capital between audiences, repeated exposure may
temporarily remove reachable people, and broad introduction may change the effect of later follow-up."

| letter | brief | family term (v8b) |
|---|---|---|
| A | rival capital | $R = 1 - k_r X$ in the rival bid $b_0 R$ |
| B | exposure removal | $A = N\beta(1 - X)$, $\dot X = k_d e - X/\tau_e$ |
| C | broad introduction prepares follow-up | purchases $c\,imp\,(g_0 + (1-g_0)P)$, $\dot P = k_p \frac{e}{1+e/E_0}(1-P)$ |
| base | auction, cap, fulfilment $F$ + idle capacity $K$ | |
| extra Y | "converted customers take time to become available again" | $\dot Y = k_c\,conv/(N\beta) - Y/\tau_c$, $A = N\beta(1 - X - Y)$ |
| extra kf | "different audiences require different amounts of fulfillment work" | $F_{eff} = F/(1 + k_f(\beta - 0.1))$ |

Without B, $X$ is still tracked as the exposure memory that drives $R$ ($k_d$ pinned, since only
$k_d k_r$ is identified). Without C, $g_0 = 1$. LOO at 1.0 σ (start `plans/ad_auction_str9_v8bship_theta.json`):

| fold | **ABC (v8b)** | BC (no rival) | AC (no exposure) | AB (no preparation) | ABC + Y | ABC + kf | BC + Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1.hold_rec | 0.907 | 0.891 | 0.810 | 0.913 | 0.920 | 0.907 | 0.901 |
| p2.pulse60_120 | 0.845 | 0.857 | 0.680 | 0.811 | 0.794 | 0.805 | 0.806 |
| p2.multilevel200 | 0.847 | 0.832 | 0.772 | 0.836 | 0.834 | 0.847 | 0.818 |
| p6.exam | 0.823 | 0.825 | 0.735 | 0.818 | 0.828 | 0.823 | 0.830 |
| p8.longhold | 0.899 | 0.894 | 0.841 | 0.886 | 0.907 | 0.900 | 0.898 |
| mean | **0.8642** | 0.8597 | 0.7675 | 0.8530 | 0.8565 | 0.8562 | 0.8505 |

Fold ensembles (same fold thetas): mean(ABC, BC) 0.8641; mean(ABC, BC, AB) 0.8605;
median(ABC, BC, AB) 0.8644. Shipped perobs (final1) on the runs it never saw: exam 0.830, p8 0.890.

- B (exposure removal) is essential: AC loses 0.097, every fold; the fit sends $k_r \to 0$ and $\tau_e$ to its
  400 bound, so rival capital cannot replace it.
- C (preparation) is active: AB loses 0.011 and the pulse fold (0.811 vs 0.845), the conversion ramp.
- A (rival capital) is the weakest: BC ties (−0.0045) and keeps the pulse fold (+0.013) and exam (+0.002),
  but loses win_rate on hold_rec (0.898 vs 0.944) and multilevel (0.884 vs 0.912): the win rate rise after
  depletion (0.245 → 0.287) is what $k_r$ carries. Full-fit $k_r$ = 0.39, far from zero; $b_0$ drops
  3.64 → 2.73 without it.
- Near-zero check of the full ABC fit: no mechanism parameter near a bound ($k_r$ 0.39, $k_d$ 0.147,
  $g_0$ 0.63; the flag on $k_p$ = 0.49 is a linear-span artefact of its log range).
- Converted customers leaving the pool (Y): $k_c$ 0.45, $\tau_c$ 24; wins p1, exam, p8 (+0.005 to +0.013)
  but loses the pulse fold (0.794 vs 0.845). Fails the acceptance rule. With BC it loses more (0.8505).
- Breadth-dependent fulfilment work: $k_f$ fits −0.017 (zero); LOO −0.008, pulse fold −0.04. Rejected.

**Decision: keep final1's ad_auction perobs.** BC is the only pair that ties and passes the pulse and exam
checks, and it is the pair the data prefers (A weakest), but it gains nothing on the mean, its exam fold
(0.825) is still below the shipped ensemble's held-out 0.830, and it gives up win_rate on two folds.
Doc if wanted as a hedge on the two-of-three rule: `plans/ad_auction_pair_bc_doc.json` (single member;
not recommended for Final without an ensemble check at the runtime path).

## Summary

| system | pairs (LOO) | active pair by the data | action |
|---|---|---|---|
| reservoir | AC 0.8750, ABC 0.8692, AB 0.8661, BC 0.7907 | A + C (irrigation return off) | none: the pick already is AC |
| ad_auction | ABC 0.8642, BC 0.8597, AB 0.8530, AC 0.7675 | B + C (rival weakest), tie with ABC | none: keep final1 |

Extras from the text (Y, kf, fouling) all fit to zero or lose the pulse fold. No winner; no doc changes.
