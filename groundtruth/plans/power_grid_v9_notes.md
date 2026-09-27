# power_grid v9: fixing the long-hold frequency map of v8g

Lab notebook, Sun Sep 27 (night). No credits. Same three runs as v8 (`p1.hold_rec` 120,
`p2.pulse60_120` 180, `p2.multilevel200` 200). Scale: calibrated organizer sigma
(`plans/sigma_calibrated.json`, [load 9.65, frequency 0.308, share 0.066]). Every fit below is a
plain lab fit (`scripts/ode_lab.py --mech AB --budget 300 --starts 10 --nfev 60 --sigma-cal 1.0`),
no score-polishing: the polished v8g doc over-fit publicly.

## 1. The defect

u012 (v8g) gained on the sequence band (0.767 -> 0.778) but lost on sustained (0.715 -> 0.698).
`plans/longhold_audit.md` located it in the frequency under a long reserve hold. We rolled every
model 4,000 ticks on the recovery base (price 1.5, charging 1) with reserve 0 or 150 and read the
last tick:

| model | ic 0 | ic 0.5 | ic 1 |
|---|---:|---:|---:|
| u008 (power_grid_min) | +0.297 | +1.138 | +1.979 |
| v8g doc (u012) | **-0.374** | +1.065 | +1.299 |
| v8g lab doc | -0.657 | +0.971 | +1.390 |

(steady-state frequency at reserve 150 minus reserve 0, Hz)

Why v8g goes negative at ic 0. With $i_f = i_0 + (1-i_0)i$, $i_0=0.15$, v8g has
$$R_{eff} = \big(c_r(1-\beta)R_s + \text{draw}\big)i_f,\qquad b = \dots + R_{eff} - P_{ch} - L.$$
On a long hold the store settles where refill equals draw, $P_{ch}=\text{draw}$, so the storage part
nets $\text{draw}\,(i_f-1)\le 0$: the store is refilled at full grid cost but delivers only a share
$i_f$ of its output. At ic 0 with the v8g doc ($c_r=1.49$, $\beta=0.555$, $p_{ch}=40.3$):
$p_{ch}(1-E)=c_r\beta R_sE \Rightarrow E=0.245$, draw $=30.4$; conventional $0.15\cdot1.49\cdot0.445\cdot150=14.9$,
storage out $4.6$, refill $-30.4$: net $-10.9$ power units, $\times k_f/(1+k_fk_g)$ = about $-0.37$ Hz. Matches.

Why v8g under-shoots at ic 1. The soft ceiling $f^\star\leftarrow f_{hi}\tanh(f^\star/f_{hi})$ with
$f_{hi}=1.56$ caps frequency at 51.56 on any hold; u008 has no ceiling and reaches 52.31.

## 2. Structure (v9 family)

All variants copy v8g and change only the balance.

- **Refill over the same connection** (all v9*): $b = \dots + R_{conv} + \text{draw}\,i_f - P_{ch}\,i_f - L$,
  $R_{conv}=c_r(1-\beta)R_s i_f$. On a hold the storage nets to zero and only the conventional
  share persists: $\Delta f_{ss}\propto k_f c_r(1-\beta)R_s i_f/(1+k_fk_g) \ge 0$ at every ic, linear in
  $i_f$ as in u008 (u008's ratio ic0/ic1 = 0.297/1.979 = 0.15 = $i_0$). Transients at ic near 1 are
  unchanged (all refill segments in our data have ic 0.94-1 or tiny charging).
- v9, v9e: governor restores toward a reserve-dependent set point, target $-k_g(\delta f - g_s k_f R_{conv})$, $g_s\in[0,1]$.
- v9b: no frequency ceiling.
- v9d, v9e: sharp ceiling (smooth min, width 0.1 Hz) instead of tanh: linear below $f_{hi}$.
- `f_hi` bound widened 2.5 -> 3.5 (never reached).

We checked the refill fix alone at the v8g doc's theta (no refit): in-sample 0.7964 (v8g 0.7965),
map +0.47 / +1.18 / +1.30. The sign is fixed at no cost; ic 1 stays capped by the ceiling.

## 3. Results

In-sample = mean over the 3 runs of `metric.score` at the calibrated sigma, Y from
`runtime.infer.rollout_from_blob(doc)`. LOO = the lab's leave-one-run-out mean (v8g lab: 0.726).
Map = 4,000-tick steady-state frequency delta at reserve 150 vs 0 (u008: +0.30 / +1.14 / +1.98).

| candidate | in-sample | LOO | ic 0 | ic 0.5 | ic 1 | notes |
|---|---:|---:|---:|---:|---:|---|
| v8g doc (u012) | 0.797 | (0.726 lab) | -0.37 | +1.07 | +1.30 | shipped |
| v8g lab | 0.790 | 0.726 | -0.66 | +0.97 | +1.39 | |
| **v9c** refill fix | **0.792** | **0.731** | +0.38 | +1.15 | +1.41 | $f_{hi}=1.73$ |
| v9 + gov set point | 0.792 | 0.728 | +0.37 | +1.14 | +1.42 | $g_s\to0.05$ |
| v9b no ceiling | 0.773 | 0.701 | +0.20 | +0.76 | +1.32 | fits worse |
| v9d sharp ceiling | 0.787 | 0.708 | +0.35 | +1.28 | +1.34 | $f_{hi}=1.59$ |
| v9e sharp + gov | 0.787 | 0.727 | +0.35 | +1.28 | +1.35 | $g_s\to0$ |

Absolute frequency at reserve 150 (v9c vs u008): ic 0 50.64 vs 50.63, ic 0.5 51.41 vs 51.47,
ic 1 51.67 vs 52.31 (0.64 Hz = 2.1 sigma; v8g doc 0.76 Hz = 2.5 sigma).

Agreement with v8g on eval-shaped 4,000-tick schedules (score with v8g as truth, calibrated sigma):
v9c sustained 0.910, order 0.947, recovery 0.965, composition 0.948. The transient part is kept;
the difference sits in the long holds, which is what we wanted to change.

Eval sanity (lab, 4 categories x 4,000 ticks): all finite, 0.6 s per episode (about 24 s for 40).

## 4. Decision

- **Recommend v9c**: `plans/power_grid_power_grid_v9c_v9c_doc.json`. LOO 0.731 >= v8g 0.726,
  in-sample 0.792, sign fixed at every interconnector, ic 0 and 0.5 within 0.06 Hz of u008.
- Rejected: the governor set point (data drive $g_s$ to 0: no evidence; extra parameter), removing
  the ceiling (v9b: -0.025 LOO; the data do show a ceiling, true frequency tops out at 51.89 twice),
  the sharp ceiling (v9d/e: lower $f_{hi}$ and lower LOO than tanh).
- Open: ic 1. The data want a ceiling near 51.7-51.9; u008 has none and sits at 52.31 on a long
  ic-1 hold at recovery load. None of our runs holds reserve at ic 1 for more than 58 ticks at low load,
  so the two cannot be told apart from what we own. We do not know how much of the 0.017
  sustained gap sits at ic 1; the 0.64 Hz (2.1 sigma) gap there stays open. A test-shaped long run
  with reserve at ic 1 on the recovery base would settle it.
