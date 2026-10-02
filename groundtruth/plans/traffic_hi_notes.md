# traffic hi: residuals, speed laws, travel delay, lane storage (Tue Sep 29, no credits)

Goal: move traffic past z8 (public 0.8385, `plans/traffic_traffic_z8_z8all7_doc.json`). Rule (power_grid lesson):
a candidate must win on most folds, including the exam (p5.testlike) and pulse40, and the gain must not come
from one run.

## 1. Residuals of the z8 folds (`scripts/lab_hi_tr_resid.py`)

Per segment of the held-out run, z8 fold fit, score and bias in sigma units:

| run, segment | what the truth does | what z8 does |
|---|---|---|
| every start (roads empty) | flows exactly 0 for 11 ticks, then jump to level | smooth gamma rise from tick 1 (3 first-order stages) |
| pulse40 [40,80) recovery | flow_a at ~15 until t 72, flow_b until t 63, then exactly 0 | flow_b jumps to 17-23 and decays slowly; bias +2.25 sigma |
| pulse40 stock drained after the pulse | a 576, b 347 vehicles | a 618, b 665 (b exit section holds 513) |
| multilevel [109,125] drain | flows flat at 16.2 until the queue empties | decay starts at once |
| multilevel [121,144] ramp 0, lane .75 | speed_a back to 48.7, speed_b stuck at 31-34 with zero flow | both back to 48; speed_b bias +2.6 |
| multilevel [144,161] ramp .35 for 7 ticks | flows appear 11 ticks later (t 153-160), rising to 15 | spread-out 4-5 |
| exam [222,290], [372,396] empty roads | speed_b sits at 47.5 / 47.2 (a 48.9) for 70+ ticks | 48.8 |
| exam [300,360] 3-6 tick blocks | flows in platoons (0, 20, 45, 48) | mean level ~11; flat |

Data noise on a settled hold is ~0.05 (hold_mid): the flow platoons are the simulator's own dynamics, so
their mean is the best open-loop answer. Readings: (i) a sharp travel delay (~7-11 ticks), (ii) queues drain
at capacity then stop, (iii) lane closure lowers route b's speed even without traffic, (iv) lane closure
limits how many vehicles a route can hold (b's stock at lane .65 is 60 % of a's), (v) a small persistent
speed offset after congestion.

## 2. Families (all neutral = z8 exactly: max |dY| < 1e-14 on all 7 runs)

`gtlab/ode/traffic_hi.py` (+ variants `traffic_hi_{g,u,t,j,m,d}.py`): speed law switch and entry chain.
`traffic_hi2.py` (+ `traffic_hi2_x3.py`), `traffic_hi3.py`, `traffic_hi4.py`: output / pipeline terms.
Terms (neutral value in brackets):

- capacity sharpness $\mathrm{smin}(x,c) = c\,q/(1+q^{p_s})^{1/p_s}$, $q = x/c$ [$p_s = 4$]; large $p_s$ = hard $\min$.
- speed laws on the lagged occupancy $m$: Greenshields $v_f((1-m/n_{ref})^+)^{\gamma}$, Underwood
  $v_f e^{-(m/n_{ref})^\gamma}$, triangular $\mathrm{smin}(v_f\gamma(1-m/n_{ref})^+ n_{ref}/m,\ v_f)$,
  journey (Little) $v_f\min(1, T_0(out+e)/(m+T_0e))^\gamma$, mix $w_j\,$journey $+ (1-w_j)\,$hill.
- entry travel chain: $K = 4$ stages of rate $K/D_{tr}$ between admission and the approach queue [none].
- crossing chain $K_X$ stages of rate $K_X r$ [1]; queue service rate $r_q r$ on $p_1, p_3$ [$r_q = 1$].
- stale journey memory: $\dot J_i = r_J\,\frac{out_i}{out_i+f_0}(\mathrm{tgt}_i - J_i)$, reported speed relaxes to
  $(1-w_j)\mathrm{tgt}_i + w_j J_i$, $J_i(0) = v_f$ in hi4 [$w_j = 0$].
- lane free speed $v_i = v_f(1 - c_{l,i}\,\mathrm{lane}^{n_v})^+$ [$c_l = 0$].
- lane storage $n_{max,i} = n_{max}(1 - c_{n,i}\,\mathrm{lane})^+$ [$c_n = 0$].

## 3. Protocol (`scripts/lab_hi_tr_loo.py`)

LOO at the calibrated 1.0 sigma over all 7 runs + full fit, one start, `max_nfev` 60, free = base + A + B +
new terms, no time budget, folds as parallel processes. Fold k starts from the z8 fold-k fit
(`plans/traffic_tm_z8base.json`), or with `--warm` from an earlier candidate's fold-k fit: never from a fit
that saw run k. `base` = z8 refit under this protocol (the control). `d2` = `d` refit once more with no
new term: 0.7458 vs 0.7452, so extra iterations alone give nothing; the gains below are structure.
p7 fold is bimodal (0.61 or 0.66 for the same model, e.g. base vs z8(tm), d vs d2): treat ±0.05 on p7 as noise.

## 4. Fold table (LOO score of the held-out run)

| cand | terms | hold_rec | pulse40 | mid40 | multilevel | hold_mid | exam | p7 | LOO | vs base |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z8 (tm protocol) | | .965 | .525 | .853 | .678 | .809 | .729 | .613 | .7389 | −.006 |
| base | z8 refit | .965 | .521 | .853 | .678 | .809 | .729 | .661 | .7452 | |
| ps | hard capacity | .965 | .522 | .852 | .674 | .803 | .729 | .662 | .7440 | −.001 |
| q | point queue $r_q$ | .965 | .522 | .852 | .675 | .811 | .728 | .619 | .7391 | −.006 |
| x3 | crossing chain 3 | .963 | .527 | .863 | .686 | .810 | .733 | .613 | .7420 | −.003 |
| g | Greenshields | .932 | .506 | .840 | .647 | .918 | .722 | .613 | .7399 | −.005 |
| u | Underwood | .988 | .522 | .837 | .666 | .834 | .732 | .661 | .7485 | +.003 |
| t | triangular (bug, see §7) | .516 | .566 | .630 | .547 | .619 | .556 | .569 | .5717 | −.174 |
| j | journey speed (Little) | .908 | .518 | .653 | .619 | .801 | .677 | .589 | .6805 | −.065 |
| m | journey/hill mix | .981 | .524 | .833 | .677 | .826 | .726 | .618 | .7406 | −.005 |
| l | lane free speed | .987 | .525 | .849 | .693 | .812 | .727 | .661 | .7508 | +.006 |
| s | stale memory ($J_0$ = y0) | .932 | .542 | .864 | .649 | .848 | .737 | .659 | .7474 | +.002 |
| n | lane storage | .967 | .548 | .852 | .674 | .847 | .739 | .605 | .7474 | +.002 |
| d | entry chain | .947 | .514 | .866 | .712 | .835 | .731 | .661 | .7522 | +.007 |
| dl | d + l | .973 | .516 | .848 | .720 | .829 | .735 | .616 | .7482 | +.003 |
| ds | d + stale | .989 | .528 | .898 | .698 | .859 | .733 | .655 | .7660 | +.021 |
| dn | d + storage | .955 | .578 | .867 | .728 | .870 | .766 | .621 | .7692 | +.024 |
| dls | d + l + stale | .973 | .533 | .901 | .720 | .854 | .741 | .682 | .7721 | +.027 |
| dsn | d + stale + storage | .989 | .541 | .892 | .728 | .853 | .749 | .666 | .7740 | +.029 |
| **all_s** | d + stale + l + storage | .986 | .563 | .891 | .718 | .862 | .772 | .690 | **.7831** | +.038 |
| all_s2 | all_s refit (fixed point) | .986 | .563 | .891 | .718 | .862 | .772 | .690 | .7831 | +.038 |
| all_n | same terms, dn lineage | .986 | .595 | .891 | .714 | .868 | .771 | .668 | .7846 | +.039 |
| all_f | all_s, $n_{max}$ held 1070 | .958 | .533 | .903 | .673 | .867 | .747 | .695 | .7680 | +.023 |
| dln | d + l + storage, dn lineage | .971 | .581 | .851 | .712 | .868 | .770 | .643 | .7708 | +.026 |
| **dln_s** | d + l + storage, all_s lineage | .970 | .551 | .854 | .712 | .862 | .770 | .652 | **.7672** | +.022 |

Per observable (flow_a flow_b speed_a speed_b), z8 / dln_s / all_s: pulse40 .44 .38 .66 .63 / .45 .41 .69 .66 /
.45 .40 .72 .67; multilevel .65 .61 .74 .71 / .70 .68 .73 .75 / .70 .68 .74 .75; exam .71 .69 .82 .70 /
.76 .72 .86 .74 / .76 .73 .82 .78; p7 .76 .33 .55 .81 / .82 .58 .63 .58 / .81 .58 .71 .67.

## 5. Checks beyond LOO

- **Physical sanity** (`scripts/lab_hi_tr_sanity.py`, 4,000-tick holds through the runtime): the dn lineage
  (dn, dln, all_n) fits route b's exit law as a cliff ($L_b = 0.655$, $n_l = 5.4$): at the pulse action
  flow_b = 0.6, while pulse40 shows 7-10. **Rejected.** dln_s and all_s keep $L_b \approx 0.93$,
  $n_l \approx 1.7$ (pulse hold flow_b 8.4). All finite; 0.7 s per 4,000-tick episode (z8 1.0 s).
- **Public-band consistency** (`scripts/lab_hi_tr_band.py`: candidate as truth on 6 schedules per category,
  implied score of each scored upload vs its public band; RMSE):

| doc | sustained | sequence |
|---|---:|---:|
| z8 | .097 | .031 |
| base / d / l / n / u | .092 / .094 / .084 / .088 / .092 | .031 / .032 / .031 / .029 / .031 |
| any with stale memory (s, ds, dls, all_s, all_f) | .075-.090 | **.046-.051** |
| dln_s | .089 | .030 |

The stale memory makes older uploads look 0.04 worse on the sequence categories than the leaderboard
measured: in test-like recovery schedules its memory $J$ freezes near 25-28 and every empty-road speed sits
~1.5 below free for the rest of the episode. The exam shows such an offset on route b (47.5 vs 48.9), but
the bands say it is not general. The stale term is kept out of the pick.
- **Unidentified regime**: with storage the full-throughput hold (lane 0, toll 0, ramp 1) settles at speed 6
  (z8 12.4); no run holds that regime. Flows agree (16.8 vs 17.7).
- **Packaging**: `flatpack.write_folder` + smoke into a scratch folder; packed predict = dev rollout
  (max diff 0.0); imports json/math/pathlib/numpy only.

## 6. Decision and forecast

- **Pick: `plans/traffic_hi_dln_s_doc.json`** (family `traffic_hi4`, mech AB): z8 + entry travel chain
  ($D_{tr}$ 7.0), lane-dependent free speed on route b ($c_{l,b}$ 1.12, $n_v$ 4.5), lane-dependent storage
  ($c_{n,a}$ 0.94, $c_{n,b}$ 1.07, $n_{max}$ 2024). Wins pulse40 (+.030), exam (+.041), multilevel (+.034),
  hold_mid (+.053), hold_rec (+.005); ties mid40 (+.001) and p7 (−.009 vs base, +.039 vs z8 tm); band
  consistent. Forecast: 0.8385 + 0.6 × 0.022 to 0.6 × 0.028 ≈ **0.85** (range 0.84-0.86; the full-throughput
  regime is the main risk).
- Higher LOO, band warning: `plans/traffic_hi_all_s_doc.json` (adds the stale memory): LOO .7831, 7/7 folds,
  but sequence band .048. Forecast 0.8385 + 0.6 × 0.038 ≈ 0.86 if the memory is real, below dln_s if not.
- Speed-law replacements (Greenshields, Underwood, triangular, journey time, flow-weighted mix) and hard
  capacity / point queues do **not** help: the hill law is not the problem; $v_f$ = 48.7-49.0 is the
  measured empty-road speed (hold_rec reads 48.7-49.1), not a fitting artefact.

## 7. Caveats

- `t` ran with a bug in `smin` (clipped $q$ at 30 returned $x/30$); fixed in all traffic_hi files before
  any later run; `t` was not re-run (its speed law was already far behind in the working part).
- Fold parameters of the lane terms move a lot ($L_b$ 0.64-6.6, $n_l$ 1-11.6, $c_{l,b}$ 0.3-1.5): the lane
  shape is weakly identified; the storage slope is set mainly by p7 (without p7: $n_{max}$ 1033, $c_{n,a}$ 0.45).
- Stale-memory $f_0$ sits on its upper bound (50) in most folds: the memory updates in proportion to flow.
