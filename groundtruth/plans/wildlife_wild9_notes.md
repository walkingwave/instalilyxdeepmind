# wildlife wild9: structure switches, two-stage fit, median ensemble (Mon Sep 28, no credits)

Data: 5 runs (p1 120, p2 400, p3 291, p4 300, p7 700). Scale: calibrated sigma (1.0x). LOO = fit on 4 runs,
score the 5th; every fold starts from the v8h p7 full-fit theta (same optimism for every candidate).
Base `wildlife_wild90` = v8h rebuilt as a switch template (`scripts/lab_wild9_make.py`); it reproduces the
v8h p7 LOO exactly (0.6751; folds .635 .724 .594 .666 .756).

## Where the loss is (v8h p7, in-sample loss share per segment)
p7 hold predators (1.66 data vs 1.59 model), p7 after the switch to (8, 1, 0) (north climbs to 20, south
falls to 8.7; model 11 / 10.6), p2 pulse and recovery (overshoot peak 206 vs 159), p3 low-habitat hunting
(floor ~19 / 15 in p3 vs 11 / 11 in p7 under nearly the same controls: a history effect no family has).

## Switches tried (LOO mean; folds p1 p2 p3 p4 p7)
| family | change | in-sample | LOO | folds |
|---|---|---|---|---|
| wild90 (= v8h) | | 0.727 | 0.6751 | .635 .724 .594 .666 .756 |
| wild9a | harvest refuge grows with cover per region, $P_{h,i} = P_h(1 + h_i u_p)$ | 0.729 | 0.6637 | .639 .738 .603 .666 **.673** |
| wild9i | predators feed on exposed prey, $Z = Pe/(Pe + 100 s)$, $e = 1 + x_i(1-u_p)$ | 0.733 | 0.6720 | .618 .753 .592 .639 .758 |
| wild9m | i + predator journeys $\propto P/(P + P_t s)$ | 0.737 | 0.6639 | .628 .741 .600 **.587** .764 |
| wild9o | predator food half-saturation fitted | | 0.6650 | .609 .713 .585 .661 .757 |
| wild9b/c/d/e/f/g/h/j | pt, phr+pt, south quota factor, refuge, $P_h \propto s$, qx, own south $b_h$ | 0.719-0.738 | not run | |
| w2 (refit, for blending) | | 0.747 | 0.6600 | .673 .726 .596 .684 .621 |

Every added mechanism raises in-sample and the pulse fold but breaks the one fold that identifies it
(phr the long hold, gx the habitat run). Rejected.

## Two-stage fit (`scripts/lab_wild9_2stage.py`)
In v8h the prey equations do not depend on predators, so prey parameters can be fitted on prey alone.
Stage 1: prey parameters on prey observables (predator sigma x1e4). Stage 2: predator parameters with prey held.
With shared transit parameters (wild90) stage 2 cannot fit the predators (p4 pred S .549): LOO 0.6704.
**wild9p**: predators get their own transit ($e_{mq}, \tau_q, s_{vq}$), 21 parameters. Fit: prey transit
$\tau$ 26 -> 88, $s_v$ .78 -> .90; predators $\tau_q$ = 18, $s_{vq}$ = .81.

| candidate | LOO | p1 | p2 pulse | p3 | p4 | p7 long hold |
|---|---|---|---|---|---|---|
| v8h p7 (current) | 0.6751 | .635 | .724 | .594 | .666 | .756 |
| **wild9p two-stage** | **0.6897** | .632 | **.761** | .639 | .680 | .737 |
| **median(v8h, wild9p, wild9m)** | **0.6846** | .634 | .748 | .607 | .677 | .757 |
| mean(v8h, wild9p) | 0.6864 | .634 | .743 | .621 | .686 | .747 |

In-sample: wild9p 0.738, median 0.733 (v8h 0.727). Both finite on 4,000-tick eval schedules, 0 % of ticks
outside the data range; wild9p 0.3-0.4 s, median 1.0 s per episode.

Docs: `plans/wildlife_wild9_s2p_doc.json` (single ode), `plans/wildlife_wild9_ens_doc.json` (median of three).
Thetas: `plans/wildlife_wild9_s2p.json`, `plans/wildlife_wild9_m.json`, v8h `plans/wildlife_wildlife_v8h_p7.json`.
