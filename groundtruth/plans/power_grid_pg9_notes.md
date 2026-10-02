# power_grid pg9: packaging w5, thermostat-population load

Mon Sep 28, no credits. Calibrated sigma [9.65, 0.308, 0.066]. Fit runs hold_rec, pulse60_120,
multilevel200, p8.longhold; p6.exam never fitted. Lab: `scripts/lab_pg9_fit.py`, build + verify:
`scripts/lab_pg9_build.py`.

## 1. w5 packaged

`submissions/20260928-1635-pg9-w5/power_grid` (predict.py + model.json from
`plans/power_grid_power_grid_w5_w5_doc.json`). Flat predict.py reproduces the lab exactly (exam 0.685,
p8 0.826, in-sample as in the w notes). 40 x 4,000 check episodes: finite, deterministic, 0.43 s/episode
on a quiet machine; static zip check clean. Drops into a Final build by copying the folder over
final1's `power_grid/`.

## 2. Thermostat-population load (`gtlab/ode/power_grid_pg9.py`)

The data show a hard load floor (~63, all cooling loads off after a price rise) and ceiling (~168, all on
after a price drop), then a rebound 45-50 ticks later. w5's damped oscillator cannot produce the floor
plateau or a rebound larger than the dip. pg9 models it the way the brief describes: two classes of loads
(tau 52 / 31 ticks, weights 0.39 / 0.61) as off/on densities on a temperature grid (32 cells),
off warms $\dot x=(1-x)/\tau$, on cools $\dot x=-x/\tau$, band centre $s(p)=s_{08}+s_1(p-0.8)+s_2(p-0.8)^2$,
width $w$, smooth switching at rate $r$; load $=b+P\cdot$(fraction on). Reset = stationary population at
price 0.8 (brief). Population dynamics are linear for a fixed price, so the runtime caches the generator
per price and steps it with one matvec.

| fit | LOO hold | LOO pulse | LOO multilevel | LOO p8 | LOO mean | exam |
|---|---:|---:|---:|---:|---:|---:|
| w5 (reference) | 0.770 | 0.688 | 0.733 | 0.650 | 0.710 | 0.685 |
| pg9 joint fit, all obs (`_a`) | 0.752 | 0.696 | 0.616 | 0.644 | 0.677 | 0.677 |
| pg9 load-only, reset offset free (`_L`) load only | 0.782 | 0.687 | 0.627 | 0.743 | | 0.702 |
| **pg9 load-only, no reset offset (`_L0`) load only** | 0.783 | 0.687 | 0.746 | 0.743 | | 0.701 |
| w5 load only | 0.787 | 0.637 | 0.763 | 0.703 | | 0.669 |
| **perobs: pg9_L0 load + w5 frequency/share** | 0.768 | 0.704 | 0.727 | 0.664 | **0.716** | **0.695** |

- The joint fit gains load but loses frequency on every fold (multilevel 0.38, p8 0.44): rejected.
- A free reset offset ($z_0=k(y_0-L_{ref})$) fitted without multilevel goes persistent ($a_z=0.02$) and
  costs the multilevel fold 0.12 on load (multilevel has the lowest $y_0$, 93). With $k=0$ (brief: every
  reset is the same population) the fold recovers.
- perobs vs w5 per fold: hold -0.001, pulse +0.017, multilevel -0.006, p8 +0.013, exam +0.011.

## 3. Decision

- **Safe pick: w5** (every fold ties or gains vs the shipped model).
- **pg9 perobs** (`submissions/20260928-1717-pg9-perobs`) is better on mean LOO (+0.006), the pulse fold
  and the exam (+0.011) but loses 0.006 on the multilevel fold, so it misses the "no fold loss" rule.
  Runtime about 1.5x w5 (two ODEs), still well under 2 s per episode.
- Not done: charging_allowance stays as in w5 (exam segments with only the charging change move the
  frequency by < 0.03 Hz, below noise); frequency is still w5's.
