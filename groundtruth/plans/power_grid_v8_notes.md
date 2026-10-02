# power_grid v8: structural pass at the organizer's scale

Lab notebook, Sat Sep 26 (evening). No credits. Same three runs as `power_grid_min`
(`p1.hold_rec` 120, `p2.pulse60_120` 180, `p2.multilevel200` 200). Scoring scale: the calibrated
organizer sigma, `plans/sigma_calibrated.json` = [load 9.65, frequency 0.308, share 0.066].
"Calibrated in-sample" = mean over runs of `metric.score(Y, run.Y, sigma)` with Y from
`runtime.infer.rollout_from_blob(doc)` (the exact shipping path).

Fitting: besides `scripts/ode_lab.py`, we fit directly on the score. Per cell the loss is
$z/(1+z)$ with $z=|e|/\sigma$, so least squares on residuals $r=\sqrt{z/(1+z)}$ minimises
exactly $\sum(1-\text{score})$. Multistart around the previous fit, 150 nfev. Our LOO below
("score-LOO") refits on two runs starting from the full-fit theta; the lab LOO is the lab's own
protocol (6 starts, 60 nfev, same sigma).

## 1. Error budget of the current best (power_grid_min, calibrated in-sample 0.737)

| run | load | freq | share | mean |
|---|---:|---:|---:|---:|
| hold_rec | 0.812 | 0.709 | 0.804 | 0.775 |
| pulse60_120 | 0.665 | 0.635 | 0.781 | 0.694 |
| multilevel200 | 0.802 | 0.588 | 0.833 | 0.741 |

Loss summed over ticks: frequency 183, load 123, share 96 (of 500 ticks each). Frequency noise
is 0.03 Hz against sigma 0.31, share noise 0.001 against 0.066: all of it is model error.

Top three leaks, read off the trajectories:
1. **Frequency after a reserve release.** When reserve drops (pulse t=60, multilevel t=58) the
   frequency falls far below what load and base supply give: pulse t=64 true 50.08 Hz at load 80
   (model 51.0), multilevel t=64 true 49.20 at load 115 (model 49.7). The gap closes over
   ~15 ticks. Both segments switch charging_allowance on (0 -> 1, 0 -> 0.69) after a 60-tick reserve
   pulse with charging 0: the reserve fleet is refilling from the grid.
2. **Frequency ceiling / slow drift under a large surplus.** True frequency tops out at 51.85 Hz
   twice (multilevel t=3 and t=124) and drifts down while load falls in the first multilevel
   segment (the reserve contribution fades: finite duration). The min model ran to its 2.5 Hz clip.
3. **Share timing and load coupling.** The share moves within one tick of a reserve step
   (pulse t=0: 0.046 true vs 0.170 model), so it follows the request, not the thermally lagged
   reserve. At reserve 0 it oscillates +-0.04 against load with a lag of ~6 ticks (hold_rec: share
   peak t=18-21 after load trough t=12-15): curtailment at the interconnector follows recent flow.

Load (the thermostat rebound) is the fourth, discussed at the end.

## 2. Structure (v8 family)

States $(D_s, x_1, x_2, R_s, \delta f, s, G, E, \Theta)$, controls $(p, r, c, i)$. RK4, 2 substeps.

Load, as in `power_grid_min`: $D(p)=d_0-d_1p$, $\dot D_s = a_s(D-D_s)$,
$\dot x_1=x_2$, $\dot x_2=-2\zeta\omega x_2-\omega^2x_1+k\,a_sD_{sat}\tanh((D-D_s)/D_{sat})$,
$L = D_s + \text{sw}(x_1)$ with (v8j/v8k) a soft limit on the downswing only,
$\text{sw}(x)=\max(x,0)-x_l\tanh(\max(-x,0)/x_l)$. $\omega=0.10$ and $D_{sat}=3$ fixed.

Reserve with a storage part (v8e on): a share $\beta$ of the reserve is storage with stock
$E\in[0,1]$; the fleet runs at
$$\text{draw}=c_r\beta R_sE,\qquad R_{eff}=\big(c_r(1-\beta)R_s+\text{draw}\big)(i_0+(1-i_0)i)$$
$$P_{ch}=p_{ch}\,c\,(1-E),\qquad \dot E=(P_{ch}-\text{draw})/\text{cap}.$$
Refilling takes grid power: it enters the balance with a minus sign.

Frequency with a governor and a ceiling:
$$b = D(0.8)+g_b+G+R_{eff}-P_{ch}-L,\quad f^\star=k_fb,\quad f^\star\leftarrow\min(f^\star, f_{hi}\tanh(f^\star/f_{hi}))$$
$$\dot{\delta f}=a_f(f^\star-\delta f),\qquad \dot G=a_g\big(\text{clip}(-k_g\delta f,\pm40)-G\big),\quad a_f=1.$$

Share (curtailment at the connection):
$$s^\star=s_0(0.5+0.5i)\,\frac{1}{1+(r/80)^2}\cdot\begin{cases}1-k_\theta(\Theta-D(0.8))/100 & \text{(v8g, v8j)}\\ (D(0.8)/\Theta)^{k_\theta} & \text{(v8k)}\end{cases},\qquad \dot\Theta=a_\theta(L-\Theta),\quad \dot s=1.5(s^\star-s)$$
with $r$ the reserve request itself (no thermal lag), $a_\theta=0.12$ (v8g/j) or $0.2$ (v8k).

$x_0$: $D_s=\Theta=\text{load}_0$, $\delta f=f_0-50$, $s=\text{share}_0$, $E=1$, the rest 0.

## 3. Variants tried (all 16 params or fewer unless noted; score-fit in-sample / score-LOO)

| variant | change | params | in-sample | score-LOO | kept |
|---|---|---:|---:|---:|---|
| min (polished on the score) | baseline | 11 | 0.746 | 0.707 | ref |
| v8 | governor state G + reserve stock (all reserve) + share vs lagged load | 18 | 0.778 | 0.731 | step |
| v8b | + slow decay of the y0 load offset | 19 | 0.778 | - | no (+0.0005) |
| v8c | v8 with a_f, g_lim, Dsat fixed; charging draw on balance; f ceiling | 16 | 0.783 | - | step |
| v8d | + softening spring (period shortens with amplitude) | 17 | 0.782 | - | no (soft -> 0) |
| v8f | + symmetric soft limit on the swing | 16 | 0.784 | - | no (xl 69, +0.001) |
| v8e | stock only on a share beta of the reserve | 17 | 0.793 | - | step |
| **v8g** | v8e, share on the request, k_c fixed 80 | 16 | **0.7965** | **0.763** | yes |
| v8h | v8g, share power law $(D_{ref}/\Theta)^{k}$, $a_\theta$ 0.2 | 16 | 0.8006 | 0.763 | tie |
| v8i | two thermostat groups (w, rho w) | 16 | 0.7956 | 0.758 | no (rho -> 1) |
| v8j | v8g, w fixed 0.10, downswing soft limit xl | 16 | 0.7979 | 0.767 | yes |
| **v8k** | v8j + v8h share law | 16 | **0.8015** | **0.776** | **best** |

Checks on v8g's fitted theta (set one term to its null, no refit): governor off (kg=0) -0.019,
no storage share (beta=0) -0.053, no ceiling (f_hi=2.5) -0.025, no load term in the share
(k_th=0) -0.013. Every added term carries weight.

## 4. Lab protocol at the calibrated sigma (`ode_lab.py --sigma-cal 1.0`, 6 starts, 60 nfev)

| model | LOO hold_rec | LOO pulse | LOO multilevel | LOO mean | lab in-sample |
|---|---:|---:|---:|---:|---:|
| power_grid_min (tag v8ref) | 0.773 | 0.652 | 0.663 | 0.696 | 0.734 |
| v8g (tag v8g_lab) | 0.788 | 0.694 | 0.695 | **0.726** | 0.790 |
| v8j (tag v8j_lab) | 0.787 | 0.692 | 0.666 | 0.715 | 0.790 |
| v8k (tag v8k_lab) | 0.783 | 0.695 | 0.680 | 0.719 | 0.793 |

l0b_lin at the same scale: 0.563. The lab LOO (cauchy loss, 60 nfev) and our score-LOO disagree
on v8j/v8k vs v8g (score-LOO +0.004/+0.012, lab LOO -0.011/-0.007, mostly the multilevel fold).
Our rule is that a change stays only if in-sample improves and the lab LOO does not drop, so
**v8g is the pick**; v8k is the alternative with the best in-sample (0.8015) and score-LOO (0.776).
Every v8 variant beats min on every lab fold.

## 5. Shipping documents (theta polished on the score, written with `--theta0 ... --starts 1 --nfev 1`)

`plans/power_grid_power_grid_v8g_v8g_doc.json` (the pick: best lab LOO) and
`plans/power_grid_power_grid_v8k_v8k_doc.json` (best in-sample and score-LOO). Calibrated in-sample through
`rollout_from_blob`:

| run | min | v8g | v8k |
|---|---|---|---|
| hold_rec | 0.812 0.709 0.804 = 0.775 | 0.843 0.844 0.842 = 0.843 | 0.844 0.847 0.864 = 0.851 |
| pulse60_120 | 0.665 0.635 0.781 = 0.694 | 0.666 0.700 0.834 = 0.733 | 0.665 0.706 0.845 = 0.739 |
| multilevel200 | 0.802 0.588 0.833 = 0.741 | 0.826 0.768 0.845 = 0.813 | 0.830 0.767 0.846 = 0.814 |
| **mean** | **0.737** | **0.7965** | **0.8015** |

v8g theta: d0 126.3, d1 20.81, w 0.1000, zeta 0.302, kick 0.543, kf 0.0454, gb -7.14, a_g 0.0253,
kg 7.03, c_r 1.49, cap 320, p_ch 40.3, s0 0.362, k_th 0.288, f_hi 1.565, beta 0.555; nothing at a
bound. v8k theta: d0 126.1, d1 20.66, zeta 0.309, kick 0.561, kf 0.0461, gb -6.58, a_g 0.0217,
kg 8.98, c_r 1.96, cap 329, p_ch 40.5, s0 0.360, k_th 0.302, f_hi 1.564, beta 0.661, xl 69.9.
Reading (v8k; v8g is within 10-25% on every parameter): frequency droop 0.046 Hz per power unit; a slow governor (time constant 46 ticks)
takes back ~30% of a sustained imbalance ($1/(1+k_fk_g)=0.71$); 66% of the reserve is storage
holding cap = 329 unit-ticks, which a full request (draw $\approx$ 194/tick) empties in ~2 ticks,
refilled at up to 40 units x charging allowance (the refill shows as a ~15-tick frequency dip); the frequency saturates at +1.56 Hz. c_r sits at 1.96 of its bound 2.0 (within 2%);
the storage part empties so fast that c_r beta is only seen in the first ticks of a pulse.

Eval-shaped 4,000-tick rollouts (lab, v8g and v8k alike): finite, 0.4-0.7 s per episode, fraction outside the
observed range +-5% = 0.00 / 0.00 / 0.01 / 0.00 (sustained / order / recovery / composition),
against 0.24 / 0.08 / 0.01 / 0.07 for min. Frequency stays in [47.7, 51.6], load in [54.9, 158.1].

## 6. What is left

Biggest leak now: **load in the pulse recovery** (pulse60_120 t=60-180, loss 38 of the 110 load
loss): after 60 ticks at price 0 the price-1.5 trough is flat for ~16 ticks at 64 and the rebound
overshoots to 119 (+25 above D(1.5)); after a 0.8 -> 1.5 step (hold_rec) the rebound is only +11.
The rebound size depends on how long the loads were pre-cooled, which a linear oscillator with a
saturating kick cannot carry; two thermostat groups (v8i) and a softening spring (v8d) were
rejected by the fit. A pre-cool memory state (thermal debt integrated during low price, scaling
the rebound) is the next structure to try. The frequency error in the same window
(t=100-112, 0.4-0.5 Hz) is that same load error times kf.
