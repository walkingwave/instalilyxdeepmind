# Actions between uploads and their measured effects, Thu 2026-09-24

Companion to `docs/MATH_LOG.md`. Every number here is a public leaderboard score on the same
40 fixed episodes per system, so differences between uploads are exact, not sampled.

Notation. $y_t \in \mathbb{R}^p$ observables, $u_t \in \mathbb{R}^m$ controls, $T = 4000$.
Per-tick score $s(e) = 1/(1 + |e|/\sigma)$, averaged over ticks, observables, episodes and four
equally weighted categories. $g(\cdot)$ is the per-observable transform (identity, $\log(1+y)$ or
logit, then standardised); $v_t = g(y_t)$. $\phi(u) \in \mathbb{R}^{n_f}$ are control features.

## 0. Starting point: u001, persistence (Sep 23)

$\hat y_t = y_0$ for all $t$. Mean 0.3416. This is the score of "the initial observation is the
whole forecast"; every gain below is relative to it.

## 1. u001 → u002: one hold from reset, relax-to-equilibrium on six systems

**Action.** Bought `hold_rec`: reset, then 120 ticks at the recovery action, on all ten systems
(1,200 credits). Fitted `l0b_lin` on the six systems whose recovery equilibrium is not degenerate.

**Model.** For each observable $j$, in transformed units,
$$z_{t+1} = a_j z_t + (1 - a_j)\,\phi(u_t)^\top W_{:,j}, \qquad z_0 = v_{0,j}, \qquad \hat v_{t} = z_{t+1},$$
$a_j$ chosen on a grid of 18 values in $(0.4, 0.9993)$, $W$ by ridge regression. With one control
vector in the data, $\phi(u)^\top W$ is a constant, so the model is exactly
$$\hat v_t = v_\infty + (v_0 - v_\infty)\,a^{t+1},$$
an exponential relaxation from the observed initial to a learned level at a learned rate. The
learned $v_\infty$ is the tail of the hold, and the same $v_\infty$ is predicted regardless of the
test schedule's controls.

**Effect** (public score, before → after):

| system | u001 | u002 | Δ |
|---|---:|---:|---:|
| ad_auction | 0.4711 | 0.6044 | +0.133 |
| reservoir | 0.3471 | 0.4625 | +0.115 |
| social_contagion | 0.3052 | 0.3672 | +0.062 |
| power_grid | 0.4867 | 0.5448 | +0.058 |
| wildlife | 0.2227 | 0.2785 | +0.056 |
| epidemic | 0.2720 | 0.1562 | −0.116 |

Mean over ten: 0.3416 → 0.3724.

**Why.** Every test episode starts from reset, so the reset transient is in all 40 episodes. On
five systems the transient dominates the first few hundred ticks and the recovery level is a fair
guess for the long tail. Epidemic is a wave: the 120-tick hold captured a falling flank
(187 → 51 cases) and the fit called 51 the equilibrium. In the test the wave rebounds (confirmed
in purchase 2: 48 → 99 cases after the pulse ends), so the forecast sits below the truth for
thousands of ticks. Persistence at 187 is wrong by less on average than a constant 51.
Single-exponential fit error on that hold was 46 % of scale, the largest of any system; that
number alone flagged the failure before the upload.

## 2. u002 → u003: identification data, per-control map, four fixes

**Action.** Purchase 2, 3,720 credits: fast systems (traffic, hospital_queue, market) got a
40/40 pulse, a 40-tick midpoint hold and a 200-tick random multilevel run (10–16 levels);
power_grid and ad_auction a 60/120 pulse and a 200-tick multilevel with long dwells; the five
slow systems a 200/200 (epidemic 120/280) pulse. Then four code fixes, then a leave-one-run-out
screen, then u003 = winner per system on all ten.

**Identifiability.** With linear features $\phi(u) = [1, u_1, \dots, u_m]$, the equilibrium map
$W \in \mathbb{R}^{(m+1) \times p}$ needs at least $m + 1$ affinely independent control vectors.
Before: 1 (hold_rec) or 2. After: 4 + 10–16 distinct vectors on the fast systems ($m = 6$ for
traffic and hospital_queue, so rank $7$ is reachable), 3 + 3–4 on the medium ones, 3 on the slow
ones (rank 3 against $m + 1 = 3$–$4$: exactly or barely determined; ridge picks the minimum-norm
solution for the rest).

**Fixes and their mathematics.**
1. Output clip widened from $[\min - 1R, \max + 1R]$ to $\pm 3R$ of the observed range $R$
   (inside hard bounds). A capped integrator cannot reach its 4,000-tick level otherwise.
2. Lag-model initial state anchored: $z_{0,k} = (v_{0,j} - c)/K + b_k$ so that
   $\hat v_0 = v_{0,j} + \sum_k b_k$, $b_k \in [-1, 1]$ sd, replacing a free $K \times p$ matrix
   fitted from a handful of collinear columns $a^t v_{0,i}$.
3. Relaxation rate $a_j$ chosen by the competition loss in physical units,
   $\sum_t w_t^2 (1 - s(y_t - \hat y_t))$, instead of squared error in transformed space, where
   logit spreads the ends of $[0, 1]$ and mis-weights them.
4. Validation runs stay held out (the builder had relabeled every run as training).

**Screen** (robust score on the held-out run, averaged over $\sigma \times \{0.5, 1, 2\}$,
paired against persistence): l0b_lin best on market (0.770), traffic (0.605), power_grid
(0.707), supply_chain (0.715), reservoir (0.597), social_contagion (0.707); l1 best on wildlife
(0.721), ad_auction (0.698), hospital_queue (0.645); nothing beat persistence on epidemic.

**Effect.**

| system | u002 | u003 | Δ | model |
|---|---:|---:|---:|---|
| traffic | 0.2636 | 0.6259 | +0.362 | l0b_lin |
| market | 0.1849 | 0.4806 | +0.296 | l0b_lin |
| supply_chain | 0.4203 | 0.7016 | +0.281 | l0b_lin |
| wildlife | 0.2785 | 0.4292 | +0.151 | l1 |
| social_contagion | 0.3672 | 0.4404 | +0.073 | l0b_lin |
| reservoir | 0.4625 | 0.5286 | +0.066 | l0b_lin |
| power_grid | 0.5448 | 0.5932 | +0.048 | l0b_lin |
| ad_auction | 0.6044 | 0.6283 | +0.024 | l1 |
| hospital_queue | 0.4428 | 0.4505 | +0.008 | l1 |
| epidemic | 0.2720 | 0.2720 | 0 | persistence |

Mean 0.3724 → 0.5150.

**Why.** The three largest gains are the systems whose recovery action is a degenerate corner
(flows, shipments, retail stock and volume go to zero under it). Persistence and the recovery
level are both useless there, and the sustained category holds interior levels: the first upload
with an actual $u \mapsto y_\infty$ map fixed exactly that. The screen predicted the public score
within 0.02 on traffic and supply_chain and over-predicted by 0.15–0.30 on market, wildlife,
social_contagion, hospital_queue and power_grid: the screen holds out one of our own short
from-reset runs, while the test contains 4,000-tick holds and pulse trains we have never
observed. That gap is the measured cost of missing long-horizon data.

## 3. u003 → u004: one hypothesis per system

**Action.** No credits. Ten single-factor tests on the same data.

**Hypotheses and effects.**

| system | change | u003 | u004 | Δ | reading |
|---|---|---:|---:|---:|---|
| epidemic | persistence → l0b_lin with $\lambda = 0.5$ | 0.2720 | 0.3265 | +0.055 | overshoot: model and persistence err with opposite signs on many ticks |
| market | l0b_lin → l1 | 0.4806 | 0.3355 | −0.145 | three lags + history multiplier over-fit four runs |
| wildlife | l1 → l0b_lin | 0.4292 | 0.5550 | +0.126 | l1 fitted our cycle's phase; the test's phase differs |
| ad_auction | l1 → l0b_lin | 0.6283 | 0.5858 | −0.043 | the 60/120 rebound is genuinely second order |
| hospital_queue | l1 → l0b_lin | 0.4505 | 0.4776 | +0.027 | |
| traffic | linear → quadratic + pairwise $\phi$ | 0.6259 | 0.5557 | −0.070 | $n_f$: 7 → 28 with ~14 distinct $u$ |
| power_grid | same | 0.5932 | 0.5503 | −0.043 | $n_f$: 5 → 15 with ~7 distinct $u$ |
| social_contagion | same | 0.4404 | 0.4122 | −0.028 | $n_f$: 4 → 10 with 3 distinct $u$ |
| reservoir | same | 0.5286 | 0.5268 | −0.002 | |
| supply_chain | same | 0.7016 | 0.6997 | −0.002 | |

**Why the quadratic map loses.** With $n_f$ features and $d$ distinct control vectors, the
regression for $W$ has rank at most $d$. When $n_f > d$ the ridge solution is the minimum-norm
one, which spreads the observed responses across squares and cross terms that the test then
evaluates at unseen $u$. The loss is monotone in $n_f - d$: traffic (21 surplus) −0.070,
power_grid (8) −0.043, social_contagion (7) −0.028, supply_chain and reservoir (near-determined,
dwell-limited) ≈ 0.

**Why the blend helps epidemic.** $\hat y = y_0 + \lambda(\hat y_m - y_0)$ has error
$e_\lambda = (1 - \lambda) e_0 + \lambda e_m$. The per-tick optimum is
$\lambda^* = e_0 / (e_0 - e_m)$, which lies in $(0, 1)$ only where $e_0$ and $e_m$ have opposite
signs. A gain of +0.055 at $\lambda = 0.5$ over both $\lambda = 0$ (0.272) and $\lambda = 1$
(0.156) says the relaxation model undershoots where persistence overshoots across most of the
horizon: the signature of a wave that first falls below and then rebounds above the initial.

## 4. u004 → u005: blend strength as the single factor

**Action.** No credits. $\lambda = 0.7$ on epidemic, $0.8$ on the seven systems whose screen
over-predicted the public score, $1.2$ on the two where screen and public agreed. Scored so far:
the four systems with slots left.

| system | $\lambda$ | best before | u005 | Δ |
|---|---:|---:|---:|---:|
| hospital_queue | 0.8 | 0.4776 | 0.4925 | +0.015 |
| traffic | 1.2 | 0.6259 | 0.6116 | −0.014 |
| supply_chain | 1.2 | 0.7016 | 0.6444 | −0.057 |
| market | 0.8 | 0.4806 | 0.3422 | −0.138 |

**Why.** $|e_\lambda| = |(1 - \lambda) e_0 + \lambda e_m|$. On market, volume collapses from
~90 to ~2 within ten ticks under most schedules, so $|e_0| \approx 88$ on that observable while
$|e_m|$ is a few units: keeping 20 % of persistence adds $0.2 \times 88 \approx 18$ units of error,
many $\sigma$, for the whole episode. Shrinkage toward $y_0$ is only a hedge where
$|e_0| \lesssim |e_m|$; it is a guaranteed loss where persistence is catastrophically wrong.
Hospital_queue is the opposite case (wait time and queue drift slowly, so $e_0$ stays comparable
to $e_m$) and gains. $\lambda = 1.2$ extrapolates beyond the fitted equilibrium; on supply_chain
the inventories are capped integrators, so 20 % overshoot of the cap is pure error.
The rule that comes out: $\lambda = 1$ unless a paired test shows opposite-sign errors, which so
far is epidemic ($0.5$) and hospital_queue ($0.8$).

## 5. Where the day ended

Best public score per system and the model behind it:

| system | score | model |
|---|---:|---|
| supply_chain | 0.7016 | l0b_lin |
| ad_auction | 0.6283 | l1 |
| traffic | 0.6259 | l0b_lin |
| power_grid | 0.5932 | l0b_lin |
| wildlife | 0.5550 | l0b_lin |
| reservoir | 0.5286 | l0b_lin |
| hospital_queue | 0.4925 | l0b_lin, $\lambda = 0.8$ |
| market | 0.4806 | l0b_lin |
| social_contagion | 0.4404 | l0b_lin |
| epidemic | 0.3265 | l0b_lin, $\lambda = 0.5$ |

Mean of best-of: **0.537** (from 0.342). Credits: 4,920 of 20,000 spent; 15,080 left.
Score per 1,000 credits so far: +0.040. Uploads: 5 (25 system-scores), 21 hypotheses answered.

Decomposition of the +0.195: data with a control-dependent equilibrium map +0.14 (u003 over u002,
averaged over ten), the reset transient alone +0.03 (u002), hypothesis tests +0.02 (u004, u005
winners). Nothing yet from the recovery and order categories beyond what a relaxation model
implies, and those are half the score.

## 6. What the numbers say to do next

1. The screen-minus-public gap (0.15–0.30 on five systems) is long-horizon behaviour we have
   not observed. Buy it: 4,000-tick-shaped holds and pulse trains with the test's $\alpha$ range
   and gaps, plus a held-out mixed run per system.
2. The relaxation model is first order; ad_auction, epidemic and wildlife have measured
   second-order responses (rebound, wave, cycle). A two-state model per observable fitted on
   rollout error, not one-step error, is the next model.
3. Keep $\phi$ linear until the number of distinct control vectors exceeds $n_f$ by a margin.
4. Keep $\lambda = 1$ except where a paired test shows opposite-sign errors.
