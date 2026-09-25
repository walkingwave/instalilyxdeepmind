# Research notes: long-horizon open-loop models from small data (2026-09-24)

[E] = published evidence, [J] = judgment.

## 1. Model classes with a track record on simulation error, small data
- [E] Nonlinear benchmarks (Silverbox, Wiener-Hammerstein, Cascaded Tanks ~1k samples, EMPS, CED): winners on simulation RMSE are state-space models trained on rollouts. Deep subspace encoder (SUBNET) holds the best W-H test error [Beintema, Tóth, Schoukens, Automatica 2023, arXiv 2210.14816; L4DC 2021 arXiv 2012.07697]. On small-data Cascaded Tanks, recurrent state-space nets beat polynomial NARX and LTI by 5–10× [Champneys et al. 2024, arXiv 2405.10779].
- [E] Grey-box / hybrid (UDE) beats black-box neural ODE on extrapolation and few data [Rackauckas et al. 2020, arXiv 2001.04385; Kamp 2023 ICINCO]. SINDy(c) is best only on clean, truly sparse dynamics and breaks under noise / hidden states [arXiv 2601.09811]. Koopman / DMDc: good near-linear, unstable on long open-loop horizons, poor on limit cycles / hard saturation [arXiv 2601.11901; 2310.10745].
- [E] AR (recurrent) models keep working with little data; non-AR needs much more [Weber et al. 2021, arXiv 2105.02027]. GP-NARX was the worst entry on CED [Champneys 2024].
- [J] Ranking for 2–4 observables, hidden queues / delays, 500–1500 ticks, structural hints: grey-box ODE with per-system structure > small linear / bilinear state-space trained on rollout error > Hammerstein / NARX with linear dynamic core > Koopman > SR / SINDy. Persistence 0.34 and relax-to-equilibrium 0.37–0.60 say the dominant error is slow dynamics + control gain, not fine nonlinearity.

## 2. Simulation-error vs one-step fitting
- [E] One-step PEM picks wrong / redundant NARX structures; simulation-error minimisation gives compact, correct ones [Piroddi & Spinelli, IJC 2003; Piroddi 2008; Aguirre et al. MSSP 2010].
- [E] RLC benchmark: one-step fit test R² 0.73 / 0.03 vs truncated simulation error (windows m = 64) 0.99 / 0.98; full-horizon fit same accuracy but ~50× slower [Forgione & Piga 2021, Eur. J. Control, arXiv 2006.02915].
- [E] Full-horizon single shooting fails because loss smoothness blows up exponentially with horizon outside the contractive region; fix = multiple shooting with the horizon as a design knob [Ribeiro, Tiels, Umenberger, Schön, Aguirre, Automatica 2020, arXiv 1905.00820]. On oscillatory data single shooting yields flattened trajectories; multiple shooting rescues it [Turan & Jäschke 2021, arXiv 2109.06786].
- [J] Recipe: multiple shooting with windows of 64–256 ticks, per-window initial states as free variables with a continuity penalty, scipy least_squares (robust loss); final short refinement on the full run if stable.

## 3. Hidden initial state from y0 only
- [E] For asymptotically stable systems a zero / random initial hidden state is competitive once the transient is past; for integrator systems a learned estimator was essential [Forgione, Muni, Piga, Gallieri 2022, arXiv 2206.12928]. Encoders need a past I/O window, unavailable at test time here.
- [J] With only y0: (a) parameterise hidden state as a deterministic function of y0 (equilibrium-consistent), fit that map jointly with the dynamics on training resets; (b) regularise toward it; (c) accept a transient cost: a 50-tick transient in 4,000 ticks costs ~1 % of score. Exception: integrator systems (reservoir, supply chain, hospital queue) where hidden inventory sets a long-lived offset; fit the y0 → hidden map carefully there.

## 4. Structural priors, saturation, delays, oscillations
- [E] Hybrid / UDE models extrapolate from short series where black-box NODEs drift [Rackauckas 2020]. Truncated simulation error + robust loss is the standard fit [Forgione & Piga 2021]. Hidden delays: short FIFO pipeline or a chain of first-order lags (Erlang), differentiable. min() / clip capacities: smooth-min while fitting, hard min at inference.
- [J, DISPUTED] The agent claimed the metric is "concave for large error" so damped mean trajectories beat full-amplitude wrong-phase ones, citing Patton & Timmermann 2007. This is wrong on the metric: s(e) = 1/(1+e/σ) has second derivative 2/(σ²(1+e/σ)³) > 0, i.e. convex in |e| everywhere. The strategy audit's Monte Carlo (MATH_LOG §1.4) shows full amplitude wins for A ≥ σ. Keep amplitude; verify phase on held-out runs.

## 5. Ranking for the constraints (numpy runtime, ~1,000 ticks, 3 days)
1. Grey-box ODE per family, multiple shooting + robust least squares, mechanism pair by held-out simulation error. Highest expected gain, 1–1.5 days. (Contradicted by the ODE audit on mocks, MATH_LOG §9: fitting 240 ticks was worse than the prior; needs multiple shooting and frozen parameters to be viable.)
2. Low-order linear / bilinear state-space with control (2–4 hidden states, Hammerstein input nonlinearity), rollout-error fit. Half a day; fallback and second ensemble member. The l1 / l2 ladder is close to this; switch its training loss to truncated simulation error.
3. Small GRU / RNN simulator trained without autograd. Only if time remains.
4. Koopman / DMDc, SINDy: skip.

Ensemble: [E] averaging K independently trained dynamics models cuts prediction variance ~1/K and stabilises long rollouts [Fan & Xiu 2022, arXiv 2203.03458]; ensembles beat single models for multi-step planning [Chua et al. NeurIPS 2018, PETS]. [J] Per-tick median (or trimmed mean) of 3–5 members as the final layer: discards a diverged member instead of averaging it in.
