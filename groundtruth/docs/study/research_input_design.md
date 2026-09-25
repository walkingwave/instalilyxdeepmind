# Research notes: input design under a step budget (2026-09-24)

[L] = literature support, [J] = judgment.

## Bottom line
Space-filling multilevel first, committee-disagreement adaptive after, is the right shape. Two corrections: (a) the largest documented lever for open-loop rollout error is fitting (simulation / multi-step error, multiple shooting), not more clever excitation; (b) evidence that adaptive beats one-shot at this budget is thin and mixed, so cap adaptive spend and make the bulk of remaining credits test-shaped, long segments.

## 1. Input design under a budget
- PRBS is persistently exciting for linear models but not for nonlinear ones; multilevel pseudo-random / amplitude-modulated PRBS (PRMLS, APRBS) is the standard fix for Hammerstein/Wiener/NARX because it covers amplitude space, not just frequency. [L] Tan & Godfrey 2002 (IEEE T-IM); Braun, Rivera, Stenman 2001 (ACC); Nelles, Nonlinear System Identification, 2001; Schoukens & Ljung 2019 (IEEE CSM, arXiv 1902.00683).
- Dwell time: plant-friendly guidelines (Rivera & Gaikwad 1995; Lee & Rivera 2005; Braun et al. 2001) set the minimum switching time from the fastest dominant tau (T_sw ≈ 2.8·tau_lo/alpha) and total length from the slowest tau (≥ 3–5 × tau_hi settling). [L] For tau ≈ 300+ a segment must last ≥ 900–1500 ticks to pin the slow gain; short random segments (200-tick multilevel) only estimate the fast modes. Integrators: an integrator's low-frequency gain is unidentifiable from short zero-mean excitation; use level tests plus a long hold at a known equilibrium. [L, Ljung 1999 ch. 13]
- Practical: for slow / integrator systems, a few long holds at distinct levels beat many short segments; for fast systems, APRBS with T_sw ≈ tau_fast/2 .. tau_fast. [L+J]

## 2. Adaptive / optimal design
- D-/A-optimal design is Fisher-information based for a fixed model structure; only as good as the structure. [L] Application-oriented design (Hjalmarsson 2009 EJC; Bombois et al. 2006): excite what the intended use is sensitive to. [L]
- Ensemble / committee disagreement: Seung et al. 1992 (QBC); Buisson-Fenet, Solowjow, Trimpe 2020 (L4DC); Larrañaga, Fasel, Brunton 2026 (arXiv 2606.12182, E-SINDy uncertainty); Xie & Bemporad 2025 (arXiv 2506.21754); Pathak et al. 2019; Sekar et al. 2020 (Plan2Explore); Shyam et al. 2019 (MAX). [L]
- Counter-evidence: Wagenmaker & Jamieson 2020 (COLT); Chatzikiriakos, Jamieson, Iannelli 2025 (arXiv 2509.11907): active excitation gains over well-chosen non-adaptive inputs are constant-factor, not order-of-magnitude, for linear systems. [L] Verdict: adaptive helps mainly where a few regions are badly under-sampled (sharp nonlinearity, thresholds, cycles). With ~1,500 samples per system it is worth a bounded slice, not the majority. [J]

## 3. Does the metric change the design?
Yes. Bounded per-tick score over 4,000 open-loop ticks means (i) low-frequency / steady-state error dominates, (ii) one-step-fit models can be badly wrong on rollouts. Lambert et al. 2020 (L4DC, "Objective mismatch"); Farina & Piroddi 2011 (IJACSP): multi-step PEM converges to simulation-error minimisation; Ribeiro et al. 2020 (Automatica, arXiv 1905.00820): full-trajectory simulation loss is non-smooth, use multiple shooting. [L] Hold levels, pulse amplitudes, gaps and durations should match the eval categories. Maximally informative excitation is for parameter recovery; the score is on rollouts. [J]

## 4. Allocation of 1,560 credits per system (proposal)
First (~800): two 400-tick holds at extreme control corners; priority to systems with tau 300+ or integrators; 2 × 250 on fast ones. If cycles / waves, make one hold 600 ticks to see ≥ 2 periods. [L: settling-time rule; J]
Then (~450), after refitting with simulation-error loss and checking residuals: one test-shaped 300-tick recovery run (3 pulses at 0.7 / 0.85 / 1.0 of pulse_ref, gaps 40 / 100 at baseline); one 150-tick single-vs-joint order pair, skipped if residuals already show additivity. [J]
Adaptive (~200): committee of l1 / l2 / ODE fits; spend where 4,000-tick forecasts disagree most across eval-like schedules. Cap at 200. [L: QBC; J: cap]
Validation (~110): one eval-like run per system never trained on; used for blending and veto. [J]
Free levers first: repeated resets for initial-condition spread; public leaderboard as noiseless validation.
Evidence that changes this: if the 120-tick hold has not settled (> 2 sigma / 100 ticks), lengthen holds to 600 and drop the order pair; if residuals after long holds are white and small, skip adaptive and buy a second recovery run at different gaps.

Sources: arxiv.org/abs/1902.00683; proceedings.mlr.press/v120/buisson-fenet20a; arxiv.org/abs/2606.12182; arxiv.org/abs/2506.21754; proceedings.mlr.press/v125/wagenmaker20a; arxiv.org/abs/2509.11907; arxiv.org/abs/2002.04523; doi 10.1002/acs.1203; arxiv.org/abs/1905.00820.
