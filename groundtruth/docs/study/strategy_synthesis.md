# Strategy synthesis (2026-09-24, late)

Inputs: MATH_LOG §1, §3, §5, §9, §11–15; research_input_design.md; research_models.md; briefs; sim_next.json. Balance: 1,880 per system.

## A. Purchase 2: modified
The 200-tick multilevel (dwell 4–50) is right only for the fast three. Committee spread never peaks on multilevel; it peaks on long pulse holds / sustained for power_grid, reservoir, ad_auction, social_contagion, on pulse60 for supply_chain and wildlife, and on interior levels (mid40) for traffic and market. Median segment ~14 ticks against τ ≥ 250 returns slopes, not gains; integrators have no low-frequency gain from short zero-mean excitation (Ljung ch. 13; segments ≥ 3–5 τ).

| group | systems | experiments | per system |
|---|---|---|---|
| fast | traffic, hospital_queue | pulse 40/40, mid hold 40, multilevel 200 (dwell 4–50) | 320 |
| fast | market | same, multilevel dwell 15–60 (T_sw ≥ τ_fast) | 320 |
| medium | power_grid, ad_auction | pulse 60 on / 120 off (rebound), multilevel 200 dwell 40–70 (3–4 settled levels) | 380 |
| slow | wildlife, reservoir, social_contagion, supply_chain | pulse 200 on / recovery 200 off | 400 |
| slow | epidemic | pulse 120 on / recovery 280 off (suppression then rebound wave) | 400 |

Total 3,720. Balance after: fast 1,560, medium 1,500, slow 1,480.

## B. Remaining credits, four decisions
- P2 (Fri Sep 25 AM): above.
- P3 (Sat Sep 26), 650 per system, after refitting on P2 with simulation-error loss and reading Friday's public scores: (a) 300-tick recovery-shaped pulse train, 4 pulses α ∈ {0.7, 0.85, 1.0, 0.85}, lengths 10–40, gaps 30–120 at recovery; (b) 150-tick single-vs-joint from recovery (composition; per-control weights identifiable for the first time on medium/slow); (c) 200-tick eval_like mixed validation run, held out until the Tuesday refit. Swap rule: if P2's 200-on segment has not settled (|slope|·100 > 2σ̂ over its last 50 ticks), replace (a) with a 400-tick hold at the bounds midpoint; expected on reservoir, social_contagion, supply_chain.
- P4 (Sun Sep 27 PM): adaptive, ≤ balance − 300. Committee refit on P2+P3; candidates = eval_like × 4 categories, order pairs, 300-tick corner holds; buy the top-2 spread per system. Capped because adaptive gains are constant-factor (Wagenmaker 2020; Chatzikiriakos 2025).
- Reserve 300 per system until Tue Sep 29 09:00: spent on whichever category the validation run and public scores put below persistence, else a second validation run. All spent by Tue noon for the final refit.
Validation: (c) plus the public tab as noiseless fixed-episode validation; ≤ 3 uploads/day, kind-level comparisons only; λ and ensemble weights tuned locally. Fix the `split="train"` tagging before P3.

## C. Model build order
1. Fri: audit fixes touching every kind: soft_clip floor (§3 #2), z0 = g(y0) with one scalar (#3), σ-weighted loss in physical space (#5), held-out tagging and stress gate on propose (§4). Refit l1 and l0b_lin on P2 with per-control features. Friday upload: per-system best of {l1, l0b_lin, persistence}; epidemic stays persistence.
2. Sat: l2 (modal state space, 2–4 hidden states per observable, saturating Hammerstein input map) trained by multiple shooting: windows 64–128, per-window initial state free with continuity penalty, Cauchy least squares (Ribeiro 2020; Forgione & Piga 2021). Allow eigenvalue 1.0 with the physical box instead of 0.9995 (0.9995^4000 = 0.135: a level decays 86 % over an episode). Then Nelder–Mead polish of the 3–5 slowest parameters on the exact metric.
3. Sun: exceptions and ensemble. Epidemic: SIR with hospital Erlang lag, β(u) = β0(1 − a·mask)(1 − b·closure), vaccination S→R; hidden compartments a fixed function of y0. Wildlife: l2 complex modes at full amplitude. Integrators: reservoir level = inflow − release − irrigation with cap; supplier inventory = production − dispatch(u) with cap; adopters logistic with r(u), K(u). Degenerate-corner systems: log1p, hard floor at 0, rate-limited collapse. Ensemble: per-tick median of 3–5 members for non-oscillatory systems; for wildlife and power_grid the single best held-out member (a median across phases is damping). Blend λ per observable, local. ODE templates: ad_auction only, 3–6 free parameters, ensemble member, no credits.

## D. Oscillation dispute, resolved
Truth A sin φ, forecast cA sin(φ+δ). For fixed δ, e = A·R·sin(φ − ψ), R² = 1 − 2c cos δ + c². Over uniform φ the score is F(AR/σ), F(a) = (1/2π)∮ dφ/(1 + a|sin φ|): decreasing, convex, F ≈ 1 − 2a/π for a ≪ 1, F ≈ (2/πa) ln(πa/2) for a ≫ 1. L2 gives c = E cos δ = e^{−v/2} because L2 is quadratic in R; this metric is not.
- A/σ ≪ 1: minimise E_δ R; ∂R/∂c at c = 1 is |sin(δ/2)| > 0, so c* < 1 (mild damping, tiny stakes).
- A/σ ≫ 1: F ~ ln a / a is dominated by δ where R ≈ 0; only c ≈ 1 reaches R = 0 (error crosses zero twice per cycle) and convexity pays: E F(R) > F(E R). c = 1.
Condition: full amplitude whenever A ≳ 3σ; never e^{−v/2}. The same argument forbids averaging members with different phases.

## E. Verdict
Near-optimal for this budget: the biggest slice goes where τ and spread say the models are blind (long pulse/recovery holds on the slow five), it buys the one structure the test uses verbatim (pulse trains), gets per-control information once, holds out a validation run, and moves the fit to simulation error. Biggest risk: engineering time on Sat/Sun (multiple-shooting l2 plus four grey-box models). Mitigation: upload the best public-verified build to Final at Mon 12:00 immediately, keep the Friday l1 build as the per-system floor, admit each new member only on a paired held-out win.
