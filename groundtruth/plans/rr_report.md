# Reset-rule audit (organizer initialization documents vs shipped x0), Tue Sep 29

Source of each rule: `data/<sys>/docs.json` -> `documents[id = "<sys>-initialization"]` (same sentence is in
`brief.brief` and in `kit/briefs.md`). Every document also says: "Unobserved memory always starts from the same
fixed reference rule described in the brief, including any deterministic dependence on the initial observable
state. There is no random prehistory." Shipped model = `submissions/20260928-2230-final3/<sys>/model.json`,
x0 in `gtlab/ode/<family>.py`.

LOO protocol (`scripts/lab_rr_loo.py`): leave-one-run-out on every owned run at the calibrated organizer
sigma (1.0), every fold a warm polish from the shipped theta with identical budget (60 s, 40 nfev) for the
shipped family and the copy; delta = copy minus shipped. Output `plans/rr_loo_<sys>_<family>[tag].json`.

| system | documented rule | our init (final3) | mismatch? | LOO delta (mean; per fold) | doc path |
|---|---|---|---|---|---|
| supply_chain | "Internal buffers and conveyors start empty; the initial stocks split into fixed class shares." | `supply_chain_v8b`: first conveyor stage seeded Q1A = Q1B = shipments0/2; S = supplier0; RA = RB = retail0/2; commitments C1 = C2 = 0; wear W = 0 | **yes** (conveyors seeded) | +0.0005 (hold_rec +0.0075, pulse +0.0001, hold_mid -0.0070, longhold +0.0015) | data/supply_chain/docs.json |
| power_grid | "Every reset starts with the same asynchronous population at reference price 0.8, without random hidden phases." | `power_grid_w5`: demand state Ds = load0, interconnector temp Th = load0, oscillator x = 0, reserve energy E = 1 | **yes** (population follows load reading, not price 0.8) | rr (Ds = Th = D(0.8), reading ignored): -0.0063, exam -0.007; rr2 (Ds = D(0.8), reading as oscillating mode x1 = load0 - D(0.8)): -0.0037, exam -0.006 | data/power_grid/docs.json |
| reservoir | "Initial instrument readings are observed; internal layers start from the same reference profile on every reset." | `reservoir_str9`: bank storage B = level0 (equilibrium with the reading); N = n0, D = d0 fixed; R = 0 | **yes** (bank stock follows the level reading) | +0.0464 raw (pulse fold +0.136, but driven by k_ret drift, see below); with k_ret, tau_ret pinned: **+0.0083** (hold_rec +0.0015, pulse +0.0209, longhold +0.0023) | data/reservoir/docs.json |
| market | "Customer warehouses start half full with full working cash, dealer books start neutral, and orders and settlement commitments start empty." | `market_y3`: Iw = 0.5, C = 1, v = 0, S = R = G = H = M = 0; P = A = Q = price0, D = depth0 | no | not run | data/market/docs.json |
| ad_auction | "people are available and unprepared, with no pending purchases or exposure recovery" | `ad_auction_v8b/min/s1`: X = 0, P = 0, queues Q1 = Q2 = 0 (v8b idle capacity K = Kmax, not covered by the doc) | no | not run | data/ad_auction/docs.json |
| traffic | "Roads and crossings start empty." | `traffic_z8`: all six pipeline stages 0, occupancy lags 0, fronts 0, split q = 0.5; speeds from reading; crew ce = ce0 | no | not run | data/traffic/docs.json |
| hospital_queue | "Initial queue composition is a fixed function of the reported initial count; services and the follow-up program start empty." | `hospital_queue_canon2/hosp9`: whole queue0 in waiting W; assessment/treatment A1 = A2 = T = 0; returns R1 = R2 = 0; fatigue/overtime 0; wait from reading | no (single class; fixed split is trivially "all waiting") | not run | data/hospital_queue/docs.json |
| epidemic | "Unobserved resident compartments start in a fixed age mix determined by the observed initial cases; waiting lists and intervention histories start empty." | `epidemic_y2`: E, I per age from cases0 via fixed EMIX and r_I; R = 0; S = rest; H = hosp0; WL, fatigue, debt, vaccination, behaviour = 0; report chain at cases0 | no | not run | data/epidemic/docs.json |
| social_contagion | "Initial members have a fixed disclosed-style community mix and no paid promises; other queues begin empty." | `social_contagion_z20/z21`: adopters split (1 - m0, m0) fixed; all waiting/onboarding cohorts 0; credibility K = 1; expectations E = 0; relationships R = 0 | no | not run | data/social_contagion/docs.json |
| wildlife | no system-specific rule (only "fixed reference conditions"; totals do not identify patch occupancy, juveniles, transit) | `wildlife_v8h/wild9m/tm*`: P, Q from reading; food R = 1; transit pools 0; predator food lag G = Z(prey0); shelter S = s0 | no (G = Z(y0) is an allowed deterministic y0 dependence) | not run | data/wildlife/docs.json |

## Findings

**supply_chain (conveyors empty).** Real and visible in the data: shipments are exactly 0 for the first 3
ticks of every run (the shipped model starts arrivals too early). The copy fixes the hold_rec shipments
score (0.978 -> 1.000) but loses the hold_mid fold by 0.007 (retail and shipments). Net +0.0005, pulse fold
unchanged. The effect only touches the first ~3-5 ticks, i.e. < 0.001 on a 4,000-tick episode. Not worth a
final slot on its own. Family: `gtlab/ode/supply_chain_v8b_rr.py`.

**power_grid (population at price 0.8).** The literal rule loses: without the load reading the first
ticks miss (hold_rec load 0.811 -> 0.760 without refit), and every fold except pulse gets worse; exam fold
-0.006/-0.007. The reset load reading carries real information (probably a non-thermostatic component we
lump into Ds). Keep the shipped init. Families: `power_grid_w5_rr.py`, `power_grid_w5_rr2.py`.

**reservoir (bank storage at a fixed reference).** The first run showed +0.046, all from the pulse fold
(0.755 -> 0.891). That is mostly an artefact: with the pulse run held out, irrigation is 0 in both training
runs, so the irrigation return k_ret is unidentified and the shipped-family polish drifted it 0 -> 0.16 while
the copy stayed at 0. Same result at 2x budget (`_b120`). With k_ret and tau_ret pinned in both families
(`_pinret`) the copy still wins every fold: +0.0015 / +0.0209 / +0.0023, mean +0.0083; the pulse fold gain is
in level (0.913 -> 0.952) and outflow (0.917 -> 0.964). Full-data refit (`plans/rr_fit_reservoir_reservoir_str9_rr.json`,
drop-in model.json under `model_json`) gives in-sample 0.8927 / 0.8996 / 0.8984 (mean 0.8969) vs the exact shipped
theta 0.8928 / 0.8982 / 0.8952 (0.8954); eval-shaped 4,000-tick rollouts finite, < 1.1 s. Caveat: b0 is poorly
identified (438-507 across folds, 385 on the full fit), only 3 runs. Candidate, small (+0.008 LOO, +0.0015 in-sample),
passes the rule (no fold loses, pulse fold gains; reservoir has no exam run). Family: `reservoir_str9_rr.py`.

**Other seven systems:** the shipped x0 already follows the documented rule (market_y3 was built from it).
No copy needed.

Note: the shipped reservoir theta has 21 entries, so the shipped model runs with d0 = 0 (module default `_K`);
the lab starts both families from d0 = 0.2 (PARAMS default). Same start for both, so the deltas are fair,
but the lab "noref" row for reservoir_str9 is not the exact shipped score (exact: mean 0.8954).
