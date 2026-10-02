# Purchase 9: final discovery round (Mon Sep 28, designed, NOT bought)

Question: with 2,747 credits left (reservoir 780, ad_auction 400, power_grid 400, hospital 230,
supply 230, epidemic / social / wildlife 189, market / traffic 160), which single run per system is
most likely to show a regime our models have never seen and the 4,000-tick test will visit?
The last two days' only gains came from such runs (traffic +0.063, wildlife, power_grid); refits on
owned data are exhausted (MATH_LOG §36, §39).

## Method (free)

1. **Coverage** (`scripts/lab_p9_coverage.py`): every owned hold of >= 60 ticks per system, as its
   control vector normalized to the bounds. Owned long holds are almost all recovery, full pulse,
   interior mid, or one harsh p7/p8 corner. **No system has a single-control or partial-pulse hold
   longer than 100 ticks except wildlife habitat / epidemic mask / vaccination**; hospital's single
   controls were held 30 ticks, market's tax 60.
2. **Candidates**: every level on the brief's recovery -> pulse axis at $\alpha = 0.85$ (single
   controls, all-but-one, pairs for $m \le 4$, full pulse) plus the bound midpoint; each as a hold for
   the whole budget and as hold (60%) + recovery (40%).
3. **Committee** (`scripts/lab_p9_committee.py`, `plans/p9_committee.json`): shipped predictors from
   u008 to final1 and pg9-w5 within 0.08 of the system's best public score, deduplicated. Committee
   loss $L = \mathrm{mean}_{i,t,k}\big(1 - 1/(1 + |y_{itk} - \tilde y_{tk}|/\sigma_k)\big)$, $\tilde y$ the
   committee median, $\sigma$ the calibrated organizer scale: the score lost by shipping the median if
   one member were the truth. Also $L$ on the last 40% of the window and for the level held to 4,000.
4. **Novelty**: RMS distance (bound-normalized) to the nearest owned hold of >= 100 ticks.
   Committee loss alone is misleading where members predate data we now own (market "mid" ranks
   first with novelty 0: p6 and p7 already hold it). Rule: **only candidates with novelty >= 0.25,
   ranked by $L$**, then judgment where the committee is blind (a control every member ignores gives
   zero disagreement by construction: hospital overtime, power_grid charging, traffic freight /
   clearance, supply lead_time_buy).

## Design (`scripts/make_p9.py` -> `plans/p9.json`, one run `p9.discover` per system, 20 credits left each)

Levels: controls named are at recovery + 0.85 (pulse - recovery); all others at recovery.

| system | run (blocks) | credits | L (last 40%) | why |
|---|---|---:|---:|---|
| reservoir | full pulse 420 -> recovery 120 -> release+irrigation only 220 | 760 | 0.070 (0.061) | owned pulses stop at 200 ticks with outflow still falling 13.7 -> 11.2 (screen fouling) and level 280 and flattening; the test holds pulses for thousands of ticks. Then refill, then drain with clean screens (withdrawal 0, aeration 1): separates fouling from drainage. release+irrigation alone has novelty 0.55, the highest of any candidate |
| ad_auction | full pulse 230 -> bid only 150 | 380 | 0.102 (0.117) | top two unowned candidates: full pulse (owned 60 ticks; audience depletion by repeated exposure) and bid alone (never held; highest long-horizon disagreement 0.121) |
| power_grid | price+interconnector 200 -> + charging cut 180 | 380 | 0.145 (0.147) | price+interconnector is the top unowned candidate (novelty 0.34). The second block changes only charging_allowance (authority 0.00 in every model) while the store is depleted, where its refill term should act: a sustained footprint for the ignored control |
| hospital_queue | overtime only 110 -> recovery 100 | 210 | 0.135 (0.201) | the largest single miss anywhere (+16.7 σ): after overtime drains the queue, wait climbs +2/tick. Owned only 30 ticks, confounded with the follow-up block after it. Recovery afterwards decides state effect (keeps climbing) vs control law (falls). Committee is blind to overtime (all members ignore it); staffing-only was its top pick (0.156) and is the alternative |
| supply_chain | orders only 130 -> + lead_time_buy cut 80 | 210 | 0.469 (0.540) | orders alone: largest disagreement of any system (supplier inventory 0 vs 362, retail 0 to 880; 4,000-tick L 0.58). Recovery has orders 0, so lead_time_buy can only act with orders flowing: the second block gives it its first sustained footprint |
| epidemic | school closure only 169 | 169 | 0.173 (0.264) | never held alone (novelty 0.46); highest late-window disagreement; the brief's closure vs masking comparison (mask alone owned 100) |
| social_contagion | seeding+incentive, no bridge, 169 | 169 | 0.360 (0.388) | top unowned candidate; owned seeding+bridge (p6) and full pulse, never incentive with seeding and bridge off |
| wildlife | hunting+corridor under full protection 169 | 169 | 0.169 (0.188) | top unowned candidate (novelty 0.42) and largest 4,000-tick disagreement (0.221): prey 14-18 vs 36-48; the brief's harvest before/after protection |
| market | transaction tax only 140 | 140 | 0.227 (0.290) | the only unowned level on the pulse axis (novelty 0.43); tax alone owned 60 ticks (depth 91 -> 44 and still falling); depth is market's weak observable |
| traffic | pulse except toll (toll 5) 80 -> freight + clearance back to recovery 60 | 140 | 0.225 (0.218) | top candidate: u016's toll-elastic demand says speeds 27 / 32, older fits 11 / 12 (p7 settled within ~100 ticks). The second block changes only freight_priority and clearance_effort (ignored by every model) under congestion |
| **total** | | **2,727** | | 20 left on every system (200) |

Rejected: market "mid" and power_grid / supply "all" (top committee loss but owned: members predate
p6/p7/p8); pure recovery tails after short pulses (owned on 8 systems); interior "mid" holds
(off the test's pulse axis); two short runs instead of one (the value is in depth past what we own).

## Commands (PowerShell, from groundtruth\, venv active)

```powershell
python scripts/make_p9.py --check          # FREE: rebuild plans/p9.json + in-memory dry-run vs real ledgers
python -m gtlab.cli plan --all --phase p9 --experiments plans/p9.json --real-dir   # FREE: materialize into data/<sys>/plan.json
python scripts/make_p9.py --budget         # FREE: p9 cap = run length in data/<sys>/budget.json, reserve = rest
python -m gtlab.cli collect --all --phase p9 --spend --dry-run --max-steps 760     # FREE: official dry run (calls nothing)
$env:GT_ALLOW_SPEND = "1"
# USES 2,727 CREDITS: reservoir 760, ad_auction 380, power_grid 380, hospital_queue 210,
# supply_chain 210, epidemic 169, social_contagion 169, wildlife 169, market 140, traffic 140
python -m gtlab.cli collect --all --phase p9 --spend --max-steps 760 --yes
Remove-Item Env:GT_ALLOW_SPEND
```

Per-system alternative (same credits): `python -m gtlab.cli collect <sys> --phase p9 --spend --max-steps <credits> --yes`.

`gtlab/design.py` gained phase `p9` (appended after `reserve` so the default-phase seeds of
`val`/`reserve` do not change).
