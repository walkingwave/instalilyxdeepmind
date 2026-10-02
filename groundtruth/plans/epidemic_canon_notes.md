# Epidemic canon: literal "textbook code" readings of the brief

Mon Sep 28. No credits. Six runs (p1 120, p2 400, p3 291, p4 300, p6 300, p8 400). LOO at
$1.0\sigma$ = (12.95, 5.23), ode_lab settings (150 s per fold, 8 starts, nfev 50), folds run in
parallel (7 processes), pair AC throughout. Script `scripts/lab_canon_epi_loo.py <family> AC <prefix>
[procs] [fix_json]`; ensembles `scripts/lab_canon_epi_ens.py`. y2 was re-run with the same script
on the same load as the baseline (0.698, matches epi9's 0.695/0.699).

## Hypothesis
The simulator was coded from a short spec, so it should look like a textbook age-structured SEIR-H
with round constants and one of the usual literal control readings. The brief's wording to match:
"School closure changes where contacts occur; masks reduce exposure; vaccination uses a shared
clinic workforce ... hospital pressure reduces clinic availability."

Literal readings written (all keep y2's disease/hospital/delay core, A/B/C mechanisms):

| family | closure | masks | combine | other |
|---|---|---|---|---|
| canon1 | setting layers: $C = w_h\,HOME + (1-c)\,w_s\,SCHOOL + WORK + w_c\,COMM$, share `displace` of school contacts moves to child community | non-home settings only, $\times(1-e_m m)$ | layered | textbook setting shapes |
| canon2 | $\beta(1-e_c c)$ on every contact | $\times(1-e_m m)$ on every contact | multiplicative | same matrix, no layers |
| canon3 | as canon2 | as canon2 | additive: $\beta\max(0, 1-e_c c-e_m m)$ | |
| canon4 | y2 (child-child removal) | y2 | y2 | clinic $=\max(0, 1-k\,H/h_{cap})$ instead of $1/(1+kH/h_{cap})$ |

## LOO fold table (held-out mean score per run)
| family | p1 | p2 | p3 | p4 | p6 | p8 | LOO | in-sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **y2 (baseline, rerun)** | .742 | .732 | .652 | .732 | .563 | .768 | **.698** | .761 |
| canon1 layers, masks outside home | .734 | .710 | .605 | .650 | .523 | .787 | .668 | .755 |
| canon2 textbook multiplicative | .694 | .636 | .589 | .642 | .499 | .699 | .626 | .747 |
| canon3 additive | .692 | .635 | .581 | .631 | .492 | .691 | .620 | .744 |
| canon4 linear clinic | .769 | .752 | .630 | .726 | .559 | .734 | .695 | .757 |
| canon4 snap1 ($k$=1, $\tau_{wane}$=90, school=1) | .774 | .762 | .632 | .712 | .571 | .726 | .696 | .751 |
| canon4 snap2 (+ los=10, $\tau_p$=6) | .770 | .760 | .623 | .707 | .563 | .725 | .691 | .757 |

Ensembles of saved fold predictions (equal mean): y2+canon4 .698, y2+canon4 snap1 .698,
y2+canon1 .688, y2+canon4+canon1 .693. No gain.

Acceptance (beat y2 at LOO, not lose p2 or p8): **nothing passes.** canon4 wins p1 (+.03) and p2
(+.02..+.03) but loses p8 (-.03..-.04) and p4.

## What the readings tell us about the code
- Textbook uniform readings are ruled out. With closure as a uniform $\beta$ factor the fitted
  $e_c$ collapses to 0.02-0.04 in both multiplicative and additive forms, and LOO drops 0.07. Closure
  acts on children's contacts only (y2's `school` sits at 1.0 in every fold: closure removes all
  child-child contacts). Additive vs multiplicative cannot be told apart (0.620 vs 0.626; masks and
  closure barely overlap because closure is weak on everything but children).
- Layered settings with masks outside the home (canon1) fit masks at $e_m = 0.95$ on non-home
  contacts with home weight 2: plausible literally, but it loses p3/p4/p6 held out (0.03-0.08).
  Best on p8 (.787), the only family that beats y2 there.
- **Clinic availability is linear in occupancy with slope 1.** Fitted free, canon4's $k$ lands at
  0.94-1.06 in all seven fits (six folds + all runs): clinic $= 1 - H/h_{cap}$, i.e. vaccination stops
  when beds are full. That is the round-constant reading, and it explains why y2's $1/(1+kH/h_{cap})$
  pins $k$ at its upper bound 5. It does not gain LOO (0.695 vs 0.698): the two forms differ only
  while $H \approx h_{cap}$, and the tie is decided by p8 (vaccination 0.0015 during a 60-tick
  full-hospital stretch).
- Other constants that are stable across every fold of y2 and canon4 (candidate round values):
  $\tau_{wane}$ 87-92 (90), $h_{cap}$ 154.1-154.6, los 9.1-10.6 (10), $\tau_p$ 5.6-6.6 (6), incubation
  $1/\sigma$ 6.2-7.2, mask_eff 0.39-0.45. Snapping them does not change LOO (0.696 / 0.691): the
  fits already sit there, so fewer free parameters buys nothing.
- The p8 fold is a different basin in every family ($\beta$ 0.16 vs 0.24, $\gamma$ 0.26 vs 0.40):
  without p8 the data do not pin the $\beta/\gamma$ split. That, not the control reading, sets the
  p8 loss.

## Weak spots unchanged
p6 (mask 1 alone): first wave 273 vs truth 253 in every family; vaccination-alone second wave
115-120 vs 144 at t=140. p3 held out stays at 0.60-0.65. No literal control reading moved either.

## Decision
Keep y2 AC. Record for the final write-up: clinic $= 1 - H/h_{cap}$ is probably the literal code
(k = 1.00 ± 0.06 across folds) and $\tau_{wane} = 90$; neither changes predictions enough to ship.
Files: `gtlab/ode/epidemic_canon1..4.py`, `plans/epidemic_canon*_AC.json` (fold scores + all-run
theta), `plans/epidemic_canon_y2base_AC.json` (baseline rerun).
