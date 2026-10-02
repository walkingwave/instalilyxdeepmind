"""Fold table for the power_grid text-mined fits (plans/power_grid_tm_<tag>.json).

    python scripts/lab_tm_pg_table.py [tag ...]

Columns: LOO per held-out run (mean over observables), LOO mean, in-sample mean, exam; the last
block repeats the per-observable LOO and exam for each tag. The w5 row from the w notes is shown
as the shipped reference (fit under a different machine load; w5ref is the same-conditions rerun).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ["p1.hold_rec", "p2.pulse60_120", "p2.multilevel200", "p8.longhold"]
W5_NOTES = {"p1.hold_rec": 0.770, "p2.pulse60_120": 0.688, "p2.multilevel200": 0.733, "p8.longhold": 0.650,
            "mean": 0.710, "exam": 0.685}


def row(tag):
    p = ROOT / "plans" / f"power_grid_tm_{tag}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    loo = {x["held_out"]: x["score"] for x in r.get("loo", [])}
    ex = list(r["exam"].values())[0]
    ins = float(np.mean([np.mean(v) for v in r["insample"].values()]))
    return r, loo, ex, ins


def main():
    tags = sys.argv[1:] or sorted(p.stem[len("power_grid_tm_"):] for p in (ROOT / "plans").glob("power_grid_tm_*.json")
                                  if not p.stem.endswith("_doc"))
    print(f"| fit | {' | '.join(r.split('.')[1] for r in RUNS)} | LOO mean | in-sample | exam | at bound |")
    print("|---|" + "---:|" * (len(RUNS) + 3) + "---|")
    print(f"| w5 (w notes) | {' | '.join(f'{W5_NOTES[r]:.3f}' for r in RUNS)} | {W5_NOTES['mean']:.3f} | 0.789 | {W5_NOTES['exam']:.3f} | none |")
    detail = []
    for t in tags:
        x = row(t)
        if x is None:
            continue
        r, loo, ex, ins = x
        cells = [f"{np.mean(loo[k]):.3f}" if k in loo else "-" for k in RUNS]
        lm = f"{np.mean([np.mean(v) for v in loo.values()]):.3f}" if loo else "-"
        print(f"| {t} ({r['family']}) | {' | '.join(cells)} | {lm} | {ins:.3f} | {np.mean(ex):.3f} | {', '.join(r['at_bound']) or 'none'} |")
        detail.append((t, loo, ex))
    print()
    for t, loo, ex in detail:
        print(t, {k.split('.')[1]: np.round(v, 3).tolist() for k, v in loo.items()}, "exam", np.round(ex, 3).tolist())


if __name__ == "__main__":
    main()
