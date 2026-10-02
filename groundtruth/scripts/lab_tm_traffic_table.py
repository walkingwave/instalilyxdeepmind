"""Fold table for the traffic text-mined candidates vs the z8 baseline (same protocol).

    python scripts/lab_tm_traffic_table.py [tags...]   (default: every plans/traffic_tm_*.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

FOLDS = ["p1.hold_rec", "p2.pulse40", "p2.mid40", "p2.multilevel200", "p3.hold_mid", "p5.testlike", "p7.longhold"]


def load(tag):
    return json.loads(Path(f"plans/traffic_tm_{tag}.json").read_text())


def main():
    tags = sys.argv[1:] or sorted(p.stem[len("traffic_tm_"):] for p in Path("plans").glob("traffic_tm_*.json")
                                  if not p.stem.endswith("_doc") and not p.stem.endswith("profile"))
    base = load("z8base") if Path("plans/traffic_tm_z8base.json").exists() else None
    print("| cand | " + " | ".join(f.split(".")[1] for f in FOLDS) + " | LOO | vs z8 | in-sample |")
    print("|---|" + "---:|" * (len(FOLDS) + 3))
    for t in tags:
        d = load(t)
        row = [d["loo"].get(f, {}).get("mean", np.nan) for f in FOLDS]
        m = np.nanmean(row)
        diff = ""
        if base is not None and t != "z8base":
            b = [base["loo"].get(f, {}).get("mean", np.nan) for f in FOLDS]
            ok = [i for i in range(len(FOLDS)) if np.isfinite(row[i]) and np.isfinite(b[i])]
            diff = f"{np.mean([row[i] - b[i] for i in ok]):+.3f}"
        cells = " | ".join("" if not np.isfinite(x) else f"{x:.3f}" for x in row)
        print(f"| {t} | {cells} | {m:.3f} | {diff} | {d.get('insample_mean', float('nan')):.3f} |")
    if base is not None:
        print()
        for t in tags:
            d = load(t)
            if "p7.longhold" in d["loo"]:
                print(t, "p7 per obs", np.round(d["loo"]["p7.longhold"]["per_obs"], 3),
                      "exam per obs", np.round(d["loo"].get("p5.testlike", {}).get("per_obs", []), 3))


if __name__ == "__main__":
    main()
