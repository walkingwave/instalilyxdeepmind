"""Fold table for the traffic hi candidates (plans/traffic_hi_<tag>.json) against the warm z8 refit
(tag base) and the shipped-protocol z8 folds (plans/traffic_tm_z8base.json).

    python scripts/lab_hi_tr_table.py [--tags base,ps,g,...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

FOLDS = ["p1.hold_rec", "p2.pulse40", "p2.mid40", "p2.multilevel200", "p3.hold_mid", "p5.testlike", "p7.longhold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t] or sorted(p.stem[len("traffic_hi_"):] for p in Path("plans").glob("traffic_hi_*.json")
                                                        if not p.stem.endswith("_doc"))
    reps = {"z8(tm)": json.loads(Path("plans/traffic_tm_z8base.json").read_text())}
    for t in tags:
        p = Path(f"plans/traffic_hi_{t}.json")
        if p.exists():
            reps[t] = json.loads(p.read_text())
    base = reps.get("base")
    print("| cand | " + " | ".join(f.split(".")[1][:9] for f in FOLDS) + " | LOO | vs base | wins | in-sample |")
    print("|---|" + "---:|" * (len(FOLDS) + 4))
    for t, r in reps.items():
        v = [r["loo"][f]["mean"] for f in FOLDS]
        d = "" if base is None else f"{np.mean(v) - base['loo_mean']:+.4f}"
        w = "" if base is None else f"{sum(x > base['loo'][f]['mean'] + 0.002 for x, f in zip(v, FOLDS))}/{sum(x < base['loo'][f]['mean'] - 0.002 for x, f in zip(v, FOLDS))}"
        print(f"| {t} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.4f} | {d} | {w} | {r.get('insample_mean', float('nan')):.4f} |")
    print("\nper observable (flow_a flow_b speed_a speed_b) on pulse40 / multilevel / exam / p7")
    for t, r in reps.items():
        print(t.ljust(8), "  ".join(" ".join(f"{x:.2f}" for x in r["loo"][f]["per_obs"]) for f in
                                    ["p2.pulse40", "p2.multilevel200", "p5.testlike", "p7.longhold"]))


if __name__ == "__main__":
    main()
