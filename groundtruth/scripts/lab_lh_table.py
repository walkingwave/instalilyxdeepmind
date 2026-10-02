"""Print the tables of plans/lh_disagree.json (late-horizon disagreement study)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "plans" / "lh_disagree.json").read_text())
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(d)
for sid in only:
    r = d[sid]
    names = [c["name"] for c in r["candidates"]]
    print(f"\n=== {sid}  obs={r['observables']}")
    for c in r["candidates"]:
        print(f"  {c['name']:8s} {c['hash']} {str(c['model_id'])[:80]}  builds={len(c['builds'])}")
    print("  pairwise |a-b|/sigma  bands 0-400 / 400-1000 / 1000-4000  (all; sus ord rec comp late)")
    for k, v in r["pairwise"].items():
        cat_late = " ".join(f"{v[c][2]:.2f}" for c in ("sustained", "order", "recovery", "composition"))
        print(f"   {k:18s} {v['all'][0]:.2f} {v['all'][1]:.2f} {v['all'][2]:.2f} | {cat_late} | obs {np.round(v['late_obs'], 2).tolist()}")
    rv = r["ref_vs_median"]
    print("  final3 vs median:", {c: np.round(rv[c], 2).tolist() for c in rv})
    print("  cross-truth score (truth=row) bands; cols:", names[1:] if False else "", )
    for tn, row in r["cross_truth"].items():
        s = "  ".join(f"{k}:{'/'.join(f'{x:.3f}' for x in v)}" for k, v in row.items())
        print(f"   T={tn:8s} {s}")
    print("  runs (score on [0,400) and [400,T)):")
    for e in r["runs"]:
        if e["T"] <= 400:
            continue
        s = "  ".join(f"{k}:{'/'.join(f'{x:.3f}' for x in e[k])}" for k in names + ["median", "switch"])
        print(f"   {e['exp']} T={e['T']}  {s}")
        for k in names + ["median"]:
            print(f"      {k:8s} late per-obs {np.round(e[k + '_obs_late'], 3).tolist()}")
    early = {k: np.mean([e[k][0] for e in r["runs"]]) for k in names + ["median", "switch"]}
    print("  all runs [0,400) mean:", {k: round(v, 3) for k, v in early.items()})
