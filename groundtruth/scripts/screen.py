"""Leave-one-run-out screen of model kinds on the real data (train runs only). Free.

For each system and kind: fit on all-but-one run, score the held-out run with robust_score
(sigma x {0.5, 1, 2}), paired against persistence on the same run. Prints mean score, paired
mean difference vs persistence and its standard error, and writes docs/study/screen.json.

    python scripts/screen.py [--kinds l0b_lin,l1,l2] [--budget 60]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, select as SEL
from gtlab.models import common as C
from gtlab.ledger import Ledger, data_dir, load_runs

warnings.filterwarnings("ignore")


def fit(kind, spec, runs, budget):
    clip = C.soft_clip(spec, runs)
    sigma = metric.sigma_proxy(runs)
    if kind == "l0b_lin":
        m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None})
    elif kind == "l0a":
        m = SEL.make_model("l0a", spec, clip, sigma)
    else:
        m = SEL.make_model(kind, spec, clip, sigma, time_budget_s=budget)
    return m.fit(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="l0b_lin,l1")
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--systems", default=",".join(S.SYSTEM_IDS))
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    out = {}
    for sid in a.systems.split(","):
        spec = S.get(sid)
        runs = [r for r in load_runs(spec, Ledger(data_dir(sid, False), sid, False)) if r.tags.get("split") != "val"]
        if len(runs) < 2:
            continue
        sigma = metric.sigma_proxy(runs)
        t0 = time.time()
        per = {k: [] for k in ["l0a"] + kinds}
        for i, hold in enumerate(runs):
            tr = runs[:i] + runs[i + 1:]
            for k in per:
                try:
                    m = fit(k, spec, tr, a.budget)
                    P = m.rollout(hold.y0, hold.U)
                    per[k].append(metric.robust_score(P, hold.Y, sigma) if np.all(np.isfinite(P)) else 0.0)
                except Exception as e:
                    per[k].append(0.0)
        base = np.array(per["l0a"])
        row = {}
        for k in per:
            v = np.array(per[k]); d = v - base
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            row[k] = {"mean": float(v.mean()), "diff": float(d.mean()), "se": se, "per_run": v.round(3).tolist()}
        best = max(row, key=lambda k: row[k]["mean"])
        out[sid] = {"n_runs": len(runs), "exps": [r.exp for r in runs], "kinds": row, "best": best}
        print(f"{sid:<17} n={len(runs)} " + " ".join(f"{k}={row[k]['mean']:.3f}({row[k]['diff']:+.3f}±{row[k]['se']:.3f})" for k in row)
              + f" best={best} [{time.time()-t0:.0f}s]", flush=True)
    Path("docs/study").mkdir(parents=True, exist_ok=True)
    Path("docs/study/screen.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
