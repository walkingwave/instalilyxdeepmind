"""LOO score of convex blends of held-out predictions saved by lab_wild9_loo.py.

    python scripts/lab_wild9_ens.py base w2 [more tags]
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs

spec = S.get("wildlife")
runs = load_runs(spec, Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
tags = sys.argv[1:]
P = {t: np.load(Path(os.environ["TEMP"]) / "wild9" / f"{t}.npz") for t in tags}

def fold_scores(pred_fn, pre="loo_"):
    return [metric.score_per_obs(pred_fn(r.exp, pre), r.Y, sigma) for r in runs]

def show(name, sc):
    m = [s.mean() for s in sc]
    print(f"{name:<28} " + " ".join(f"{x:.3f}" for x in m) + f" | mean {np.mean(m):.4f}")

print(" " * 29 + " ".join(r.exp[:5] for r in runs))
for t in tags:
    show(t, fold_scores(lambda e, p, t=t: P[t][p + e]))
a, b = tags[0], tags[1]
for w in (0.25, 0.5, 0.75):
    show(f"{w:.2f} {a} + {1-w:.2f} {b}", fold_scores(lambda e, p: w * P[a][p + e] + (1 - w) * P[b][p + e]))
if len(tags) >= 3:
    show("mean all", fold_scores(lambda e, p: np.mean([P[t][p + e] for t in tags], 0)))
    show("median all", fold_scores(lambda e, p: np.median([P[t][p + e] for t in tags], 0)))
# per-observable blend: prey from a, predators from b and vice versa
show(f"prey {a} / pred {b}", fold_scores(lambda e, p: np.where([1, 0, 1, 0], P[a][p + e], P[b][p + e])))
show(f"prey {b} / pred {a}", fold_scores(lambda e, p: np.where([1, 0, 1, 0], P[b][p + e], P[a][p + e])))
