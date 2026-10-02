"""epi9: score mean / median combinations of saved LOO predictions (lab_epi9_loo.py npz files)."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
runs = load_runs(S.get("epidemic"), Ledger(data_dir("epidemic", False), "epidemic", False))
sig = np.array(json.loads(Path("plans/sigma_calibrated.json").read_text())["epidemic"]["sigma"], float)
Z = {Path(f).stem.replace("loo_epidemic_", ""): np.load(f) for f in sys.argv[1:]}
names = list(Z)
def score(members, how):
    fo = []
    for r in runs:
        P = np.stack([Z[m][f"Y_{r.exp}"] for m in members])
        Y = P.mean(0) if how == "mean" else np.median(P, 0)
        fo.append(metric.score_per_obs(Y, r.Y, sig).mean())
    return fo
print("folds:", [r.exp.split(".")[0] for r in runs])
rows = []
for k in range(1, len(names) + 1):
    for sub in itertools.combinations(names, k):
        for how in (["mean"] if k < 3 else ["mean", "median"]):
            fo = score(sub, how)
            rows.append((np.mean(fo), "+".join(sub), how, fo))
for m, sub, how, fo in sorted(rows, reverse=True)[:25] + [r for r in rows if "+" not in r[1]]:
    print(f"{m:.4f} {how:<6} {sub:<40} " + " ".join(f"{x:.3f}" for x in fo))
