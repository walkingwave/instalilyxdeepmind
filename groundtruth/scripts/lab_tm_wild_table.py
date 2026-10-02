"""Fold table for wildlife_tm candidates vs the current median ensemble (v8h, wild9p, wild9m), from saved
held-out predictions, plus the p3-vs-p7 hunting floor check.

    python scripts/lab_tm_wild_table.py ab ac bc ... [--ens tag1+tag2+tag3 ...]
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs

runs = load_runs(S.get("wildlife"), Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
T = Path(os.environ["TEMP"])
args = sys.argv[1:]
ens = [a[6:] for a in args if a.startswith("--ens=")]
tags = [a for a in args if not a.startswith("--")]
P = {"v8h": np.load(T / "wild9" / "base.npz"), "w9p": np.load(T / "wild9" / "s2p.npz"), "w9m": np.load(T / "wild9" / "m.npz")}
for t in tags:
    P[t] = np.load(T / "wildtm" / f"{t}.npz")


def pred(name, key):
    if "+" in name:
        return np.median([P[m][key] for m in name.split("+")], 0)
    return P[name][key]


rows = ["v8h+w9p+w9m", "v8h", "w9p", "w9m"] + tags + ens
print(f"{'model':<24} {'LOO':>6} " + " ".join(f"{r.exp[:5]:>6}" for r in runs) + f" {'ins':>6}")
for name in rows:
    f = [metric.score_per_obs(pred(name, "loo_" + r.exp), r.Y, sigma).mean() for r in runs]
    i = [metric.score_per_obs(pred(name, "ins_" + r.exp), r.Y, sigma).mean() for r in runs]
    print(f"{name:<24} {np.mean(f):6.4f} " + " ".join(f"{x:6.3f}" for x in f) + f" {np.mean(i):6.3f}")

rr = {r.exp: r for r in runs}
print("\nfloor check (prey N / S): data vs held-out prediction")
pts = [("p3.compose", 245, "p3 end of hunt"), ("p3.compose", 221, "p3 +20"), ("p7.longhold", 20, "p7 +20"),
       ("p7.longhold", 45, "p7 +45"), ("p7.longhold", 300, "p7 +300")]
for exp, t, lab in pts:
    s = f"{lab:<16} data {rr[exp].Y[t, 0]:6.1f} {rr[exp].Y[t, 2]:6.1f} |"
    for name in rows:
        y = pred(name, "loo_" + exp)[t]
        s += f" {name[:8]} {y[0]:5.1f} {y[2]:5.1f}"
    print(s)
