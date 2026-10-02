"""hi lab: per-fold held-out table for supply_chain members and medians (no credits).

    python scripts/lab_hi_sc_combine.py LABEL=plans/supply_chain_hi_<fam>_<tag>.json[,more fold files] ...
v8b / v8c held-out predictions come from the ens9 cache (cold refits); hi members are re-rolled
from the per-fold theta stored by lab_hi_sc_loo.py. Scores at the calibrated sigma (1.0x),
predictions clipped to the pick's clip box before scoring.
"""
import importlib, itertools, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core

CACHE = Path(os.environ.get("ENS9_SCRATCH", "C:/Users/DYLANH~1/AppData/Local/Temp/gtscratch/ens9")) / "supply_chain"
spec = S.get("supply_chain")
runs = load_runs(spec, Ledger(data_dir("supply_chain", False), "supply_chain", False))
sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["supply_chain"]["sigma"], float)
pick = json.loads((ROOT / "submissions/20260928-1440-final1/supply_chain/model.json").read_text())
lo, hi = np.array(pick["clip_lo"]), np.array(pick["clip_hi"])
folds = [r.exp for r in runs]
P = {}
for lab in ("v8b", "v8c"):
    P[lab] = {e: np.asarray(json.loads((CACHE / f"{lab}__{e}.json").read_text())["Y"], float) for e in folds}
for arg in sys.argv[1:]:
    lab, files = arg.split("=")
    P[lab] = {}
    for fn in files.split(","):
        rep = json.loads(Path(fn).read_text())
        mod = importlib.import_module(f"gtlab.ode.{rep['family']}")
        for x in rep["loo"]:
            r = [r for r in runs if r.exp == x["held_out"]][0]
            P[lab][r.exp] = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, x["theta"]), "AB", n_sub=int(getattr(mod, "N_SUB", 2)))
labs = [l for l in P if all(e in P[l] for e in folds)]
combos = {l: [l] for l in labs}
for k in (2, 3):
    for c in itertools.combinations(labs, k):
        combos["med(" + ",".join(c) + ")"] = list(c)
rows = {}
for name, mem in combos.items():
    fs, obs = [], []
    for r in runs:
        Y = np.median(np.stack([np.clip(P[m][r.exp][: r.T], lo, hi) for m in mem]), axis=0)
        s = metric.score_per_obs(Y, r.Y, sigma)
        fs.append(float(s.mean())); obs.append(s)
    rows[name] = {"fold": dict(zip(folds, fs)), "mean": float(np.mean(fs)), "per_obs": np.mean(obs, 0).tolist()}
base = rows["v8b"]["fold"]
print(f"{'model':<34}" + "".join(f"{e[:14]:>16}" for e in folds) + f"{'mean':>8}{'wins':>6}")
for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
    w = sum(r["fold"][e] > base[e] + 1e-4 for e in folds)
    print(f"{name:<34}" + "".join(f"{r['fold'][e]:16.4f}" for e in folds) + f"{r['mean']:8.4f}{w:>4}/4")
Path(ROOT / "plans/supply_chain_hi_folds.json").write_text(json.dumps(rows, indent=1))
