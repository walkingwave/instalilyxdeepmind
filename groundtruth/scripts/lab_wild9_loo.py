"""LOO + full fit for one wildlife family; saves fold thetas and held-out predictions for ensembling.

    python scripts/lab_wild9_loo.py <family> <tag> [--budget 300] [--starts 8] [--nfev 60] [--no-loo] [--theta0 json]
"""
import argparse, importlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F

ap = argparse.ArgumentParser()
ap.add_argument("family"); ap.add_argument("tag")
ap.add_argument("--budget", type=float, default=300); ap.add_argument("--starts", type=int, default=8)
ap.add_argument("--nfev", type=int, default=60); ap.add_argument("--no-loo", action="store_true")
ap.add_argument("--theta0", default=None); ap.add_argument("--mech", default="AB")
ap.add_argument("--out", default=None)
a = ap.parse_args()
spec = S.get("wildlife")
mod = importlib.import_module(f"gtlab.ode.{a.family}")
mech = frozenset(a.mech)
runs = load_runs(spec, Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
names = [p[0] for p in mod.PARAMS]
theta0 = None
if a.theta0:
    d = json.loads(Path(a.theta0).read_text())["theta"]
    theta0 = [d.get(n, p[1]) for n, p in zip(names, mod.PARAMS)]
kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=2, early_T=None, spread=0.5, theta0=theta0)
out = {"family": a.family, "names": names, "loo": {}, "folds": {}}
preds = {}
with C.single_thread():
    if not a.no_loo:
        for k, r in enumerate(runs):
            tr = [x for i, x in enumerate(runs) if i != k]
            t0 = time.time()
            th, info = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget / 2, **kw)
            Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=2)
            s = metric.score_per_obs(Y, r.Y, sigma)
            out["loo"][r.exp] = s.tolist(); out["folds"][r.exp] = [float(v) for v in th]
            preds["loo_" + r.exp] = Y
            print(f"LOO {r.exp:<22} {np.round(s, 3)} mean {s.mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
        print(f"LOO mean {np.mean([np.mean(v) for v in out['loo'].values()]):.4f}", flush=True)
    th, info = F.fit_ode(mod, runs, sigma, mech, time_budget=a.budget, **kw)
out["theta"] = {n: float(v) for n, v in zip(names, th)}; out["theta_vec"] = [float(v) for v in th]
out["insample"] = {}
for r in runs:
    Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=2)
    preds["ins_" + r.exp] = Y
    s = metric.score_per_obs(Y, r.Y, sigma); out["insample"][r.exp] = s.tolist()
    print(f"ins {r.exp:<22} {np.round(s, 3)} mean {s.mean():.3f}", flush=True)
print(f"ins mean {np.mean([np.mean(v) for v in out['insample'].values()]):.4f}", flush=True)
print("theta", {k: round(v, 5) for k, v in out["theta"].items()}, flush=True)
o = Path(a.out or f"plans/wildlife_wild9_{a.tag}.json")
o.write_text(json.dumps(out, indent=1))
import os
np.savez(Path(os.environ.get("TEMP", ".")) / "wild9" / f"{a.tag}.npz", **preds)
