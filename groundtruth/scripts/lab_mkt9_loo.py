"""Leave-one-run-out on market with the ode_lab protocol, saving each fold's held-out prediction.

    python scripts/lab_mkt9_loo.py --family market_y3 --mech AB --tag y3 [--free a,b] [--theta0 f.json]
        [--folds 0,1,...] [--full]

Protocol = scripts/ode_lab.py (--budget 300 --starts 8 --nfev 60 --sigma-cal 1.0): fold budget 150 s.
Writes plans/market_mkt9_<tag>_loo.json: per fold theta, scores (P/V/D), held-out prediction.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--free", default=None)
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    spec = S.get("market")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir("market", False), "market", False))
    sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["market"]["sigma"], float)
    free = a.free.split(",") if a.free else None
    theta0 = json.loads(Path(a.theta0).read_text())["theta_vec" if "theta_vec" in json.loads(Path(a.theta0).read_text()) else "theta"] if a.theta0 else None
    kw = dict(n_starts=8, max_nfev=60, n_sub=n_sub, early_T=None, spread=0.5, free=free, theta0=theta0)
    out = Path("plans") / f"market_mkt9_{a.tag}_loo.json"
    rep = json.loads(out.read_text()) if out.exists() else {"family": a.family, "mech": a.mech, "folds": {}}
    folds = [int(k) for k in a.folds.split(",")] if a.folds else list(range(len(runs)))
    with C.single_thread():
        for k in folds:
            tr = [r for i, r in enumerate(runs) if i != k]
            t0 = time.time()
            th, info = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget / 2, **kw)
            Y = core.rollout(mod, runs[k].y0, runs[k].U, core.theta_dict(mod, th), mech, n_sub=n_sub)
            s = metric.score_per_obs(Y, runs[k].Y, sigma)
            rep["folds"][runs[k].exp] = {"theta": [float(v) for v in th], "score": s.tolist(), "pred": Y.tolist()}
            out.write_text(json.dumps(rep))
            print(f"{a.tag} fold {runs[k].exp:<18} {np.round(s, 3)} mean {s.mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
        if a.full:
            th, info = F.fit_ode(mod, runs, sigma, mech, time_budget=a.budget, **kw)
            rep["full"] = {"theta": [float(v) for v in th], "insample": {}}
            for r in runs:
                Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
                rep["full"]["insample"][r.exp] = metric.score_per_obs(Y, r.Y, sigma).tolist()
            out.write_text(json.dumps(rep))
    m = [np.mean(v["score"]) for v in rep["folds"].values()]
    print(f"{a.tag}: {len(m)} folds, mean {np.mean(m):.3f}")


if __name__ == "__main__":
    main()
