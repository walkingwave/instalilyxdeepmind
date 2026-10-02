"""reservoir hi: one custom train/test split for an ODE family at the calibrated sigma. Free.

    python scripts/lab_hi_res_fold.py --family reservoir_str9 --train p2.pulse200_200 \
        --theta0 defaults|file.json --fix k_ret=0,tau_ret=32.66,w_L=1 --tag x [--budget 200]
Writes plans/reservoir_hi_fold_<tag>.json (theta + per-run per-observable scores).
"""
import argparse, importlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--train", required=True, help="comma list of run exps")
    ap.add_argument("--theta0", default="defaults")
    ap.add_argument("--fix", default="", help="name=value,... held fixed")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", type=float, default=200)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--spread", type=float, default=0.5)
    a = ap.parse_args()
    spec = S.get("reservoir")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir("reservoir", False), "reservoir", False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["reservoir"]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    init = {p[0]: p[1] for p in mod.PARAMS}
    if a.theta0 != "defaults":
        d = json.loads(Path(a.theta0).read_text())
        if isinstance(d.get("theta"), dict):
            init.update({k: v for k, v in d["theta"].items() if k in init})
        else:
            th = d.get("theta_vec") or d.get("theta") or d["model"]["theta"]
            init.update(dict(zip(names, th)))
    fixed = {}
    for kv in filter(None, a.fix.split(",")):
        k, v = kv.split("=")
        fixed[k] = float(v)
    init.update(fixed)
    theta0 = [init[n] for n in names]
    free = [n for n in names if n not in fixed]
    train = a.train.split(",")
    tr = [r for r in runs if r.exp in train]
    t0 = time.time()
    with C.single_thread():
        th, info = F.fit_ode(mod, tr, sigma, frozenset("AB"), n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub,
                             early_T=None, spread=a.spread, free=free, theta0=theta0, time_budget=a.budget)
    rep = {"family": a.family, "train": train, "fixed": fixed, "theta0": a.theta0, "time": time.time() - t0,
           "theta": {n: float(v) for n, v in zip(names, th)}, "theta_vec": [float(v) for v in th], "scores": {}}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset("AB"), n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, sigma)
        rep["scores"][r.exp] = s.tolist()
        flag = "train" if r.exp in train else "HELD"
        print(f"{a.tag:14s} {r.exp:18s} {flag:5s} {np.round(s, 3)} {s.mean():.4f}", flush=True)
    (ROOT / "plans" / f"reservoir_hi_fold_{a.tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
