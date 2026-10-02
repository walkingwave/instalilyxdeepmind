"""Traffic text-mined terms: deterministic leave-one-run-out at the calibrated 1.0 sigma.

    python scripts/lab_tm_traffic_loo.py --family traffic_tm1 --tag tm1a --extra kx3,kjo
        [--mech AB] [--nfev 60] [--starts 1] [--init '{"kx3": 0.005}'] [--folds p7.longhold,p5.testlike]

Every fold and the full fit start from the family's PARAMS init (plus --init overrides), fit the
base parameters + the active mechanism's parameters + --extra, with no time budget (so the result
does not depend on machine load). Scores the held-out run per observable. Writes
plans/traffic_tm_<tag>.json and plans/traffic_tm_<tag>_doc.json (full fit on every run).
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--extra", default="")
    ap.add_argument("--drop", default="", help="base names to hold fixed")
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--init", default="{}")
    ap.add_argument("--folds", default="")
    ap.add_argument("--no-full", action="store_true")
    a = ap.parse_args()
    spec = S.get("traffic")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    runs = load_runs(spec, Ledger(data_dir("traffic", False), "traffic", False))
    sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["traffic"]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    init = {p[0]: p[1] for p in mod.PARAMS}
    init.update(json.loads(a.init))
    theta0 = [init[n] for n in names]
    drop = set(x for x in a.drop.split(",") if x)
    free = [n for n in mod.free_for(a.mech) if n not in drop] + [x for x in a.extra.split(",") if x]
    free = list(dict.fromkeys(free))
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=int(getattr(mod, "N_SUB", 2)), early_T=None,
              spread=0.5, free=free, theta0=theta0, time_budget=None)
    only = set(x for x in a.folds.split(",") if x)
    rep = {"family": a.family, "tag": a.tag, "mech": a.mech, "free": free, "init": init, "nfev": a.nfev,
           "starts": a.starts, "sigma": sigma.tolist(), "loo": {}}
    n_sub = kw["n_sub"]
    with C.single_thread():
        for k, r in enumerate(runs):
            if only and r.exp not in only:
                continue
            t0 = time.time()
            tr = [x for i, x in enumerate(runs) if i != k]
            th, info = F.fit_ode(mod, tr, sigma, mech, **kw)
            Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
            s = metric.score_per_obs(Y, r.Y, sigma)
            rep["loo"][r.exp] = {"per_obs": s.tolist(), "mean": float(s.mean()), "cost": float(info["cost"]),
                                 "theta": {n: float(v) for n, v in zip(names, th)}}
            print(f"LOO {r.exp:<18} {np.round(s, 3)} mean {s.mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
        if rep["loo"]:
            rep["loo_mean"] = float(np.mean([v["mean"] for v in rep["loo"].values()]))
            print(f"LOO mean {rep['loo_mean']:.4f}")
        if not a.no_full:
            t0 = time.time()
            th, info = F.fit_ode(mod, runs, sigma, mech, **kw)
            rep["theta"] = {n: float(v) for n, v in zip(names, th)}
            rep["theta_vec"] = [float(v) for v in th]
            rep["insample"] = {}
            for r in runs:
                Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
                rep["insample"][r.exp] = metric.score_per_obs(Y, r.Y, sigma).tolist()
            rep["insample_mean"] = float(np.mean([np.mean(v) for v in rep["insample"].values()]))
            print(f"full fit [{time.time() - t0:.0f}s] in-sample {rep['insample_mean']:.4f}")
            print("theta", {n: round(v, 4) for n, v in rep["theta"].items() if n in free})
            blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": sorted(a.mech), "n_sub": n_sub}
            doc = C.make_doc(spec, blob, runs=runs, clip=C.soft_clip(spec, runs, margin=1.0),
                             info={"model_id": f"ode:{a.family}", "n_runs": len(runs),
                                   "n_ticks": int(sum(r.T for r in runs)), "cost": float(info["cost"])})
            Path(f"plans/traffic_tm_{a.tag}_doc.json").write_text(json.dumps(doc, indent=1))
    Path(f"plans/traffic_tm_{a.tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
