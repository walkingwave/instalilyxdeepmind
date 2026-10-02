"""Traffic hi families: deterministic leave-one-run-out at the calibrated 1.0 sigma, warm-started.

    python scripts/lab_hi_tr_loo.py --family traffic_hi_g --tag g --extra ps [--scale '{"n_ref": 2}']
        [--init '{"gam": 1}'] [--fix ps] [--jobs 8]

Fold k starts from the z8 fold-k fit (or, with --warm, the fold-k fit of an earlier candidate) of plans/traffic_tm_z8base.json (fit without run k, so no
held-out information enters the start), the full fit from the shipped z8all7 theta; new parameters
start at the family init (or --init), --scale multiplies a warm-start value. Every fit: one start,
max_nfev 60, free = base + A + B parameters + --extra, no time budget. Folds run as parallel
processes. Writes plans/traffic_hi_<tag>.json (per-fold scores) and plans/traffic_hi_<tag>_doc.json.
"""
import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F

BASE_REP = "plans/traffic_tm_z8base.json"
SHIPPED = "plans/traffic_traffic_z8_z8all7_doc.json"


def start_theta(mod, fold, a):
    init = {p[0]: p[1] for p in mod.PARAMS}
    if a.warm:
        w = json.loads(Path(a.warm).read_text())
        warm = w["theta"] if fold == "full" else w["loo"][fold]["theta"]
    elif fold == "full":
        z8 = importlib.import_module("gtlab.ode.traffic_z8")
        warm = dict(zip([p[0] for p in z8.PARAMS], json.loads(Path(SHIPPED).read_text())["model"]["theta"]))
    else:
        warm = json.loads(Path(BASE_REP).read_text())["loo"][fold]["theta"]
    for n in init:
        if n in warm:
            init[n] = warm[n]
    for n, s in json.loads(a.scale).items():
        init[n] = init[n] * s
    init.update(json.loads(a.init))
    lo = {p[0]: p[2] for p in mod.PARAMS}
    hi = {p[0]: p[3] for p in mod.PARAMS}
    return [min(max(init[p[0]], lo[p[0]]), hi[p[0]]) for p in mod.PARAMS]


def run_one(a, fold, out):
    spec = S.get("traffic")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    runs = load_runs(spec, Ledger(data_dir("traffic", False), "traffic", False))
    sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["traffic"]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    fix = set(x for x in a.fix.split(",") if x)
    free = [n for n in mod.free_for(a.mech) if n not in fix] + [x for x in a.extra.split(",") if x]
    free = list(dict.fromkeys(free))
    th0 = start_theta(mod, fold, a)
    n_sub = int(getattr(mod, "N_SUB", 2))
    kw = dict(n_starts=1, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=0.5, free=free,
              theta0=th0, time_budget=None)
    tr = [r for r in runs if r.exp != fold]
    t0 = time.time()
    with C.single_thread():
        th, info = F.fit_ode(mod, tr, sigma, mech, **kw)
    res = {"fold": fold, "theta": {n: float(v) for n, v in zip(names, th)}, "theta_vec": [float(v) for v in th],
           "cost": float(info["cost"]), "free": free, "secs": time.time() - t0, "scores": {}}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        res["scores"][r.exp] = metric.score_per_obs(Y, r.Y, sigma).tolist()
    Path(out).write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--extra", default="")
    ap.add_argument("--fix", default="")
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--init", default="{}")
    ap.add_argument("--scale", default="{}")
    ap.add_argument("--warm", default="", help="plans/traffic_hi_<tag>.json: start fold k from its fold-k fit")
    ap.add_argument("--one", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tmp", default="")
    a = ap.parse_args()
    if a.one:
        run_one(a, a.one, a.out)
        return
    spec = S.get("traffic")
    runs = load_runs(spec, Ledger(data_dir("traffic", False), "traffic", False))
    folds = [r.exp for r in runs] + ["full"]
    tmp = Path(a.tmp or ".")
    tmp.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, __file__, "--family", a.family, "--tag", a.tag, "--mech", a.mech, "--extra", a.extra,
            "--fix", a.fix, "--nfev", str(a.nfev), "--init", a.init, "--scale", a.scale, "--warm", a.warm]
    procs = []
    for fo in folds:
        out = tmp / f"hi_{a.tag}_{fo}.json"
        procs.append((fo, out, subprocess.Popen(args + ["--one", fo, "--out", str(out)])))
    for fo, out, p in procs:
        p.wait()
    rep = {"family": a.family, "tag": a.tag, "mech": a.mech, "extra": a.extra, "fix": a.fix, "init": a.init, "warm": a.warm,
           "scale": a.scale, "nfev": a.nfev, "loo": {}}
    for fo, out, p in procs:
        res = json.loads(out.read_text())
        if fo == "full":
            rep["theta"] = res["theta"]
            rep["theta_vec"] = res["theta_vec"]
            rep["insample"] = res["scores"]
            rep["insample_mean"] = float(np.mean([np.mean(v) for v in res["scores"].values()]))
            rep["free"] = res["free"]
            rep["full_cost"] = res["cost"]
        else:
            s = res["scores"][fo]
            rep["loo"][fo] = {"per_obs": s, "mean": float(np.mean(s)), "cost": res["cost"], "theta": res["theta"]}
    rep["loo_mean"] = float(np.mean([v["mean"] for v in rep["loo"].values()]))
    print(a.tag, " ".join(f"{k.split('.')[1][:8]} {v['mean']:.3f}" for k, v in rep["loo"].items()),
          f"LOO {rep['loo_mean']:.4f} ins {rep['insample_mean']:.4f}")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    n_sub = int(getattr(mod, "N_SUB", 2))
    blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": sorted(a.mech), "n_sub": n_sub}
    doc = C.make_doc(spec, blob, runs=runs, clip=C.soft_clip(spec, runs, margin=1.0),
                     info={"model_id": f"ode:{a.family}", "n_runs": len(runs),
                           "n_ticks": int(sum(r.T for r in runs)), "cost": rep["full_cost"]})
    Path(f"plans/traffic_hi_{a.tag}_doc.json").write_text(json.dumps(doc, indent=1))
    Path(f"plans/traffic_hi_{a.tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
