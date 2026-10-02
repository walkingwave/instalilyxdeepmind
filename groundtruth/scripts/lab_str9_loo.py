"""str9 lab: leave-one-run-out at the calibrated organizer scale (1.0 sigma) for one ODE family,
plus a full fit, per-observable fold table, and a model doc. Free, no gateway calls.

    python scripts/lab_str9_loo.py --system reservoir --family reservoir_str9 --tag a [--budget 300]
        [--starts 8] [--nfev 60] [--theta0 file.json|v8ship] [--free p1,p2] [--no-loo]
Writes plans/<system>_str9_<family>_<tag>.json and ..._doc.json.
"""
import argparse, importlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--free", default=None)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--folds", default=None, help="comma list of run exps to hold out (default all)")
    ap.add_argument("--spread", type=float, default=0.5)
    a = ap.parse_args()
    spec = S.get(a.system)
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    theta0 = None
    if a.theta0:
        d = json.loads(Path(a.theta0).read_text())
        if "theta" in d and isinstance(d["theta"], dict):
            init = {p[0]: p[1] for p in mod.PARAMS}
            init.update({k: v for k, v in d["theta"].items() if k in init})
            theta0 = [init[n] for n in names]
        else:
            th = d.get("theta_vec") or d.get("theta") or d["model"]["theta"]
            theta0 = list(th) + [p[1] for p in mod.PARAMS[len(th):]]
    free = a.free.split(",") if a.free else None
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=a.spread, free=free, theta0=theta0)
    rep = {"system": a.system, "family": a.family, "runs": [(r.exp, r.T) for r in runs], "sigma": sigma.tolist(), "loo": []}
    print(f"{a.system}/{a.family} runs {rep['runs']}", flush=True)
    folds = a.folds.split(",") if a.folds else [r.exp for r in runs]
    with C.single_thread():
        if not a.no_loo:
            for k, rk in enumerate(runs):
                if rk.exp not in folds:
                    continue
                tr = [r for i, r in enumerate(runs) if i != k]
                t0 = time.time()
                th, info = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget / 2, **kw)
                Y = core.rollout(mod, rk.y0, rk.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
                s = metric.score_per_obs(Y, rk.Y, sigma)
                rep["loo"].append({"held_out": rk.exp, "ode": s.tolist(), "theta": [float(v) for v in th]})
                print(f"  LOO {rk.exp:<28} {np.round(s, 3)} mean {s.mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
            if rep["loo"]:
                rep["loo_mean"] = float(np.mean([np.mean(x["ode"]) for x in rep["loo"]]))
                print(f"  LOO mean {rep['loo_mean']:.4f}", flush=True)
        th, info = F.fit_ode(mod, runs, sigma, mech, time_budget=a.budget, **kw)
    rep["insample"] = {}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, sigma)
        rep["insample"][r.exp] = s.tolist()
        print(f"  in {r.exp:<28} {np.round(s, 3)} mean {s.mean():.4f}")
    rep["theta"] = {n: float(v) for n, v in zip(names, th)}
    rep["theta_vec"] = [float(v) for v in th]
    print("  theta", {k: round(v, 5) for k, v in rep["theta"].items()})
    rng = np.random.default_rng(0)
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time()
        Y = core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        print(f"  eval[{cat}] finite={bool(np.all(np.isfinite(Y)))} {time.time() - t0:.2f}s min={np.round(Y.min(0), 3)} max={np.round(Y.max(0), 3)}")
    out = ROOT / "plans" / f"{a.system}_str9_{a.family}_{a.tag}.json"
    out.write_text(json.dumps(rep, indent=1))
    blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": sorted(a.mech), "n_sub": n_sub}
    doc = C.make_doc(spec, blob, runs=runs, clip=C.soft_clip(spec, runs, margin=1.0),
                     info={"model_id": f"ode:{a.family}", "n_runs": len(runs)})
    if a.system == "reservoir":
        doc["post"] = []
    (ROOT / "plans" / f"{a.system}_str9_{a.family}_{a.tag}_doc.json").write_text(json.dumps(doc, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
