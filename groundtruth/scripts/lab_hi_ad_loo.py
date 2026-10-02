"""ad_auction hi lab: leave-one-run-out at 1.0 sigma (calibrated) for one ODE family, same protocol
as the str9 / pair labs (Cauchy LSQ, 8 starts, 60 evals, budget/2 per fold, budget for the full fit).
Free, no gateway calls.

    python scripts/lab_hi_ad_loo.py --family ad_auction_hi1 --tag a [--folds p6.exam,p8.longhold]
        [--no-full] [--no-loo] [--theta0 file.json] [--budget 300]
Writes plans/ad_auction_hi_<family>_<tag>[_<fold>].json and, with the full fit, ..._doc.json.
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

V8B_THETA = [266.51705246677216, 3.8319568601909597, 0.44095712682781824, 0.13824034735164845, 63.95965934504543,
             0.26793555409085906, 5.43316932471045, 5.793772654366348, 5.39384526673784, 0.37140060555771154,
             0.3762393353980513, 0.5058757683206756, 0.33318076592460205, 28.603860972230635]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--free", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--spread", type=float, default=0.5)
    a = ap.parse_args()
    spec = S.get("ad_auction")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset("AB")
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir("ad_auction", False), "ad_auction", False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["ad_auction"]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    if a.theta0:
        d = json.loads(Path(a.theta0).read_text())
        th = d.get("theta_vec") or d.get("theta")
        if isinstance(th, dict):
            init = {p[0]: p[1] for p in mod.PARAMS}
            init.update({k: v for k, v in th.items() if k in init})
            theta0 = [init[n] for n in names]
        else:
            theta0 = list(th) + [p[1] for p in mod.PARAMS[len(th):]]
    else:
        theta0 = [p[1] for p in mod.PARAMS]
    free = a.free.split(",") if a.free else None
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=a.spread, free=free, theta0=theta0)
    folds = a.folds.split(",") if a.folds else [r.exp for r in runs]
    base = f"ad_auction_hi_{a.family}_{a.tag}"
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
                rec = {"family": a.family, "held_out": rk.exp, "ode": s.tolist(), "theta": [float(v) for v in th]}
                (ROOT / "plans" / f"{base}_{rk.exp}.json").write_text(json.dumps(rec, indent=1))
                print(f"  LOO {rk.exp:<20} {np.round(s, 3)} mean {s.mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
        if a.no_full:
            return
        th, info = F.fit_ode(mod, runs, sigma, mech, time_budget=a.budget, **kw)
    rep = {"family": a.family, "sigma": sigma.tolist(), "insample": {}}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, sigma)
        rep["insample"][r.exp] = s.tolist()
        print(f"  in {r.exp:<20} {np.round(s, 3)} mean {s.mean():.4f}")
    rep["theta"] = {n: float(v) for n, v in zip(names, th)}
    rep["theta_vec"] = [float(v) for v in th]
    print("  theta", {k: round(v, 5) for k, v in rep["theta"].items()})
    rng = np.random.default_rng(0)
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        Y = core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        print(f"  eval[{cat}] finite={bool(np.all(np.isfinite(Y)))} min={np.round(Y.min(0), 3)} max={np.round(Y.max(0), 3)}")
    (ROOT / "plans" / f"{base}.json").write_text(json.dumps(rep, indent=1))
    blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": ["A", "B"], "n_sub": n_sub}
    doc = C.make_doc(spec, blob, runs=runs, clip=C.soft_clip(spec, runs, margin=1.0),
                     info={"model_id": f"ode:{a.family}", "n_runs": len(runs)})
    (ROOT / "plans" / f"{base}_doc.json").write_text(json.dumps(doc, indent=1))
    print("wrote", base)


if __name__ == "__main__":
    main()
