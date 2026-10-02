"""power_grid text-mined terms lab fit: leave-one-run-out + full fit of one family, exam (p6.exam)
never fitted. Same protocol as scripts/lab_pg9_fit.py.

    python scripts/lab_tm_pg_fit.py --family power_grid_tm1 --tag tm1 [--budget 300] [--starts 8] [--nfev 60]
        [--init plans/x.json] [--free a,b] [--workers 5] [--no-loo]

--init takes a report or doc json with a theta dict (or a family + theta vec) and uses the values of
the parameters with the same name as the start point; the others keep the family defaults.

Same fit call as scripts/ode_lab.py (gtlab.ode.fit.fit_ode, cauchy, calibrated sigma 1.0x, mech AB).
Fit runs: p1.hold_rec, p2.pulse60_120, p2.multilevel200, p8.longhold. Folds run in parallel processes.
Writes plans/power_grid_tm_<tag>.json (thetas, LOO / in-sample / exam per observable) and
plans/power_grid_tm_<tag>_doc.json (model doc from the full fit, clip from the fit runs).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402

SID = "power_grid"
NOFIT = {"p6.exam"}


def setup():
    spec = S.get(SID)
    runs = load_runs(spec, Ledger(data_dir(SID, False), SID, False))
    sig = np.asarray(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[SID]["sigma"], float)
    return spec, [r for r in runs if r.exp not in NOFIT], [r for r in runs if r.exp in NOFIT], sig


def score(mod, th, r, sig, n_sub):
    Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset("AB"), n_sub=n_sub)
    return (1.0 / (1.0 + np.abs(Y - r.Y) / sig)).mean(0)


def job(args):
    family, k, kw = args
    mod = importlib.import_module(f"gtlab.ode.{family}")
    spec, fit_runs, exams, sig = setup()
    n_sub = int(getattr(mod, "N_SUB", 2))
    tr = fit_runs if k < 0 else [r for i, r in enumerate(fit_runs) if i != k]
    t0 = time.time()
    fsig = sig.copy()
    if kw.pop("load_only", False):
        fsig[1:] *= 1e6                      # fit the load observable only
    with C.single_thread():
        th, info = F.fit_ode(mod, tr, fsig, frozenset("AB"), n_sub=n_sub, **kw)
    out = {"k": k, "theta": [float(v) for v in th], "cost": float(info["cost"]), "sec": time.time() - t0}
    if k >= 0:
        out["held_out"] = fit_runs[k].exp
        out["score"] = score(mod, th, fit_runs[k], sig, n_sub).tolist()
    else:
        out["insample"] = {r.exp: score(mod, th, r, sig, n_sub).tolist() for r in fit_runs}
        out["exam"] = {r.exp: score(mod, th, r, sig, n_sub).tolist() for r in exams}
    print(f"[{family} fold {k}] done {out['sec']:.0f}s " + (f"{out.get('held_out')} {np.round(out['score'], 3)}" if k >= 0 else
          f"exam {[np.round(v, 3).tolist() for v in out['exam'].values()]}"), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="power_grid_tm1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--init-report", default=None, help="report json of a base family: each fold starts from the "
                    "base fold theta with the same held-out run, the full fit from the base full theta (by name)")
    ap.add_argument("--set", default=None, help="name=value,... overrides of the start point (after --init)")
    ap.add_argument("--free", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--load-only", action="store_true", help="fit on the load residuals only (for a per-observable map)")
    a = ap.parse_args()
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    spec, fit_runs, exams, sig = setup()
    theta0 = json.loads(Path(a.theta0).read_text())["theta_vec"] if a.theta0 else None
    if a.init:
        src = json.loads(Path(a.init).read_text())
        if isinstance(src.get("theta"), dict):
            named = src["theta"]
        else:
            m0 = src["model"] if "model" in src else src
            fam0 = importlib.import_module(f"gtlab.ode.{m0['family']}")
            named = dict(zip([p[0] for p in fam0.PARAMS], m0["theta"]))
        theta0 = [float(np.clip(named.get(n, ini), lo, hi)) for n, ini, lo, hi, lg in mod.PARAMS]
        print("init from", a.init, "shared:", [n for n, *_ in mod.PARAMS if n in named])
    base_kw = dict(n_starts=a.starts, max_nfev=a.nfev, early_T=None, spread=a.spread, theta0=theta0,
                   free=a.free.split(",") if a.free else None, load_only=a.load_only)
    over = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.set.split(",")} if a.set else {}

    def start(named):
        return [float(np.clip(over.get(n, named.get(n, ini)), lo, hi)) for n, ini, lo, hi, lg in mod.PARAMS]

    if over and theta0 is None and not a.init_report:
        base_kw["theta0"] = start({})
    elif over and theta0 is not None:
        base_kw["theta0"] = start(dict(zip([p[0] for p in mod.PARAMS], theta0)))
    fold0 = {}
    if a.init_report:
        rb = json.loads(Path(a.init_report).read_text())
        fb = importlib.import_module(f"gtlab.ode.{rb['family']}")
        nb = [p[0] for p in fb.PARAMS]
        fold0[-1] = start(dict(zip(nb, rb["theta_vec"])))
        for x in rb["loo"]:
            k = [r.exp for r in fit_runs].index(x["held_out"])
            fold0[k] = start(dict(zip(nb, x["theta"])))
        print("init per fold from", a.init_report)
    jobs = [(a.family, -1, dict(base_kw, time_budget=a.budget, **({"theta0": fold0[-1]} if fold0 else {})))]
    if not a.no_loo:
        jobs += [(a.family, k, dict(base_kw, time_budget=a.budget / 2, **({"theta0": fold0[k]} if fold0 else {})))
                 for k in range(len(fit_runs))]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(job, jobs))
    full = res[0]
    loo = res[1:]
    names = [p[0] for p in mod.PARAMS]
    th = full["theta"]
    at_bound = [n for (n, ini, lo, hi, lg), v in zip(mod.PARAMS, th) if v <= lo * 1.001 + 1e-12 or v >= hi * 0.999]
    n_sub = int(getattr(mod, "N_SUB", 2))
    rng = np.random.default_rng(0)
    ev = {}
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time()
        Y = core.rollout(mod, fit_runs[0].y0, U, core.theta_dict(mod, th), frozenset("AB"), n_sub=n_sub)
        ev[cat] = {"finite": bool(np.all(np.isfinite(Y))), "sec": time.time() - t0, "min": Y.min(0).tolist(), "max": Y.max(0).tolist()}
    rep = {"family": a.family, "tag": a.tag, "sigma": sig.tolist(), "theta": dict(zip(names, th)), "theta_vec": th,
           "at_bound": at_bound, "loo": [{"held_out": x["held_out"], "score": x["score"], "theta": x["theta"]} for x in loo],
           "insample": full["insample"], "exam": full["exam"], "eval": ev, "args": vars(a)}
    if loo:
        rep["loo_mean"] = float(np.mean([np.mean(x["score"]) for x in loo]))
    out = ROOT / "plans" / f"power_grid_tm_{a.tag}.json"
    out.write_text(json.dumps(rep, indent=1))
    blob = {"kind": "ode", "family": a.family, "theta": th, "mech": ["A", "B"], "n_sub": n_sub}
    doc = C.make_doc(spec, blob, runs=fit_runs, clip=C.soft_clip(spec, fit_runs, margin=1.0),
                     info={"model_id": f"ode:{a.family}", "n_runs": len(fit_runs), "cost": full["cost"]})
    (ROOT / "plans" / f"power_grid_tm_{a.tag}_doc.json").write_text(json.dumps(doc, indent=1))
    print("theta", {k: round(v, 4) for k, v in rep["theta"].items()})
    print("at bound", at_bound)
    for x in loo:
        print(f"LOO {x['held_out']:<18} {np.round(x['score'], 3)} {np.mean(x['score']):.3f}")
    if loo:
        print(f"LOO mean {rep['loo_mean']:.3f}")
    for e, v in full["insample"].items():
        print(f"in  {e:<18} {np.round(v, 3)} {np.mean(v):.3f}")
    for e, v in full["exam"].items():
        print(f"EXAM {e:<17} {np.round(v, 3)} {np.mean(v):.3f}")
    for c, v in ev.items():
        print(f"eval {c:<12} finite={v['finite']} {v['sec']:.2f}s min {np.round(v['min'], 2)} max {np.round(v['max'], 2)}")
    print(out)


if __name__ == "__main__":
    main()
