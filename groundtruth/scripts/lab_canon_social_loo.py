"""social_contagion canon lab: leave-one-run-out for the canon families, same protocol as ens9 (no credits).

    python scripts/lab_canon_social_loo.py --families canon1,canon2 [--mech BC] [--workers 6] [--budget 60]
        [--starts 6] [--nfev 50] [--seed 0] [--full] [--tag x]

Each fold: refit cold from the family default init (LHS multistart, spread 0.5, no early phase) on all runs but
one at the calibrated sigma (1.0x), predict the held-out run. Predictions cached as JSON in
CANON_SCRATCH/<family><tag>__<held>.json (same layout as the ens9 cache). --full also fits on all six runs and
caches <family><tag>__FULL.json with theta. Then prints the fold table against the current pick
med(z20, z21) from the ens9 cache, plus median combinations with the pick.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402

SYSTEM = "social_contagion"
SCRATCH = Path(os.environ.get("CANON_SCRATCH", "C:/Users/DYLANH~1/AppData/Local/Temp/gtscratch/canon_social"))
ENS9 = Path(os.environ.get("ENS9_SCRATCH", "C:/Users/DYLANH~1/AppData/Local/Temp/gtscratch/ens9")) / SYSTEM


def setup():
    spec = S.get(SYSTEM)
    runs = load_runs(spec, Ledger(data_dir(SYSTEM, False), SYSTEM, False))
    cal = json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[SYSTEM]
    return spec, runs, np.asarray(cal["sigma"], float)


def _job(fam, mech, held, budget, starts, nfev, seed):
    _, runs, sigma = setup()
    mod = importlib.import_module(f"gtlab.ode.social_contagion_{fam}")
    tr = [r for r in runs if r.exp != held]
    t0 = time.time()
    with C.single_thread():
        th, info = F.fit_ode(mod, tr, sigma, frozenset(mech), n_starts=starts, max_nfev=nfev, n_sub=2, early_T=None,
                             time_budget=budget, theta0=None, spread=0.5, seed=seed)
        thd = core.theta_dict(mod, th)
        preds = {r.exp: core.rollout(mod, r.y0, r.U, thd, frozenset(mech), n_sub=2).tolist() for r in runs}
    pay = {"theta": [float(v) for v in th], "cost": float(info["cost"]), "train": [r.exp for r in tr],
           "sec": time.time() - t0, "mech": mech}
    if held != "FULL":
        te = next(r for r in runs if r.exp == held)
        pay["Y"] = preds[held]
        pay["score"] = metric.score_per_obs(np.asarray(pay["Y"]), te.Y, sigma).tolist()
    else:
        pay["preds"] = preds
        pay["insample"] = {r.exp: metric.score_per_obs(np.asarray(preds[r.exp]), r.Y, sigma).tolist() for r in runs}
    return pay


def load_pred(label, exp):
    for d in (SCRATCH, ENS9):
        f = d / f"{label}__{exp}.json"
        if f.exists():
            return np.asarray(json.loads(f.read_text())["Y"], float)
    return None


def table(labels, runs, sigma):
    cols = {}
    for lab in labels:
        parts = lab.split("+")
        rows = []
        for r in runs:
            P = [load_pred(p, r.exp) for p in parts]
            if any(p is None for p in P):
                rows = None
                break
            Y = np.median(np.stack(P), axis=0)
            rows.append(metric.score_per_obs(Y, r.Y, sigma))
        if rows is not None:
            cols[lab] = rows
    print(f"{'fold':<20}" + "".join(f"{k:>22}" for k in cols))
    for i, r in enumerate(runs):
        print(f"{r.exp:<20}" + "".join(f"{v[i].mean():>8.3f} ({v[i][0]:.2f}/{v[i][1]:.2f})" for v in cols.values()))
    print(f"{'mean':<20}" + "".join(f"{np.mean([x.mean() for x in v]):>22.4f}" for v in cols.values()))
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--mech", default="BC")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--nfev", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--table", default="", help="extra comma list of labels (a+b = median) to tabulate")
    a = ap.parse_args()
    _, runs, sigma = setup()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fams = [f for f in a.families.split(",") if f]
    jobs = []
    for fam in fams:
        helds = [r.exp for r in runs] + (["FULL"] if a.full else [])
        for h in helds:
            out = SCRATCH / f"{fam}{a.tag}__{h}.json"
            if not out.exists():
                jobs.append((fam, h, out))
    if jobs:
        print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_job, fam, a.mech, h, a.budget * (2 if h == "FULL" else 1), a.starts, a.nfev, a.seed):
                    (fam, h, out) for fam, h, out in jobs}
            for n, fu in enumerate(as_completed(futs)):
                fam, h, out = futs[fu]
                try:
                    pay = fu.result()
                except Exception as e:  # noqa: BLE001
                    print(f"ERR {fam} {h} {e!r}", flush=True)
                    continue
                out.write_text(json.dumps(pay))
                msg = f"{np.mean(pay['score']):.3f} {np.round(pay['score'], 3)}" if "score" in pay else "full"
                print(f"[{n + 1}/{len(jobs)} {time.time() - t0:.0f}s] {fam}{a.tag} held {h} {msg} cost {pay['cost']:.1f}",
                      flush=True)
    labels = ["z20", "z21", "z20+z21"] + [f + a.tag for f in fams]
    labels += [f"{f}{a.tag}+z20+z21" for f in fams]
    labels += [x for x in a.table.split(",") if x]
    table(labels, runs, sigma)


if __name__ == "__main__":
    main()
