"""ens9: fit-seed stability check of the LOO member refits (no credits).

    python scripts/lab_ens9_seed.py --jobs "supply_chain:v8b,v8c;social_contagion:z20,z21" [--seed 1]

Same protocol as lab_ens9_loo.py with another multistart seed; cached as <label>__<held>__s<seed>.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import metric  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402
import lab_ens9_loo as L  # noqa: E402


def job(system, label, held, seed, budget):
    spec, runs = L.get_runs(system)
    sigma = L.cal_sigma(system)
    tr = [r for r in runs if r.exp != held]
    te = next(r for r in runs if r.exp == held)
    blob = L.member_blob(system, label)
    mod = importlib.import_module(f"gtlab.ode.{blob['family']}")
    n_sub = int(blob.get("n_sub", 2))
    t0 = time.time()
    with C.single_thread():
        th, info = F.fit_ode(mod, tr, sigma, frozenset(blob["mech"]), n_starts=6, max_nfev=50, n_sub=n_sub, early_T=None,
                             time_budget=budget, theta0=None, spread=0.5, seed=seed)
        Y = np.asarray(core.rollout(mod, te.y0, te.U, core.theta_dict(mod, th), frozenset(blob["mech"]), n_sub=n_sub))
    return {"Y": Y.tolist(), "score": metric.score_per_obs(Y, te.Y, sigma).tolist(), "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for part in a.jobs.split(";"):
        s, labs = part.split(":")
        _, runs = L.get_runs(s)
        for l in labs.split(","):
            jobs += [(s, l, r.exp) for r in runs]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, s, l, h, a.seed, a.budget): (s, l, h) for s, l, h in jobs}
        for fu in as_completed(futs):
            s, l, h = futs[fu]
            pay = fu.result()
            (L.SCRATCH / s / f"{l}__{h}__s{a.seed}.json").write_text(json.dumps(pay))
            base = json.loads((L.SCRATCH / s / f"{l}__{h}.json").read_text())
            res.setdefault((s, l), []).append((h, np.mean(base["score"]), np.mean(pay["score"])))
    for (s, l), rows in res.items():
        rows.sort()
        print(f"{s} {l}: seed0 {np.mean([r[1] for r in rows]):.4f} seed{a.seed} {np.mean([r[2] for r in rows]):.4f}  "
              + " ".join(f"{h}:{x:.3f}/{y:.3f}" for h, x, y in rows))


if __name__ == "__main__":
    main()
