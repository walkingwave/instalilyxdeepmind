"""social_contagion soc9 lab: leave-one-run-out with saved fold thetas and held-out predictions.

    python scripts/lab_soc9_loo.py --family social_contagion_z20 --mech BC --tag z20 [--budget 300] [--starts 8]
        [--nfev 60] [--drop2]

Same fit settings as ode_lab (calibrated sigma 1.0, early_T None, spread 0.5). Writes
<scratch>/soc9/<tag>.npz: fold thetas, per-fold predictions of EVERY run (so fold ensembles can be scored
offline). --drop2 also fits every two-run-out subset (for nested bagging).
"""
import argparse
import importlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from gtlab import metric, systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402

OUT = Path(os.environ.get("SOC9_OUT", r"C:\Users\DYLANH~1\AppData\Local\Temp\gtscratch\soc9"))
SYSTEM = "social_contagion"


def setup():
    spec = S.get(SYSTEM)
    runs = load_runs(spec, Ledger(data_dir(SYSTEM, False), SYSTEM, False))
    cal = json.loads(Path("plans/sigma_calibrated.json").read_text())[SYSTEM]
    return spec, runs, np.asarray(cal["sigma"], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--mech", default="BC")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--drop2", action="store_true")
    ap.add_argument("--nofull", action="store_true")
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--only", default=None, help="semicolon list of subset keys, e.g. 3,4;3,5")
    a = ap.parse_args()
    spec, runs, sigma = setup()
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    n_sub = int(getattr(mod, "N_SUB", 2))
    theta0 = None
    if a.theta0:
        theta0 = json.loads(Path(a.theta0).read_text())["theta_vec"]
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=0.5, theta0=theta0)
    OUT.mkdir(parents=True, exist_ok=True)
    K = len(runs)
    subsets = [(k,) for k in range(K)]
    if a.drop2:
        subsets += list(itertools.combinations(range(K), 2))
    if not a.nofull:
        subsets = [()] + subsets
    if a.only:
        keep = set(a.only.split(";"))
        subsets = [t for t in subsets if ",".join(map(str, t)) in keep]
    res = {}
    path = OUT / f"{a.tag}.npz"
    if path.exists():
        old = np.load(path, allow_pickle=True)["res"].item()
        res.update(old)
    with C.single_thread():
        for sub in subsets:
            key = ",".join(map(str, sub))
            if key in res:
                continue
            tr = [r for i, r in enumerate(runs) if i not in sub]
            t0 = time.time()
            bud = a.budget if not sub else a.budget / 2
            th, info = F.fit_ode(mod, tr, sigma, mech, time_budget=bud, **kw)
            thd = core.theta_dict(mod, th)
            preds = [core.rollout(mod, r.y0, r.U, thd, mech, n_sub=n_sub) for r in runs]
            res[key] = {"theta": np.asarray(th, float), "preds": preds, "cost": float(info["cost"])}
            msg = ""
            if len(sub) == 1:
                k = sub[0]
                s = metric.score_per_obs(preds[k], runs[k].Y, sigma)
                msg = f"held {runs[k].exp:<18} {np.round(s, 3)} mean {s.mean():.3f}"
            print(f"[{a.tag}] sub={key or 'full'} cost {info['cost']:.1f} {msg} [{time.time() - t0:.0f}s]", flush=True)
            np.savez(path, res=np.array(res, dtype=object), exps=np.array([r.exp for r in runs]),
                     family=a.family, mech=a.mech)
    if any(str(k) not in res for k in range(K)):
        return
    loo = [metric.score_per_obs(res[str(k)]["preds"][k], runs[k].Y, sigma) for k in range(K)]
    print(f"[{a.tag}] LOO {np.mean(loo):.4f} folds " + " ".join(f"{runs[k].exp[:2]}:{loo[k].mean():.3f}" for k in range(K)))


if __name__ == "__main__":
    main()
