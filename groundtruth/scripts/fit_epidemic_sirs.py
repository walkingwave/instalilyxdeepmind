"""Fit the minimal SIRS+H epidemic family on the real runs; in-sample and leave-one-run-out.

    python scripts/fit_epidemic_sirs.py [--budget 240] [--starts 12] [--out plans/epidemic_sirs.json]
"""
import argparse
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
from gtlab.ode import epidemic_sirs as mod

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def score_runs(theta, runs, sigma):
    out = []
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, theta), frozenset("AB"), n_sub=mod.N_SUB)
        out.append(metric.score_per_obs(Y, r.Y, sigma))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=240)
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--nfev", type=int, default=80)
    ap.add_argument("--out", default="plans/epidemic_sirs.json")
    a = ap.parse_args()
    spec = S.get("epidemic")
    runs = load_runs(spec, Ledger(data_dir("epidemic", False), "epidemic", False))
    sigma = metric.sigma_proxy(runs)
    print("runs", [(r.exp, r.T) for r in runs], "sigma", sigma)
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=mod.N_SUB, early_T=None, spread=0.5)
    # leave-one-run-out
    with C.single_thread():
        for k in range(len(runs)):
            tr = [r for i, r in enumerate(runs) if i != k]
            t0 = time.time()
            th, info = F.fit_ode(mod, tr, sigma, frozenset("AB"), time_budget=a.budget / 2, **kw)
            s_in = score_runs(th, tr, sigma)
            s_out = score_runs(th, [runs[k]], sigma)[0]
            pers = metric.score_per_obs(np.tile(runs[k].y0, (runs[k].T, 1)), runs[k].Y, sigma)
            print(f"LOO hold-out {runs[k].exp}: in-sample {[np.round(s, 3) for s in s_in]} held-out {np.round(s_out, 3)} "
                  f"(persistence {np.round(pers, 3)}) cost {info['cost']:.1f} [{time.time() - t0:.0f}s]")
        # full fit
        t0 = time.time()
        th, info = F.fit_ode(mod, runs, sigma, frozenset("AB"), time_budget=a.budget, **kw)
    print("full fit cost", round(info["cost"], 1), f"[{time.time() - t0:.0f}s]")
    for r, s in zip(runs, score_runs(th, runs, sigma)):
        print(f"  {r.exp}: {np.round(s, 3)}")
    names = [p[0] for p in mod.PARAMS]
    print({n: round(float(v), 5) for n, v in zip(names, th)})
    # 4000-tick sanity on eval-like schedules
    from gtlab import design as D
    rng = np.random.default_rng(0)
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        Y = core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th), frozenset("AB"), n_sub=mod.N_SUB)
        print(f"  eval[{cat}] finite={np.all(np.isfinite(Y))} min={np.round(Y.min(0), 1)} max={np.round(Y.max(0), 1)} end={np.round(Y[-1], 1)}")
    Path(a.out).write_text(json.dumps({"family": mod.FAMILY, "theta": [float(v) for v in th], "mech": ["A", "B"],
                                       "n_sub": mod.N_SUB, "param_names": names, "cost": float(info["cost"])}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
