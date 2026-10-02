"""Profile of the traffic tm1 buffer terms: hold (kx3, kjo, k_j) at fixed values, refit the rest on all
runs starting from the z8 baseline full fit, report in-sample cost and per-run score.

    python scripts/lab_tm_traffic_profile.py [--nfev 30]
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
import gtlab.ode.traffic_tm1 as mod

GRID = [(0.0, 0.0, 0.0), (0.003, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.003, 0.0), (0.003, 0.003, 0.0),
        (0.003, 0.003, 0.3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfev", type=int, default=30)
    a = ap.parse_args()
    spec = S.get("traffic")
    runs = load_runs(spec, Ledger(data_dir("traffic", False), "traffic", False))
    sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["traffic"]["sigma"], float)
    base = json.loads(Path("plans/traffic_tm_z8base.json").read_text())["theta"]
    names = [p[0] for p in mod.PARAMS]
    init = {p[0]: p[1] for p in mod.PARAMS}
    init.update(base)
    free = mod.free_for("AB")
    out = []
    with C.single_thread():
        for kx3, kjo, kj in GRID:
            t0 = time.time()
            th0 = dict(init, kx3=kx3, kjo=kjo, k_j=kj)
            th, info = F.fit_ode(mod, runs, sigma, frozenset("AB"), n_starts=1, max_nfev=a.nfev, early_T=None,
                                 free=free, theta0=[th0[n] for n in names], time_budget=None,
                                 n_sub=int(mod.N_SUB))
            sc = {}
            for r in runs:
                Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset("AB"), n_sub=int(mod.N_SUB))
                sc[r.exp] = float(metric.score_per_obs(Y, r.Y, sigma).mean())
            row = {"kx3": kx3, "kjo": kjo, "k_j": kj, "cost": float(info["cost"]), "insample": sc,
                   "mean": float(np.mean(list(sc.values())))}
            out.append(row)
            print(f"kx3 {kx3} kjo {kjo} k_j {kj}: cost {info['cost']:.1f} mean {row['mean']:.4f} "
                  + " ".join(f"{k.split('.')[1]} {v:.3f}" for k, v in sc.items()) + f" [{time.time() - t0:.0f}s]",
                  flush=True)
    Path("plans/traffic_tm_profile.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
