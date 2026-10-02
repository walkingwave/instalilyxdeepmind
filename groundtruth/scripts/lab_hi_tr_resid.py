"""Traffic residuals of the z8 LOO fold fits on their held-out runs, per observable and segment.

    python scripts/lab_hi_tr_resid.py [--rep plans/traffic_tm_z8base.json] [--family traffic_z8]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core


def segments(U):
    ch = [0] + [t for t in range(1, len(U)) if np.any(np.abs(U[t] - U[t - 1]) > 1e-9)] + [len(U)]
    return list(zip(ch[:-1], ch[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="plans/traffic_tm_z8base.json")
    ap.add_argument("--family", default="traffic_z8")
    ap.add_argument("--runs", default="p2.pulse40,p2.multilevel200,p5.testlike,p7.longhold")
    a = ap.parse_args()
    spec = S.get("traffic")
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    rep = json.loads(Path(a.rep).read_text())
    mech = frozenset(rep["mech"])
    sigma = np.asarray(rep.get("sigma") or json.loads(Path("plans/sigma_calibrated.json").read_text())["traffic"]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    runs = load_runs(spec, Ledger(data_dir("traffic", False), "traffic", False))
    for r in runs:
        if r.exp not in a.runs.split(","):
            continue
        th = rep["loo"][r.exp]["theta"]
        Y = core.rollout(mod, r.y0, r.U, {n: th[n] for n in names}, mech, n_sub=int(getattr(mod, "N_SUB", 2)))
        s = metric.score_per_obs(Y, r.Y, sigma)
        print(f"\n=== {r.exp} T={r.T} score {np.round(s, 3)} mean {s.mean():.3f}")
        for t0, t1 in segments(r.U):
            u = r.U[t0]
            e = (Y[t0:t1] - r.Y[t0:t1]) / sigma
            sc = 1 / (1 + np.abs(e))
            k = min(t1, t0 + 10)
            print(f"[{t0:4d},{t1:4d}) u={np.round(u, 2)}")
            print(f"   truth mean {np.round(r.Y[t0:t1].mean(0), 1)} end {np.round(r.Y[t1 - 1], 1)}")
            print(f"   model mean {np.round(Y[t0:t1].mean(0), 1)} end {np.round(Y[t1 - 1], 1)}")
            print(f"   score {np.round(sc.mean(0), 2)}  first10 {np.round(sc[:k - t0].mean(0), 2)}  bias/sig {np.round(e.mean(0), 2)}")


if __name__ == "__main__":
    main()
