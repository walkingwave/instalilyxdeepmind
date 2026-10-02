"""Fold-wise ensemble check for mechanism-pair fits: for every held-out run, average (mean or median)
the predictions of several pair fits made without that run (fold thetas saved by lab_pair_loo.py).

    python scripts/lab_pair_ens.py --system ad_auction --tags abc_ref,bc [--how mean|median]
"""
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--how", default="mean")
    a = ap.parse_args()
    spec = S.get(a.system)
    runs = {r.exp: r for r in load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))}
    reps = [json.loads((ROOT / "plans" / f"{a.system}_pair_{t}.json").read_text()) for t in a.tags.split(",")]
    sigma = np.asarray(reps[0]["sigma"])
    tot = []
    for k, fold in enumerate(reps[0]["loo"]):
        r = runs[fold["held_out"]]
        Ys = []
        for rep in reps:
            mod = importlib.import_module(f"gtlab.ode.{rep['family']}")
            th = core.theta_dict(mod, rep["loo"][k]["theta"])
            Ys.append(core.rollout(mod, r.y0, r.U, th, frozenset(rep["mech"]), n_sub=int(getattr(mod, "N_SUB", 2))))
        Y = np.median(Ys, 0) if a.how == "median" else np.mean(Ys, 0)
        s = metric.score_per_obs(Y, r.Y, sigma)
        tot.append(s.mean())
        print(f"{r.exp:<22} {np.round(s, 3)} mean {s.mean():.4f}")
    print(f"LOO mean {np.mean(tot):.4f}")


if __name__ == "__main__":
    main()
