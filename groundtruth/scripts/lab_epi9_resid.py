"""epi9: per-run, per-observable, per-segment score of a fitted epidemic plan on every owned run.

    python scripts/lab_epi9_resid.py plans/epidemic_epidemic_y2_AC.json [--seg]
"""
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core

SIG = np.array([12.953956472279227, 5.230371399915308])


def main():
    p = json.loads(Path(sys.argv[1]).read_text())
    mod = importlib.import_module(f"gtlab.ode.{p['family']}")
    mech = frozenset(p["mech"])
    spec = S.get("epidemic")
    runs = load_runs(spec, Ledger(data_dir("epidemic", False), "epidemic", False))
    th = core.theta_dict(mod, np.array(p["theta_vec"]))
    n_sub = int(getattr(mod, "N_SUB", 2))
    tot = []
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, th, mech, n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, SIG)
        tot.append(s.mean())
        print(f"{r.exp:<18} T={r.T} score {np.round(s, 3)} mean {s.mean():.3f}")
        if "--seg" in sys.argv:
            t = 0
            for k, g in itertools.groupby([tuple(np.round(u, 4)) for u in r.U]):
                n = len(list(g))
                sl = slice(t, t + n)
                ss = metric.score_per_obs(Y[sl], r.Y[sl], SIG)
                bias = (Y[sl] - r.Y[sl]).mean(0) / SIG
                print(f"    t{t:>4}+{n:<4} u={k} score {np.round(ss, 3)} bias/sig {np.round(bias, 2)}"
                      f" truth end {np.round(r.Y[t + n - 1], 1)} pred end {np.round(Y[t + n - 1], 1)}")
                t += n
    print(f"mean over runs {np.mean(tot):.3f}")


if __name__ == "__main__":
    main()
