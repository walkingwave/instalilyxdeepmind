"""reservoir hi: long-hold quality/level asymptotes (t = 700, 1500, 3999) of fitted thetas under
constant control corners, from the p8 reset state. Free.

    python scripts/lab_hi_res_asym.py <family>:<fold json or doc> ...
"""
import importlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core

CORNERS = {"recovery": [2, 0, 0, 1], "zero": [0, 0, 0, 0], "pulse": [12, 8, 1, 0], "p8pulse": [12, 0, 1, 0],
           "rel12_aer": [12, 0, 0, 1], "deep_aer": [2, 0, 1, 1]}


def main():
    spec = S.get("reservoir")
    runs = load_runs(spec, Ledger(data_dir("reservoir", False), "reservoir", False))
    y0 = [r for r in runs if r.exp.startswith("p8")][0].y0
    print(f"{'model':22s} " + " ".join(f"{c:>22s}" for c in CORNERS))
    for arg in sys.argv[1:]:
        fam, path = arg.split(":", 1)
        mod = importlib.import_module(f"gtlab.ode.{fam}")
        d = json.loads(Path(path).read_text())
        th = d.get("theta_vec") or d["model"]["theta"]
        cells = []
        for c, u in CORNERS.items():
            Y = core.rollout(mod, y0, np.tile(np.array(u, float), (4000, 1)), core.theta_dict(mod, th), frozenset("AB"), n_sub=2)
            cells.append(f"q {Y[699, 3]:.4f}/{Y[1499, 3]:.4f}/{Y[3999, 3]:.4f} L{Y[3999, 0]:4.0f}")
        print(f"{Path(path).stem[-22:]:22s} " + " ".join(f"{x:>22s}" for x in cells))


if __name__ == "__main__":
    main()
