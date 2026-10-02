import importlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core

spec = S.get("wildlife")
runs = load_runs(spec, Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
fam = sys.argv[1] if len(sys.argv) > 1 else "wildlife_v8h"
plan = sys.argv[2] if len(sys.argv) > 2 else "plans/wildlife_wildlife_v8h_p7.json"
mech = sys.argv[3] if len(sys.argv) > 3 else "AB"
mod = importlib.import_module(f"gtlab.ode.{fam}")
th = json.loads(Path(plan).read_text())["theta_vec"]
tot = sum(r.T for r in runs) * 4
for r in runs:
    Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset(mech), n_sub=getattr(mod, "N_SUB", 2))
    sc = 1 / (1 + np.abs(Y - r.Y) / sigma)
    print(f"== {r.exp} T={r.T} score {np.round(sc.mean(0),3)} mean {sc.mean():.3f}")
    # segments of constant controls
    U = r.U
    cuts = [0] + [t for t in range(1, r.T) if np.any(np.abs(U[t] - U[t-1]) > 1e-9)] + [r.T]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3: continue
        ls = (1 - sc[a:b]).sum(0) / tot
        print(f"  [{a:4d},{b:4d}) u={np.round(U[a],2)} loss {ls.sum():.4f} {np.round(ls,4)} | end data {np.round(r.Y[b-1],2)} model {np.round(Y[b-1],2)}")
