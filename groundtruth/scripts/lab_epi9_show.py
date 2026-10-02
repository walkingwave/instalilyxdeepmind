"""epi9: print truth vs prediction for one run and tick range."""
import importlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
p = json.loads(Path(sys.argv[1]).read_text()); exp = sys.argv[2]; a, b, st = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
mod = importlib.import_module(f"gtlab.ode.{p['family']}"); mech = frozenset(p["mech"])
runs = load_runs(S.get("epidemic"), Ledger(data_dir("epidemic", False), "epidemic", False))
r = [r for r in runs if r.exp == exp][0]
Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, np.array(p["theta_vec"])), mech, n_sub=int(getattr(mod, "N_SUB", 2)))
print("y0", r.y0)
for t in range(a, min(b, r.T), st):
    print(t, np.round(r.U[t], 4), "truth", np.round(r.Y[t], 1), "pred", np.round(Y[t], 1))
