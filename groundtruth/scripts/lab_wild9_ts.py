import importlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
spec = S.get("wildlife")
runs = {r.exp: r for r in load_runs(spec, Ledger(data_dir("wildlife", False), "wildlife", False))}
fam, plan, exp, a, b, st = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
mod = importlib.import_module(f"gtlab.ode.{fam}")
th = json.loads(Path(plan).read_text())["theta_vec"]
r = runs[exp]
Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset("AB"), n_sub=getattr(mod, "N_SUB", 2))
print("y0", np.round(r.y0, 2))
for t in range(a, b, st):
    print(t, np.round(r.U[t], 2), "data", np.round(r.Y[t], 2), "model", np.round(Y[t], 2))
