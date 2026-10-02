import sys, json, importlib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
spec = S.get("hospital_queue")
runs = load_runs(spec, Ledger(data_dir("hospital_queue", False), "hospital_queue", False))
sig = np.array(json.load(open("plans/sigma_calibrated.json"))["hospital_queue"]["sigma"])
fam = sys.argv[1]; rep = sys.argv[2]
mod = importlib.import_module(f"gtlab.ode.{fam}")
d = json.load(open(rep))
th = core.theta_dict(mod, d["theta_vec"]) if "theta_vec" in d else core.theta_dict(mod, d["model"]["theta"])
mech = frozenset(d.get("mech", "AB"))
show = sys.argv[3] if len(sys.argv) > 3 else None
for r in runs:
    Y, X = core.rollout(mod, r.y0, r.U, th, mech, n_sub=2, return_states=True)
    s = metric.score_per_obs(Y, r.Y, sig)
    # smoothed discharges comparison
    print(f"{r.exp:<24} {np.round(s,3)} mean {s.mean():.3f}")
    if show == r.exp:
        a, b, st = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        for t in range(a, b, st):
            print(t, np.round(r.Y[t], 1), np.round(Y[t], 1), np.round(X[t], 1), np.round(r.U[t], 2))
