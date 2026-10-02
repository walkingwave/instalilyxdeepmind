"""epi9: leave-one-run-out exactly as ode_lab (budget 300 -> 150 s per fold, 8 starts, nfev 50, sigma 1.0 cal),
saving each fold's theta and held-out prediction so predictions of several families can be combined.

    python scripts/lab_epi9_loo.py <family> <mech> <out.npz>
"""
import importlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F
fam, mech, out = sys.argv[1], frozenset(sys.argv[2]), sys.argv[3]
mod = importlib.import_module(f"gtlab.ode.{fam}")
runs = load_runs(S.get("epidemic"), Ledger(data_dir("epidemic", False), "epidemic", False))
sig = np.array(json.loads(Path("plans/sigma_calibrated.json").read_text())["epidemic"]["sigma"], float)
n_sub = int(getattr(mod, "N_SUB", 2))
res = {}
with C.single_thread():
    for k, r in enumerate(runs):
        tr = [q for q in runs if q.exp != r.exp]
        t0 = time.time()
        th, info = F.fit_ode(mod, tr, sig, mech, time_budget=150, n_starts=8, max_nfev=50, n_sub=n_sub,
                             early_T=None, spread=0.5, free=None, theta0=None)
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, sig)
        res[f"Y_{r.exp}"] = Y
        res[f"th_{r.exp}"] = np.asarray(th, float)
        print(f"  LOO {r.exp:<20} ode {np.round(s, 3)} mean {s.mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
np.savez(out, **res)
print("DONE", out)
