"""epi9: fit one LOO fold (same settings as ode_lab) and save theta; print held-out segments.

    python scripts/lab_epi9_fold.py <family> <mech> <held_out_exp> <out.json>
"""
import importlib, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F
fam, mech, ho, out = sys.argv[1], frozenset(sys.argv[2]), sys.argv[3], sys.argv[4]
mod = importlib.import_module(f"gtlab.ode.{fam}")
runs = load_runs(S.get("epidemic"), Ledger(data_dir("epidemic", False), "epidemic", False))
sig = np.array([12.953956472279227, 5.230371399915308])
n_sub = int(getattr(mod, "N_SUB", 2))
tr = [r for r in runs if r.exp != ho]
with C.single_thread():
    th, info = F.fit_ode(mod, tr, sig, mech, time_budget=150, n_starts=8, max_nfev=50, n_sub=n_sub,
                         early_T=None, spread=0.5, free=None, theta0=None)
r = [r for r in runs if r.exp == ho][0]
Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
print(ho, metric.score_per_obs(Y, r.Y, sig))
Path(out).write_text(json.dumps({"family": fam, "mech": "".join(sorted(mech)), "theta_vec": [float(v) for v in th],
                                 "theta": dict(zip([p[0] for p in mod.PARAMS], map(float, th)))}))
