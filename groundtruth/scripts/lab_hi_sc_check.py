"""hi lab: parity of the discrete family through the shared rollout, and in-sample score at theta0."""
import importlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
fam = sys.argv[1] if len(sys.argv) > 1 else "supply_chain_hi1"
mod = importlib.import_module(f"gtlab.ode.{fam}")
spec = S.get("supply_chain")
runs = load_runs(spec, Ledger(data_dir("supply_chain", False), "supply_chain", False))
sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["supply_chain"]["sigma"], float)
th = core.theta_dict(mod)
if len(sys.argv) > 2:
    d = json.loads(Path(sys.argv[2]).read_text())
    th = dict(zip([p[0] for p in mod.PARAMS], d["theta_vec"] if "theta_vec" in d else d["model"]["theta"]))
for r in runs:
    Y1 = core.rollout(mod, r.y0, r.U, th, "AB", n_sub=1)
    Y2 = mod.simulate(r.y0, r.U, th)
    s = metric.score_per_obs(Y1, r.Y, sigma)
    print(r.exp, "parity", float(np.abs(Y1 - Y2).max()), "score", np.round(s, 3), round(float(s.mean()), 4))
