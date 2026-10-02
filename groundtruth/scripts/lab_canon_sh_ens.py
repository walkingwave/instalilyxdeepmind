"""canon lab: LOO of the average (= median of two) of two lab fold fits, per fold, at 1.0 sigma.
    python scripts/lab_canon_sh_ens.py plans/<lab A>.json plans/<lab B>.json [weight_of_B]
Uses each lab json's per-fold theta (same held-out run), queue capped at 333 for hospital_queue."""
import sys, json, importlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, metric
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core

pa, pb = sys.argv[1], sys.argv[2]
w = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
A, B = json.load(open(pa)), json.load(open(pb))
system = A["system"]
spec = S.get(system)
runs = {r.exp: r for r in load_runs(spec, Ledger(data_dir(system, False), system, False))}
sig = np.asarray(json.load(open(ROOT / "plans/sigma_calibrated.json"))[system]["sigma"], float)
cap = np.array([1e9, 333.0, 1e9]) if system == "hospital_queue" else np.full(3, 1e9)
ma, mb = importlib.import_module(f"gtlab.ode.{A['family']}"), importlib.import_module(f"gtlab.ode.{B['family']}")
fa = {x["held_out"]: x["theta"] for x in A["loo"]}
fb = {x["held_out"]: x["theta"] for x in B["loo"]}
rows = []
for ex, r in runs.items():
    if ex not in fa or ex not in fb:
        continue
    Ya = np.minimum(core.rollout(ma, r.y0, r.U, core.theta_dict(ma, fa[ex]), "AB", n_sub=2), cap)
    Yb = np.minimum(core.rollout(mb, r.y0, r.U, core.theta_dict(mb, fb[ex]), "AB", n_sub=2), cap)
    s = [metric.score_per_obs(Y, r.Y, sig) for Y in (Ya, Yb, (1 - w) * Ya + w * Yb)]
    rows.append([x.mean() for x in s])
    print(f"{ex:<26} A {s[0].mean():.3f} B {s[1].mean():.3f} mix {np.round(s[2], 3)} {s[2].mean():.3f}")
rows = np.array(rows)
print(f"LOO mean  A {rows[:, 0].mean():.4f}  B {rows[:, 1].mean():.4f}  mix(w={w}) {rows[:, 2].mean():.4f}")
