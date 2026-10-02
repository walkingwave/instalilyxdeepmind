"""Long-hold and eval-shape checks for a soc9 full fit: python scripts/lab_soc9_hold.py <tag> <family>"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_soc9_loo import OUT, setup  # noqa: E402
from gtlab import design as D  # noqa: E402
from gtlab.ode import core  # noqa: E402

tag, fam = sys.argv[1], sys.argv[2]
spec, runs, sigma = setup()
mod = importlib.import_module(f"gtlab.ode.{fam}")
th = core.theta_dict(mod, np.load(OUT / f"{tag}.npz", allow_pickle=True)["res"].item()[""]["theta"])
mech = frozenset("BC")


def roll(y0, u, T=4000):
    return core.rollout(mod, np.array(y0, float), np.tile(np.array(u, float), (T, 1)), th, mech, n_sub=2)


Y = roll([48.6, 35.9], [0, 0, 0])
print("zero from (48.6,35.9): t120", Y[119].round(0), "t4000", Y[-1].round(0))
print("zero from (190,130): t4000", roll([190, 130], [0, 0, 0])[-1].round(0))
print("incentive 0/1/2 at s5 b0.5:", [roll([48.6, 35.9], [5, c, 0.5])[-1].round(0).tolist() for c in (0, 1, 2)])
print("bridge 0/0.5/1 at s5 c1:", [roll([48.6, 35.9], [5, 1, b])[-1].round(0).tolist() for b in (0, 0.5, 1)])
print("seeding 0/1/3/5/7/10 a:", [round(float(roll([48.6, 35.9], [s, 1, 0.5])[-1, 0])) for s in (0, 1, 3, 5, 7, 10)])
Yall = np.concatenate([r.Y for r in runs]); lo, hi = Yall.min(0), Yall.max(0); rg = hi - lo
rng = np.random.default_rng(0)
for cat in ("sustained", "order", "recovery", "composition"):
    U = D.eval_like(spec, cat, 4000, rng)
    t0 = time.time(); Y = core.rollout(mod, runs[0].y0, U, th, mech, n_sub=2); dt = time.time() - t0
    out = float(np.mean((Y > hi + 0.05 * rg) | (Y < lo - 0.05 * rg)))
    print(f"eval {cat:<12} finite {np.all(np.isfinite(Y))} {dt:.2f}s outside {out:.2f}")
