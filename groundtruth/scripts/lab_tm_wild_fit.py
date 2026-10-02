"""Two-stage fit of a wildlife_tm family under one mechanism pair: prey parameters on the prey observables
(predator sigma x1e4), then predator parameters with prey held. LOO over the five runs + full fit.
Parameters of switched-off terms are held (never fitted). Saves held-out predictions for the ensemble table.

    python scripts/lab_tm_wild_fit.py <family> <tag> --mech AC [--pin H=1] [--set cJ=470,Kc=200]
"""
import argparse, importlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F

ap = argparse.ArgumentParser()
ap.add_argument("family"); ap.add_argument("tag")
ap.add_argument("--mech", default="AB")
ap.add_argument("--pin", default="", help="name=value,... held fixed")
ap.add_argument("--set", default="", help="name=value,... start values")
ap.add_argument("--b1", type=float, default=100); ap.add_argument("--b2", type=float, default=50)
ap.add_argument("--starts", type=int, default=8); ap.add_argument("--nfev", type=int, default=60)
ap.add_argument("--predw", type=float, default=1e4)
ap.add_argument("--pred-free", default="q0,q1,cq,tz,emq,tauq,svq,Kq")
ap.add_argument("--no-loo", action="store_true")
a = ap.parse_args()
spec = S.get("wildlife")
mod = importlib.import_module(f"gtlab.ode.{a.family}")
mech = frozenset(a.mech)
flags = set(getattr(mod, "FLAGS", []))
runs = load_runs(spec, Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
names = [p[0] for p in mod.PARAMS]
kv = lambda s: {k: float(v) for k, v in (x.split("=") for x in s.split(",") if x)}
pins, sets = kv(a.pin), kv(a.set)
theta0 = np.array([p[1] for p in mod.PARAMS], float)
for k, v in {**sets, **pins}.items():
    theta0[names.index(k)] = v
off = set(pins)
if "A" not in mech: off |= {"cJ"}
if "B" not in mech: off |= {"w", "kR", "bh"}
if "C" not in mech: off |= {"Kc", "Kq"}
if "cq" not in flags: off |= {"Kq"}
if "pd" not in flags: off |= {"ap"}
if "sh" not in flags: off |= {"ts", "xh", "s0"}
off &= set(names)
pred_free = [n for n in a.pred_free.split(",") if n in names and n not in off]
prey_free = [n for n in names if n not in pred_free and n not in off]
print("prey free", prey_free, "| pred free", pred_free, "| held", sorted(off), flush=True)
s1 = sigma * np.array([1, a.predw, 1, a.predw])


def fit(tr):
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=2, early_T=None, spread=0.5)
    th1, _ = F.fit_ode(mod, tr, s1, mech, time_budget=a.b1, free=prey_free, theta0=list(theta0), **kw)
    th2, _ = F.fit_ode(mod, tr, sigma, mech, time_budget=a.b2, free=pred_free, theta0=list(th1), **kw)
    return th2


out = {"family": a.family, "mech": sorted(mech), "pins": pins, "names": names, "loo": {}, "folds": {}}
preds = {}
with C.single_thread():
    if not a.no_loo:
        for k, r in enumerate(runs):
            t0 = time.time()
            th = fit([x for i, x in enumerate(runs) if i != k])
            Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=2)
            s = metric.score_per_obs(Y, r.Y, sigma)
            out["loo"][r.exp] = s.tolist(); out["folds"][r.exp] = [float(v) for v in th]; preds["loo_" + r.exp] = Y
            print(f"LOO {r.exp:<22} {np.round(s, 3)} mean {s.mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
        print(f"LOO mean {np.mean([np.mean(v) for v in out['loo'].values()]):.4f}", flush=True)
    th = fit(runs)
out["theta"] = {n: float(v) for n, v in zip(names, th)}; out["theta_vec"] = [float(v) for v in th]; out["insample"] = {}
for r in runs:
    Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=2); preds["ins_" + r.exp] = Y
    s = metric.score_per_obs(Y, r.Y, sigma); out["insample"][r.exp] = s.tolist()
    print(f"ins {r.exp:<22} {np.round(s, 3)} mean {s.mean():.3f}", flush=True)
print(f"ins mean {np.mean([np.mean(v) for v in out['insample'].values()]):.4f}", flush=True)
print("theta", {k: round(v, 5) for k, v in out["theta"].items()}, flush=True)
Path(f"plans/wildlife_tm_{a.tag}.json").write_text(json.dumps(out, indent=1))
d = Path(os.environ.get("TEMP", ".")) / "wildtm"; d.mkdir(exist_ok=True)
np.savez(d / f"{a.tag}.npz", **preds)
