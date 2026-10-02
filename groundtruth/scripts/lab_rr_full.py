"""Full-data warm polish of a reset-rule copy from the shipped final3 theta, plus in-sample scores (1.0 sigma)
and a 4000-tick eval-shaped sanity check. Writes plans/rr_fit_<system>_<family>.json: the shipped final3
model.json for that system with the member family/theta replaced (drop-in model file). Free, no gateway.

    python scripts/lab_rr_full.py --system reservoir --base reservoir_str9 --family reservoir_str9_rr [--budget 120]
"""
import argparse, copy, importlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F

SUB = ROOT / "submissions/20260928-2230-final3"


def find_member(model, family):
    if model.get("kind") == "ode" and model.get("family") == family:
        return model
    for m in model.get("members", []) or []:
        r = find_member(m, family)
        if r is not None:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=120)
    ap.add_argument("--nfev", type=int, default=60)
    a = ap.parse_args()
    spec = S.get(a.system)
    doc = json.loads((SUB / a.system / "model.json").read_text())
    mem = find_member(doc["model"], a.base)
    mech = frozenset(mem["mech"])
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    n_sub = int(mem.get("n_sub", getattr(mod, "N_SUB", 2)))
    names = [p[0] for p in mod.PARAMS]
    th0 = list(mem["theta"])[:len(names)] + [p[1] for p in mod.PARAMS[len(mem["theta"]):]]
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"], float)
    with C.single_thread():
        th, info = F.fit_ode(mod, runs, sigma, mech, n_starts=1, max_nfev=a.nfev, n_sub=n_sub, early_T=None,
                             n_polish=1, time_budget=a.budget, theta0=th0)
    ins = {}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, sigma)
        ins[r.exp] = s.tolist()
        print(f"  in {r.exp:<24} {np.round(s, 3)} mean {s.mean():.4f}")
    rng = np.random.default_rng(0)
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time()
        Y = core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        print(f"  eval[{cat}] finite={bool(np.all(np.isfinite(Y)))} {time.time() - t0:.2f}s "
              f"min={np.round(Y.min(0), 3)} max={np.round(Y.max(0), 3)}")
    new = copy.deepcopy(doc)
    m2 = find_member(new["model"], a.base)
    m2["family"] = a.family
    m2["theta"] = [float(v) for v in th]
    out = ROOT / "plans" / f"rr_fit_{a.system}_{a.family}.json"
    out.write_text(json.dumps({"model_json": new, "insample": ins, "theta": dict(zip(names, map(float, th))),
                               "train_cost": float(info["cost"])}, indent=1))
    print("wrote", out, {n: round(float(v), 5) for n, v in zip(names, th)})


if __name__ == "__main__":
    main()
