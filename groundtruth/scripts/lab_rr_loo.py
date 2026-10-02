"""Reset-rule test: leave-one-run-out at the calibrated organizer scale (1.0 sigma) on every run, comparing
the shipped family (final3) against copies whose x0 follows the organizer initialization document.

    python scripts/lab_rr_loo.py --system supply_chain --base supply_chain_v8b --variants supply_chain_v8b_rr
        [--budget 60] [--nfev 40]
Every fold is a warm-started polish from the shipped theta (extra parameters of a copy start at their
PARAMS defaults), identical start and budget for every family, so differences are the effect of the reset
rule. Also scores the shipped theta without refit ("noref") on every run. Free, no gateway calls.
Writes plans/rr_loo_<system>_<variant>.json.
"""
import argparse, importlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
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
    ap.add_argument("--variants", required=True)
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--nfev", type=int, default=40)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pin", default="", help="comma list of parameters held at their start value in every fold")
    a = ap.parse_args()
    spec = S.get(a.system)
    mem = find_member(json.loads((SUB / a.system / "model.json").read_text())["model"], a.base)
    assert mem is not None, "family not in the shipped model"
    mech = frozenset(mem["mech"])
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"], float)
    fams = ([] if a.no_base else [a.base]) + a.variants.split(",")
    for fam in fams:
        mod = importlib.import_module(f"gtlab.ode.{fam}")
        n_sub = int(mem.get("n_sub", getattr(mod, "N_SUB", 2)))
        names = [p[0] for p in mod.PARAMS]
        th0 = list(mem["theta"])[:len(names)] + [p[1] for p in mod.PARAMS[len(mem["theta"]):]]
        rep = {"system": a.system, "family": fam, "base": a.base, "mech": sorted(mech),
               "runs": [(r.exp, r.T) for r in runs], "budget": a.budget, "nfev": a.nfev, "noref": {}, "folds": []}
        for r in runs:
            Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th0), mech, n_sub=n_sub)
            s = metric.score_per_obs(Y, r.Y, sigma)
            rep["noref"][r.exp] = s.tolist()
            print(f"  [{fam}] noref {r.exp:<24} {np.round(s, 3)} mean {s.mean():.4f}", flush=True)
        with C.single_thread():
            for k, rk in enumerate(runs):
                tr = [r for i, r in enumerate(runs) if i != k]
                t0 = time.time()
                th, info = F.fit_ode(mod, tr, sigma, mech, n_starts=1, max_nfev=a.nfev, n_sub=n_sub, early_T=None,
                                     n_polish=1, time_budget=a.budget, theta0=th0,
                                     free=[n for n in names if n not in a.pin.split(",")] if a.pin else None)
                Y = core.rollout(mod, rk.y0, rk.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
                s = metric.score_per_obs(Y, rk.Y, sigma)
                rep["folds"].append({"held_out": rk.exp, "score": s.tolist(), "mean": float(s.mean()),
                                     "train_cost": float(info["cost"]), "theta": [float(x) for x in th]})
                print(f"  [{fam}] LOO {rk.exp:<24} {np.round(s, 3)} mean {s.mean():.4f} cost {info['cost']:.1f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        rep["loo_mean"] = float(np.mean([f["mean"] for f in rep["folds"]]))
        print(f"  [{fam}] LOO mean {rep['loo_mean']:.4f}", flush=True)
        (ROOT / "plans" / f"rr_loo_{a.system}_{fam}{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
