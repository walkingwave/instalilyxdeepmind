"""Round-constant snap test: leave-one-run-out at the calibrated organizer scale (1.0 sigma) on every
run, comparing the shipped family refit with all parameters free ("base") against refits with the
flagged parameters pinned at their round values ("snap3" = every flag within 3%, "snap1" = within 1%).

    python scripts/lab_snap_loo.py --system power_grid --family power_grid_w5 [--budget 90] [--nfev 40]
        [--variants base,snap3,snap1] [--extra name=value,...]
Reads the shipped theta from submissions/20260928-1810-final2/<system>/model.json and the flags from
plans/snap_scan.json (scripts/lab_snap_scan.py). Every fold is a warm-started polish from the shipped
theta (identical start and budget for every variant), so differences are the effect of the pin.
Writes plans/snap_loo_<system>_<family>.json. Free, no gateway calls.
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

SUB = ROOT / "submissions/20260928-1810-final2"


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
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=90)
    ap.add_argument("--nfev", type=int, default=40)
    ap.add_argument("--variants", default="base,snap3,snap1")
    ap.add_argument("--extra", default="", help="extra manual pins name=value, applied in a variant 'manual'")
    ap.add_argument("--maxrel", type=float, default=0.03)
    ap.add_argument("--out-tag", dest="out_tag", default="", help="suffix for the output file (e.g. _ctl)")
    a = ap.parse_args()
    spec = S.get(a.system)
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mem = find_member(json.loads((SUB / a.system / "model.json").read_text())["model"], a.family)
    assert mem is not None, "family not in the shipped model"
    mech = frozenset(mem["mech"])
    n_sub = int(mem.get("n_sub", getattr(mod, "N_SUB", 2)))
    names = [p[0] for p in mod.PARAMS]
    th_ship = list(mem["theta"]) + [p[1] for p in mod.PARAMS[len(mem["theta"]):]]
    scan = json.loads((ROOT / "plans/snap_scan.json").read_text())["systems"][a.system]
    rows = next(x["params"] for x in scan if x["family"] == a.family and
                np.allclose([r["value"] for r in x["params"]], mem["theta"][:len(x["params"])]))
    flags = [r for r in rows if r["flag"]]
    pinsets = {"base": {},
               "snap3": {r["name"]: r["snap_value"] for r in flags if r["rel"] <= a.maxrel},
               "snap1": {r["name"]: r["snap_value"] for r in flags if r["rel"] <= 0.01}}
    # control: same parameters pinned the same distance from the fit on the other side (not round)
    pinsets["mirror3"] = {r["name"]: min(max(2 * r["value"] - r["snap_value"], r["lo"]), r["hi"])
                          for r in flags if r["rel"] <= a.maxrel}
    pinsets["mirror1"] = {r["name"]: min(max(2 * r["value"] - r["snap_value"], r["lo"]), r["hi"])
                          for r in flags if r["rel"] <= 0.01}
    if a.extra:
        pinsets["manual"] = {k: float(v) for k, v in (kv.split("=") for kv in a.extra.split(","))}
    variants = [v for v in a.variants.split(",") if v in pinsets]
    if "manual" in pinsets and "manual" not in variants:
        variants.append("manual")
    # skip duplicate pin sets
    seen, vv = [], []
    for v in variants:
        key = tuple(sorted(pinsets[v].items()))
        if key in seen:
            continue
        seen.append(key); vv.append(v)
    variants = vv
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"], float)
    rep = {"system": a.system, "family": a.family, "mech": sorted(mech), "runs": [(r.exp, r.T) for r in runs],
           "sigma": sigma.tolist(), "pins": {v: pinsets[v] for v in variants}, "budget": a.budget, "nfev": a.nfev,
           "shipped_insample": {}, "variants": {}}
    print(f"{a.system}/{a.family} mech {''.join(sorted(mech))} runs {rep['runs']}", flush=True)
    for v in variants:
        print(f"  pins[{v}] {pinsets[v]}", flush=True)
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th_ship), mech, n_sub=n_sub)
        rep["shipped_insample"][r.exp] = metric.score_per_obs(Y, r.Y, sigma).tolist()
    with C.single_thread():
        for v in variants:
            pins = pinsets[v]
            th0 = [pins.get(n, x) for n, x in zip(names, th_ship)]
            free = [n for n in names if n not in pins]
            folds = []
            for k, rk in enumerate(runs):
                tr = [r for i, r in enumerate(runs) if i != k]
                t0 = time.time()
                th, info = F.fit_ode(mod, tr, sigma, mech, n_starts=1, max_nfev=a.nfev, n_sub=n_sub, early_T=None,
                                     n_polish=1, time_budget=a.budget, theta0=th0, free=free)
                Y = core.rollout(mod, rk.y0, rk.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
                s = metric.score_per_obs(Y, rk.Y, sigma)
                folds.append({"held_out": rk.exp, "score": s.tolist(), "mean": float(s.mean()),
                              "train_cost": float(info["cost"]), "theta": [float(x) for x in th]})
                print(f"  [{v}] LOO {rk.exp:<24} {np.round(s, 3)} mean {s.mean():.4f} cost {info['cost']:.1f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            m = float(np.mean([f["mean"] for f in folds]))
            rep["variants"][v] = {"folds": folds, "loo_mean": m}
            print(f"  [{v}] LOO mean {m:.4f}", flush=True)
            (ROOT / "plans" / f"snap_loo_{a.system}_{a.family}{a.out_tag}.json").write_text(json.dumps(rep, indent=1))
    # fold spread of each parameter under base (identifiability), relative to its value
    if "base" in rep["variants"]:
        TH = np.array([f["theta"] for f in rep["variants"]["base"]["folds"]])
        rep["base_fold_spread"] = {n: {"min": float(TH[:, j].min()), "max": float(TH[:, j].max()),
                                       "rel_sd": float(TH[:, j].std() / max(abs(TH[:, j].mean()), 1e-12))}
                                   for j, n in enumerate(names)}
    (ROOT / "plans" / f"snap_loo_{a.system}_{a.family}{a.out_tag}.json").write_text(json.dumps(rep, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
