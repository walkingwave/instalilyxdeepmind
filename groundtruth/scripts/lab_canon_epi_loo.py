"""canon: leave-one-run-out for an epidemic family, folds in parallel (ode_lab settings: 150 s per fold,
8 starts, nfev 50, sigma 1.0 calibrated). Also fits all six runs (in-sample). Saves thetas + held-out
predictions to <out>.npz and prints the fold table.

    python scripts/lab_canon_epi_loo.py <family> <mech> <out_prefix> [procs] [fix_json]

fix_json (optional): {"name": value, ...} parameters held fixed (snapped constants).
"""
import importlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SIG = np.array([12.953956472279227, 5.230371399915308])


def _load():
    from gtlab import systems as S
    from gtlab.ledger import Ledger, data_dir, load_runs
    return load_runs(S.get("epidemic"), Ledger(data_dir("epidemic", False), "epidemic", False))


def job(args):
    fam, mech, ho, fix = args
    import os
    os.chdir(ROOT)
    from gtlab import metric
    from gtlab.models import common as C
    from gtlab.ode import core, fit as F
    mod = importlib.import_module(f"gtlab.ode.{fam}")
    runs = _load()
    names = [p[0] for p in mod.PARAMS]
    theta0 = np.array([fix.get(n, p[1]) for n, p in zip(names, mod.PARAMS)], float)
    free = [n for n in names if n not in fix] if fix else None
    tr = [r for r in runs if r.exp != ho]
    t0 = time.time()
    with C.single_thread():
        th, info = F.fit_ode(mod, tr, SIG, frozenset(mech), time_budget=150, n_starts=8, max_nfev=50,
                             n_sub=int(getattr(mod, "N_SUB", 2)), early_T=None, spread=0.5, free=free,
                             theta0=theta0)
    out = {"ho": ho, "th": [float(v) for v in th], "sec": time.time() - t0, "scores": {}, "Y": {}}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset(mech), n_sub=int(getattr(mod, "N_SUB", 2)))
        s = metric.score_per_obs(Y, r.Y, SIG)
        out["scores"][r.exp] = [float(v) for v in s]
        out["Y"][r.exp] = Y
    return out


def main():
    fam, mech, pref = sys.argv[1], sys.argv[2], sys.argv[3]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    fix = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    runs = _load()
    exps = [r.exp for r in runs]
    jobs = [(fam, mech, e, fix) for e in exps] + [(fam, mech, "__all__", fix)]
    with Pool(procs) as p:
        res = p.map(job, jobs)
    npz = {}
    row, ins = [], None
    for o in res:
        if o["ho"] == "__all__":
            ins = o
            continue
        s = np.array(o["scores"][o["ho"]])
        row.append((o["ho"], s))
        npz[f"Y_{o['ho']}"] = o["Y"][o["ho"]]
        npz[f"th_{o['ho']}"] = np.array(o["th"])
    npz["th_all"] = np.array(ins["th"])
    np.savez(pref + ".npz", **npz)
    print(f"{fam} {mech} fix={fix}")
    for e, s in row:
        print(f"  LOO {e:<18} {np.round(s, 3)} mean {s.mean():.3f}")
    loo = np.mean([s.mean() for _, s in row])
    insm = np.mean([np.mean(v) for v in ins["scores"].values()])
    print(f"  LOO mean {loo:.4f}   in-sample (all 6) {insm:.4f}  " +
          " ".join(f"{k.split('.')[0]}={np.mean(v):.3f}" for k, v in ins["scores"].items()))
    Path(pref + ".json").write_text(json.dumps({"family": fam, "mech": mech, "fix": fix,
        "loo": {e: [float(v) for v in s] for e, s in row}, "loo_mean": float(loo),
        "insample": ins["scores"], "insample_mean": float(insm),
        "theta_all": dict(zip([p[0] for p in importlib.import_module(f"gtlab.ode.{fam}").PARAMS], ins["th"]))}, indent=1))


if __name__ == "__main__":
    main()
