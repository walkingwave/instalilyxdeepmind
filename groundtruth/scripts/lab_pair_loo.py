"""Mechanism-pair lab: full fit + leave-one-run-out at the calibrated organizer scale (1.0 sigma) for one
family, one mechanism set and a list of pinned parameters. Free, no gateway calls.

    python scripts/lab_pair_loo.py --system reservoir --family reservoir_pair --mech AC --tag ac \
        --theta0 plans/reservoir_str9_c_theta0_clean.json --pin k_ret,tau_ret,w_L,d0,g_f,r_fl,r_a \
        [--set k_ret=0 w_L=1] [--budget 300] [--freeze-fold p2.pulse200_200 --freeze k_ret,tau_ret]

Pinned parameters stay at theta0 (after --set). --freeze-fold adds one extra evaluation of that fold
with the --freeze parameters held at the full-fit values (they are unseen in that fold's training
runs). Writes plans/<system>_pair_<tag>.json and ..._doc.json.
"""
import argparse, importlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F


def load_theta0(mod, path, sets):
    init = {p[0]: p[1] for p in mod.PARAMS}
    if path:
        d = json.loads(Path(path).read_text())
        if isinstance(d.get("theta"), dict):
            init.update({k: v for k, v in d["theta"].items() if k in init})
        else:
            th = d.get("theta_vec") or d.get("theta") or d["model"]["theta"]
            for (n, *_), v in zip(mod.PARAMS, th):
                init[n] = v
    for s in sets or []:
        k, v = s.split("=")
        init[k] = float(v)
    return [float(init[p[0]]) for p in mod.PARAMS]


def near_zero(mod, th):
    out = []
    for (n, _, lo, hi, logp), v in zip(mod.PARAMS, th):
        span = hi - lo
        if (v - lo) < 0.02 * span or (lo < 0 < hi and abs(v) < 0.02 * span):
            out.append((n, float(v), lo, hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--mech", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--pin", default="")
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--freeze-fold", default=None)
    ap.add_argument("--freeze", default="")
    a = ap.parse_args()
    spec = S.get(a.system)
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    theta0 = load_theta0(mod, a.theta0, a.set)
    pins = [p for p in a.pin.split(",") if p]
    free = [n for n in names if n not in pins]
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=a.spread)
    rep = {"system": a.system, "family": a.family, "mech": sorted(a.mech), "pins": pins, "sets": a.set,
           "theta0": theta0, "runs": [(r.exp, r.T) for r in runs], "sigma": sigma.tolist(), "loo": []}
    print(f"{a.system}/{a.family} mech {a.mech} tag {a.tag} free {len(free)}", flush=True)
    with C.single_thread():
        t0 = time.time()
        th_full, _ = F.fit_ode(mod, runs, sigma, mech, time_budget=a.budget, free=free, theta0=theta0, **kw)
        print(f"  full fit [{time.time() - t0:.0f}s]", flush=True)
        for k, rk in enumerate(runs):
            tr = [r for i, r in enumerate(runs) if i != k]
            t0 = time.time()
            th, _ = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget / 2, free=free, theta0=theta0, **kw)
            Y = core.rollout(mod, rk.y0, rk.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
            s = metric.score_per_obs(Y, rk.Y, sigma)
            rep["loo"].append({"held_out": rk.exp, "ode": s.tolist(), "theta": [float(v) for v in th]})
            print(f"  LOO {rk.exp:<22} {np.round(s, 3)} mean {s.mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
            if a.freeze_fold == rk.exp and a.freeze:
                fz = a.freeze.split(",")
                th0f = list(theta0)
                for n in fz:
                    th0f[names.index(n)] = float(th_full[names.index(n)])
                thf, _ = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget / 2,
                                   free=[n for n in free if n not in fz], theta0=th0f, **kw)
                Y = core.rollout(mod, rk.y0, rk.U, core.theta_dict(mod, thf), mech, n_sub=n_sub)
                s = metric.score_per_obs(Y, rk.Y, sigma)
                rep["frozen_fold"] = {"held_out": rk.exp, "frozen": fz, "ode": s.tolist()}
                print(f"  LOO {rk.exp:<22} frozen {fz}: {np.round(s, 3)} mean {s.mean():.4f}", flush=True)
    rep["loo_mean"] = float(np.mean([np.mean(x["ode"]) for x in rep["loo"]]))
    print(f"  LOO mean {rep['loo_mean']:.4f}", flush=True)
    rep["insample"] = {}
    for r in runs:
        Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th_full), mech, n_sub=n_sub)
        s = metric.score_per_obs(Y, r.Y, sigma)
        rep["insample"][r.exp] = s.tolist()
        print(f"  in {r.exp:<22} {np.round(s, 3)} mean {s.mean():.4f}")
    rep["theta"] = {n: float(v) for n, v in zip(names, th_full)}
    rep["theta_vec"] = [float(v) for v in th_full]
    rep["near_bound"] = [x for x in near_zero(mod, th_full) if x[0] in free]
    print("  theta", {k: round(v, 5) for k, v in rep["theta"].items()})
    print("  near lower bound / zero (free only):", rep["near_bound"])
    rng = np.random.default_rng(0)
    rep["eval"] = {}
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time()
        Y = core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th_full), mech, n_sub=n_sub)
        rep["eval"][cat] = {"finite": bool(np.all(np.isfinite(Y))), "sec": time.time() - t0,
                            "min": Y.min(0).tolist(), "max": Y.max(0).tolist()}
        print(f"  eval[{cat}] finite={rep['eval'][cat]['finite']} {time.time() - t0:.2f}s "
              f"min={np.round(Y.min(0), 3)} max={np.round(Y.max(0), 3)}")
    out = ROOT / "plans" / f"{a.system}_pair_{a.tag}.json"
    out.write_text(json.dumps(rep, indent=1))
    blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": sorted(a.mech), "n_sub": n_sub}
    doc = C.make_doc(spec, blob, runs=runs, clip=C.soft_clip(spec, runs, margin=1.0),
                     info={"model_id": f"ode:{a.family}", "n_runs": len(runs)})
    if a.system == "reservoir":
        doc["post"] = []
    (ROOT / "plans" / f"{a.system}_pair_{a.tag}_doc.json").write_text(json.dumps(doc, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
