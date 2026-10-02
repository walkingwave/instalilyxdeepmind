"""hi lab (supply_chain, discrete-time families): leave-one-run-out at the calibrated organizer scale
(1.0 sigma), per-fold per-observable table, named-segment scores, and a model doc. Free, no gateway.

    python scripts/lab_canon_sh_loo.py --system supply_chain --family supply_chain_canon1 --tag a
        [--theta0 DOC_OR_LAB_JSON] [--free p1,p2] [--budget 240] [--starts 4] [--nfev 40]
        [--folds exp1,exp2] [--no-full]
Writes plans/<system>_canon_<family>_<tag>.json and ..._doc.json (post rules copied from the pick).
Hospital scores apply the shipped queue cap (333) before scoring, as the pick does.
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

PICK = {"supply_chain": ROOT / "plans/alt1_supply_chain_doc.json",
        "hospital_queue": ROOT / "plans/hospital_queue_hospital_queue_hosp9_h9w1_doc.json"}
SEG = {"supply_chain": {"p2.pulse200_200": [(0, 200), (200, 300), (300, 400)],
                        "p3.hold_mid": [(0, 200), (200, 300), (300, 450)],
                        "p7.longhold": [(760, 800)]},
       "hospital_queue": {"p3.compose": [(0, 180), (180, 270), (270, 330)],
                          "p4.pulse_long_recovery": [(0, 40), (40, 130), (130, 300)],
                          "p6.voi": [(40, 130), (130, 300)],
                          "p1.hold_rec": [(0, 30), (30, 120)]}}


def post(system, Y):
    if system == "hospital_queue":
        Y = Y.copy()
        Y[:, 1] = np.minimum(Y[:, 1], 333.0)
    return Y


def theta_from(mod, path):
    init = {p[0]: p[1] for p in mod.PARAMS}
    if path:
        d = json.loads(Path(path).read_text())
        if "model" in d:
            fam = importlib.import_module(f"gtlab.ode.{d['model']['family']}")
            src = dict(zip([p[0] for p in fam.PARAMS], d["model"]["theta"]))
        elif isinstance(d.get("theta"), dict):
            src = d["theta"]
        else:
            raise ValueError("unknown theta0 file")
        for k, v in src.items():
            if k in init:
                lo, hi = [(p[2], p[3]) for p in mod.PARAMS if p[0] == k][0]
                init[k] = float(min(max(v, lo), hi))
    return [init[p[0]] for p in mod.PARAMS]


def scores(system, mod, th, r, sigma, n_sub):
    Y = post(system, core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), "AB", n_sub=n_sub))
    s = metric.score_per_obs(Y, r.Y, sigma)
    seg = {}
    for a, b in SEG[system].get(r.exp, []):
        b = min(b, r.T)
        if b > a:
            seg[f"{a}-{b}"] = metric.score_per_obs(Y[a:b], r.Y[a:b], sigma).tolist()
    return s, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--theta0", default=None)
    ap.add_argument("--free", default=None)
    ap.add_argument("--set", default=None, help="k=v,k=v: override theta0 entries (fixed if not free)")
    ap.add_argument("--exclude", default=None, help="comma list: fit all params except these")
    ap.add_argument("--budget", type=float, default=240)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--nfev", type=int, default=40)
    ap.add_argument("--spread", type=float, default=0.15)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--no-loo", action="store_true")
    a = ap.parse_args()
    spec = S.get(a.system)
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"], float)
    names = [p[0] for p in mod.PARAMS]
    th0 = theta_from(mod, a.theta0 or str(PICK[a.system]))
    if a.set:
        for kv in a.set.split(","):
            k, v = kv.split("=")
            th0[names.index(k)] = float(v)
    free = a.free.split(",") if a.free else None
    if a.exclude:
        free = [n for n in names if n not in a.exclude.split(",")]
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=a.spread, free=free,
              theta0=th0, n_polish=a.starts)
    rep = {"set": a.set, "exclude": a.exclude, "system": a.system, "family": a.family, "tag": a.tag, "free": free, "sigma": sigma.tolist(),
           "runs": [(r.exp, r.T) for r in runs], "theta0": th0, "loo": [], "insample0": {}}
    print(f"{a.system}/{a.family}/{a.tag} runs {rep['runs']}", flush=True)
    for r in runs:
        s, seg = scores(a.system, mod, th0, r, sigma, n_sub)
        rep["insample0"][r.exp] = {"s": s.tolist(), "seg": seg}
    folds = a.folds.split(",") if a.folds else [r.exp for r in runs]
    with C.single_thread():
        if not a.no_loo:
            for k, rk in enumerate(runs):
                if rk.exp not in folds:
                    continue
                tr = [r for i, r in enumerate(runs) if i != k]
                t0 = time.time()
                th, info = F.fit_ode(mod, tr, sigma, "AB", time_budget=a.budget, **kw)
                s, seg = scores(a.system, mod, th, rk, sigma, n_sub)
                rep["loo"].append({"held_out": rk.exp, "s": s.tolist(), "seg": seg, "theta": [float(v) for v in th]})
                print(f"  LOO {rk.exp:<26} {np.round(s, 3)} mean {s.mean():.4f} seg "
                      f"{ {k2: round(float(np.mean(v)), 3) for k2, v in seg.items()} } [{time.time() - t0:.0f}s]", flush=True)
            if rep["loo"]:
                rep["loo_mean"] = float(np.mean([np.mean(x["s"]) for x in rep["loo"]]))
                print(f"  LOO mean {rep['loo_mean']:.4f}", flush=True)
        if not a.no_full:
            th, info = F.fit_ode(mod, runs, sigma, "AB", time_budget=a.budget * 1.5, **kw)
            rep["theta_vec"] = [float(v) for v in th]
            rep["theta"] = dict(zip(names, rep["theta_vec"]))
            rep["insample"] = {}
            for r in runs:
                s, seg = scores(a.system, mod, th, r, sigma, n_sub)
                rep["insample"][r.exp] = {"s": s.tolist(), "seg": seg}
                print(f"  in {r.exp:<26} {np.round(s, 3)} mean {s.mean():.4f}", flush=True)
            print("  theta", {k: round(v, 5) for k, v in rep["theta"].items()}, flush=True)
            rng = np.random.default_rng(0)
            rep["eval"] = {}
            for cat in ("sustained", "order", "recovery", "composition"):
                U = D.eval_like(spec, cat, 4000, rng)
                t0 = time.time()
                Y = post(a.system, core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th), "AB", n_sub=n_sub))
                rep["eval"][cat] = {"finite": bool(np.all(np.isfinite(Y))), "sec": time.time() - t0,
                                    "min": Y.min(0).tolist(), "max": Y.max(0).tolist()}
                print(f"  eval[{cat}] finite={rep['eval'][cat]['finite']} {time.time() - t0:.2f}s "
                      f"min={np.round(Y.min(0), 2)} max={np.round(Y.max(0), 2)}", flush=True)
            pick = json.loads(PICK[a.system].read_text())
            blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": ["A", "B"], "n_sub": n_sub}
            doc = C.make_doc(spec, blob, runs=runs, clip=(pick["clip_lo"], pick["clip_hi"]),
                             info={"model_id": f"ode:{a.family}", "n_runs": len(runs)})
            doc["post"] = pick.get("post", [])
            (ROOT / "plans" / f"supply_chain_hi_{a.family}_{a.tag}_doc.json").write_text(json.dumps(doc, indent=1))
    out = ROOT / "plans" / f"supply_chain_hi_{a.family}_{a.tag}.json"
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
