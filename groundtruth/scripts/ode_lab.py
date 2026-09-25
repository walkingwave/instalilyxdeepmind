"""Grey-box lab: fit one ODE family on a system's real runs and report everything a build needs.

    python scripts/ode_lab.py --system reservoir --family reservoir_min [--budget 240] [--starts 10]
        [--nfev 60] [--mech AB] [--free p1,p2] [--no-loo] [--tag x]

Steps (all free, no credits):
  1. leave-one-run-out: fit on all runs but one, score the held-out run per observable
     (own sigma_proxy), against persistence and against l0b_lin fitted on the same folds
  2. full fit on every run; in-sample per observable
  3. 4,000-tick rollouts on eval-shaped schedules of all four categories: finite, time,
     fraction of ticks outside the observed data range (+-5% of range)
  4. writes plans/<system>_<family>[_tag].json (theta + report) and
     plans/<system>_<family>[_tag]_doc.json (a model doc that scripts/build_cfg.py / package can use)
Exit code 0 always; the report's "verdict" says ship / maybe / no.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import design as D, metric, select as SEL, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def per_obs(mod, theta, mech, r, sigma, n_sub):
    Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, theta), mech, n_sub=n_sub)
    return metric.score_per_obs(Y, r.Y, sigma), Y


def l0b_scores(spec, train, test, sigma):
    clip = C.soft_clip(spec, train, margin=1.0)
    m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(train)
    return metric.score_per_obs(m.rollout(test.y0, test.U), test.Y, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--family", required=True, help="module name under gtlab.ode, e.g. reservoir_min")
    ap.add_argument("--budget", type=float, default=240, help="seconds for the full fit (LOO folds get half each)")
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--free", default=None, help="comma list of parameter names to fit (others fixed at init)")
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--theta0", default=None, help="json file with a theta list to start from")
    a = ap.parse_args()
    spec = S.get(a.system)
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    assert list(mod.OBS) == list(spec.observables) and list(mod.CTRL) == list(spec.controls), "OBS/CTRL order mismatch"
    mech = frozenset(a.mech)
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = metric.sigma_proxy(runs)
    free = a.free.split(",") if a.free else None
    theta0 = json.loads(Path(a.theta0).read_text())["theta"] if a.theta0 else None
    kw = dict(n_starts=a.starts, max_nfev=a.nfev, n_sub=n_sub, early_T=None, spread=0.5, free=free, theta0=theta0)
    names = [p[0] for p in mod.PARAMS]
    rep = {"system": a.system, "family": a.family, "mech": a.mech, "runs": [(r.exp, r.T) for r in runs],
           "sigma_proxy": sigma.tolist(), "loo": [], "insample": {}}
    print(f"{a.system} / {a.family}: runs {rep['runs']} sigma {sigma}")
    with C.single_thread():
        if not a.no_loo and len(runs) >= 2:
            for k in range(len(runs)):
                tr = [r for i, r in enumerate(runs) if i != k]
                t0 = time.time()
                th, info = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget / 2, **kw)
                s_out, _ = per_obs(mod, th, mech, runs[k], sigma, n_sub)
                pers = metric.score_per_obs(np.tile(runs[k].y0, (runs[k].T, 1)), runs[k].Y, sigma)
                base = l0b_scores(spec, tr, runs[k], sigma)
                rep["loo"].append({"held_out": runs[k].exp, "ode": s_out.tolist(), "l0b_lin": base.tolist(),
                                   "persistence": pers.tolist(), "cost": float(info["cost"])})
                print(f"  LOO {runs[k].exp:<20} ode {np.round(s_out, 3)} mean {s_out.mean():.3f} | l0b_lin {np.round(base, 3)} "
                      f"mean {base.mean():.3f} | pers {pers.mean():.3f} [{time.time() - t0:.0f}s]")
        t0 = time.time()
        th, info = F.fit_ode(mod, runs, sigma, mech, time_budget=a.budget, **kw)
    print(f"  full fit cost {info['cost']:.1f} [{time.time() - t0:.0f}s]")
    for r in runs:
        s, _ = per_obs(mod, th, mech, r, sigma, n_sub)
        rep["insample"][r.exp] = s.tolist()
        print(f"  in-sample {r.exp:<20} {np.round(s, 3)} mean {s.mean():.3f}")
    theta = {n: float(v) for n, v in zip(names, th)}
    at_bound = [n for (n, ini, lo, hi, lg), v in zip(mod.PARAMS, th) if v <= lo * 1.001 + 1e-12 or v >= hi * 0.999]
    print("  theta", {k: round(v, 5) for k, v in theta.items()})
    if at_bound:
        print("  AT BOUND:", at_bound)
    # eval-shaped 4000-tick sanity
    Yall = np.concatenate([r.Y for r in runs]); ymin, ymax = Yall.min(0), Yall.max(0); rng_ = ymax - ymin
    rng = np.random.default_rng(0)
    rep["eval"] = {}
    ok = True
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time()
        Y = core.rollout(mod, runs[0].y0, U, core.theta_dict(mod, th), mech, n_sub=n_sub)
        dt = time.time() - t0
        fin = bool(np.all(np.isfinite(Y)))
        out = float(np.mean((Y > ymax + 0.05 * rng_) | (Y < ymin - 0.05 * rng_)))
        rep["eval"][cat] = {"finite": fin, "seconds": dt, "frac_outside": out, "min": Y.min(0).tolist(), "max": Y.max(0).tolist()}
        ok = ok and fin and dt < 25
        print(f"  eval[{cat:<11}] finite={fin} {dt:.1f}s outside={out:.2f} min={np.round(Y.min(0), 1)} max={np.round(Y.max(0), 1)}")
    loo_ode = np.mean([np.mean(x["ode"]) for x in rep["loo"]]) if rep["loo"] else float("nan")
    loo_base = np.mean([np.mean(x["l0b_lin"]) for x in rep["loo"]]) if rep["loo"] else float("nan")
    ins = np.mean([np.mean(v) for v in rep["insample"].values()])
    verdict = "no" if not ok else ("ship" if (rep["loo"] and loo_ode > loo_base + 0.03) else ("maybe" if ins > 0.75 else "no"))
    rep.update({"theta": theta, "theta_vec": [float(v) for v in th], "cost": float(info["cost"]), "at_bound": at_bound,
                "loo_mean_ode": float(loo_ode), "loo_mean_l0b_lin": float(loo_base), "insample_mean": float(ins), "verdict": verdict})
    tag = f"_{a.tag}" if a.tag else ""
    out = Path("plans") / f"{a.system}_{a.family}{tag}.json"
    out.write_text(json.dumps(rep, indent=1))
    blob = {"kind": "ode", "family": a.family, "theta": rep["theta_vec"], "mech": sorted(a.mech), "n_sub": n_sub}
    doc = C.make_doc(spec, blob, runs=runs, clip=C.soft_clip(spec, runs, margin=1.0),
                     info={"model_id": f"ode:{a.family}", "n_runs": len(runs), "n_ticks": int(sum(r.T for r in runs)), "cost": float(info["cost"])})
    (Path("plans") / f"{a.system}_{a.family}{tag}_doc.json").write_text(json.dumps(doc, indent=1))
    print(f"VERDICT {verdict}: LOO ode {loo_ode:.3f} vs l0b_lin {loo_base:.3f}; in-sample {ins:.3f} -> {out}")


if __name__ == "__main__":
    main()
