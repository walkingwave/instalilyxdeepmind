"""Per-observable leave-one-run-out: ODE family vs l0b_lin vs their per-tick median.

    python scripts/perobs_loo.py --system market --family market_min [--budget 90] [--out plans/x.json]

For each fold: refit the ODE (time_budget) and l0b_lin (clip 1x, eq bound) on the other runs,
predict the held-out run, score per observable for ode, l0b, median, plus the fraction of ticks
where the two members straddle the truth. Prints the table and the per-observable winner.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, select as SEL, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core, fit as F
from gtlab.runtime import infer as rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--budget", type=float, default=90)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = S.get(a.system)
    mod = importlib.import_module(f"gtlab.ode.{a.family}")
    mech = frozenset(a.mech)
    n_sub = int(getattr(mod, "N_SUB", 2))
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sigma = metric.sigma_proxy(runs)
    p = spec.p
    acc = {k: np.zeros(p) for k in ("ode", "l0b", "median")}
    straddle = np.zeros(p)
    folds = []
    with C.single_thread():
        for k in range(len(runs)):
            tr = [r for i, r in enumerate(runs) if i != k]
            te = runs[k]
            th, _ = F.fit_ode(mod, tr, sigma, mech, time_budget=a.budget, n_starts=6, max_nfev=50, n_sub=n_sub, early_T=None, spread=0.5)
            Yo = core.rollout(mod, te.y0, te.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
            clip = C.soft_clip(spec, tr, margin=1.0)
            m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(tr)
            Yl = m.rollout(te.y0, te.U)
            lo, hi = rt.clip_vectors(C.meta_doc(spec, clip))
            Yo = rt.finalize(Yo, te.y0, lo, hi)
            Ym = 0.5 * (Yo + Yl)
            so, sl, sm = (metric.score_per_obs(Y, te.Y, sigma) for Y in (Yo, Yl, Ym))
            st = np.mean(((Yo - te.Y) * (Yl - te.Y)) < 0, axis=0)
            acc["ode"] += so; acc["l0b"] += sl; acc["median"] += sm; straddle += st
            folds.append({"held_out": te.exp, "ode": so.tolist(), "l0b": sl.tolist(), "median": sm.tolist(), "straddle": st.tolist()})
            print(f"fold {te.exp:<20} ode {np.round(so,3)} l0b {np.round(sl,3)} median {np.round(sm,3)} straddle {np.round(st,2)}", flush=True)
    n = len(runs)
    print(f"\n{'observable':<18} {'ode':>6} {'l0b':>6} {'median':>7} {'straddle':>8}  pick")
    picks = {}
    for j, o in enumerate(spec.observables):
        vals = {k: acc[k][j] / n for k in acc}
        best = max(vals, key=vals.get)
        picks[o] = best
        print(f"{o:<18} {vals['ode']:6.3f} {vals['l0b']:6.3f} {vals['median']:7.3f} {straddle[j]/n:8.2f}  {best}")
    out = a.out or f"plans/{a.system}_perobs_loo.json"
    Path(out).write_text(json.dumps({"system": a.system, "family": a.family, "folds": folds,
                                     "mean": {k: (acc[k] / n).tolist() for k in acc}, "straddle": (straddle / n).tolist(), "picks": picks}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
