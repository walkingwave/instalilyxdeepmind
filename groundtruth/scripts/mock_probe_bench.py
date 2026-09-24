"""Tonight's question: does a lag model fit on the 240-step probe beat persistence?

Fits l0a/l0b/l1 on mock probe runs (hold_rec + hold_pulse) and scores each against the
mock's NOISELESS truth on eval-like 4000-step episodes (all four categories).
Usage: python scripts/mock_probe_bench.py --data-root /tmp/mockroot [--systems a,b] [--l1-budget 90]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, design
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.mocks.base import MockSystem
from gtlab import select as SEL
from gtlab.models import common as C


def bench(sid, root, l1_budget, n_per_cat=3, T=4000):
    spec = S.get(sid)
    led = Ledger(data_dir(sid, True, root), sid, mock=True)
    runs = load_runs(spec, led)
    for r in runs:
        r.tags["split"] = "train"
    clip = C.soft_clip(spec, runs)
    sigma = metric.sigma_proxy(runs)
    models = {}
    for kind in ("l0a", "l0b", "l0b_lin", "l1"):
        t = time.time()
        if kind == "l0b_lin":
            m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None})
        else:
            m = SEL.make_model(kind, spec, clip, sigma, time_budget_s=l1_budget)
        m.fit(runs)
        models[kind] = (m, time.time() - t)
    world = MockSystem(sid, mech=("A", "B"), seed=0)
    rng = np.random.default_rng(123)
    eps = []
    for cat in ("sustained", "order", "recovery", "composition"):
        for _ in range(n_per_cat):
            U = design.eval_like(spec, cat, T, rng)
            y0 = world.sample_y0(rng)
            Yt, _ = world.simulate(y0, U)
            y0n = world._noisy(y0)
            eps.append((cat, y0n, U, Yt))
    sig_true = np.concatenate([e[3] for e in eps]).std(axis=0) + 1e-9
    out = {}
    for kind, (m, ft) in models.items():
        sc = []
        for cat, y0n, U, Yt in eps:
            P = m.rollout(y0n, U)
            if not np.all(np.isfinite(P)):
                sc.append(0.0)
                continue
            sc.append(metric.score(P, Yt, sig_true))
        out[kind] = {"score": float(np.mean(sc)), "fit_s": round(ft, 1)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--systems", default=",".join(S.SYSTEM_IDS))
    ap.add_argument("--l1-budget", type=float, default=90)
    a = ap.parse_args()
    allres = {}
    for sid in a.systems.split(","):
        try:
            res = bench(sid, a.data_root, a.l1_budget)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"error": str(e)}
        allres[sid] = res
        print(sid, json.dumps(res), flush=True)
    print("MEAN", {k: round(float(np.mean([r[k]["score"] for r in allres.values() if k in r])), 4)
                   for k in ("l0a", "l0b", "l0b_lin", "l1")})
