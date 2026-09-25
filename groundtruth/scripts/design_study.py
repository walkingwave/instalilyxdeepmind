"""Which experiments buy the most score per credit? Offline study on the mock simulators.

For each system, design family and budget, simulate the purchased runs (noisy, from reset), fit
the fast models, and score them against NOISELESS eval-like episodes of all four categories.
    python scripts/design_study.py --systems all --budgets 120,240,420,700,1000 --out /tmp/study.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, design as D
from gtlab.data import Run
from gtlab.mocks.base import MockSystem
from gtlab import select as SEL
from gtlab.models import common as C

CATS = ("sustained", "order", "recovery", "composition")


# ----------------------------------------------------------------------------- designs
def d_plan(spec, rng, B):
    """The current p1 plan, in plan order, truncated to budget."""
    ex = D.default_experiments(spec, "p1", seed=0)
    out, left = [], B
    for e in ex:
        if left <= 0:
            break
        U = np.asarray(e["U"])[:left]
        out.append(U)
        left -= len(U)
    return out


def d_evalmix(spec, rng, B, L=120):
    """Short runs shaped exactly like the test (mixed categories)."""
    out, left = [], B
    while left > 0:
        T = min(L, left)
        cat = CATS[len(out) % 4]
        out.append(D.eval_like(spec, cat, T, rng) if T >= 20 else D.hold(spec, D.rec(spec), T))
        left -= T
    return out


def d_evalmix_long(spec, rng, B):
    return d_evalmix(spec, rng, B, L=max(120, B // 2))


def d_holds(spec, rng, B, L=240):
    """Long holds at varied levels (recovery, pulse, interior, corners): steady states + slow modes."""
    out, left = [], B
    while left > 0:
        T = min(L, left)
        rows, t = [], 0
        while t < T:
            d = int(min(T - t, max(20, D._loguni(rng, 40, 200))))
            kind = rng.choice(["recovery", "pulse", "uniform", "corner", "mid"], p=[.2, .25, .3, .15, .1])
            rows.append(D.hold(spec, D.level(spec, kind, rng), d))
            t += d
        out.append(D.clip(spec, np.concatenate(rows)[:T]))
        left -= T
    return out


def d_multilevel(spec, rng, B, L=160):
    out, left = [], B
    while left > 0:
        T = min(L, left)
        out.append(D.multilevel(spec, rng, T))
        left -= T
    return out


def d_hybrid(spec, rng, B):
    """Proposal: 1 recovery hold (reset transient + baseline) then alternate long-hold runs and
    test-shaped runs; composition/pulse coverage comes from the test-shaped runs."""
    first = min(120, B)
    out = [D.hold(spec, D.rec(spec), first)]
    left = B - first
    k = 0
    while left > 0:
        T = min(200, left)
        if k % 2 == 0:
            out += d_holds(spec, rng, T, L=T)
        else:
            cat = ["recovery", "composition", "order", "sustained"][(k // 2) % 4]
            out.append(D.eval_like(spec, cat, T, rng) if T >= 20 else D.hold(spec, D.rec(spec), T))
        left -= T
        k += 1
    return out


DESIGNS = {"plan": d_plan, "evalmix120": d_evalmix, "evalmix_long": d_evalmix_long,
           "holds": d_holds, "multilevel": d_multilevel, "hybrid": d_hybrid}


# ----------------------------------------------------------------------------- study
def episodes(spec, world, n_per_cat, T, seed=777):
    rng = np.random.default_rng(seed)
    eps = []
    for cat in CATS:
        for _ in range(n_per_cat):
            U = D.eval_like(spec, cat, T, rng)
            y0 = world.sample_y0(rng)
            Yt, _ = world.simulate(y0, U)
            eps.append((cat, world._noisy(y0), U, Yt))
    sig = np.concatenate([e[3] for e in eps]).std(axis=0) + 1e-9
    return eps, sig


def buy(spec, world, Us, rng):
    runs = []
    for i, U in enumerate(Us):
        y0 = world.sample_y0(rng)
        Yt, Yn = world.simulate(y0, U)
        runs.append(Run(spec.id, f"x{i}", world._noisy(y0), U, Yn, Ytrue=Yt, tags={"split": "train"}))
    return runs


def score_model(m, eps, sig):
    per = {c: [] for c in CATS}
    for cat, y0, U, Yt in eps:
        P = m.rollout(y0, U)
        per[cat].append(metric.score(P, Yt, sig) if np.all(np.isfinite(P)) else 0.0)
    pc = {c: float(np.mean(v)) for c, v in per.items()}
    return float(np.mean(list(pc.values()))), pc


def run_system(sid, budgets, designs, models, n_per_cat, T, l1_budget, mech, seed):
    spec = S.get(sid)
    world = MockSystem(sid, mech=mech, seed=seed)
    eps, sig = episodes(spec, world, n_per_cat, T)
    res = {}
    # persistence reference
    m0 = SEL.make_model("l0a", spec, C.soft_clip(spec, []), np.ones(spec.p))
    res["l0a"] = score_model(m0, eps, sig)
    for dn in designs:
        for B in budgets:
            rng = np.random.default_rng([seed, B, len(dn)])
            runs = buy(spec, world, DESIGNS[dn](spec, rng, B), rng)
            clip = C.soft_clip(spec, runs)
            sigma = metric.sigma_proxy(runs)
            for kind in models:
                t0 = time.time()
                try:
                    if kind == "l0b_lin":
                        m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None})
                    else:
                        m = SEL.make_model(kind, spec, clip, sigma, time_budget_s=l1_budget)
                    m.fit(runs)
                    s, pc = score_model(m, eps, sig)
                except Exception as e:
                    s, pc = 0.0, {"error": str(e)[:80]}
                res[f"{dn}|{B}|{kind}"] = (s, pc)
    return sid, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--budgets", default="120,240,420,700,1000")
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--models", default="l0b_lin,l1")
    ap.add_argument("--n-per-cat", type=int, default=4)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--l1-budget", type=float, default=20)
    ap.add_argument("--mech", default="AB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sids = S.SYSTEM_IDS if a.systems == "all" else a.systems.split(",")
    budgets = [int(b) for b in a.budgets.split(",")]
    allres = {}
    for sid in sids:
        t = time.time()
        _, r = run_system(sid, budgets, a.designs.split(","), a.models.split(","), a.n_per_cat, a.T,
                          a.l1_budget, tuple(a.mech), a.seed)
        allres[sid] = r
        Path(a.out).write_text(json.dumps(allres))
        print(f"{sid} done in {time.time() - t:.0f}s", flush=True)
