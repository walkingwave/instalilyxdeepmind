"""Simulations on the real data collected so far, to guide the next purchase. Free.

1. Extrapolation test: fit each kind on the first `--head` ticks of every run, score the rest.
2. Committee disagreement: fit a committee (kinds x sigma multipliers x block bootstrap), roll it
   on candidate schedules (each test category, plus the planned p2 experiments), report the spread
   between members in sigma units, per system and candidate. High spread = the data would settle
   something the models cannot.

    python scripts/sim_next.py [--head 80] [--kinds l0b_lin,l1] [--T 300]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, design as D, select as SEL
from gtlab.models import common as C
from gtlab.data import Run
from gtlab.ledger import Ledger, data_dir, load_runs

warnings.filterwarnings("ignore")
CATS = ("sustained", "order", "recovery", "composition")


def fit(kind, spec, runs, sig_mult=1.0, budget=15):
    clip = C.soft_clip(spec, runs)
    sigma = metric.sigma_proxy(runs) * sig_mult
    if kind == "l0b_lin":
        m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None})
    else:
        m = SEL.make_model(kind, spec, clip, sigma, time_budget_s=budget)
    m.fit(runs)
    return m


def block_bootstrap(run, rng, block=20):
    """Resample contiguous blocks of one run into a same-length pseudo-run (keeps y0)."""
    T = run.T
    idx = []
    while len(idx) < T:
        s = int(rng.integers(0, max(1, T - block)))
        idx += list(range(s, min(T, s + block)))
    idx = np.array(idx[:T])
    return Run(run.system, run.exp, run.y0, run.U[idx], run.Y[idx], tags=dict(run.tags))


def extrapolation(spec, runs, kinds, head):
    out = {}
    tr = [Run(r.system, r.exp, r.y0, r.U[:head], r.Y[:head], tags=dict(r.tags)) for r in runs]
    sigma = metric.sigma_proxy(runs)
    for kind in ["l0a"] + list(kinds):
        try:
            m = fit(kind, spec, tr)
            sc = []
            for r in runs:
                P = m.rollout(r.y0, r.U)
                sc.append(metric.score(P[head:], r.Y[head:], sigma))
            out[kind] = float(np.mean(sc))
        except Exception as e:
            out[kind] = f"ERR {type(e).__name__}"
    return out


def candidates(spec, T, rng, plan_dir):
    cands = {c: D.eval_like(spec, c, T, rng) for c in CATS}
    cands["long_hold_pulse"] = D.hold(spec, D.pulse_level(spec, 1.0), T)
    cands["long_hold_mid"] = D.hold(spec, D.level(spec, "mid", rng), T)
    p = Path(plan_dir) / "plan.json"
    if p.exists():
        doc = json.loads(p.read_text())
        for e in doc.get("phases", {}).get("p2", {}).get("experiments", []):
            cands["p2." + e["exp_id"].split(".", 1)[-1]] = np.asarray(e["U"], float)
    return cands


def disagreement(spec, runs, kinds, cands, rng, n_boot=3):
    sigma = metric.sigma_proxy(runs)
    members = []
    for kind in kinds:
        for sm in (0.5, 1.0, 2.0):
            for b in range(n_boot):
                rs = runs if b == 0 else [block_bootstrap(r, rng) for r in runs]
                try:
                    members.append(fit(kind, spec, rs, sig_mult=sm, budget=8))
                except Exception:
                    pass
    y0 = runs[0].y0
    out = {}
    for name, U in cands.items():
        P = np.array([m.rollout(y0, U) for m in members])          # [M, T, p]
        med = np.median(P, axis=0)
        spread = np.abs(P - med[None]) / sigma[None, None, :]        # sigma units
        out[name] = float(np.mean(np.median(spread, axis=0)))       # robust spread, mean over ticks/obs
    return out, len(members)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=int, default=80)
    ap.add_argument("--kinds", default="l0b_lin,l1")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--systems", default=",".join(S.SYSTEM_IDS))
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    rng = np.random.default_rng(0)
    res = {}
    for sid in a.systems.split(","):
        spec = S.get(sid)
        d = data_dir(sid, False)
        runs = load_runs(spec, Ledger(d, sid, False))
        if not runs:
            continue
        t0 = time.time()
        ex = extrapolation(spec, runs, kinds, a.head)
        cands = candidates(spec, a.T, rng, d)
        dis, M = disagreement(spec, runs, kinds, cands, rng)
        res[sid] = {"extrapolation": ex, "disagreement": dis, "members": M}
        top = sorted(dis.items(), key=lambda kv: -kv[1])
        print(f"{sid:<17} extrap(head={a.head}) " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in ex.items())
              + f" | committee n={M} spread(sigma): " + " ".join(f"{k}={v:.2f}" for k, v in top) + f"  [{time.time()-t0:.0f}s]", flush=True)
    Path("docs/study").mkdir(parents=True, exist_ok=True)
    Path("docs/study/sim_next.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
