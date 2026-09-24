"""Dry-run the leaderboard loop against mock simulators (zero credits, no portal).

The "leaderboard" is a mock oracle: each proposed ZIP is scored per system against the mock's
noiseless truth on 40 fixed eval-like episodes (fixed like the real public set).
    python scripts/sim_autotune.py --data-root /tmp/mockroot --rounds 6
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import autotune as AT, systems as S, metric, design
from gtlab.mocks.base import MockSystem
from gtlab.runtime import infer as rt


def episodes(sid, n_per_cat=10, T=4000):
    spec = S.get(sid)
    world = MockSystem(sid, mech=("A", "B"), seed=0)
    rng = np.random.default_rng(2026)
    eps = []
    for cat in ("sustained", "order", "recovery", "composition"):
        for _ in range(n_per_cat):
            U = design.eval_like(spec, cat, T, rng)
            y0 = world.sample_y0(rng)
            Yt, _ = world.simulate(y0, U)
            eps.append((world._noisy(y0), U, Yt))
    sig = np.concatenate([e[2] for e in eps]).std(axis=0) + 1e-9
    return eps, sig


def oracle(build_dir, sid, cache):
    if sid not in cache:
        cache[sid] = episodes(sid, n_per_cat=3, T=2000)
    eps, sig = cache[sid]
    folder = Path(build_dir) / sid
    doc = json.loads((folder / "model.json").read_text())
    sc = [metric.score(rt.rollout_from_blob(doc, y0, U, doc=doc, base_dir=folder), Yt, sig)
          for y0, U, Yt in eps]
    return float(np.mean(sc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    AT.MOCK = True
    AT.TUNE = Path(a.data_root) / "tune_sim"
    AT.REG = AT.TUNE / "registry.json"
    shutil.rmtree(AT.TUNE, ignore_errors=True)
    cache = {}
    for r in range(a.rounds):
        day = f"2026-09-{23 + r // 3}"
        AT.toronto_today = lambda day=day: day
        rc = AT.main(["--data-root", a.data_root, "propose", "--no-check"])
        reg = AT.load()
        u = reg["uploads"][-1]
        if u["uploaded"]:
            print("nothing new; stop")
            break
        u["uploaded"] = True
        u["scores"] = {sid: oracle(AT.ROOT / u["dir"], sid, cache) for sid in u["systems"]}
        AT.save(reg)
        print(f"round {r}: {u['id']} mean over included = {np.mean(list(u['scores'].values())):.4f}\n")
    AT.main(["status"])
    reg = AT.load()
    best = {sid: max((s for c, s in AT.history(reg, sid)), default=0) for sid in S.SYSTEM_IDS}
    print("best-per-system mean:", round(float(np.mean(list(best.values()))), 4))
