"""Combine saved held-out fold predictions (plans/market_mkt9_<tag>_loo.json): members, mean, median, weighted.

    python scripts/lab_mkt9_ens.py y3 d3 z12 y3BC [--w 0.5]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs


def main():
    tags = [t for t in sys.argv[1:] if not t.startswith("--")]
    spec = S.get("market")
    runs = load_runs(spec, Ledger(data_dir("market", False), "market", False))
    sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["market"]["sigma"], float)
    reps = {t: json.loads((Path("plans") / f"market_mkt9_{t}_loo.json").read_text())["folds"] for t in tags}
    names = [r.exp for r in runs if all(r.exp in reps[t] for t in tags)]
    rmap = {r.exp: r for r in runs}
    preds = {t: {n: np.asarray(reps[t][n]["pred"]) for n in names} for t in tags}
    rows = {}
    for t in tags:
        rows[t] = {n: metric.score_per_obs(preds[t][n], rmap[n].Y, sigma) for n in names}
    for k in range(2, len(tags) + 1):
        for combo in itertools.combinations(tags, k):
            rows["mean(" + ",".join(combo) + ")"] = {
                n: metric.score_per_obs(np.mean([preds[t][n] for t in combo], 0), rmap[n].Y, sigma) for n in names}
            if k >= 3:
                rows["med(" + ",".join(combo) + ")"] = {
                    n: metric.score_per_obs(np.median([preds[t][n] for t in combo], 0), rmap[n].Y, sigma) for n in names}
    for w in (0.25, 0.33):
        for t in tags[1:]:
            rows[f"{tags[0]}+{w}{t}"] = {
                n: metric.score_per_obs((1 - w) * preds[tags[0]][n] + w * preds[t][n], rmap[n].Y, sigma) for n in names}
    short = [n.split(".")[1][:9] for n in names]
    print(f"{'model':<34}" + "".join(f"{s:>10}" for s in short) + f"{'LOO':>8}{'P/V/D':>18}")
    for key, d in rows.items():
        m = [d[n].mean() for n in names]
        pvd = np.mean([d[n] for n in names], 0)
        print(f"{key[:34]:<34}" + "".join(f"{x:>10.3f}" for x in m) + f"{np.mean(m):>8.3f}   {pvd[0]:.3f}/{pvd[1]:.3f}/{pvd[2]:.3f}")


if __name__ == "__main__":
    main()
