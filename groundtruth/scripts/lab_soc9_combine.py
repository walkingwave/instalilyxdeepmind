"""Score fold ensembles from lab_soc9_loo outputs.

    python scripts/lab_soc9_combine.py z20 y10AB [y10BC ...]      # each tag alone, all pairs/triples (mean, median)
    python scripts/lab_soc9_combine.py --bag z20                    # nested bagging from the drop2 fits
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_soc9_loo import OUT, setup  # noqa: E402
from gtlab import metric  # noqa: E402

spec, runs, sigma = setup()
K = len(runs)


def load(tag):
    res = {}
    for t in tag.split(","):
        res.update(np.load(OUT / f"{t}.npz", allow_pickle=True)["res"].item())
    return res


def row(name, P):
    s = [metric.score_per_obs(P[k], runs[k].Y, sigma) for k in range(K)]
    m = np.mean(s)
    folds = " ".join(f"{s[k].mean():.3f}" for k in range(K))
    ob = " ".join(f"{s[k][0]:.2f}/{s[k][1]:.2f}" for k in range(K))
    print(f"{name:<28} LOO {m:.4f} | {folds} | {ob}")
    return m


args = sys.argv[1:]
print("folds: " + " ".join(r.exp for r in runs))
if args and args[0] == "--bag":
    for tag in args[1:]:
        R = load(tag)
        base = [R[str(k)]["preds"][k] for k in range(K)]
        row(f"{tag} P0", base)
        mem = {}
        for k in range(K):
            ms = [R[str(k)]["preds"][k]]
            for j in range(K):
                if j == k:
                    continue
                key = ",".join(map(str, sorted((k, j))))
                if key in R:
                    ms.append(R[key]["preds"][k])
            mem[k] = np.stack(ms)
        print("members per fold", [len(mem[k]) for k in range(K)])
        row(f"{tag} bag mean", [mem[k].mean(0) for k in range(K)])
        row(f"{tag} bag median", [np.median(mem[k], 0) for k in range(K)])
        # mean of 4 (P0 + 3 random members, fixed seed) = shippable balanced tree
        rng = np.random.default_rng(0)
        for n in (2, 4):
            P = []
            for k in range(K):
                idx = [0] + list(rng.choice(np.arange(1, len(mem[k])), n - 1, replace=False))
                P.append(mem[k][idx].mean(0))
            row(f"{tag} bag mean{n} (rand)", P)
    sys.exit(0)

Rs = {t: load(t) for t in args}
F = {t: [Rs[t][str(k)]["preds"][k] for k in range(K)] for t in args}
for t in args:
    row(t, F[t])
for n in (2, 3):
    for combo in itertools.combinations(args, n):
        row("mean(" + "+".join(combo) + ")", [np.mean([F[t][k] for t in combo], 0) for k in range(K)])
        if n == 3:
            row("med(" + "+".join(combo) + ")", [np.median([F[t][k] for t in combo], 0) for k in range(K)])
