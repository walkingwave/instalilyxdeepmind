"""Offset-free and pairwise summaries of plans/ph_<sys>.json (from lab_ph3_history.py).

For each truth: residual SD after removing the mean offset, RMSE after a linear fit, and the pairwise
test: for every pair of history predictors (none identical to the truth) compare implied delta with
public delta (RMSE, sign agreement). 'top' restricts to pairs among the best public predictors, where
the decision lives.

    python scripts/lab_ph3_pairs.py plans/ph_power_grid.json [--top 4]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def summarize(path, top):
    res = json.loads((ROOT / path).read_text())
    hist = res["history"]
    best = sorted(hist, key=lambda k: -hist[k]["public"])[:top]
    cand_h = {c["hash"] for c in res["candidates"].values()}
    out = {}
    print(f"== {res['system']}  top public: " + ", ".join(f"{k} {hist[k]['public']:.4f}" for k in best))
    for tn, T in res["truths"].items():
        rows = {k: v for k, v in T["rows"].items() if "public" in v and v["hash"] != T["hash"]}
        com = {k: v for k, v in rows.items() if v["hash"] not in cand_h}
        x = np.array([v["cat_mean"] for v in com.values()])
        y = np.array([v["public"] for v in com.values()])
        sd = float(np.std(y - x))
        b, a0 = np.polyfit(x, y, 1)
        lin = float(np.sqrt(np.mean((a0 + b * x - y) ** 2)))
        pairs = list(itertools.combinations(rows, 2))
        dd = [((rows[i]["cat_mean"] - rows[j]["cat_mean"]), (rows[i]["public"] - rows[j]["public"])) for i, j in pairs]
        d_rmse = float(np.sqrt(np.mean([(p - q) ** 2 for p, q in dd])))
        sign = float(np.mean([np.sign(p) == np.sign(q) for p, q in dd if abs(q) > 1e-3]))
        tp = [(i, j) for i, j in pairs if i in best and j in best]
        tdd = {f"{i}-{j}": (rows[i]["cat_mean"] - rows[j]["cat_mean"], rows[i]["public"] - rows[j]["public"]) for i, j in tp}
        t_rmse = float(np.sqrt(np.mean([(p - q) ** 2 for p, q in tdd.values()]))) if tdd else float("nan")
        t_sign = float(np.mean([np.sign(p) == np.sign(q) for p, q in tdd.values() if abs(q) > 1e-3])) if tdd else float("nan")
        # leave-one-out linear forecast of each common predictor's public score (frontier = top public)
        loo = {}
        keys = list(com)
        for k in keys:
            xs = np.array([com[j]["cat_mean"] for j in keys if j != k])
            ys = np.array([com[j]["public"] for j in keys if j != k])
            bb, aa = np.polyfit(xs, ys, 1)
            loo[k] = float(aa + bb * com[k]["cat_mean"] - com[k]["public"])
        fr = [loo[k] for k in keys if k in best]
        # known candidates predicted from this truth (full common-set linear calibration)
        known = {}
        for cn, c in res["candidates"].items():
            if c["hash"] == T["hash"] or not c.get("in_history_as"):
                continue
            imp = T["rows"]["cand:" + cn]["cat_mean"] if "cand:" + cn in T["rows"] else None
            if imp is None:
                imp = next(v["cat_mean"] for v in T["rows"].values() if v["hash"] == c["hash"])
            known[cn] = {"forecast": float(a0 + b * imp), "public": hist[c["in_history_as"]]["public"]}
        fcs = {cn: v["linear_fc"] for cn, v in T["forecast"].items()}
        out[tn] = {"loo_rmse": float(np.sqrt(np.mean(np.square(list(loo.values()))))),
                   "loo_frontier_rmse": float(np.sqrt(np.mean(np.square(fr)))) if fr else float("nan"),
                   "loo": loo, "known_candidates": known, "linear_forecasts": fcs}
        print(f"  {tn:26s} LOO rmse {out[tn]['loo_rmse']:.3f} frontier {out[tn]['loo_frontier_rmse']:.3f} | "
              + " ".join(f"{cn} fc {v['forecast']:.4f} vs pub {v['public']:.4f}" for cn, v in known.items())
              + " | lin fc " + " ".join(f"{k}:{v:.4f}" for k, v in fcs.items()))
        out[tn].update({"resid_sd": sd, "lin_rmse": lin, "slope": float(b), "pair_rmse": d_rmse, "pair_sign": sign,
                   "top_pair_rmse": t_rmse, "top_pair_sign": t_sign, "top_pairs": tdd, "n_common": len(com)})
        print(f"  {tn:26s} resid_sd {sd:.3f} lin_rmse {lin:.3f} slope {b:+.2f} | pairs rmse {d_rmse:.3f} "
              f"sign {sign:.2f} | top rmse {t_rmse:.3f} sign {t_sign:.2f} | "
              + " ".join(f"{k}:{p:+.3f}/{q:+.3f}" for k, (p, q) in tdd.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    allo = {f: summarize(f, a.top) for f in a.files}
    Path(ROOT / "plans" / "ph_pairs_sup_mkt_pg_epi.json").write_text(json.dumps(allo, indent=1))


if __name__ == "__main__":
    main()
