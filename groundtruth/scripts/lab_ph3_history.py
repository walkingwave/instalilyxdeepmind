"""Public-history consistency for one system (free, no gateway).

Every scored public upload Q is a probe with a known public score. Take a candidate model C as the
noiseless truth, roll C and every distinct Q on eval-like episodes (gtlab.check.make_episodes, pinned
lo/hi episodes dropped), score Q against C at the calibrated sigma, and compare implied vs public
scores (overall and sustained/sequence bands where the portal receipt exists). The candidate that
reproduces the history best is the most credible truth. A predictor scores 1.0 against itself, so
predictors identical to the truth are dropped from its statistics, and the common-set statistics drop
every predictor identical to ANY candidate.

Calibrated forecast of predictor X under truth C (C != X): public ~ a + b * implied fitted on the
common set, and the offset form implied(X) + mean(public - implied).

    python scripts/lab_ph3_history.py power_grid --cand final4=submissions/20260929-2000-final4 \
        --cand alt1b=submissions/20260929-1840-alt1b --cand w5=submissions/20260929-1823-u017 \
        --n 40 --seed 11 --out plans/ph_power_grid.json
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S  # noqa: E402
from gtlab.check import make_episodes  # noqa: E402

warnings.filterwarnings("ignore")
SEQ = ("order", "recovery", "composition")


def load(folder, sid, tag):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    if not p.exists():
        return None, "persistence"
    mj = p.parent / "model.json"
    h = hashlib.sha1(p.read_bytes() + (mj.read_bytes() if mj.exists() else b"")).hexdigest()[:8]
    sp = importlib.util.spec_from_file_location(f"ph3_{sid}_{tag}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod, h


def roll(mod, spec, ep):
    obs = list(spec.observables)
    if mod is None:
        return np.tile(np.array([ep["initial"][o] for o in obs], float), (len(ep["interventions"]), 1))
    out = mod.predict(dict(ep["initial"]), [dict(r) for r in ep["interventions"]], spec.context())
    Y = np.array([[row[o] for o in obs] for row in out], float)
    return np.where(np.isfinite(Y), Y, np.nan)


def score(Yh, Yt, sig):
    e = np.abs(np.nan_to_num(Yh, nan=1e12) - Yt)
    return float(np.mean(1.0 / (1.0 + e / sig[None, :])))


def stats(pairs):
    """pairs: list of (implied, public). Returns gap, corr, offset (public - implied), rmse."""
    if len(pairs) < 2:
        return None
    x, y = np.array(pairs, float).T
    return {"n": len(pairs), "mae": float(np.mean(np.abs(x - y))), "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "offset": float(np.mean(y - x)), "corr": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else float("nan"),
            "rank_corr": float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--cand", action="append", default=[], help="name=submissions/<dir>")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    a = ap.parse_args()
    sid = a.sid
    spec = S.get(sid)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    bands = {r["upload"]: r for r in json.loads((ROOT / "tune" / "bands.json").read_text())["rows"]
             if r["system"] == sid}

    # distinct scored history predictors, grouped by predictor hash
    hist = {}
    for u in reg:
        pub = (u.get("scores") or {}).get(sid)
        if pub is None:
            continue
        mod, h = load(u["dir"], sid, u["id"])
        g = hist.setdefault(h, {"mod": mod, "ids": [], "pub": [], "sus": [], "seq": []})
        g["ids"].append(u["id"])
        g["pub"].append(pub)
        if u["id"] in bands:
            g["sus"].append(bands[u["id"]]["sustained"])
            g["seq"].append(bands[u["id"]]["sequence"])
    for h, g in hist.items():
        g["name"] = "=".join(g["ids"]) if len(g["ids"]) <= 2 else f"{g['ids'][0]}..{g['ids'][-1]}"
        g["public"] = float(np.mean(g["pub"]))
        g["sustained"] = float(np.mean(g["sus"])) if g["sus"] else None
        g["sequence"] = float(np.mean(g["seq"])) if g["seq"] else None

    cands = {}
    for c in a.cand:
        name, d = c.split("=", 1)
        mod, h = load(d, sid, name)
        cands[name] = {"mod": mod, "hash": h, "dir": d}
    cand_hashes = {c["hash"] for c in cands.values()}

    eps, src = make_episodes(spec, n=a.n, T=4000, seed=a.seed)
    eps = [e for e in eps if not e["category"].startswith("pinned")]
    cats = [e["category"] for e in eps]
    print(f"{sid}: {len(eps)} episodes (init {src}), {len(hist)} distinct history predictors, sigma {sig}", flush=True)

    t0 = time.time()
    R = {}
    for h, g in hist.items():
        R[h] = [roll(g["mod"], spec, e) for e in eps]
    for name, c in cands.items():
        if c["hash"] not in R:
            R[c["hash"]] = [roll(c["mod"], spec, e) for e in eps]
    print(f"rollouts {time.time() - t0:.0f}s", flush=True)

    def implied(hq, ht):
        s = np.array([score(R[hq][i], R[ht][i], sig) for i in range(len(eps))])
        ci = np.array(cats)
        sus = float(s[ci == "sustained"].mean())
        seq = float(np.mean([s[ci == k].mean() for k in SEQ]))
        return {"overall": float(s.mean()), "sustained": sus, "sequence": seq, "cat_mean": 0.25 * sus + 0.75 * seq}

    truths = dict((n, c["hash"]) for n, c in cands.items())
    for h, g in hist.items():  # history predictors as truths too (reference points)
        if h not in truths.values():
            truths["hist:" + g["name"]] = h

    res = {"system": sid, "sigma": sig.tolist(), "n_episodes": len(eps), "seed": a.seed,
           "history": {g["name"]: {"hash": h, "public": g["public"], "sustained": g["sustained"],
                                   "sequence": g["sequence"]} for h, g in hist.items()},
           "candidates": {n: {"hash": c["hash"], "dir": c["dir"],
                              "in_history_as": next((g["name"] for h, g in hist.items() if h == c["hash"]), None)}
                          for n, c in cands.items()},
           "truths": {}}
    for tn, th in truths.items():
        rows = {}
        for h, g in hist.items():
            rows[g["name"]] = dict(implied(h, th), public=g["public"], sustained_pub=g["sustained"],
                                   sequence_pub=g["sequence"], hash=h)
        for cn, c in cands.items():
            rows.setdefault("cand:" + cn, dict(implied(c["hash"], th), hash=c["hash"]))
        others = [(r["cat_mean"], r["public"]) for r in rows.values() if "public" in r and r["hash"] != th]
        common = [(r["cat_mean"], r["public"]) for r in rows.values()
                  if "public" in r and r["hash"] not in cand_hashes and r["hash"] != th]
        band = [(r["sustained"], r["sustained_pub"]) for r in rows.values()
                if r.get("sustained_pub") is not None and r["hash"] not in cand_hashes and r["hash"] != th]
        band += [(r["sequence"], r["sequence_pub"]) for r in rows.values()
                 if r.get("sequence_pub") is not None and r["hash"] not in cand_hashes and r["hash"] != th]
        st = {"others": stats(others), "common": stats(common), "bands_common": stats(band)}
        # calibrated forecasts for each candidate that is not the truth
        fc = {}
        if len(common) >= 3:
            x, y = np.array(common, float).T
            b, a0 = np.polyfit(x, y, 1)
            off = float(np.mean(y - x))
            for cn, c in cands.items():
                if c["hash"] == th:
                    continue
                imp = rows["cand:" + cn]["cat_mean"] if ("cand:" + cn) in rows else \
                    next(r["cat_mean"] for r in rows.values() if r["hash"] == c["hash"])
                fc[cn] = {"implied": imp, "offset_fc": imp + off, "linear_fc": float(a0 + b * imp)}
        res["truths"][tn] = {"hash": th, "stats": st, "forecast": fc, "rows": rows}
        s = st["common"]
        print(f"truth {tn:28s} common n={s['n'] if s else 0} mae {s['mae'] if s else float('nan'):.3f} "
              f"corr {s['corr'] if s else float('nan'):.3f} off {s['offset'] if s else float('nan'):+.3f} | "
              + " ".join(f"{k}:{v['offset_fc']:.3f}/{v['linear_fc']:.3f}" for k, v in fc.items()), flush=True)
    if a.out:
        Path(ROOT / a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
