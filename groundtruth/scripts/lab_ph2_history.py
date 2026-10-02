"""Public-history consistency of candidate models (free, no gateway).

Take candidate C as the noiseless truth on 40 eval-like episodes (gtlab.check.make_episodes, T=4000),
score every past scored upload Q (deduplicated on predict.py+model.json) against it with the
calibrated sigma, and compare Q's implied score with its actual public score (registry) and bands
(tune/bands.json). The candidate closest to the truth should reproduce the public history best.
Every history predictor is also tried as truth (context). Implied scores of a truth against itself
are dropped.

Forecast of candidate X from truth C != X: implied(X | C) + mean(public - implied | C).

    python scripts/lab_ph2_history.py hospital_queue [--seed 11] [--n 40]
Writes plans/ph_<sys>_history.json.
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
from gtlab import systems as S                     # noqa: E402
from gtlab.check import make_episodes              # noqa: E402
from gtlab.runtime import infer                    # noqa: E402

warnings.filterwarnings("ignore")
CATS = ["sustained", "order", "recovery", "composition"]
CANDS = {"final4": "submissions/20260929-2000-final4", "alt1b": "submissions/20260929-1840-alt1b",
         "final1": "submissions/20260928-1440-final1"}
DOCS = {"hospital_queue": {"h9w1": "plans/hospital_queue_hospital_queue_hosp9_h9w1_doc.json",
                           "canon2": "plans/hospital_queue_canon_hospital_queue_canon2_hr_doc.json"}}


def fhash(folder, sid):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    if not p.exists():
        return None
    mj = p.parent / "model.json"
    return hashlib.sha1(p.read_bytes() + (mj.read_bytes() if mj.exists() else b"")).hexdigest()[:10]


def load(folder, sid, tag):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    sp = importlib.util.spec_from_file_location(f"ph2_{sid}_{tag}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sid = a.sid
    spec = S.get(sid)
    obs = list(spec.observables)
    ctx = spec.context()
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    eps, src = make_episodes(spec, n=a.n, T=4000, seed=a.seed)
    cat = np.array([e["category"] for e in eps])
    print(sid, "episodes", len(eps), "y0 src", src, {c: int((cat == c).sum()) for c in CATS}, flush=True)

    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    bands = {r["upload"]: r for r in json.loads((ROOT / "tune" / "bands.json").read_text())["rows"]
             if r["system"] == sid}
    # predictors keyed by hash
    P = {}
    for u in reg:
        pub = (u.get("scores") or {}).get(sid)
        if pub is None:
            continue
        h = fhash(u["dir"], sid)
        if h is None:
            if u["id"] != "u001":
                continue
            h = "persistence"
        g = P.setdefault(h, {"hash": h, "dir": u["dir"], "uploads": [], "pub": [], "sus": [], "seq": []})
        g["uploads"].append(u["id"])
        g["pub"].append(pub)
        if u["id"] in bands:
            g["sus"].append(bands[u["id"]]["sustained"])
            g["seq"].append(bands[u["id"]]["sequence"])
    hist = list(P.keys())
    for nm, d in CANDS.items():
        h = fhash(d, sid)
        g = P.setdefault(h, {"hash": h, "dir": d, "uploads": [], "pub": [], "sus": [], "seq": []})
        g.setdefault("cand", []).append(nm)
    for nm, f in DOCS.get(sid, {}).items():
        P["doc:" + nm] = {"hash": "doc:" + nm, "doc": f, "uploads": [], "pub": [], "sus": [], "seq": [],
                          "cand": [nm]}
    for h, g in P.items():
        g["name"] = "/".join(g.get("cand", []) + g["uploads"][-1:]) or h

    Y = {}
    for h, g in P.items():
        t0 = time.time()
        out = []
        if h == "persistence":
            for e in eps:
                y0 = np.array([e["initial"][o] for o in obs], float)
                out.append(np.repeat(y0[None, :], 4000, axis=0))
        elif "doc" in g:
            doc = json.loads((ROOT / g["doc"]).read_text())
            idx = [doc["observables"].index(o) for o in obs]
            for e in eps:
                y0 = np.array([e["initial"][o] for o in doc["observables"]], float)
                U = np.array([[r[c] for c in doc["controls"]] for r in e["interventions"]], float)
                out.append(np.asarray(infer.rollout_from_blob(doc, y0, U), float)[:, idx])
        else:
            mod = load(g["dir"], sid, h)
            for e in eps:
                res = mod.predict(dict(e["initial"]), [dict(r) for r in e["interventions"]], ctx)
                out.append(np.array([[r[o] for o in obs] for r in res], float))
        Y[h] = np.nan_to_num(np.stack(out), nan=1e12, posinf=1e12, neginf=-1e12)
        print(f"  rolled {g['name']:28s} {time.time() - t0:5.1f}s", flush=True)

    def sc(q, t):
        s = np.mean(1.0 / (1.0 + np.abs(Y[q] - Y[t]) / sig[None, None, :]), axis=(1, 2))
        per = {c: float(s[cat == c].mean()) for c in CATS}
        per["overall"] = float(np.mean([per[c] for c in CATS]))
        per["sequence"] = float(np.mean([per["order"], per["recovery"], per["composition"]]))
        return per

    res = {"system": sid, "seed": a.seed, "n": a.n, "y0_src": src, "predictors": {}, "truths": {}}
    for h, g in P.items():
        res["predictors"][h] = {k: g[k] for k in ("name", "uploads", "pub", "sus", "seq") if k in g}
        res["predictors"][h]["cand"] = g.get("cand", [])
    for t in P:
        rows = []
        for q in hist:
            if q == t:
                continue
            s = sc(q, t)
            g = P[q]
            rows.append({"q": q, "name": g["name"], "pub": float(np.mean(g["pub"])), "imp": s["overall"],
                         "imp_sus": s["sustained"], "imp_seq": s["sequence"],
                         "pub_sus": float(np.mean(g["sus"])) if g["sus"] else None,
                         "pub_seq": float(np.mean(g["seq"])) if g["seq"] else None})
        pub = np.array([r["pub"] for r in rows])
        imp = np.array([r["imp"] for r in rows])
        d = imp - pub
        bs = [(r["imp_sus"] - r["pub_sus"], r["imp_seq"] - r["pub_seq"]) for r in rows if r["pub_sus"] is not None]
        bs = np.array(bs) if bs else np.zeros((0, 2))
        # forecasts for every candidate predictor (not the truth itself)
        fc = {}
        for x, gx in P.items():
            if x == t or not gx.get("cand"):
                continue
            s = sc(x, t)
            fc["/".join(gx["cand"])] = {"imp": s["overall"], "cal": s["overall"] - float(d.mean()),
                                        "imp_sus": s["sustained"], "imp_seq": s["sequence"]}
        res["truths"][t] = {"name": P[t]["name"], "cand": P[t].get("cand", []), "nq": len(rows),
                            "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())),
                            "offset": float(d.mean()), "sd_gap": float(d.std()),
                            "corr": float(np.corrcoef(pub, imp)[0, 1]) if len(rows) > 2 else None,
                            "band_rmse_sus": float(np.sqrt((bs[:, 0] ** 2).mean())) if len(bs) else None,
                            "band_rmse_seq": float(np.sqrt((bs[:, 1] ** 2).mean())) if len(bs) else None,
                            "rows": rows, "forecast": fc}
    order = sorted(res["truths"], key=lambda t: res["truths"][t]["sd_gap"])
    print(f"\n{'truth':30s} nq   MAE  offset sd_gap  corr  bandS  bandQ")
    for t in order:
        r = res["truths"][t]
        print(f"{r['name']:30s} {r['nq']:2d} {r['mae']:.3f} {r['offset']:+.3f} {r['sd_gap']:.3f} "
              f"{r['corr']:+.2f} {r['band_rmse_sus'] or 0:.3f} {r['band_rmse_seq'] or 0:.3f}")
        if r["cand"]:
            for q in r["rows"]:
                print(f"     {q['name']:26s} pub {q['pub']:.3f} imp {q['imp']:.3f}")
            for k, v in r["forecast"].items():
                print(f"     -> {k:20s} imp {v['imp']:.3f} cal {v['cal']:.3f}")
    out = ROOT / "plans" / f"ph_{sid}_history{a.suffix}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
