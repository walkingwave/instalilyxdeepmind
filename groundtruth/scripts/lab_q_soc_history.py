"""Public-history judge for cheap social_contagion combinations (free, no gateway).

Same method as lab_ph2_history.py / lab_ph2_analyze.py, with u019 (final4 = median z20,z21,canon3,
public 0.6358) in the history and combinations built from rollouts of existing fits:
components z20 (u016), y10 BC (u015), x7 (u013), z21 and canon3 (members of final4 model.json).

For every truth C (each distinct history predictor + z21 + canon3):
  implied score s_hat_Q(C) of each scored history predictor Q (self and identical dropped),
  offset fit pub = imp - d, affine fit pub = a + b imp, residual SDs.
  Forecast of candidate X: a + b s_hat_X(C); delta vs z20: b (s_hat_X(C) - s_hat_z20(C)).
Pooled over truths with weights 1/sd_aff^2 and also with likelihood weights exp(-SSE/2 s^2).
Backtest: hide each scored predictor, refit, forecast it and its delta vs z20.

    python scripts/lab_q_soc_history.py [--seed 11] [--n 40]
Writes plans/q_soc_history_s<seed>.json.
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
SID = "social_contagion"
CATS = ["sustained", "order", "recovery", "composition"]
F4 = "submissions/20260929-2000-final4"
COMP = {"z20": "submissions/20260928-1430-u016", "y10": "submissions/20260927-2151-u015",
        "x7": "submissions/20260927-0233-u013"}


def fhash(folder):
    p = ROOT / folder.replace("\\", "/") / SID / "predict.py"
    if not p.exists():
        return None
    mj = p.parent / "model.json"
    return hashlib.sha1(p.read_bytes() + (mj.read_bytes() if mj.exists() else b"")).hexdigest()[:10]


def load(folder, tag):
    p = ROOT / folder.replace("\\", "/") / SID / "predict.py"
    sp = importlib.util.spec_from_file_location(f"qsoc_{tag}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    spec = S.get(SID)
    obs = list(spec.observables)
    ctx = spec.context()
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[SID]["sigma"], float)
    eps, src = make_episodes(spec, n=a.n, T=4000, seed=a.seed)
    cat = np.array([e["category"] for e in eps])
    Y0 = np.stack([np.array([e["initial"][o] for o in obs], float) for e in eps])   # [E, p]
    print("obs", obs, "episodes", len(eps), "src", src, flush=True)

    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    P = {}
    for u in reg:
        pub = (u.get("scores") or {}).get(SID)
        if pub is None:
            continue
        h = fhash(u["dir"])
        if h is None:
            if u["id"] != "u001":
                continue
            h = "persistence"
        g = P.setdefault(h, {"dir": u["dir"], "uploads": [], "pub": []})
        g["uploads"].append(u["id"])
        g["pub"].append(pub)
    hist = list(P)
    for h, g in P.items():
        g["name"] = g["uploads"][-1] if h != "persistence" else "u001persist"

    def roll_predict(d, tag):
        mod = load(d, tag)
        out = []
        for e in eps:
            res = mod.predict(dict(e["initial"]), [dict(x) for x in e["interventions"]], ctx)
            out.append(np.array([[r[o] for o in obs] for r in res], float))
        return np.stack(out)

    def roll_doc(doc):
        idx = [doc["observables"].index(o) for o in obs]
        out = []
        for e in eps:
            y0 = np.array([e["initial"][o] for o in doc["observables"]], float)
            U = np.array([[r[c] for c in doc["controls"]] for r in e["interventions"]], float)
            out.append(np.asarray(infer.rollout_from_blob(doc, y0, U), float)[:, idx])
        return np.stack(out)

    Y = {}
    for h, g in P.items():
        t0 = time.time()
        if h == "persistence":
            Y[h] = np.repeat(Y0[:, None, :], 4000, axis=1)
        else:
            Y[h] = roll_predict(g["dir"], h)
        print(f"  rolled {g['name']:14s} {time.time() - t0:5.1f}s", flush=True)
    # components
    C = {}
    for k, d in COMP.items():
        C[k] = Y[fhash(d)]
    f4 = json.loads((ROOT / F4 / SID / "model.json").read_text())
    for m in f4["model"]["members"]:
        k = m["family"].split("_")[-1]
        if k in C:
            continue
        doc = dict(f4)
        doc["model"] = m
        t0 = time.time()
        C[k] = roll_doc(doc)
        print(f"  rolled member {k:8s} {time.time() - t0:5.1f}s", flush=True)
    C["final4"] = Y[fhash(F4)]
    med3 = np.median(np.stack([C["z20"], C["z21"], C["canon3"]]), axis=0)
    print("check final4 == median(z20,z21,canon3): max abs diff",
          float(np.abs(med3 - C["final4"]).max()), flush=True)

    ia = obs.index("adopters_a")
    ib = obs.index("adopters_b")

    def perobs(Ya, Yb):
        Z = Yb.copy()
        Z[..., ia] = Ya[..., ia]
        return Z

    def blend(M, lam):
        return Y0[:, None, :] + lam * (M - Y0[:, None, :])

    z20, z21, c3, y10, x7 = C["z20"], C["z21"], C["canon3"], C["y10"], C["x7"]
    CAND = {
        "z20": z20, "z21": z21, "canon3": c3, "y10": y10, "x7": x7, "final4": C["final4"],
        "med(z20,z21)": 0.5 * (z20 + z21),
        "med(z20,y10,x7)": np.median(np.stack([z20, y10, x7]), axis=0),
        "med(z20,y10,canon3)": np.median(np.stack([z20, y10, c3]), axis=0),
        "med(z20,z21,y10)": np.median(np.stack([z20, z21, y10]), axis=0),
        "mean(z20,y10)": 0.5 * (z20 + y10),
        "perobs a=z21,b=z20": perobs(z21, z20),
        "perobs a=med(z20,z21),b=z20": perobs(0.5 * (z20 + z21), z20),
        "perobs a=z20,b=med(z20,z21)": perobs(z20, 0.5 * (z20 + z21)),
        "blend z20 0.95": blend(z20, 0.95),
        "blend z20 0.9": blend(z20, 0.9),
        "blend z20 1.05": blend(z20, 1.05),
    }
    # truths: every distinct history predictor + z21 + canon3
    T = {h: Y[h] for h in hist}
    T["z21"] = z21
    T["canon3"] = c3
    tname = {h: P[h]["name"] for h in hist}
    tname["z21"], tname["canon3"] = "z21", "canon3"

    def sc(A, B):
        s = np.mean(1.0 / (1.0 + np.abs(A - B) / sig[None, None, :]), axis=(1, 2))
        return float(np.mean([s[cat == c].mean() for c in CATS]))

    imp = {t: {q: sc(Y[q], T[t]) for q in hist if q != t} for t in T}
    impc = {t: {x: sc(M, T[t]) for x, M in CAND.items()} for t in T}
    pubs = {h: float(np.mean(P[h]["pub"])) for h in hist}

    def fit(t, excl=()):
        qs = [q for q in imp[t] if q not in excl and imp[t][q] < 0.999]
        pub = np.array([pubs[q] for q in qs])
        im = np.array([imp[t][q] for q in qs])
        d = float(np.mean(im - pub))
        b, a0 = np.polyfit(im, pub, 1)
        ra = a0 + b * im - pub
        return {"d": d, "a": float(a0), "b": float(b), "sd_off": float(np.std(im - d - pub)),
                "sd_aff": float(np.std(ra)), "sse": float(np.sum(ra ** 2)), "n": len(qs)}

    res = {"system": SID, "seed": a.seed, "n": a.n, "src": src, "pubs": {P[h]["name"]: pubs[h] for h in hist},
           "truths": {}, "cands": {}, "backtest": {}}
    F = {t: fit(t) for t in T}
    print(f"\n{'truth':14s}  n   off     sd_off  a      b     sd_aff")
    for t in sorted(T, key=lambda t: F[t]["sd_aff"]):
        f = F[t]
        res["truths"][tname[t]] = f
        print(f"{tname[t]:14s} {f['n']:2d} {f['d']:+.3f}  {f['sd_off']:.3f}  {f['a']:+.3f} {f['b']:.3f}  {f['sd_aff']:.3f}")

    s_ref = np.median([F[t]["sd_aff"] for t in T])

    def pooled(vals, t_list, fits):
        w1 = np.array([1 / max(fits[t]["sd_aff"], 1e-3) ** 2 for t in t_list])
        ll = np.array([-0.5 * fits[t]["sse"] / s_ref ** 2 for t in t_list])
        w2 = np.exp(ll - ll.max())
        v = np.array(vals)
        return float(np.sum(v * w1) / w1.sum()), float(np.sum(v * w2) / w2.sum())

    tl = list(T)
    print(f"\n{'candidate':30s} aff_pool  lik_pool | d_vs_z20 aff / lik | min..max delta over truths")
    for x in CAND:
        fa = [F[t]["a"] + F[t]["b"] * impc[t][x] for t in tl]
        dz = [F[t]["b"] * (impc[t][x] - impc[t]["z20"]) for t in tl]
        pa, pl = pooled(fa, tl, F)
        da, dl = pooled(dz, tl, F)
        res["cands"][x] = {"aff_pool": pa, "lik_pool": pl, "dz_aff": da, "dz_lik": dl,
                           "by_truth": {tname[t]: {"imp": impc[t][x], "aff": fa[i], "dz": dz[i]}
                                        for i, t in enumerate(tl)}}
        print(f"{x:30s} {pa:.4f}   {pl:.4f}  | {da:+.4f} / {dl:+.4f} | {min(dz):+.4f}..{max(dz):+.4f}")

    # backtest: hide each scored predictor X, refit, forecast X and delta vs z20 (z20 must be known)
    hz = fhash(COMP["z20"])
    print("\nbacktest (hide X; forecast level and delta vs z20):")
    errs, derrs = [], []
    for x in hist:
        tt = [t for t in T if t != x]
        Fx = {t: fit(t, excl=(x,)) for t in tt}
        fl = [Fx[t]["a"] + Fx[t]["b"] * imp[t][x] for t in tt]
        pa, pl = pooled(fl, tt, Fx)
        row = {"pub": pubs[x], "aff_pool": pa, "lik_pool": pl}
        if x != hz:
            tt2 = [t for t in tt if t != hz]
            dd = [Fx[t]["b"] * (imp[t][x] - imp[t][hz]) for t in tt2]
            da, dl = pooled(dd, tt2, Fx)
            row.update({"d_pub": pubs[x] - pubs[hz], "d_aff": da, "d_lik": dl})
            if pubs[x] > 0.6:
                derrs.append((da - row["d_pub"], dl - row["d_pub"]))
        errs.append((pa - pubs[x], pl - pubs[x]))
        res["backtest"][P[x]["name"]] = row
        extra = (f"  delta pub {row['d_pub']:+.4f} fc {row['d_aff']:+.4f}/{row['d_lik']:+.4f}"
                 if "d_pub" in row else "")
        print(f"   {P[x]['name']:12s} pub {pubs[x]:.4f} fc {pa:.4f}/{pl:.4f}{extra}")
    e = np.array(errs)
    de = np.array(derrs)
    res["backtest"]["_summary"] = {"level_mae": np.abs(e).mean(0).tolist(), "level_bias": e.mean(0).tolist(),
                                   "delta_top_mae": np.abs(de).mean(0).tolist(),
                                   "delta_top_bias": de.mean(0).tolist(), "delta_top_n": len(de)}
    print("   level MAE aff/lik", np.abs(e).mean(0).round(4), "bias", e.mean(0).round(4))
    print("   delta vs z20 (pub>0.6) MAE aff/lik", np.abs(de).mean(0).round(4), "bias", de.mean(0).round(4))
    out = ROOT / "plans" / f"q_soc_history_s{a.seed}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
