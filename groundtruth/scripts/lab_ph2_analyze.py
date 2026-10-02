"""Second pass over plans/ph_<sys>_history.json (from lab_ph2_history.py). No rollouts.

Per truth C: offset fit (pub = imp - d) and affine fit (pub = a + b imp) of the public history;
residual SD of each. Forecast of each candidate X from each truth C != X with both fits.
Backtest: for each scored history predictor X, hide it, fit on the rest, forecast it from every
truth C != X; report the error of each forecast rule (single truth, and pooled over truths
weighted by 1/residual variance), overall and on the top-3 public predictors.

    python scripts/lab_ph2_analyze.py hospital_queue social_contagion wildlife
Writes plans/ph_<sys>_analysis.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def fits(rows, excl=()):
    rr = [r for r in rows if r["q"] not in excl]
    pub = np.array([r["pub"] for r in rr])
    imp = np.array([r["imp"] for r in rr])
    d = float(np.mean(imp - pub))
    b, a = np.polyfit(imp, pub, 1)
    sd_off = float(np.std(imp - d - pub))
    sd_aff = float(np.std(a + b * imp - pub))
    return d, float(a), float(b), sd_off, sd_aff


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--suffix=")]
    suf = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--suffix=")), "")
    for sid in args:
        R = json.loads((ROOT / "plans" / f"ph_{sid}_history{suf}.json").read_text())
        P, T = R["predictors"], R["truths"]
        hist = [h for h, g in P.items() if g["pub"]]
        pubs = {h: float(np.mean(P[h]["pub"])) for h in hist}
        imp = {t: {r["q"]: r["imp"] for r in T[t]["rows"]} for t in T}
        out = {"system": sid, "truths": {}, "forecast": {}, "backtest": {}}
        print(f"\n=== {sid}")
        print(f"{'truth':28s}  off    sd_off  a      b     sd_aff")
        for t in sorted(T, key=lambda t: fits(T[t]["rows"])[4]):
            d, a, b, so, sa = fits(T[t]["rows"])
            out["truths"][t] = {"name": T[t]["name"], "offset": d, "a": a, "b": b, "sd_off": so, "sd_aff": sa}
            print(f"{T[t]['name']:28s} {d:+.3f}  {so:.3f}  {a:+.3f} {b:.3f}  {sa:.3f}")
        # candidate forecasts
        for t in T:
            d, a, b, so, sa = fits(T[t]["rows"])
            for x, f in T[t]["forecast"].items():
                out["forecast"].setdefault(x, {})[T[t]["name"]] = {
                    "imp": f["imp"], "off": f["imp"] - d, "aff": a + b * f["imp"], "sd_off": so, "sd_aff": sa}
        # backtest over scored history predictors
        errs = {"off_best": [], "aff_best": [], "off_pool": [], "aff_pool": []}
        top = sorted(hist, key=lambda h: -pubs[h])[:3]
        rows_bt = []
        for x in hist:
            fo, fa, wo, wa = [], [], [], []
            for t in T:
                if t == x or x not in imp[t]:
                    continue
                if len([r for r in T[t]["rows"] if r["q"] != x]) < 4:
                    continue
                d, a, b, so, sa = fits(T[t]["rows"], excl=(x,))
                fo.append(imp[t][x] - d); wo.append(1 / max(so, 1e-3) ** 2)
                fa.append(a + b * imp[t][x]); wa.append(1 / max(sa, 1e-3) ** 2)
            fo, fa, wo, wa = map(np.array, (fo, fa, wo, wa))
            po = float(np.sum(fo * wo) / wo.sum())
            pa = float(np.sum(fa * wa) / wa.sum())
            bo = float(fo[np.argmax(wo)])
            ba = float(fa[np.argmax(wa)])
            e = {"off_best": bo - pubs[x], "aff_best": ba - pubs[x], "off_pool": po - pubs[x], "aff_pool": pa - pubs[x]}
            for k in errs:
                errs[k].append((x in top, e[k]))
            rows_bt.append((P[x]["name"], pubs[x], po, pa))
            out["backtest"][x] = {"name": P[x]["name"], "pub": pubs[x], **{k: v + pubs[x] for k, v in e.items()}}
        print("backtest (forecast - pub):")
        for nm, pb, po, pa in rows_bt:
            print(f"   {nm:26s} pub {pb:.3f}  pooled-off {po:.3f}  pooled-aff {pa:.3f}")
        for k, v in errs.items():
            allv = np.array([e for _, e in v])
            tv = np.array([e for t_, e in v if t_])
            out["backtest"][f"_{k}"] = {"mae": float(np.abs(allv).mean()), "bias": float(allv.mean()),
                                        "top3_bias": float(tv.mean()), "top3_mae": float(np.abs(tv).mean())}
            print(f"   rule {k:9s} MAE {np.abs(allv).mean():.3f} bias {allv.mean():+.3f}  top3 bias {tv.mean():+.3f}")
        print("candidate forecasts (off / aff), by truth:")
        for x, fd in out["forecast"].items():
            ws = np.array([1 / v["sd_aff"] ** 2 for v in fd.values()])
            fa = np.array([v["aff"] for v in fd.values()])
            wo = np.array([1 / v["sd_off"] ** 2 for v in fd.values()])
            fo = np.array([v["off"] for v in fd.values()])
            fd["_pooled"] = {"off": float(np.sum(fo * wo) / wo.sum()), "aff": float(np.sum(fa * ws) / ws.sum())}
            print(f"   {x:12s} " + "  ".join(f"{t}:{v['off']:.3f}/{v['aff']:.3f}" for t, v in fd.items()
                                          if t != "_pooled")
                  + f"  | pooled {fd['_pooled']['off']:.3f}/{fd['_pooled']['aff']:.3f}")
        (ROOT / "plans" / f"ph_{sid}_analysis{suf}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
