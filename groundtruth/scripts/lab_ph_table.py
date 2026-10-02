"""Pool plans/ph_<sys>.json over seeds and print the consistency table + final4 forecasts.

    python scripts/lab_ph_table.py <system> [--seeds 11,23] [--clone 0.93] [--ok 0.06]

Forecasts of final4 under truth C (C not final4 or a near-clone of it, mutual implied > --clone):
  offset:  S_C(final4) + mean_top(pub - S_C)               (bias on strong past uploads)
  anchor:  pub(A) + S_C(final4) - S_C(A), A = best public model, C also not A or a clone of A
Consistent truths: pooled gap_top <= --ok.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--seeds", default="11,23")
    ap.add_argument("--clone", type=float, default=0.93)
    ap.add_argument("--ok", type=float, default=0.06)
    a = ap.parse_args()
    sid = a.system
    J = []
    for s in a.seeds.split(","):
        suf = "" if s == "11" else f"_s{s}"
        J.append(json.loads((ROOT / "plans" / f"ph_{sid}{suf}.json").read_text()))
    M = J[0]["models"]
    hs = list(M)
    imp = {t: {q: float(np.mean([j["truths"][t]["implied"][q]["all"] for j in J])) for q in hs} for t in hs}
    scored = [h for h in hs if M[h]["pub"] is not None]
    f4 = next(h for h in hs if "final4" in M[h]["cands"])
    A = max(scored, key=lambda h: M[h]["pub"])
    clone = lambda x, y: x == y or min(imp[x][y], imp[y][x]) > a.clone
    rows = []
    for t in hs:
        hist = [h for h in scored if h != t]
        x = np.array([imp[t][h] for h in hist]); y = np.array([M[h]["pub"] for h in hist])
        top = y >= 0.7
        r = {"truth": M[t]["name"], "gap": np.mean(np.abs(y - x)), "corr": np.corrcoef(x, y)[0, 1],
             "off": np.mean(y - x), "gap_top": np.mean(np.abs(y - x)[top]), "off_top": np.mean((y - x)[top]),
             "gap_nc": np.mean([abs(M[h]["pub"] - imp[t][h]) for h in hist if M[h]["pub"] >= 0.7 and not clone(t, h)] or [np.nan]),
             "S_f4": imp[t][f4], "S_A": imp[t][A], "fc_off": None, "fc_anc": None}
        if not clone(t, f4):
            r["fc_off"] = imp[t][f4] + r["off_top"]
            if not clone(t, A):
                r["fc_anc"] = M[A]["pub"] + imp[t][f4] - imp[t][A]
        rows.append(r)
    print(f"## {sid}  (pooled seeds {a.seeds}, {sum(j['n_eps'] for j in J)} episodes; A = {M[A]['name']} pub {M[A]['pub']:.4f})")
    print("| truth C | gap | corr | offset | gap (pub>=0.7) | offset (pub>=0.7) | gap (pub>=0.7, no clones of C) | S_C(final4) | S_C(A) | fc offset | fc anchor |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda v: "-" if v is None else f"{v:.3f}"
    for r in sorted(rows, key=lambda r: r["gap_top"]):
        print(f"| {r['truth']} | {r['gap']:.3f} | {r['corr']:.3f} | {r['off']:+.3f} | {r['gap_top']:.3f} | "
              f"{r['off_top']:+.3f} | {r['gap_nc']:.3f} | {r['S_f4']:.3f} | {r['S_A']:.3f} | {f(r['fc_off'])} | {f(r['fc_anc'])} |")
    ok = [r for r in rows if r["gap_top"] <= a.ok]
    fo = [r["fc_off"] for r in ok if r["fc_off"] is not None]
    fa = [r["fc_anc"] for r in ok if r["fc_anc"] is not None]
    print(f"\nconsistent truths (gap_top <= {a.ok}): {[r['truth'] for r in ok]}")
    print(f"final4 forecast: offset {np.mean(fo) if fo else float('nan'):.3f} (n={len(fo)}, range "
          f"{min(fo) if fo else 0:.3f}-{max(fo) if fo else 0:.3f}); anchor {np.mean(fa) if fa else float('nan'):.3f} (n={len(fa)})")
    print("pairwise implied (row = truth):")
    names = [M[h]["name"][:14] for h in hs]
    print("            " + " ".join(f"{n:>14s}" for n in names))
    for t in hs:
        print(f"{M[t]['name'][:12]:12s}" + " ".join(f"{imp[t][q]:14.3f}" for q in hs))


if __name__ == "__main__":
    main()
