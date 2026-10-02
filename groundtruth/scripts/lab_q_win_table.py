"""Pool plans/q_win_ph_<sys>[_sN].json and forecast every candidate against the u019 public score.

    python scripts/lab_q_win_table.py <system> [--seeds 11,23] [--clone 0.93] [--ok 0.06]

Truth set: consistent truths C (pooled gap on strong uploads, pub >= 0.7, <= --ok).
Per candidate X (u019 = shipped winner, now a scored upload):
  offset : S_C(X) + mean_top(pub - S_C)                          C not X, not a clone of X
  anchor : pub(u019) + S_C(X) - S_C(u019)                        C not X/u019 and no clone of either (strict)
  anchorL: same, C only != X and != u019 (loose: clones allowed as truth)
  slope  : pub(u019) + b_C (S_C(X) - S_C(u019)), b_C = slope of pub on implied over past uploads (loose set)
"""
import argparse
import json
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
        p = ROOT / "plans" / f"q_win_ph_{sid}{suf}.json"
        if p.exists():
            J.append(json.loads(p.read_text()))
    M = J[0]["models"]
    hs = list(M)
    imp = {t: {q: float(np.mean([j["truths"][t]["implied"][q]["all"] for j in J])) for q in hs} for t in hs}
    scored = [h for h in hs if M[h]["pub"] is not None]
    W = next(h for h in hs if "u019" in M[h]["cands"])
    pubW = M[W]["pub"]
    clone = lambda x, y: x == y or min(imp[x][y], imp[y][x]) > a.clone
    stats = {}
    for t in hs:
        hist = [h for h in scored if h != t]
        x = np.array([imp[t][h] for h in hist]); y = np.array([M[h]["pub"] for h in hist])
        top = y >= 0.7
        b = np.polyfit(x, y, 1)[0] if len(hist) >= 3 else 1.0
        stats[t] = {"gap_top": float(np.mean(np.abs(y - x)[top])), "off_top": float(np.mean((y - x)[top])),
                    "corr": float(np.corrcoef(x, y)[0, 1]), "b": float(b)}
    ok = [t for t in hs if stats[t]["gap_top"] <= a.ok]
    print(f"## {sid}  seeds {a.seeds} ({sum(j['n_eps'] for j in J)} eps); u019 = {M[W]['name']} pub {pubW:.4f}")
    print("truth consistency (sorted by gap_top):")
    for t in sorted(hs, key=lambda t: stats[t]["gap_top"]):
        s = stats[t]
        print(f"  {M[t]['name'][:34]:34s} gap_top {s['gap_top']:.3f} off_top {s['off_top']:+.3f} corr {s['corr']:.3f}"
              f" slope {s['b']:.2f} {'OK' if t in ok else ''}")
    cands = [h for h in hs if M[h]["cands"]]
    res = {"system": sid, "u019_pub": pubW, "consistent": [M[t]["name"] for t in ok], "cands": {}}
    print("| candidate | mutual w/ u019 | offset fc (n) | anchor strict (n) | anchor loose (n) | slope-cal (n) | min/max loose delta |")
    print("|---|---|---|---|---|---|---|")
    for X in cands:
        fo = [imp[t][X] + stats[t]["off_top"] for t in ok if not clone(t, X)]
        fa = [pubW + imp[t][X] - imp[t][W] for t in ok if not clone(t, X) and not clone(t, W)]
        dl = [imp[t][X] - imp[t][W] for t in ok if t not in (X, W)]
        fl = [pubW + d for d in dl]
        fs = [pubW + stats[t]["b"] * (imp[t][X] - imp[t][W]) for t in ok if t not in (X, W)]
        mu = min(imp[X][W], imp[W][X])
        m = lambda v: f"{np.mean(v):.4f} ({len(v)})" if v else "- (0)"
        print(f"| {M[X]['name']} | {mu:.3f} | {m(fo)} | {m(fa)} | {m(fl)} | {m(fs)} | "
              f"{(min(dl) if dl else 0):+.4f}/{(max(dl) if dl else 0):+.4f} |")
        res["cands"][M[X]["name"]] = {"mutual_u019": mu, "offset": fo, "anchor_strict": fa, "anchor_loose": fl,
                                      "slope_cal": fs, "delta_loose": dl}
    print("implied under each consistent truth (row = truth; cols = candidates):")
    print("            " + " ".join(f"{M[q]['name'][:14]:>14s}" for q in cands))
    for t in ok:
        print(f"{M[t]['name'][:12]:12s}" + " ".join(f"{imp[t][q]:14.4f}" for q in cands))
    (ROOT / "plans" / f"q_win_fc_{sid}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
