"""ad_auction hi lab: residuals of a model doc on every run (free, no gateway).

    python scripts/lab_hi_ad_resid.py [--doc submissions/20260928-1440-final1/ad_auction/model.json]
Per run: score per observable at 1.0 sigma, mean signed error / sigma, and the same split by
segment (ticks since the last control change: 0-9, 10-49, 50+) and by control regime.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.runtime import infer as rt

SHIP = "submissions/20260928-1440-final1/ad_auction/model.json"


def since_change(U):
    ch = np.r_[True, np.any(np.abs(np.diff(U, axis=0)) > 1e-9, axis=1)]
    out = np.zeros(len(U), int)
    c = 0
    for t in range(len(U)):
        c = 0 if ch[t] else c + 1
        out[t] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=SHIP)
    ap.add_argument("--regimes", action="store_true")
    a = ap.parse_args()
    spec = S.get("ad_auction")
    runs = load_runs(spec, Ledger(data_dir("ad_auction", False), "ad_auction", False))
    sig = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["ad_auction"]["sigma"], float)
    doc = json.loads((ROOT / a.doc).read_text())
    tot = []
    for r in runs:
        Y = rt.rollout_from_blob(doc, r.y0, r.U, doc=doc)
        s = metric.score_per_obs(Y, r.Y, sig)
        e = (Y - r.Y) / sig
        sc = since_change(r.U)
        tot.append(s.mean())
        print(f"{r.exp:<24} T={r.T:4d} score {np.round(s,3)} mean {s.mean():.4f} bias {np.round(e.mean(0),2)}")
        for lo, hi in ((0, 10), (10, 50), (50, 10**6)):
            m = (sc >= lo) & (sc < hi)
            if m.sum() == 0:
                continue
            ss = (1 / (1 + np.abs(e[m]))).mean(0)
            print(f"    since {lo:>3}-{hi if hi < 10**6 else 'inf':<4} n={m.sum():4d} score {np.round(ss,3)} bias {np.round(e[m].mean(0),2)}")
        if a.regimes:
            keys = [tuple(np.round(u, 2)) for u in r.U]
            for k in sorted(set(keys)):
                m = np.array([kk == k for kk in keys])
                ss = (1 / (1 + np.abs(e[m]))).mean(0)
                print(f"    u={k} n={m.sum():4d} score {np.round(ss,3)} bias {np.round(e[m].mean(0),2)} obs {np.round(r.Y[m].mean(0),2)} pred {np.round(Y[m].mean(0),2)}")
    print(f"mean over runs {np.mean(tot):.4f}")


if __name__ == "__main__":
    main()
