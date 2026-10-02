"""Traffic hi: public-band consistency of candidate docs (free, no gateway).

Take candidate P as the truth on test-shaped schedules (design.eval_like: sustained, order, recovery,
composition; T = 4,000, fixed seeds), score every scored traffic upload Q against it at the calibrated
sigma, and compare with Q's public bands (sustained, sequence). Lower RMSE = P is more consistent with
what the leaderboard saw.

    python scripts/lab_hi_tr_band.py plans/a_doc.json plans/b_doc.json [--n 8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import systems as S, design as D
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.runtime import infer
import longhold_audit as LA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    sid = "traffic"
    spec = S.get(sid)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    Y0 = np.array([r.y0 for r in runs], float)
    preds = [g for g in LA.predictors(sid) if g["mod"] is not None]
    cats = ["sustained", "order", "recovery", "composition"]
    sched = {c: [(Y0[i % len(Y0)], D.eval_like(spec, c, 4000, np.random.default_rng(7000 + 97 * k + i)))
                 for i in range(a.n)] for k, c in enumerate(cats)}
    Q = {}
    for g in preds:
        Q[g["name"]] = {c: [LA.roll(g["mod"], spec, y0, U) for y0, U in sched[c]] for c in cats}
    print("uploads:", [(g["name"], round(g["sustained"], 3), round(g["sequence"], 3)) for g in preds])
    for p in a.docs:
        doc = json.loads(Path(p).read_text())
        T = {c: [infer.rollout_from_blob(doc, y0, U) for y0, U in sched[c]] for c in cats}
        rows = []
        for g in preds:
            s = {c: np.mean([LA.score(Q[g["name"]][c][i], T[c][i], sig) for i in range(a.n)]) for c in cats}
            seq = np.mean([s["order"], s["recovery"], s["composition"]])
            rows.append((g["name"], s["sustained"], seq, g["sustained"], g["sequence"]))
        es = np.sqrt(np.mean([(r[1] - r[3]) ** 2 for r in rows]))
        eq = np.sqrt(np.mean([(r[2] - r[4]) ** 2 for r in rows]))
        print(f"{p}: band RMSE sustained {es:.3f} sequence {eq:.3f}")
        for r in rows:
            print(f"   {r[0]:8s} implied sus {r[1]:.3f} (pub {r[3]:.3f})  seq {r[2]:.3f} (pub {r[4]:.3f})")


if __name__ == "__main__":
    main()
