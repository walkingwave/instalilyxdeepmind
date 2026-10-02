"""Coverage of owned runs per system (free, reads ledgers only).

For every run: hold segments (constant control vector) of >= MIN ticks, printed as the control
vector normalized to [0, 1] of its bound (0 = low bound, 1 = high bound), plus the end-of-hold
observation. Also the per-system balance and, per control, the longest hold at each third of range.

python scripts/lab_p9_coverage.py [SYS ...] [--min 40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtlab import systems  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402

ALL = ["epidemic", "market", "traffic", "power_grid", "supply_chain", "wildlife", "reservoir",
       "ad_auction", "social_contagion", "hospital_queue"]


def segments(U, min_len):
    out, s = [], 0
    for t in range(1, len(U) + 1):
        if t == len(U) or not np.allclose(U[t], U[s]):
            if t - s >= min_len:
                out.append((s, t))
            s = t
    return out


def load(sysid):
    spec = systems.get(sysid)
    led = Ledger(data_dir(sysid, False), sysid, False)
    return spec, led, load_runs(spec, led)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--min", type=int, default=40)
    a = ap.parse_args()
    for s in a.systems or ALL:
        spec, led, runs = load(s)
        lo, hi = np.array(spec.lo()), np.array(spec.hi())
        bal = 2000 - led.steps_charged()
        print(f"\n=== {s}  balance {bal}  runs {len(runs)}  ticks {sum(r.T for r in runs)}")
        print("  controls:", ", ".join(f"{c}[{spec.bounds[c][0]},{spec.bounds[c][1]}] rec={spec.recovery[c]} pulse={spec.pulse[c]}"
                                      for c in spec.controls))
        print("  obs:", ", ".join(spec.observables))
        for r in runs:
            segs = segments(r.U, a.min)
            for (i, j) in segs:
                z = (r.U[i] - lo) / np.where(hi > lo, hi - lo, 1)
                yend = r.Y[j - 1]
                print(f"  {r.exp:28s} [{i:4d},{j:4d}) len {j - i:4d}  u01=" +
                      " ".join(f"{v:4.2f}" for v in z) + "  y_end=" + " ".join(f"{v:7.2f}" for v in yend))


if __name__ == "__main__":
    main()
