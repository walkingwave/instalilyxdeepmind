"""Traffic hi: 4,000-tick sanity of candidate docs through the shipping runtime (predict path):
long-hold end levels, finiteness, excursion vs the data range, and time per episode.

    python scripts/lab_hi_tr_sanity.py plans/traffic_traffic_z8_z8all7_doc.json plans/traffic_hi_<tag>_doc.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.runtime import infer

CTRL = ["signal_timing", "lane_closure", "toll", "ramp_metering", "freight_priority", "clearance_effort"]
HOLDS = {
    "pulse": [0.15, 0.65, 0.0, 1.0, 1.0, 0.0],
    "pulse.8": [0.22, 0.52, 1.0, 0.8, 0.9, 0.2],
    "mid": [0.5, 0.38, 2.5, 0.5, 0.5, 0.5],
    "full": [0.5, 0.0, 0.0, 1.0, 0.5, 0.5],
    "lane.75": [0.5, 0.75, 1.0, 1.0, 0.5, 0.5],
    "p7": [0.2, 0.5, 1.2, 0.9, 0.9, 0.1],
    "h9": [0.5, 0.2, 0.95, 0.9, 0.5, 0.5],
}


def main():
    spec = S.get("traffic")
    runs = load_runs(spec, Ledger(data_dir("traffic", False), "traffic", False))
    y0 = runs[0].y0
    ymax = np.max([r.Y.max(0) for r in runs], axis=0)
    for p in sys.argv[1:]:
        doc = json.loads(Path(p).read_text())
        print(p)
        for name, u in HOLDS.items():
            U = np.tile(np.asarray(u, float), (4000, 1))
            t0 = time.time()
            Y = infer.rollout_from_blob(doc, y0, U)
            dt = time.time() - t0
            print(f"  {name:8s} end {np.round(Y[-1], 1)} t200 {np.round(Y[199], 1)} finite {np.isfinite(Y).all()} "
                  f"max/datamax {np.round(Y.max(0) / ymax, 2)} {dt:.2f}s")


if __name__ == "__main__":
    main()
