import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
runs = load_runs(S.get("wildlife"), Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
others = sys.argv[1:]
P = {t: np.load(Path(os.environ["TEMP"]) / "wild9" / f"{t}.npz") for t in ["base"] + others}
res = []
grid = [0, 0.25, 0.5, 0.75, 1]
for other in others:
    for wp in grid:
        for wq in grid:
            wv = np.array([wp, wq, wp, wq])
            sc = [metric.score_per_obs((1 - wv) * P['base']['loo_' + r.exp] + wv * P[other]['loo_' + r.exp], r.Y, sigma) for r in runs]
            m = [s.mean() for s in sc]
            res.append((np.mean(m), other, wp, wq, m))
res.sort(key=lambda x: -x[0])
print("folds p1 p2 p3 p4 p7")
for r in res[:12]:
    print(f"{r[0]:.4f} {r[1]} prey_w {r[2]} pred_w {r[3]} folds", " ".join(f"{x:.3f}" for x in r[4]))
