"""canon: score equal-weight means of saved LOO fold predictions (npz files from lab_canon_epi_loo.py).

    python scripts/lab_canon_epi_ens.py a.npz b.npz [...]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
SIG = np.array([12.953956472279227, 5.230371399915308])
runs = load_runs(S.get("epidemic"), Ledger(data_dir("epidemic", False), "epidemic", False))
Z = [np.load(p) for p in sys.argv[1:]]
row = []
for r in runs:
    Y = np.mean([z[f"Y_{r.exp}"] for z in Z], axis=0)
    row.append(metric.score_per_obs(Y, r.Y, SIG).mean())
print(" ".join(f"{v:.3f}" for v in row), f"LOO {np.mean(row):.4f}")
