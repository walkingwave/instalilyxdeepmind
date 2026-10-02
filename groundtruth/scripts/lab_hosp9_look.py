import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
spec = S.get("hospital_queue")
runs = load_runs(spec, Ledger(data_dir("hospital_queue", False), "hospital_queue", False))
for r in runs:
    print(r.exp, r.T, "y0", np.round(r.y0,1))
name = sys.argv[1]; a=int(sys.argv[2]); b=int(sys.argv[3]); st=int(sys.argv[4]) if len(sys.argv)>4 else 1
r = [r for r in runs if r.exp==name][0]
print(spec.controls)
for t in range(a,b,st):
    print(t, np.round(r.Y[t],2), np.round(r.U[t],2))
