import sys, json, importlib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab.ode import core
P = importlib.import_module("gtlab.ode.hospital_queue_p3")
thp = core.theta_dict(P, json.load(open("plans/hospital_queue_p3_doc.json"))["model"]["theta"])
rec = [20, 0, 0.4, 0.6, 0, 1]; pul = [5, 20, 0.75, 1, 1, 0]
holds = {"rec": rec, "st8": [8, 0, 0.4, 0.6, 0, 1], "st8_ot1": [8, 0, 0.4, 0.6, 1, 1], "diag0.8": [8, 0, 0.8, 0.6, 0, 1],
         "pulse": pul, "mid": [10.5, 10, 0.45, 0.5, 0.5, 0.5], "st3": [3, 0, 0.4, 0.6, 0, 1], "st12_el10": [12, 10, 0.4, 0.6, 0, 1],
         "st1": [1, 20, 0.8, 0, 1, 1]}
mods = [("pub", P, thp)]
for path in sys.argv[1:]:
    d = json.load(open(path)); m = importlib.import_module(f"gtlab.ode.{d['family']}")
    mods.append((Path(path).stem[-6:], m, core.theta_dict(m, d["theta_vec"])))
y0 = np.array([3.5, 50.0, 8.0])
for k, u in holds.items():
    U = np.tile(np.array(u, float), (4000, 1))
    row = []
    for n, m, th in mods:
        Y = core.rollout(m, y0, U, th, "AB", n_sub=2)
        row.append(f"{n}: w@50 {Y[50,0]:.0f} w@400 {Y[400,0]:.0f} end {Y[-1,0]:.1f}/{Y[-1,1]:.0f}/{Y[-1,2]:.2f}")
    print(f"{k:<10}", " | ".join(row))
# pulse 40 then recovery
U = np.array([pul]*40 + [rec]*3960, float)
for n, m, th in mods:
    Y = core.rollout(m, y0, U, th, "AB", n_sub=2)
    print("pulse40+rec", n, np.round(Y[[20, 39, 60, 100, 150, 300, 1000], 0], 1))
