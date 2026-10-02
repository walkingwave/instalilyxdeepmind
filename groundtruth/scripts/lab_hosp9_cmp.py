"""Fold table of a lab json vs the public p3 doc, plus long-run agreement with p3 (wait/queue/disch)."""
import sys, json, importlib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, design as D
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
spec = S.get("hospital_queue")
runs = load_runs(spec, Ledger(data_dir("hospital_queue", False), "hospital_queue", False))
sig = np.array(json.load(open("plans/sigma_calibrated.json"))["hospital_queue"]["sigma"])
P = importlib.import_module("gtlab.ode.hospital_queue_p3")
thp = core.theta_dict(P, json.load(open("plans/hospital_queue_p3_doc.json"))["model"]["theta"])
base = {r.exp: metric.score_per_obs(np.minimum(core.rollout(P, r.y0, r.U, thp, "AB", n_sub=2), [1e9, 333, 1e9]), r.Y, sig) for r in runs}
for path in sys.argv[1:]:
    d = json.load(open(path))
    mod = importlib.import_module(f"gtlab.ode.{d['family']}")
    th = core.theta_dict(mod, d["theta_vec"])
    print(f"\n== {path}\n   theta:", {k: round(v, 4) for k, v in d["theta"].items() if k in ("tau_w", "eps_w", "c_w", "tau_dn", "b_ot", "tau_g", "w0") or True})
    lo = []
    for x in d["loo"]:
        b = base[x["held_out"]]; o = np.array(x["ode"])
        lo.append((o.mean(), b.mean()))
        print(f"   {x['held_out']:<24} new {np.round(o,3)} {o.mean():.3f} | pub {np.round(b,3)} {b.mean():.3f} | diff {o.mean()-b.mean():+.3f}")
    lo = np.array(lo)
    print(f"   LOO mean new {lo[:,0].mean():.4f} pub {lo[:,1].mean():.4f} diff {lo[:,0].mean()-lo[:,1].mean():+.4f}")
    rng = np.random.default_rng(7)
    ag = {}
    for cat in ("sustained", "recovery", "order", "composition"):
        ss = []
        for k in range(4):
            U = D.eval_like(spec, cat, 4000, rng)
            y0 = runs[k % len(runs)].y0
            A = core.rollout(P, y0, U, thp, "AB", n_sub=2)
            B = core.rollout(mod, y0, U, th, "AB", n_sub=2)
            ss.append(metric.score_per_obs(B, A, sig))
        ag[cat] = np.round(np.mean(ss, 0), 3).tolist()
    print("   agreement with public p3 over 4000 ticks (per obs):", ag)
