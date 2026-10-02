"""Write wild9 docs (single two-stage fit and the median ensemble) and check them through the runtime:
in-sample score at the calibrated sigma, 4,000-tick eval-shaped rollouts (finite, range, seconds)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import design as D, metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.runtime import infer

spec = S.get("wildlife")
runs = load_runs(spec, Ledger(data_dir("wildlife", False), "wildlife", False))
sigma = np.asarray(json.loads(Path("plans/sigma_calibrated.json").read_text())["wildlife"]["sigma"], float)
clip = C.soft_clip(spec, runs, margin=1.0)
th = lambda p: json.loads(Path(p).read_text())["theta_vec"]
ode = lambda fam, p: {"kind": "ode", "family": fam, "theta": th(p), "mech": ["A", "B"], "n_sub": 2}
m_base = ode("wildlife_v8h", "plans/wildlife_wildlife_v8h_p7.json")
m_s2p = ode("wildlife_wild9p", "plans/wildlife_wild9_s2p.json")
m_m = ode("wildlife_wild9m", "plans/wildlife_wild9_m.json")
docs = {
    "s2p": (m_s2p, "ode:wildlife_wild9p two-stage fit (prey on prey, predators with own transit)"),
    "ens": ({"kind": "ensemble", "members": [m_base, m_s2p, m_m]}, "wild9 median: v8h p7 refit, wild9p two-stage, wild9m (gx+pt)"),
}
Yall = np.concatenate([r.Y for r in runs]); ymin, ymax = Yall.min(0), Yall.max(0); rg = ymax - ymin
for name, (blob, mid) in docs.items():
    doc = C.make_doc(spec, blob, runs=runs, clip=clip, info={"model_id": mid, "n_runs": len(runs)})
    Path(f"plans/wildlife_wild9_{name}_doc.json").write_text(json.dumps(doc, indent=1))
    sc = [metric.score_per_obs(infer.rollout_from_blob(doc, r.y0, r.U), r.Y, sigma).mean() for r in runs]
    print(name, "in-sample", " ".join(f"{x:.3f}" for x in sc), f"mean {np.mean(sc):.4f}")
    rng = np.random.default_rng(0)
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time(); Y = infer.rollout_from_blob(doc, runs[0].y0, U); dt = time.time() - t0
        out = float(np.mean((Y > ymax + 0.05 * rg) | (Y < ymin - 0.05 * rg)))
        print(f"  {cat:<11} finite={bool(np.all(np.isfinite(Y)))} {dt:.2f}s outside={out:.3f} min={np.round(Y.min(0), 1)} max={np.round(Y.max(0), 1)}")
