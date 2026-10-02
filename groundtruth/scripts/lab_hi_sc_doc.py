"""hi lab: median(v8b final1, v8c alt1, hi2 full fit) doc for supply_chain + runtime checks (no credits)."""
import copy, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.runtime import infer

spec = S.get("supply_chain")
runs = load_runs(spec, Ledger(data_dir("supply_chain", False), "supply_chain", False))
sigma = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["supply_chain"]["sigma"], float)
b = json.loads((ROOT / "submissions/20260928-1440-final1/supply_chain/model.json").read_text())
c = json.loads((ROOT / "plans/alt1_supply_chain_doc.json").read_text())
h = json.loads((ROOT / "plans/supply_chain_hi_supply_chain_hi2_a_doc.json").read_text())
doc = copy.deepcopy(b)
doc["model"] = {"kind": "ensemble", "members": [b["model"], c["model"], h["model"]]}
doc["info"] = {"model_id": "median(v8b final1, v8c alt1, hi2 full fit)"}
out = ROOT / "plans/supply_chain_hi_med3_doc.json"
out.write_text(json.dumps(doc, indent=1))
def roll(d, y0, U):
    return infer.finalize(infer.rollout_from_blob(d, y0, U, doc=d), y0, np.array(d["clip_lo"]), np.array(d["clip_hi"])) if hasattr(infer, "finalize") else infer.rollout_from_blob(d, y0, U, doc=d)
for r in runs:
    Ys = [roll(d, r.y0, r.U) for d in (b, c, h)]
    Ym = roll(doc, r.y0, r.U)
    man = np.median(np.stack(Ys), 0)
    print(r.exp, "runtime-manual", float(np.abs(Ym - man).max()), "in-sample",
          [round(float(metric.score_per_obs(Y, r.Y, sigma).mean()), 4) for Y in Ys + [Ym]])
rng = np.random.default_rng(0)
for cat in ("sustained", "order", "recovery", "composition"):
    U = D.eval_like(spec, cat, 4000, rng)
    t0 = time.time(); Y = roll(doc, runs[0].y0, U); dt = time.time() - t0
    t0 = time.time(); roll(h, runs[0].y0, U); dth = time.time() - t0
    print(cat, "finite", bool(np.all(np.isfinite(Y))), f"med3 {dt:.2f}s hi2 {dth:.2f}s", np.round(Y.max(0), 1))
print("wrote", out)
