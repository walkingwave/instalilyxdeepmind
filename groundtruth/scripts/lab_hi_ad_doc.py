"""ad_auction hi lab: build a model doc from an option string (same syntax as lab_hi_ad_select.py)
using full-data thetas, keep final1's clip / post rules, then smoke-test it: every run through the
runtime, a flat predict.py build in a scratch folder, and 4,000-tick timing. Free, no gateway.

    python scripts/lab_hi_ad_doc.py --hi hi1:ad_auction_hi1:a --opt "med(hi1,v8b);hi1;med(hi1,v8b)" --name hi1m
Writes plans/ad_auction_hi_<name>_doc.json.
"""
import argparse, json, sys, tempfile, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S, design as D
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.package import write_system_folder
from gtlab.runtime import infer as rt

SHIP = ROOT / "submissions/20260928-1440-final1/ad_auction/model.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", action="append", default=[])
    ap.add_argument("--opt", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    ship = json.loads(SHIP.read_text())
    sm = ship["model"]["members"]
    blobs = {"v8b": sm[0], "s1": sm[1]["members"][1], "min": sm[2]["members"][1]}
    for s in a.hi:
        lab, fam, tag = s.split(":")
        rep = json.loads((ROOT / "plans" / f"ad_auction_hi_{fam}_{tag}.json").read_text())
        blobs[lab] = {"kind": "ode", "family": fam, "theta": rep["theta_vec"], "mech": ["A", "B"], "n_sub": 2}
    parts = a.opt.split(";")
    if len(parts) == 1:
        parts = parts * 3

    def blob(name):
        if name.startswith("med("):
            return {"kind": "ensemble", "members": [blobs[l] for l in name[4:-1].split(",")]}
        return blobs[name]
    uniq = list(dict.fromkeys(parts))
    model = blob(uniq[0]) if len(uniq) == 1 else {"kind": "perobs", "map": [uniq.index(p) for p in parts],
                                                  "members": [blob(p) for p in uniq]}
    doc = json.loads(json.dumps(ship))
    doc["model"] = model
    doc["info"] = {"model_id": f"ad_auction_hi:{a.name}:{a.opt}"}
    out = ROOT / "plans" / f"ad_auction_hi_{a.name}_doc.json"
    out.write_text(json.dumps(doc, indent=1))
    spec = S.get("ad_auction")
    runs = load_runs(spec, Ledger(data_dir("ad_auction", False), "ad_auction", False))
    sig = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["ad_auction"]["sigma"], float)
    for r in runs:
        Y = rt.rollout_from_blob(doc, r.y0, r.U, doc=doc)
        s = metric.score_per_obs(Y, r.Y, sig)
        print(f"  in-sample {r.exp:<20} {np.round(s, 3)} {s.mean():.4f}")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "ad_auction"
        write_system_folder(d, "ad_auction", json.loads(json.dumps(doc)))
        sp = importlib.util.spec_from_file_location("pred_hi_ad", d / "predict.py")
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        rng = np.random.default_rng(0)
        obs, ctrls = spec.observables, list(doc["controls"])
        worst = 0.0
        for cat in ("sustained", "order", "recovery", "composition"):
            U = D.eval_like(spec, cat, 4000, rng)
            init = {o: float(v) for o, v in zip(obs, runs[0].y0)}
            acts = [{c: float(v) for c, v in zip(ctrls, u)} for u in U]
            t0 = time.time()
            res = mod.predict(init, acts, {"family": "ad_auction"})
            dt = time.time() - t0
            worst = max(worst, dt)
            Y = np.array([[row[o] for o in obs] for row in res])
            Yd = rt.rollout_from_blob(doc, runs[0].y0, U, doc=doc)
            print(f"  flat predict[{cat}] {dt:.2f}s finite={bool(np.all(np.isfinite(Y)))} max|flat-dev|={np.abs(Y - Yd).max():.2e} max={np.round(Y.max(0), 2)}")
    print(f"wrote {out}  (worst episode {worst:.2f}s -> 40 episodes ~{40 * worst:.0f}s)")


if __name__ == "__main__":
    main()
