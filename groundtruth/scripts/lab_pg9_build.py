"""power_grid pg9: package one model doc as a power_grid-only submission folder and verify it.

    python scripts/lab_pg9_build.py --doc plans/power_grid_power_grid_w5_w5_doc.json --tag pg9-w5
    python scripts/lab_pg9_build.py --verify submissions/<stamp>-pg9-w5   # verify an existing folder

Checks (no credits): parity of the flat predict.py with the lab rollout on every owned run, score of
every run at the calibrated sigma next to the final1 power_grid folder, 40 check-style episodes x
4,000 ticks (finite, per-episode time, range), determinism.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, check as CK  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.package import build  # noqa: E402

SID = "power_grid"
REF = ROOT / "submissions" / "20260928-1440-final1"


def load_predict(folder, tag):
    p = Path(folder) / SID / "predict.py"
    s = importlib.util.spec_from_file_location(f"pg9_{tag}", str(p))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    init = {o: float(v) for o, v in zip(obs, y0)}
    iv = [{c: float(v) for c, v in zip(ctrls, row)} for row in U]
    out = mod.predict(init, iv, spec.context())
    return np.array([[row[o] for o in obs] for row in out], float)


def score(Y, T, sig):
    return (1.0 / (1.0 + np.abs(Y - T) / sig)).mean(0)


def verify(folder, n_ep=40):
    spec = S.get(SID)
    runs = load_runs(spec, Ledger(data_dir(SID, False), SID, False))
    sig = np.asarray(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[SID]["sigma"], float)
    new, ref = load_predict(folder, "new"), load_predict(REF, "ref")
    print(f"{'run':<18} {'final1':>28} {'candidate':>28}")
    rows = {}
    for r in runs:
        a, b = score(roll(ref, spec, r.y0, r.U), r.Y, sig), score(roll(new, spec, r.y0, r.U), r.Y, sig)
        rows[r.exp] = {"final1": a.tolist(), "cand": b.tolist()}
        print(f"{r.exp:<18} {np.round(a, 3)} {a.mean():.3f}   {np.round(b, 3)} {b.mean():.3f}")
    eps, src = CK.make_episodes(spec, n=n_ep, T=4000, seed=0)
    ts, fin, lo, hi = [], True, np.full(3, np.inf), np.full(3, -np.inf)
    for ep in eps:
        t0 = time.time()
        out = new.predict(ep["initial"], ep["interventions"], spec.context())
        ts.append(time.time() - t0)
        Y = np.array([[row[o] for o in spec.observables] for row in out], float)
        fin = fin and len(out) == 4000 and bool(np.all(np.isfinite(Y)))
        lo, hi = np.minimum(lo, Y.min(0)), np.maximum(hi, Y.max(0))
    out2 = new.predict(eps[0]["initial"], eps[0]["interventions"], spec.context())
    Y1 = np.array([[row[o] for o in spec.observables] for row in new.predict(eps[0]["initial"], eps[0]["interventions"], spec.context())])
    Y2 = np.array([[row[o] for o in spec.observables] for row in out2])
    det = bool(np.array_equal(Y1, Y2))
    print(f"episodes {n_ep} x 4000 (init from {src}): finite={fin} deterministic={det} "
          f"time mean {np.mean(ts):.2f}s max {np.max(ts):.2f}s total {np.sum(ts):.1f}s")
    print(f"range min {np.round(lo, 3)} max {np.round(hi, 3)}")
    src_txt = (Path(folder) / SID / "predict.py").read_text()
    imports = sorted({ln.split()[1].split(".")[0] for ln in src_txt.splitlines() if ln.startswith(("import ", "from "))})
    print("imports", imports, "files", sorted(p.name for p in (Path(folder) / SID).iterdir() if p.is_file()))
    return {"runs": rows, "finite": fin, "deterministic": det, "t_mean": float(np.mean(ts)), "t_max": float(np.max(ts)),
            "min": lo.tolist(), "max": hi.tolist(), "imports": imports}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc")
    ap.add_argument("--tag", default="pg9")
    ap.add_argument("--verify")
    ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    if a.verify:
        folder = Path(a.verify)
    else:
        doc = json.loads(Path(a.doc).read_text())
        assert doc["system"] == SID
        folder = build([SID], a.tag, {SID: doc})
        print("built", folder)
    rep = verify(folder, a.episodes)
    (Path(folder) / "verify_pg9.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
