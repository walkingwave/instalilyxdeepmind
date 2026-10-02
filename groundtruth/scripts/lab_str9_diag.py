"""str9 diagnostics: score a shipped predictor on every owned run, per observable and per
category-shaped segment (time since last control change). Free, no gateway calls.

    python scripts/lab_str9_diag.py --system reservoir [--folder submissions/20260928-1440-final1]
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S
from gtlab.ledger import Ledger, data_dir, load_runs


def load_predict(folder, sid):
    p = ROOT / folder / sid / "predict.py"
    s = importlib.util.spec_from_file_location(f"d_{sid}", str(p))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    out = mod.predict({o: float(v) for o, v in zip(obs, y0)},
                      [{c: float(v) for c, v in zip(ctrls, row)} for row in U], spec.context())
    return np.array([[row[o] for o in obs] for row in out], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--folder", default="submissions/20260928-1440-final1")
    a = ap.parse_args()
    spec = S.get(a.system)
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    sig = np.array(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"])
    mod = load_predict(a.folder, a.system)
    print(a.system, "obs", spec.observables, "sigma", sig)
    allS = []
    for r in runs:
        Y = roll(mod, spec, r.y0, r.U)
        s = 1 / (1 + np.abs(Y - r.Y) / sig)
        # ticks since last control change
        ch = np.r_[True, np.any(np.abs(np.diff(r.U, axis=0)) > 1e-9, axis=1)]
        since = np.zeros(len(ch), int); c = 0
        for t in range(len(ch)):
            c = 0 if ch[t] else c + 1; since[t] = c
        segs = {"<10": since < 10, "10-50": (since >= 10) & (since < 50), ">=50": since >= 50}
        print(f"{r.exp:32s} T={r.T:4d} mean {s.mean():.3f} per-obs {np.round(s.mean(0),3)} bias(sig) {np.round((Y-r.Y).mean(0)/sig,2)}")
        for k, m in segs.items():
            if m.sum():
                print(f"    since-change {k:6s} n={m.sum():4d} {np.round(s[m].mean(0),3)}")
        allS.append(s.mean())
    print("mean over runs", np.mean(allS))

if __name__ == "__main__":
    main()
