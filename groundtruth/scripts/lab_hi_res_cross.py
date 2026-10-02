"""reservoir hi: cross-score two packaged predictors on eval-like 4,000-tick episodes (one taken as
truth for the other) at the calibrated sigma, from several reset states. Free.

    python scripts/lab_hi_res_cross.py --a <folder with reservoir/predict.py> --b <folder> [--n 8]
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import design as D, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from lab_str9_diag import roll


def load(folder, tag):
    s = importlib.util.spec_from_file_location(f"x_{tag}", str(Path(folder) / "reservoir" / "predict.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--T", type=int, default=4000)
    a = ap.parse_args()
    spec = S.get("reservoir")
    A, B = load(a.a, "a"), load(a.b, "b")
    sig = np.array(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["reservoir"]["sigma"])
    runs = load_runs(spec, Ledger(data_dir("reservoir", False), "reservoir", False))
    rng = np.random.default_rng(7)
    tot = []
    for cat in ("sustained", "order", "recovery", "composition"):
        ss = []
        for k in range(a.n):
            U = D.eval_like(spec, cat, a.T, rng)
            y0 = runs[k % len(runs)].y0
            Ya, Yb = roll(A, spec, y0, U), roll(B, spec, y0, U)
            ss.append((1 / (1 + np.abs(Ya - Yb) / sig)).mean(0))
            full = (Ya[:, 0] > 930).mean()
        s = np.mean(ss, 0); tot.append(s)
        print(f"{cat:12s} score(a|truth=b) {np.round(s, 3)} {s.mean():.4f}  full-pool frac {full:.2f}")
    t = np.mean(tot, 0)
    print(f"{'overall':12s} {np.round(t, 3)} {t.mean():.4f}")


if __name__ == "__main__":
    main()
