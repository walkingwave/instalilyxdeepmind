"""str9 verify: package a model doc into a scratch system folder (predict.py + model.json, the
shipping path), score it on every owned run at the calibrated sigma against the shipped predictor,
and time 4,000-tick episodes of all four categories. Free.

    python scripts/lab_str9_verify.py --system reservoir --doc plans/..._doc.json --out <scratch dir>
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import design as D, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.package import write_system_folder
from lab_str9_diag import roll


def load(folder, tag):
    s = importlib.util.spec_from_file_location(f"v_{tag}", str(folder / "predict.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ship", default="submissions/20260928-1440-final1")
    a = ap.parse_args()
    spec = S.get(a.system)
    doc = json.loads(Path(a.doc).read_text())
    folder = Path(a.out) / a.system
    write_system_folder(folder, a.system, doc)
    new, old = load(folder, "new"), load(ROOT / a.ship / a.system, "old")
    sig = np.array(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())[a.system]["sigma"])
    runs = load_runs(spec, Ledger(data_dir(a.system, False), a.system, False))
    for r in runs:
        so = (1 / (1 + np.abs(roll(old, spec, r.y0, r.U) - r.Y) / sig)).mean(0)
        sn = (1 / (1 + np.abs(roll(new, spec, r.y0, r.U) - r.Y) / sig)).mean(0)
        print(f"{r.exp:28s} ship {np.round(so, 3)} {so.mean():.4f} | new {np.round(sn, 3)} {sn.mean():.4f}")
    rng = np.random.default_rng(1)
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(spec, cat, 4000, rng)
        t0 = time.time(); Y = roll(new, spec, runs[0].y0, U); dt = time.time() - t0
        Yo = roll(old, spec, runs[0].y0, U)
        print(f"eval[{cat}] finite={bool(np.isfinite(Y).all())} {dt:.2f}s min {np.round(Y.min(0), 3)} max {np.round(Y.max(0), 3)} "
              f"| new-vs-ship mean|diff|/sigma {np.round(np.abs(Y - Yo).mean(0) / sig, 2)}")


if __name__ == "__main__":
    main()
