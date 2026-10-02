"""ens9: per-fold held-out table for named options (no credits).

    python scripts/lab_ens9_folds.py --system market --opts "y3|med(y3,d3,z8)|y3;med(y3,z8);y3"

Each option is either one name for all observables or 'a;b;c' per observable (names as in
lab_ens9_select.py). Prints fold means and per-observable means; first option is the reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gtlab import metric  # noqa: E402
from gtlab.runtime import infer as rt  # noqa: E402
import lab_ens9_loo as L  # noqa: E402
import lab_ens9_select as SE  # noqa: E402
from lab_ens9_assemble import parse  # noqa: E402


def table(system, opts_str):
    spec, runs = L.get_runs(system)
    doc = json.loads((L.FINAL / system / "model.json").read_text())
    sigma = L.cal_sigma(system)
    labels = list(L.MEMBERS[system]) + ["l0b"]
    P, err = SE.load_preds(system, runs, labels, doc)
    p = len(spec.observables)
    res = {}
    for os_ in opts_str.split("|"):
        ch = parse(os_) if ";" in os_ else parse(os_)[:1] * p
        rows = []
        for r in runs:
            cols = []
            for j, o in enumerate(ch):
                Y = np.median(np.stack([P[l][r.exp] for l in o]), axis=0)
                cols.append(metric.score_per_obs(rt.apply_post(doc, Y.copy(), r.U, r.y0), r.Y, sigma)[j])
            rows.append(cols)
        res[os_] = np.array(rows)
    names = list(res)
    ref = res[names[0]]
    print(f"== {system}  (first = reference)")
    print(f"{'fold':<24}" + "".join(f"{n[:22]:>24}" for n in names))
    for k, r in enumerate(runs):
        print(f"{r.exp:<24}" + "".join(f"{res[n][k].mean():>24.4f}" for n in names))
    print(f"{'MEAN':<24}" + "".join(f"{res[n].mean():>24.4f}" for n in names))
    print(f"{'fold wins vs ref':<24}" + "".join(f"{int(np.sum(res[n].mean(1) > ref.mean(1) + 1e-12)):>24d}" for n in names))
    for j, ob in enumerate(spec.observables):
        print(f"  {ob:<22}" + "".join(f"{res[n][:, j].mean():>24.4f}" for n in names))
    return {n: {"fold": {r.exp: float(res[n][k].mean()) for k, r in enumerate(runs)}, "mean": float(res[n].mean()),
                "per_obs": res[n].mean(0).tolist()} for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--opts", required=True)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    out = table(a.system, a.opts)
    if a.save:
        f = Path(a.save)
        d = json.loads(f.read_text()) if f.exists() else {}
        d[a.system] = out
        f.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
