"""ad_auction hi lab: per-fold held-out table for member combinations (free, no gateway).

Members: cached leave-one-run-out predictions of the shipped members (v8b, s1, min, v8c; ens9 scratch),
v8r (v8b refitted per fold with the hi-lab protocol, str9 lab thetas)
plus hi families re-rolled from their fold thetas (plans/ad_auction_hi_<family>_<tag>_<fold>.json).
Options are per observable: 'name' or 'med(a,b,..)'; an option string 'w;s;c' sets win_rate, spend,
conversions. The first option is the reference (final1 = v8b;med(v8b,s1);med(v8b,min)).

    python scripts/lab_hi_ad_select.py --hi hi1:ad_auction_hi1:a [--hi hi2:ad_auction_hi2:a] --opts "o1|o2|..."
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
from gtlab.runtime import infer as rt

SCRATCH = Path(os.environ.get("ENS9_SCRATCH", "C:/Users/DYLANH~1/AppData/Local/Temp/gtscratch/ens9")) / "ad_auction"
SHIP = ROOT / "submissions/20260928-1440-final1/ad_auction/model.json"
REF = "v8b;med(v8b,s1);med(v8b,min)"


def load(his):
    spec = S.get("ad_auction")
    runs = load_runs(spec, Ledger(data_dir("ad_auction", False), "ad_auction", False))
    doc = json.loads(SHIP.read_text())
    lo, hi = rt.clip_vectors(doc)
    sig = np.asarray(json.loads((ROOT / "plans/sigma_calibrated.json").read_text())["ad_auction"]["sigma"], float)
    P = {}
    for l in ("v8b", "s1", "min", "v8c", "l0b"):
        P[l] = {r.exp: rt.finalize(np.asarray(json.loads((SCRATCH / f"{l}__{r.exp}.json").read_text())["Y"], float), r.y0, lo, hi)
                for r in runs}
    # v8r = v8b refitted per fold with the same protocol as the hi labs (str9 lab: base3 + refit_w)
    mod = importlib.import_module("gtlab.ode.ad_auction_v8b")
    P["v8r"] = {}
    for fn in ("ad_auction_str9_ad_auction_v8b_base3.json", "ad_auction_str9_ad_auction_v8b_refit_w.json"):
        for rec in json.loads((ROOT / "plans" / fn).read_text())["loo"]:
            r = next(x for x in runs if x.exp == rec["held_out"])
            Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, rec["theta"]), frozenset("AB"), n_sub=2)
            P["v8r"][r.exp] = rt.finalize(Y, r.y0, lo, hi)
    for spec_s in his:
        lab, fam, tag = spec_s.split(":")
        mod = importlib.import_module(f"gtlab.ode.{fam}")
        P[lab] = {}
        for r in runs:
            fp = ROOT / "plans" / f"ad_auction_hi_{fam}_{tag}_{r.exp}.json"
            if not fp.exists():
                continue
            th = json.loads(fp.read_text())["theta"]
            Y = core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), frozenset("AB"), n_sub=int(getattr(mod, "N_SUB", 2)))
            P[lab][r.exp] = rt.finalize(Y, r.y0, lo, hi)
    return runs, doc, sig, P


def col(P, name, exp, j):
    if name.startswith("med("):
        ls = name[4:-1].split(",")
        return np.median(np.stack([P[l][exp][:, j] for l in ls]), axis=0)
    return P[name][exp][:, j]


def evaluate(runs, doc, sig, P, opt):
    parts = opt.split(";")
    if len(parts) == 1:
        parts = parts * 3
    rows = {}
    for r in runs:
        try:
            Y = np.stack([col(P, parts[j], r.exp, j) for j in range(3)], axis=1)
        except KeyError:
            continue
        Y = rt.apply_post(doc, Y.copy(), r.U, r.y0)
        rows[r.exp] = metric.score_per_obs(Y, r.Y, sig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", action="append", default=[])
    ap.add_argument("--opts", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs, doc, sig, P = load(a.hi)
    opts = [REF] + [o for o in a.opts.split("|") if o and o != REF]
    res = {o: evaluate(runs, doc, sig, P, o) for o in opts}
    ref = res[REF]
    exps = [r.exp for r in runs]
    print(f"{'option':<44}" + "".join(f"{e[:12]:>13}" for e in exps) + f"{'mean':>8}{'wins':>6}   per-obs mean")
    out = {}
    for o in opts:
        rows = res[o]
        if len(rows) < len(exps):
            print(f"{o:<44} incomplete ({len(rows)} folds)")
            continue
        fm = [rows[e].mean() for e in exps]
        wins = sum(rows[e].mean() > ref[e].mean() + 1e-4 for e in exps)
        po = np.mean([rows[e] for e in exps], axis=0)
        print(f"{o:<44}" + "".join(f"{v:13.4f}" for v in fm) + f"{np.mean(fm):8.4f}{wins:6d}   {np.round(po, 3)}")
        out[o] = {"folds": {e: rows[e].tolist() for e in exps}, "mean": float(np.mean(fm)), "wins_vs_ref": int(wins)}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
