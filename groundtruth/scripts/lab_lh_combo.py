"""Horizon-aware combinations on a chosen candidate subset. Free: no gateway calls.

For each system and subset (builds from lab_lh_disagree.BUILDS, final3 first):
  combos: final3 alone, median(subset), switch(T0) = final3 before T0 then median, for T0 in 400/700
  scenario scores on the 20 eval-like schedules with each subset member taken as truth:
    expected (uniform prior over members) and worst case, per band [0,400) [400,1000) [1000,4000)
  data scores on every run: [0,400) and [400,T)

    python scripts/lab_lh_combo.py --spec '{"wildlife":["final3","alt1","u017"]}'
Writes plans/lh_combo.json (merged).
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import lab_lh_disagree as L                                                   # noqa: E402
from gtlab import systems as S, design as D                                   # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs                           # noqa: E402

T0S = (400, 700)


def run(args):
    sid, subset = args
    spec = S.get(sid)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    Y0 = np.array([r.y0 for r in runs], float)
    mods = {}
    for b in L.BUILDS:
        nm = b.split("-", 2)[-1]
        if nm in subset:
            mods[nm] = L.load_mod(b, sid)[0]
    names = [n for n in subset if n in mods]
    ref = names[0]
    sched = []
    for ci, cat in enumerate(L.CATS):
        for k in range(L.NPER):
            rng = np.random.default_rng(5100 + 100 * ci + k)
            sched.append((cat, Y0[(ci * L.NPER + k) % len(Y0)], np.asarray(D.eval_like(spec, cat, L.T, rng), float)))
    R = {n: [L.roll(mods[n], spec, y0, U) for _, y0, U in sched] for n in names}

    def combos(P):
        med = np.nanmedian(np.stack([P[n] for n in names]), axis=0)
        out = {ref: P[ref], "median": med}
        for t0 in T0S:
            out[f"switch{t0}"] = L.switch(P[ref], med, t0)
        return out

    C = [combos({n: R[n][j] for n in names}) for j in range(len(sched))]
    res = {"system": sid, "subset": names, "scen": {}, "scen_cat": {}, "data": []}
    for cn in C[0]:
        per_truth = np.array([[[L.sc(C[j][cn][a:b], R[tn][j][a:b], sig) for a, b in L.BANDS]
                               for j in range(len(sched))] for tn in names])       # [truth, sched, band]
        m = per_truth.mean(axis=1)
        res["scen"][cn] = {"expected": m.mean(axis=0).tolist(), "worst": m.min(axis=0).tolist(),
                           "per_truth": dict(zip(names, m.tolist()))}
        res["scen_cat"][cn] = {cat: per_truth[:, [j for j, s in enumerate(sched) if s[0] == cat], 2].mean(axis=1).tolist()
                               for cat in L.CATS}
    for r in runs:
        U = np.asarray(r.U, float)
        Yd = np.asarray(r.Y, float)
        Cr = combos({n: L.roll(mods[n], spec, r.y0, U) for n in names})
        segs = [(0, min(400, r.T))] + ([(400, r.T)] if r.T > 400 else [])
        res["data"].append({"exp": getattr(r, "exp", "?"), "T": int(r.T),
                            **{cn: [L.sc(Y[a:b], Yd[a:b], sig) for a, b in segs] for cn, Y in Cr.items()}})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    spec = json.loads(a.spec)
    out_p = ROOT / "plans" / "lh_combo.json"
    prev = json.loads(out_p.read_text()) if out_p.exists() else {}
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(run, list(spec.items())):
            key = r["system"] + ":" + "+".join(r["subset"])
            prev[key] = r
            out_p.write_text(json.dumps(prev, indent=1))
            print("\n==", key)
            for cn, v in r["scen"].items():
                print(f"  {cn:10s} exp {np.round(v['expected'], 3).tolist()} worst {np.round(v['worst'], 3).tolist()}"
                      f"  late-by-cat {np.round(np.mean(list(r['scen_cat'][cn].values()), axis=1), 3).tolist()}")
            for e in r["data"]:
                if e["T"] > 400:
                    print("   data", e["exp"], e["T"], {k: np.round(v, 3).tolist() for k, v in e.items() if k not in ("exp", "T")})
            early = {k: round(float(np.mean([e[k][0] for e in r["data"]])), 3) for k in r["scen"]}
            print("   data [0,400) all runs", early, flush=True)


if __name__ == "__main__":
    main()
