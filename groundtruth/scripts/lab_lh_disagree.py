"""Late-horizon disagreement between final3 and the other good candidates. Free: no gateway calls.

Per system: every distinct shipped predict.py among recent builds (final3 first), rolled on 20
eval-like 4,000-tick schedules (5 per category, design.eval_like, fixed seeds, y0 cycled from our
runs). Units: calibrated sigma (plans/sigma_calibrated.json).

Measures
  pairwise mean |a-b|/sigma per horizon band [0,400) [400,1000) [1000,4000) and category
  cross-truth: taking each candidate as truth, the score of final3, of the candidate median, and
  of switch(T0): final3 for t < T0, median for t >= T0
  long-hold check: every run with T > 400, scored on [0,400) and [400,T) against the data

    python scripts/lab_lh_disagree.py [--systems a,b] [--procs 3]
Writes plans/lh_disagree.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, design as D                                   # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs                           # noqa: E402

warnings.filterwarnings("ignore")
SYSTEMS = ["epidemic", "market", "traffic", "power_grid", "supply_chain", "wildlife", "reservoir",
           "ad_auction", "social_contagion", "hospital_queue"]
BUILDS = ["20260928-2230-final3", "20260928-2330-alt1", "20260929-1823-u017", "20260928-1810-final2",
          "20260928-1440-final1", "20260928-1430-u016", "20260927-2151-u015"]
T, T0 = 4000, 400
BANDS = [(0, 400), (400, 1000), (1000, 4000)]
CATS = ["sustained", "order", "recovery", "composition"]
NPER = 5


def switch(Y_early, Y_late, t0):
    """Horizon switch: early model before tick t0, late model from t0 on (pure function)."""
    Y = np.array(Y_late, float, copy=True)
    Y[:t0] = np.asarray(Y_early, float)[:t0]
    return Y


def blend_switch(Y_early, Y_late, t0, width=0):
    """Smooth version: weight on the late model ramps linearly over [t0, t0+width)."""
    Te = len(Y_late)
    w = np.clip((np.arange(Te) - t0) / max(1, width), 0.0, 1.0) if width > 0 else \
        (np.arange(Te) >= t0).astype(float)
    return (1 - w)[:, None] * np.asarray(Y_early, float) + w[:, None] * np.asarray(Y_late, float)


def load_mod(build, sid):
    p = ROOT / "submissions" / build / sid / "predict.py"
    mj = p.parent / "model.json"
    h = hashlib.sha1(p.read_bytes() + mj.read_bytes()).hexdigest()[:8]
    sp = importlib.util.spec_from_file_location(f"lh_{sid}_{build.replace('-', '_')}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod, h


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    out = mod.predict({o: float(v) for o, v in zip(obs, y0)},
                      [{c: float(v) for c, v in zip(ctrls, r)} for r in U], spec.context())
    Y = np.array([[row[o] for o in obs] for row in out], float)
    return np.where(np.isfinite(Y), Y, np.nan)


def sc(Yh, Yt, sig):
    e = np.abs(np.nan_to_num(Yh, nan=1e12) - Yt)
    return float(np.mean(1.0 / (1.0 + e / sig[None, :])))


def run_system(sid):
    t0 = time.time()
    spec = S.get(sid)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    Y0 = np.array([r.y0 for r in runs], float)
    cands, seen = [], {}
    for b in BUILDS:
        if not (ROOT / "submissions" / b / sid / "predict.py").exists():
            continue
        mod, h = load_mod(b, sid)
        if h in seen:
            seen[h].append(b)
            continue
        seen[h] = [b]
        mid = json.loads((ROOT / "submissions" / b / "manifest.json").read_text())["systems"][sid].get("model_id")
        cands.append({"name": b.split("-", 2)[-1], "hash": h, "mod": mod, "model_id": mid, "builds": seen[h]})
    names = [c["name"] for c in cands]
    sched = []
    for ci, cat in enumerate(CATS):
        for k in range(NPER):
            rng = np.random.default_rng(5100 + 100 * ci + k)
            sched.append((cat, Y0[(ci * NPER + k) % len(Y0)], np.asarray(D.eval_like(spec, cat, T, rng), float)))
    R = {c["name"]: [roll(c["mod"], spec, y0, U) for _, y0, U in sched] for c in cands}
    med = [np.nanmedian(np.stack([R[n][i] for n in names]), axis=0) for i in range(len(sched))]
    ref = names[0]

    def band_err(A, B, a, b):
        return float(np.nanmean(np.abs(A[a:b] - B[a:b]) / sig))

    res = {"system": sid, "observables": list(spec.observables), "sigma": sig.tolist(),
           "candidates": [{k: c[k] for k in ("name", "hash", "model_id", "builds")} for c in cands]}
    # pairwise disagreement per band / category (+ per observable, late band)
    pw = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {}
            for cat in CATS + ["all"]:
                idx = [j for j, s in enumerate(sched) if cat == "all" or s[0] == cat]
                row[cat] = [float(np.mean([band_err(R[a][j], R[b][j], lo, hi) for j in idx])) for lo, hi in BANDS]
            row["late_obs"] = np.nanmean([np.nanmean(np.abs(R[a][j][1000:] - R[b][j][1000:]) / sig, axis=0)
                                          for j in range(len(sched))], axis=0).tolist()
            pw[f"{a}|{b}"] = row
    res["pairwise"] = pw
    res["ref_vs_median"] = {cat: [float(np.mean([band_err(R[ref][j], med[j], lo, hi) for j, s in enumerate(sched)
                                                 if cat == "all" or s[0] == cat])) for lo, hi in BANDS]
                            for cat in CATS + ["all"]}
    # cross truth
    ct = {}
    for tn in names:
        row = {}
        for pn in names:
            if pn != tn:
                row[pn] = [float(np.mean([sc(R[pn][j][lo:hi], R[tn][j][lo:hi], sig) for j in range(len(sched))]))
                           for lo, hi in BANDS]
        row["median"] = [float(np.mean([sc(med[j][lo:hi], R[tn][j][lo:hi], sig) for j in range(len(sched))]))
                         for lo, hi in BANDS]
        row["switch"] = [float(np.mean([sc(switch(R[ref][j], med[j], T0)[lo:hi], R[tn][j][lo:hi], sig)
                                        for j in range(len(sched))])) for lo, hi in BANDS]
        ct[tn] = row
    res["cross_truth"] = ct
    # long holds and full-data fit (in-sample for every candidate)
    lh = []
    for r in runs:
        U = np.asarray(r.U, float)
        Yd = np.asarray(r.Y, float)
        P = {c["name"]: roll(c["mod"], spec, r.y0, U) for c in cands}
        Pm = np.nanmedian(np.stack([P[n] for n in names]), axis=0)
        Ps = switch(P[ref], Pm, T0)
        ent = {"exp": getattr(r, "exp", getattr(r, "name", "?")), "T": int(r.T)}
        segs = [(0, min(T0, r.T))] + ([(T0, r.T)] if r.T > T0 else [])
        for nm, Y in list(P.items()) + [("median", Pm), ("switch", Ps)]:
            ent[nm] = [sc(Y[a:b], Yd[a:b], sig) for a, b in segs]
            if r.T > T0:
                ent[nm + "_obs_late"] = np.mean(1 / (1 + np.abs(Y[T0:] - Yd[T0:]) / sig), axis=0).tolist()
        lh.append(ent)
    res["runs"] = lh
    res["secs"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    sy = a.systems.split(",")
    out_p = ROOT / "plans" / "lh_disagree.json"
    prev = json.loads(out_p.read_text()) if out_p.exists() else {}
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(run_system, sy):
            prev[r["system"]] = r
            out_p.write_text(json.dumps(prev, indent=1))
            print(r["system"], f"{r['secs']:.0f}s", [c["name"] for c in r["candidates"]], flush=True)


if __name__ == "__main__":
    main()
