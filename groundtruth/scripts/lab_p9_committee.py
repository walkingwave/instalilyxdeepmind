"""Purchase-9 candidate ranking by committee disagreement and coverage novelty. Free: no gateway.

Committee per system = shipped predictors of the grey-box era (u008 .. final1, pg9-w5) whose public
score is within 0.08 of the system's best, deduplicated on predict.py + model.json. Candidates are long holds on the brief's recovery -> pulse axis
(alpha = 0.85 per pulsed control: every single control, every all-but-one, the full pulse) and at
the bound midpoint, each as (a) one hold for the whole budget and (b) the hold for 60% of it then
the recovery action (recovery after a long pulse). Rolled from the most typical owned reset.

Per candidate:
  L    = committee loss: mean over members, ticks, observables of 1 - 1/(1+|y_i - median|/sigma_cal),
         i.e. the score lost by shipping the median if one member were the truth (score units)
  Llate= the same on the last 40% of the window (the part a long run uniquely owns)
  L4k  = same for the level held to tick 4000 (hold family only): long-horizon relevance
  nov  = distance of the hold level to the nearest owned hold of >= 100 ticks
         (controls normalized to their bounds, RMS over controls)

    python scripts/lab_p9_committee.py [--systems a,b] [--procs 2]
Writes plans/p9_committee.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S                                   # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs             # noqa: E402

warnings.filterwarnings("ignore")
SYSTEMS = ["epidemic", "market", "traffic", "power_grid", "supply_chain", "wildlife", "reservoir",
           "ad_auction", "social_contagion", "hospital_queue"]
DIRS = ["20260925-1925-u008-ode", "20260926-0031-u009", "20260926-0135-u010b", "20260926-2120-u011",
        "20260926-2337-u012", "20260927-0233-u013", "20260927-2139-u014", "20260927-2151-u015",
        "20260928-1430-u016", "20260928-1440-final1", "20260928-1635-pg9-w5"]
RESERVE = 20
ALPHA = 0.85


GAP = 0.08          # members: public score within GAP of the system's best (unscored dirs kept)


def committee(sid):
    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    sc = {u["dir"].replace("\\", "/").split("/")[-1]: (u.get("scores") or {}).get(sid) for u in reg}
    best = max((v for v in sc.values() if v is not None), default=0.0)
    seen, out = set(), []
    for d in DIRS:
        p = ROOT / "submissions" / d / sid / "predict.py"
        if not p.exists():
            continue
        if sc.get(d) is not None and sc[d] < best - GAP:
            continue
        mj = p.parent / "model.json"
        h = hashlib.sha1(p.read_bytes() + (mj.read_bytes() if mj.exists() else b"")).hexdigest()[:10]
        if h in seen:
            continue
        seen.add(h)
        sp = importlib.util.spec_from_file_location(f"p9_{sid}_{h}", str(p))
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        out.append((d.split("-", 2)[-1], mod))
    return out


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    try:
        res = mod.predict({o: float(v) for o, v in zip(obs, y0)},
                          [{c: float(v) for c, v in zip(ctrls, r)} for r in U], spec.context())
        Y = np.array([[row[o] for o in obs] for row in res], float)
    except Exception:
        return None
    return np.where(np.isfinite(Y), Y, np.nan)


def closs(Ys, sig, a=0, b=None):
    Z = np.stack([Y[a:b] for Y in Ys])
    med = np.nanmedian(Z, axis=0)
    e = np.abs(np.nan_to_num(Z - med[None], nan=1e9)) / sig[None, None, :]
    return float(np.mean(1 - 1 / (1 + e)))


def owned_holds(spec, runs, lo, hi, min_len=100):
    out = []
    for r in runs:
        s = 0
        for t in range(1, r.T + 1):
            if t == r.T or not np.allclose(r.U[t], r.U[s]):
                if t - s >= min_len:
                    out.append(((r.U[s] - lo) / (hi - lo), t - s, r.exp))
                s = t
    return out


def candidates(spec, B):
    ctrls = list(spec.controls)
    r = np.array([spec.recovery[c] for c in ctrls], float)
    p = np.array([spec.pulse[c] for c in ctrls], float)
    lo, hi = np.array(spec.lo(), float), np.array(spec.hi(), float)
    m = len(ctrls)
    levels = {}
    for j in range(m):
        v = r.copy(); v[j] = r[j] + ALPHA * (p[j] - r[j])
        levels[f"pulse[{ctrls[j]}]"] = v
        if m > 2:
            w = r + ALPHA * (p - r); w[j] = r[j]
            levels[f"pulse[all-{ctrls[j]}]"] = w
    if m == 3 or m == 4:
        for i, j in itertools.combinations(range(m), 2):
            v = r.copy(); v[[i, j]] = r[[i, j]] + ALPHA * (p[[i, j]] - r[[i, j]])
            levels[f"pulse[{ctrls[i]}+{ctrls[j]}]"] = v
    levels["pulse[all]"] = r + ALPHA * (p - r)
    levels["mid"] = (lo + hi) / 2
    out = []
    for name, v in levels.items():
        out.append({"name": f"hold {name} {B}", "level": v,
                    "U": np.repeat(v[None], B, axis=0), "family": "hold"})
        L1 = int(round(0.6 * B))
        out.append({"name": f"{name} {L1} + rec {B - L1}", "level": v,
                    "U": np.concatenate([np.repeat(v[None], L1, 0), np.repeat(r[None], B - L1, 0)]),
                    "family": "hold+rec"})
    return out


def study(sid):
    t0 = time.time()
    spec = S.get(sid)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    led = Ledger(data_dir(sid, False), sid, False)
    runs = load_runs(spec, led)
    bal = 2000 - led.steps_charged()
    B = bal - RESERVE
    lo, hi = np.array(spec.lo(), float), np.array(spec.hi(), float)
    Y0 = np.array([r.y0 for r in runs], float)
    med = np.median(Y0, axis=0)
    y0 = Y0[int(np.argmin(np.sum(((Y0 - med) / sig) ** 2, axis=1)))]
    owned = owned_holds(spec, runs, lo, hi)
    com = committee(sid)
    rows = []
    for c in candidates(spec, B):
        Ys = [roll(mod, spec, y0, c["U"]) for _, mod in com]
        Ys = [Y for Y in Ys if Y is not None]
        z = (c["level"] - lo) / (hi - lo)
        nov = min((float(np.sqrt(np.mean((z - h) ** 2))) for h, _, _ in owned), default=1.0)
        row = {"name": c["name"], "family": c["family"], "T": int(len(c["U"])),
               "L": closs(Ys, sig), "Llate": closs(Ys, sig, int(0.6 * B)), "nov": nov,
               "ends": {n: [round(float(v), 2) for v in Y[-1]] for (n, _), Y in zip(com, Ys)},
               "level": [float(v) for v in c["level"]]}
        if c["family"] == "hold":
            U4 = np.repeat(c["level"][None], 4000, 0)
            Y4 = [roll(mod, spec, y0, U4) for _, mod in com]
            Y4 = [Y for Y in Y4 if Y is not None]
            row["L4k"] = closs(Y4, sig)
            row["L4k_late"] = closs(Y4, sig, B)
        rows.append(row)
    rows.sort(key=lambda x: -x["L"])
    return sid, {"balance": bal, "budget": B, "members": [n for n, _ in com], "y0": [float(v) for v in y0],
                 "owned_holds": [{"u01": [round(float(v), 2) for v in h], "len": n, "exp": e} for h, n, e in owned],
                 "rows": rows, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    sids = a.systems.split(",")
    outp = ROOT / "plans" / "p9_committee.json"
    res = json.loads(outp.read_text()) if outp.exists() else {}
    with Pool(a.procs) as pool:
        for sid, r in pool.imap_unordered(study, sids):
            res[sid] = r
            outp.write_text(json.dumps(res, indent=1))
            print(f"\n=== {sid} bal {r['balance']} members {r['members']} ({r['secs']} s)")
            for x in r["rows"][:12]:
                print(f"  {x['name']:55s} L {x['L']:.3f} late {x['Llate']:.3f} nov {x['nov']:.2f}"
                      + (f" L4k {x['L4k']:.3f} L4k_late {x['L4k_late']:.3f}" if "L4k" in x else ""))


if __name__ == "__main__":
    main()
