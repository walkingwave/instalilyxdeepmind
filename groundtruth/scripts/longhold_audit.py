"""Long-hold (sustained category) audit of every scored predictor. Free: no gateway calls.

For each system we load every scored upload's predict.py (deduplicated on predict.py+model.json),
roll it on 20 sustained-style schedules (design.eval_like "sustained", T=4000, fixed seeds), 10
constant holds at uniform interior control vectors, one-control sweeps for the brief-implied
monotone pairs, and init-memory pairs (same hold, low vs high y0). Units: calibrated sigma
(plans/sigma_calibrated.json).

Measures per predictor:
  disagreement with u012 and with the committee median, ticks [0,450) vs [450,4000)
  steady state on holds: drift (mean 3800-4000 minus mean 3300-3500), late oscillation,
  settle time, control authority (spread of the steady state across holds), init memory,
  monotone sweeps (sign of d ss / d control vs the brief)
  truth-consistency: taking predictor P as truth on the sustained schedules, the implied
  score of every other scored predictor Q; compared with Q's public sustained band.

    python scripts/longhold_audit.py [--systems a,b] [--procs 4]
Writes plans/longhold_audit.json (the .md is written from it by hand).
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
T, CUT, NSUS, NHOLD = 4000, 450, 20, 10
# brief-implied steady-state signs: (control, observable, sign of d ss / d control)
MONO = {
    "epidemic": [("vaccination_rate", "daily_cases", -1), ("mask_mandate", "daily_cases", -1),
                 ("school_closure", "daily_cases", -1), ("mask_mandate", "hospital_load", -1)],
    "market": [("transaction_tax", "volume", -1), ("interest_rate", "price", -1)],
    "traffic": [("lane_closure", "speed_a", -1), ("lane_closure", "speed_b", -1),
                ("ramp_metering", "flow_a", -1)],
    "power_grid": [("price_signal", "load", -1), ("reserve_dispatch", "frequency", 1)],
    "supply_chain": [("order_quantity", "shipments", 1), ("order_quantity", "inventory_supplier", -1),
                     ("production_effort", "inventory_supplier", 1)],
    "wildlife": [("hunting_quota", "prey_north", -1), ("hunting_quota", "prey_south", -1),
                 ("habitat_protection", "prey_north", 1)],
    "reservoir": [("release_rate", "outflow", 1), ("release_rate", "level", -1),
                  ("irrigation_allocation", "level", -1)],
    "ad_auction": [("bid", "win_rate", 1), ("bid", "spend", 1), ("budget_cap", "spend", 1)],
    "social_contagion": [("seeding", "adopters_a", 1), ("seeding", "adopters_b", 1),
                         ("incentive", "adopters_a", 1)],
    "hospital_queue": [("staffing", "wait_time", -1), ("staffing", "queue", -1),
                       ("elective_scheduling", "queue", 1), ("staffing", "discharges", 1)],
}


def load_mod(folder, sid, tag):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    if not p.exists():
        return None, None
    mj = p.parent / "model.json"
    h = hashlib.sha1(p.read_bytes() + (mj.read_bytes() if mj.exists() else b"")).hexdigest()[:10]
    sp = importlib.util.spec_from_file_location(f"lha_{sid}_{tag}", str(p))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod, h


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    out = mod.predict({o: float(v) for o, v in zip(obs, y0)},
                      [{c: float(v) for c, v in zip(ctrls, r)} for r in U], spec.context())
    Y = np.array([[row[o] for o in obs] for row in out], float)
    return np.where(np.isfinite(Y), Y, np.nan)


def predictors(sid):
    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    bands = json.loads((ROOT / "tune" / "bands.json").read_text())["rows"]
    bmap = {r["upload"]: r for r in bands if r["system"] == sid}
    groups = {}
    for u in reg:
        if u["id"] not in bmap:
            continue
        mod, h = load_mod(u["dir"], sid, u["id"])
        kind = ((u.get("systems") or {}).get(sid) or {}).get("kind", "?")
        if mod is None:
            if u["id"] != "u001":
                continue
            h, kind = "persistence", "l0a"
        g = groups.setdefault(h, {"mod": mod, "uploads": [], "hash": h})
        g["uploads"].append({"upload": u["id"], "kind": kind, "sustained": bmap[u["id"]]["sustained"],
                             "sequence": bmap[u["id"]]["sequence"]})
    out = []
    for g in groups.values():
        last = g["uploads"][-1]
        g["name"] = last["upload"]
        g["kind"] = last["kind"]
        g["sustained"] = float(np.mean([x["sustained"] for x in g["uploads"]]))
        g["sequence"] = float(np.mean([x["sequence"] for x in g["uploads"]]))
        out.append(g)
    return out


def score(Yh, Yt, sig):
    e = np.abs(np.nan_to_num(Yh, nan=1e12) - Yt)
    return float(np.mean(1.0 / (1.0 + e / sig[None, :])))


def audit(sid):
    t0 = time.time()
    spec = S.get(sid)
    obs, ctrls = list(spec.observables), list(spec.controls)
    sig = np.array(json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())[sid]["sigma"], float)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    Y0 = np.array([r.y0 for r in runs], float)
    med = np.median(Y0, axis=0)
    y0t = Y0[int(np.argmin(np.sum(((Y0 - med) / sig) ** 2, axis=1)))]
    sc = np.sum((Y0 - med) / sig, axis=1)
    y0lo, y0hi = Y0[int(np.argmin(sc))], Y0[int(np.argmax(sc))]
    lo, hi = np.array(spec.lo(), float), np.array(spec.hi(), float)

    sus = [(Y0[i % len(Y0)], D.eval_like(spec, "sustained", T, np.random.default_rng(7000 + i)))
           for i in range(NSUS)]
    rng = np.random.default_rng(9100)
    hold_levels = [lo + rng.uniform(0.05, 0.95, spec.m) * (hi - lo) for _ in range(NHOLD)]
    holds = [(y0t, D.hold(spec, lv, T)) for lv in hold_levels]
    mid = (lo + hi) / 2
    sweeps = []
    for c, o, sgn in MONO.get(sid, []):
        j = ctrls.index(c)
        for f in np.linspace(0.05, 0.95, 5):
            lv = mid.copy()
            lv[j] = lo[j] + f * (hi[j] - lo[j])
            sweeps.append((c, o, sgn, float(f), (y0t, D.hold(spec, lv, T))))
    mem = []
    for nm, lv in (("recovery", D.rec(spec)), ("mid", mid)):
        mem.append((nm, (y0lo, D.hold(spec, lv, T)), (y0hi, D.hold(spec, lv, T))))

    P = predictors(sid)
    R = {}
    for g in P:
        f = (lambda y0, U: np.tile(y0, (len(U), 1))) if g["mod"] is None else \
            (lambda y0, U, m=g["mod"]: roll(m, spec, y0, U))
        R[g["name"]] = {
            "sus": [f(y0, U) for y0, U in sus],
            "hold": [f(y0, U) for y0, U in holds],
            "sweep": [f(*x[4]) for x in sweeps],
            "mem": [(f(*a), f(*b)) for _, a, b in mem],
        }
    names = [g["name"] for g in P]
    ref = "u012" if "u012" in R else names[-1]
    comm = [np.nanmedian(np.stack([R[n]["sus"][i] for n in names]), axis=0) for i in range(NSUS)]

    res = {"system": sid, "observables": obs, "sigma": sig.tolist(), "ref": ref, "predictors": []}
    for g in P:
        n = g["name"]
        r = R[n]
        d_ref = np.array([[np.nanmean(np.abs(a[:CUT] - b[:CUT]) / sig, axis=0),
                           np.nanmean(np.abs(a[CUT:] - b[CUT:]) / sig, axis=0)]
                          for a, b in zip(r["sus"], R[ref]["sus"])])
        d_com = np.array([[np.nanmean(np.abs(a[:CUT] - b[:CUT]) / sig, axis=0),
                           np.nanmean(np.abs(a[CUT:] - b[CUT:]) / sig, axis=0)]
                          for a, b in zip(r["sus"], comm)])
        # steady state on holds
        H = np.stack(r["hold"])                                  # [NHOLD, T, p]
        ss = np.nanmean(H[:, -200:], axis=1)
        drift = np.nanmean(np.abs(ss - np.nanmean(H[:, 3300:3500], axis=1)) / sig, axis=0)
        osc = np.nanmean(np.nanstd(H[:, 3500:], axis=1) / sig, axis=0)
        settle = []
        for h in H:
            dev = np.nanmax(np.abs(h - h[-1]) / sig, axis=1)
            bad = np.where(~(dev < 0.25))[0]
            settle.append(int(bad[-1] + 1) if len(bad) else 0)
        auth = np.nanstd(ss, axis=0) / sig                       # spread of ss across holds
        early = np.nanmean(H[:, 300:450], axis=1)
        late_move = np.nanmean(np.abs(ss - early) / sig, axis=0)   # change after tick 450
        y0move = np.nanmean(np.abs(ss - y0t) / sig, axis=0)
        # response to level switches after tick 450 on sustained schedules
        switch_resp = []
        for (y0, U), Y in zip(sus, r["sus"]):
            ch = np.where(np.any(np.diff(U, axis=0) != 0, axis=1))[0] + 1
            for t in ch:
                if t >= CUT and t + 400 <= T:
                    switch_resp.append(np.abs(np.nanmean(Y[t + 300:t + 400], axis=0)
                                              - np.nanmean(Y[t - 50:t], axis=0)) / sig)
        sw = np.mean(switch_resp, axis=0).tolist() if switch_resp else None
        mono = []
        k = 0
        for c, o, sgn in MONO.get(sid, []):
            vals = [float(np.nanmean(r["sweep"][k + i][-200:, obs.index(o)])) for i in range(5)]
            k += 5
            dv = np.diff(vals)
            slope = np.sign(vals[-1] - vals[0])
            rngv = (max(vals) - min(vals)) / sig[obs.index(o)]
            so = sig[obs.index(o)]
            rev = int(np.sum(np.sign(dv) * sgn < 0) - np.sum((np.sign(dv) * sgn < 0) & (np.abs(dv) < 0.1 * so)))
            ok = bool(rngv < 0.25 or (slope == sgn and rev <= 1))
            mono.append({"control": c, "obs": o, "want": sgn, "ss": vals, "range_sigma": float(rngv),
                         "sign": int(slope), "ok": ok, "flat": bool(rngv < 0.25)})
        memd = {nm: (np.abs(np.nanmean(a[-200:], axis=0) - np.nanmean(b[-200:], axis=0)) / sig).tolist()
                for (nm, _, _), (a, b) in zip(mem, r["mem"])}
        res["predictors"].append({
            "name": n, "kind": g["kind"], "hash": g["hash"], "uploads": [x["upload"] for x in g["uploads"]],
            "sustained": g["sustained"], "sequence": g["sequence"],
            "dis_ref_early": d_ref[:, 0].mean(0).tolist(), "dis_ref_late": d_ref[:, 1].mean(0).tolist(),
            "score_vs_ref_early": float(np.mean([score(a[:CUT], b[:CUT], sig) for a, b in zip(r["sus"], R[ref]["sus"])])),
            "score_vs_ref_late": float(np.mean([score(a[CUT:], b[CUT:], sig) for a, b in zip(r["sus"], R[ref]["sus"])])),
            "dis_comm_early": d_com[:, 0].mean(0).tolist(), "dis_comm_late": d_com[:, 1].mean(0).tolist(),
            "sus_late_dist_y0": np.mean([np.nanmean(np.abs(Y[CUT:] - y0) / sig, axis=0)
                                         for (y0, _), Y in zip(sus, r["sus"])], axis=0).tolist(),
            "persist_if_truth_per_obs": np.mean([np.nanmean(1.0 / (1.0 + np.abs(Y - y0) / sig), axis=0)
                                                 for (y0, _), Y in zip(sus, r["sus"])], axis=0).tolist(),
            "persist_if_truth_late_per_obs": np.mean([np.nanmean(1.0 / (1.0 + np.abs(Y[CUT:] - y0) / sig), axis=0)
                                                      for (y0, _), Y in zip(sus, r["sus"])], axis=0).tolist(),
            "hold_drift": drift.tolist(), "hold_osc": osc.tolist(),
            "settle_median": float(np.median(settle)), "settle_max": int(max(settle)),
            "ss_authority": auth.tolist(), "ss_late_move": late_move.tolist(), "ss_vs_y0": y0move.tolist(),
            "switch_resp_after450": sw, "mono": mono, "init_memory": memd,
            "ss_holds": ss.tolist(),
        })
    # truth-consistency: P as truth, implied sustained score of each Q vs public band
    tc = []
    for tp in names:
        imp, pub = [], []
        for q in names:
            if q == tp:
                continue
            imp.append(np.mean([score(a, b, sig) for a, b in zip(R[q]["sus"], R[tp]["sus"])]))
            pub.append([g for g in P if g["name"] == q][0]["sustained"])
        imp, pub = np.array(imp), np.array(pub)
        rho = float(np.corrcoef(np.argsort(np.argsort(imp)), np.argsort(np.argsort(pub)))[0, 1]) if len(imp) > 2 else None
        tc.append({"truth": tp, "implied": dict(zip([q for q in names if q != tp], imp.round(4).tolist())),
                   "rmse": float(np.sqrt(np.mean((imp - pub) ** 2))), "bias": float(np.mean(imp - pub)),
                   "rank_corr": rho})
    res["truth_consistency"] = tc
    res["hold_levels"] = [lv.tolist() for lv in hold_levels]
    res["y0_typical"] = y0t.tolist()
    res["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(Path(r"C:\Users\DYLANH~1\AppData\Local\Temp\claude\scratch\lha") / f"{sid}.npz",
                        **{f"{n}_sus{i}": R[n]["sus"][i] for n in names for i in range(0, NSUS, 5)},
                        **{f"{n}_hold{i}": R[n]["hold"][i] for n in names for i in range(NHOLD)})
    print(sid, "done", res["seconds"], "s", len(names), "predictors", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    sids = a.systems.split(",")
    outp = ROOT / "plans" / "longhold_audit.json"
    out = json.loads(outp.read_text()) if outp.exists() else {}
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(audit, sids):
            out[r["system"]] = r
            outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
