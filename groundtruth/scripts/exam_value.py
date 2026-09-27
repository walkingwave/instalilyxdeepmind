"""Is a 400-tick test-shaped exam run worth its credits? Free: no gateway calls.

For every system we replay each distinct scored public predictor on every owned ledger run, score it
at the calibrated organizer sigma, and measure how well each run (and the mean over runs) ranks the
predictors like the public board:

    s_{i,r} = mean_{t,j} 1 / (1 + |yhat_{i,t,j} - y_{r,t,j}| / sigma_j)
    Spearman rho(s_{.,r}, public),  pairwise-order accuracy
    PA = #{(i,k): sign(s_i - s_k) = sign(pub_i - pub_k), |pub_i - pub_k| > tie} / #{pairs}

In-sample bias: a predictor uploaded after a run was bought may have been fitted on it. A run is
"unseen" for a predictor when its reset timestamp is later than the upload stamp (persistence is
always unseen). Out-of-training (OOT) agreement uses only the predictors that had not seen the run.

    python scripts/exam_value.py
Writes plans/exam_value.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S                                                  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs                           # noqa: E402

warnings.filterwarnings("ignore")
SYSTEMS = ["epidemic", "market", "traffic", "power_grid", "supply_chain", "wildlife", "reservoir",
           "ad_auction", "social_contagion", "hospital_queue"]
EXAM = "p5.testlike"
TIE = 0.003
HARD = 0.05
CACHE = Path(r"C:\Users\DYLANH~1\AppData\Local\Temp\claude\scratch\exv")


def shape(exp):
    e = exp.lower()
    if "testlike" in e:
        return "testlike"
    if "compose" in e or "joint" in e:
        return "compose"
    if "multilevel" in e:
        return "multilevel"
    if "pulse" in e:
        return "pulse"
    return "hold"


def load_predict(folder, sid, tag):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    if not p.exists():
        return None
    s = importlib.util.spec_from_file_location(f"exv_{sid}_{tag}", str(p))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    init = {o: float(v) for o, v in zip(obs, y0)}
    iv = [{c: float(v) for c, v in zip(ctrls, row)} for row in U]
    out = mod.predict(init, iv, spec.context())
    return np.array([[row[o] for o in obs] for row in out], float)


def upload_ts(folder):
    m = re.search(r"(\d{8}-\d{4})", folder.replace("\\", "/"))
    return time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M")) if m else 0.0


def agree(x, pub, mask=None):
    x, pub = np.asarray(x, float), np.asarray(pub, float)
    if mask is not None:
        x, pub = x[np.asarray(mask, bool)], pub[np.asarray(mask, bool)]
    ok = np.isfinite(x) & np.isfinite(pub)
    x, pub = x[ok], pub[ok]
    n = len(x)
    if n < 3:
        return {"n": int(n), "rho": None, "pa": None, "pairs": 0, "pa_hard": None, "pairs_hard": 0}
    c = t = ch = th = 0
    for i, k in itertools.combinations(range(n), 2):
        d = abs(pub[i] - pub[k])
        if d <= TIE:
            continue
        hit = int(np.sign(x[i] - x[k]) == np.sign(pub[i] - pub[k]))
        t += 1
        c += hit
        if d <= HARD:                          # close pairs: the decisions we actually face
            th += 1
            ch += hit
    rho = spearmanr(x, pub).correlation
    return {"n": int(n), "rho": None if not np.isfinite(rho) else round(float(rho), 3),
            "pa": round(c / t, 3) if t else None, "pairs": int(t),
            "pa_hard": round(ch / th, 3) if th else None, "pairs_hard": int(th)}


def main():
    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    bands = json.loads((ROOT / "tune" / "bands.json").read_text())["rows"]
    cal = json.loads((ROOT / "plans" / "sigma_calibrated.json").read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {"tie": TIE, "hard": HARD, "systems": {}}
    for sid in SYSTEMS:
        t0 = time.time()
        spec = S.get(sid)
        led = Ledger(data_dir(sid, False), sid, False)
        runs = load_runs(spec, led)
        sig = np.asarray(cal[sid]["sigma"], float)
        run_ts = {}
        for row in led.rows:
            if row.get("t") == "reset" and "ts" in row:
                run_ts.setdefault(row["run_id"], row["ts"])
        rts = [run_ts.get(r.tags.get("run_id"), 0.0) for r in runs]
        preds = []
        for u in reg:
            pub = (u.get("scores") or {}).get(sid)
            if pub is None:
                continue
            kind = str(((u.get("systems") or {}).get(sid) or {}).get("kind", "?"))
            ck = CACHE / f"{sid}_{u['id']}_{len(runs)}.pkl"
            if ck.exists():
                P = pickle.loads(ck.read_bytes())
            else:
                mod = load_predict(u["dir"], sid, u["id"])
                try:
                    P = ([np.tile(r.y0, (r.T, 1)) for r in runs] if mod is None
                         else [roll(mod, spec, r.y0, r.U) for r in runs])
                except Exception as e:
                    print(sid, u["id"], "predict failed", type(e).__name__, e)
                    continue
                ck.write_bytes(pickle.dumps(P))
            dup = next((d for d in preds if all(np.max(np.abs(a - b)) < 1e-9
                                                  for a, b in zip(d["P"], P))), None)
            bnd = [b for b in bands if b["upload"] == u["id"] and b["system"] == sid]
            if dup is not None:
                dup["ids"].append(u["id"])
                dup["pubs"].append(pub)
                dup["bands"] += bnd
                continue
            ut = upload_ts(u["dir"])
            persist = all(np.max(np.abs(Pr - Pr[0])) < 1e-12 for Pr in P)
            preds.append({"ids": [u["id"]], "pubs": [pub], "kind": kind, "P": P, "bands": bnd,
                          "unseen": [persist or kind == "l0a" or t > ut for t in rts]})
        n = len(preds)
        pub = np.array([np.mean(p["pubs"]) for p in preds])
        sus = np.array([np.mean([b["sustained"] for b in p["bands"]]) if p["bands"] else np.nan
                        for p in preds])
        seq = np.array([np.mean([b["sequence"] for b in p["bands"]]) if p["bands"] else np.nan
                        for p in preds])
        Sc = np.array([[float(np.mean(1.0 / (1.0 + np.abs(Ph - r.Y) / sig[None, :])))
                        for Ph, r in zip(p["P"], runs)] for p in preds])      # [n, R]
        U = np.array([p["unseen"] for p in preds])                             # [n, R]
        rr = []
        for j, r in enumerate(runs):
            rr.append({"exp": r.exp, "T": int(r.T), "shape": shape(r.exp),
                       "all": agree(Sc[:, j], pub), "oot": agree(Sc[:, j], pub, U[:, j]),
                       "sus": agree(Sc[:, j], sus), "seq": agree(Sc[:, j], seq),
                       "n_unseen": int(U[:, j].sum())})
        own = [j for j, r in enumerate(runs) if r.exp != EXAM]
        ex = [j for j, r in enumerate(runs) if r.exp == EXAM]
        mean_all = Sc.mean(axis=1)
        mean_own = Sc[:, own].mean(axis=1)
        # tick-weighted mean over owned runs (long runs count more)
        w = np.array([runs[j].T for j in own], float)
        mean_own_w = (Sc[:, own] * w).sum(axis=1) / w.sum()
        # each predictor averaged over the owned runs it had not seen (nan if none)
        oot_mean = np.array([Sc[i, [j for j in own if U[i, j]]].mean() if any(U[i, j] for j in own)
                             else np.nan for i in range(n)])
        anc = {a[0]: a[2] for a in cal[sid].get("anchors", [])}
        cal_is = np.array([next((anc[k] for k in anc if set(k.split("=")) & set(p["ids"])), np.nan)
                           for p in preds])
        rec = {
            "n_predictors": n, "public_range": [float(pub.min()), float(pub.max())],
            "predictors": [{"ids": p["ids"], "kind": p["kind"], "public": float(pub[i]),
                            "run_scores": [round(float(x), 4) for x in Sc[i]],
                            "unseen": [bool(x) for x in U[i]]} for i, p in enumerate(preds)],
            "runs": rr,
            "mean_all_runs": agree(mean_all, pub),
            "mean_owned_runs": agree(mean_own, pub),
            "mean_owned_runs_tickweighted": agree(mean_own_w, pub),
            "mean_owned_unseen_only": agree(oot_mean, pub),
            "calibration_insample": agree(cal_is, pub),
            "mean_owned_vs_sustained": agree(mean_own, sus),
            "mean_owned_vs_sequence": agree(mean_own, seq),
        }
        # in-sample inflation: does the local-vs-public residual grow with the share of runs seen?
        seen_frac = 1.0 - U[:, own].mean(axis=1)
        A = np.c_[np.ones(n), pub]
        resid = mean_own - A @ np.linalg.lstsq(A, mean_own, rcond=None)[0]
        rec["inflation"] = {"seen_frac": [round(float(x), 3) for x in seen_frac],
                            "resid": [round(float(x), 4) for x in resid],
                            "corr": (round(float(np.corrcoef(seen_frac, resid)[0, 1]), 3)
                                     if np.std(seen_frac) > 0 else None)}
        late = np.array([p["ids"][0] >= "u008" for p in preds])        # ODE-era candidates
        rec["late_subset"] = {"n": int(late.sum()),
                              "mean_owned": agree(mean_own, pub, late),
                              "runs": {runs[j].exp: agree(Sc[:, j], pub, late) for j in range(len(runs))}}
        pa_own = [(rr[j]["all"]["pa"] or 0, j) for j in own]
        jb = max(pa_own)[1]
        rec["best_owned_run"] = {"exp": runs[jb].exp, **rr[jb]["all"], "oot": rr[jb]["oot"]}
        oo = [(rr[j]["oot"]["pa"], j) for j in own if rr[j]["oot"]["pa"] is not None]
        rec["best_owned_run_oot"] = ({"exp": runs[max(oo)[1]].exp, **rr[max(oo)[1]]["oot"]}
                                     if oo else None)
        if ex:
            j = ex[0]
            m = U[:, j]                           # predictors that had not seen the exam
            rec["exam"] = {"all": rr[j]["all"], "oot": rr[j]["oot"]}
            # same predictor subset, owned evidence only
            rec["exam_subset_compare"] = {
                "subset_ids": [p["ids"] for i, p in enumerate(preds) if m[i]],
                "exam": agree(Sc[:, j], pub, m),
                "mean_owned": agree(mean_own, pub, m),
                "best_owned_run": max(((agree(Sc[:, k], pub, m)["pa"] or 0, runs[k].exp)
                                       for k in own)),
                "owned_runs": {runs[k].exp: agree(Sc[:, k], pub, m) for k in own},
            }
            print("   exam subset:", json.dumps({k: v for k, v in rec["exam_subset_compare"].items()
                                                 if k != "subset_ids"}, default=str))
        out["systems"][sid] = rec
        print(f"{sid}: n={n} own-mean PA {rec['mean_owned_runs']['pa']} rho "
              f"{rec['mean_owned_runs']['rho']} | best run {rec['best_owned_run']['exp']} "
              f"PA {rec['best_owned_run']['pa']} | exam {rec.get('exam')}  {time.time()-t0:.0f}s")
        for x in rr:
            print(f"   {x['exp']:24s} {x['shape']:10s} all {x['all']['pa']} rho {x['all']['rho']}"
                  f" | oot n={x['oot']['n']} {x['oot']['pa']} rho {x['oot']['rho']}"
                  f" | sus {x['sus']['rho']} seq {x['seq']['rho']} | hard {x['all']['pa_hard']}"
                  f"/{x['all']['pairs_hard']} late {rec['late_subset']['runs'][x['exp']]['pa']}"
                  f" lateHard {rec['late_subset']['runs'][x['exp']]['pa_hard']}")
        print("   mean-owned hard", rec["mean_owned_runs"]["pa_hard"], "late", rec["late_subset"]["mean_owned"],
              "unseen-only", rec["mean_owned_unseen_only"]["pa"], "cal", rec["calibration_insample"]["pa"],
              "tickw", rec["mean_owned_runs_tickweighted"]["pa"])
        print("   inflation corr(seen_frac, local-public resid)", rec["inflation"]["corr"])
        print("   pubs", [(p["ids"][0], round(float(pub[i]), 4)) for i, p in enumerate(preds)])
    (ROOT / "plans" / "exam_value.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
