"""Calibrate the organizer's per-observable sigma from scored public uploads. Free: no gateway calls.

For every system we replay each scored upload's shipped predict.py on our own ledger runs
(y0 = run.y0, U = run.U, compared with run.Y) and find the multiplier k such that the run-based
score at sigma = k * sigma_proxy matches the public score across uploads:

    ours_i(k) = mean_runs mean_{t,j} 1 / (1 + |yhat_i - y| / (k_j sigma_proxy_j))
    k* = argmin_k sum_i (ours_i(k) - public_i)^2

Per-observable k_j are fitted (log-space, weak ridge towards k*) and kept only when identifiable
(enough distinct anchors, well-conditioned Jacobian, real SSE drop). Consistency check: holds vs
sustained band, the other runs vs sequence band. Noise check: sigma vs 2x diff-based noise.
Optional alternative (--alt): eval_like schedules with the best-scored upload as truth.

    python scripts/calibrate_sigma.py [--systems a,b] [--alt]
Writes plans/sigma_calibrated.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, metric, design as D                          # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs                           # noqa: E402

warnings.filterwarnings("ignore")
SYSTEMS = ["epidemic", "market", "traffic", "power_grid", "supply_chain", "wildlife", "reservoir",
           "ad_auction", "social_contagion", "hospital_queue"]
LOGK = np.linspace(np.log(0.005), np.log(5.0), 400)
CATS = ("sustained", "order", "recovery", "composition")


def sustained_like(exp):
    e = exp.lower()
    return ("hold" in e) or ("multilevel" in e) or ("pulse200" in e) or ("pulse120_280" in e)


def noise_of(runs):
    d = np.concatenate([np.diff(r.Y, axis=0) for r in runs if r.T > 2], axis=0)
    return np.median(np.abs(d - np.median(d, axis=0)), axis=0) * 1.4826 / np.sqrt(2)


def load_predict(folder, sid, tag):
    p = ROOT / folder.replace("\\", "/") / sid / "predict.py"
    if not p.exists():
        return None, None
    h = hashlib.sha1(p.read_bytes() + (p.parent / "model.json").read_bytes()
                     if (p.parent / "model.json").exists() else p.read_bytes()).hexdigest()[:10]
    spec_ = importlib.util.spec_from_file_location(f"cal_{sid}_{tag}", str(p))
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod, h


def roll(mod, spec, y0, U):
    obs, ctrls = list(spec.observables), list(spec.controls)
    init = {o: float(v) for o, v in zip(obs, y0)}
    iv = [{c: float(v) for c, v in zip(ctrls, row)} for row in U]
    out = mod.predict(init, iv, spec.context())
    return np.array([[row[o] for o in obs] for row in out], float)


def upload_ts(folder):
    m = re.search(r"(\d{8}-\d{4})", folder.replace("\\", "/"))
    return time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M")) if m else 0.0


def roll_any(mod, spec, y0, U):
    return np.tile(np.asarray(y0, float), (len(U), 1)) if mod is None else roll(mod, spec, y0, U)


def run_scores(E, sig, kvec, runs_mask=None):
    """E: list over runs of |err| [T,p]. Returns mean over runs of per-run mean score."""
    s = sig * kvec
    vals = [np.mean(1.0 / (1.0 + e / s[None, :])) for i, e in enumerate(E)
            if runs_mask is None or runs_mask[i]]
    return float(np.mean(vals)) if vals else float("nan")


def per_obs_scores(E, sig, kvec):
    s = sig * kvec
    return np.mean([np.mean(1.0 / (1.0 + e / s[None, :]), axis=0) for e in E], axis=0)


def fit_scalar_k(Es, pubs, sig, p, mask=None):
    def sse(lk):
        k = np.full(p, np.exp(lk))
        return sum((run_scores(E, sig, k, mask) - y) ** 2 for E, y in zip(Es, pubs))
    grid = np.array([sse(lk) for lk in LOGK[::8]])
    i0 = int(np.argmin(grid))
    lo, hi = LOGK[::8][max(0, i0 - 1)], LOGK[::8][min(len(grid) - 1, i0 + 1)]
    r = minimize_scalar(sse, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x)), float(r.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--alt", action="store_true", help="also eval_like check (truth = best upload)")
    ap.add_argument("--altT", type=int, default=1000)
    ap.add_argument("--keep-l1", action="store_true", help="keep l1 anchors in the primary fit")
    a = ap.parse_args()
    reg = json.loads((ROOT / "tune" / "registry.json").read_text())["uploads"]
    bands = json.loads((ROOT / "tune" / "bands.json").read_text())["rows"]
    outp = ROOT / "plans" / "sigma_calibrated.json"
    out = json.loads(outp.read_text()) if outp.exists() else {}
    t0 = time.time()
    for sid in a.systems.split(","):
        spec = S.get(sid)
        runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
        sig = metric.sigma_proxy(runs)
        noise = noise_of(runs)
        p = len(spec.observables)
        sus_mask = [sustained_like(r.exp) for r in runs]
        run_ts = {}
        for row in Ledger(data_dir(sid, False), sid, False).rows:
            if row.get("t") == "reset" and "ts" in row:
                run_ts.setdefault(row["run_id"], row["ts"])
        # ---- anchors: every scored upload of this system, deduplicated on identical predictions
        anchors = []
        for u in reg:
            pub = (u.get("scores") or {}).get(sid)
            if pub is None:
                continue
            kind = (u.get("systems") or {}).get(sid, {}).get("kind", "?")
            mod, h = load_predict(u["dir"], sid, u["id"])
            if mod is None and kind != "l0a":
                print(sid, u["id"], "no predict.py")
                continue
            try:
                if mod is None:                      # persistence zip only: y_t = y0
                    P = [np.tile(r.y0, (r.T, 1)) for r in runs]
                else:
                    P = [roll(mod, spec, r.y0, r.U) for r in runs]
            except Exception as e:
                print(sid, u["id"], "predict failed", type(e).__name__, e)
                continue
            E = [np.abs(Ph - r.Y) for Ph, r in zip(P, runs)]
            dup = None
            for an in anchors:
                if all(np.max(np.abs(x - y)) < 1e-9 for x, y in zip(an["P"], P)):
                    dup = an
                    break
            bnd = [b for b in bands if b["upload"] == u["id"] and b["system"] == sid]
            if dup is not None:
                dup["ids"].append(u["id"])
                dup["pubs"].append(pub)
                if bnd and dup["band"] is None:
                    dup["band"] = bnd[0]
                continue
            up_ts = upload_ts(u["dir"])
            unseen = [run_ts[r.tags.get("run_id")] > up_ts for r in runs]
            anchors.append({"id": u["id"], "ids": [u["id"]], "pubs": [pub], "kind": kind, "P": P,
                            "E": E, "band": bnd[0] if bnd else None, "hash": h,
                            "unseen": unseen if kind != "l0a" else [True] * len(runs)})
        for an in anchors:
            an["pub"] = float(np.mean(an["pubs"]))
        all_anchors = anchors
        # l1 anchors are excluded from the primary fit: their state init is fitted on noisy y0 of the
        # same runs (in-sample overfit), so their run-based score says nothing about the test.
        excluded = [an["id"] for an in anchors if str(an["kind"]).startswith("l1") and not a.keep_l1]
        k_with_l1 = fit_scalar_k([an["E"] for an in anchors], np.array([an["pub"] for an in anchors]),
                                 sig, p)[0]
        anchors = [an for an in anchors if an["id"] not in excluded]
        Es = [an["E"] for an in anchors]
        pubs = np.array([an["pub"] for an in anchors])
        n = len(anchors)
        print(f"{sid}: {n} anchors {[an['ids'] for an in anchors]} excluded {excluded}  {time.time()-t0:.0f}s")

        # ---- scalar k
        k1, sse1 = fit_scalar_k(Es, pubs, sig, p)
        kv = np.full(p, k1)
        ours = np.array([run_scores(E, sig, kv) for E in Es])
        ss_tot = float(np.sum((pubs - pubs.mean()) ** 2))
        r2 = 1 - sse1 / ss_tot if ss_tot > 0 else float("nan")
        # profile: range of k with SSE within +50% of min (+ 1e-4 absolute)
        prof = []
        for lk in LOGK[::4]:
            kk = np.full(p, np.exp(lk))
            prof.append(sum((run_scores(E, sig, kk) - y) ** 2 for E, y in zip(Es, pubs)))
        prof = np.array(prof)
        ok = prof <= sse1 * 1.5 + 1e-4
        k_lo, k_hi = float(np.exp(LOGK[::4][ok].min())), float(np.exp(LOGK[::4][ok].max()))
        # tick-pooled variant
        def pooled(E, kvv):
            s = sig * kvv
            return float(np.mean(np.concatenate([1.0 / (1.0 + e / s[None, :]) for e in E], axis=0)))
        lk_pool = minimize_scalar(lambda lk: sum((pooled(E, np.full(p, np.exp(lk))) - y) ** 2
                                                 for E, y in zip(Es, pubs)),
                                  bounds=(LOGK[0], LOGK[-1]), method="bounded").x
        k_pool = float(np.exp(lk_pool))
        # leave-one-anchor-out prediction error
        loo = []
        if n >= 3:
            for i in range(n):
                m = [j for j in range(n) if j != i]
                ki, _ = fit_scalar_k([Es[j] for j in m], pubs[m], sig, p)
                loo.append(run_scores(Es[i], sig, np.full(p, ki)) - pubs[i])
        # ---- per-observable k (log space, ridge 1e-3 towards k1)
        lam = 1e-3
        def sse_vec(lkv):
            kvv = np.exp(lkv)
            r_ = sum((run_scores(E, sig, kvv) - y) ** 2 for E, y in zip(Es, pubs))
            return r_ + lam * float(np.sum((lkv - np.log(k1)) ** 2))
        res = minimize(sse_vec, np.full(p, np.log(k1)), method="L-BFGS-B",
                       bounds=[(LOGK[0], LOGK[-1])] * p)
        kpo = np.exp(res.x)
        sse_po = float(sum((run_scores(E, sig, kpo) - y) ** 2 for E, y in zip(Es, pubs)))
        loo_po = []
        if n >= p + 2:
            for i in range(n):
                m = [j for j in range(n) if j != i]
                def f_i(lkv, m=m):
                    return sum((run_scores(Es[j], sig, np.exp(lkv)) - pubs[j]) ** 2 for j in m)                         + lam * float(np.sum((lkv - np.log(k1)) ** 2))
                ri = minimize(f_i, np.full(p, np.log(k1)), method="L-BFGS-B",
                              bounds=[(LOGK[0], LOGK[-1])] * p)
                loo_po.append(run_scores(Es[i], sig, np.exp(ri.x)) - pubs[i])
        loo_po_rmse = float(np.sqrt(np.mean(np.square(loo_po)))) if loo_po else None
        # Jacobian d ours_i / d log k_j at k1 (identifiability)
        J = np.zeros((n, p))
        for j in range(p):
            dk = np.full(p, k1)
            dk[j] *= np.exp(0.05)
            J[:, j] = [(run_scores(E, sig, dk) - o) / 0.05 for E, o in zip(Es, ours)]
        sv = np.linalg.svd(J, compute_uv=False) if n else np.zeros(1)
        cond = float(sv[0] / max(sv[-1], 1e-12)) if len(sv) >= p else float("inf")
        loo_rmse = float(np.sqrt(np.mean(np.square(loo)))) if loo else None
        ident = bool(n >= p + 2 and cond < 30 and sse_po < 0.5 * sse1 and loo_po_rmse is not None
                     and loo_rmse is not None and loo_po_rmse < 0.8 * loo_rmse)
        # ---- out-of-sample variant: each anchor scored only on runs bought after it was uploaded
        oos = [(an["E"], an["pub"], an["unseen"], an["id"]) for an in anchors if any(an["unseen"])]
        k_oos = None
        oos_rows = []
        if len(oos) >= 2:
            def sse_oos(lk):
                kk = np.full(p, np.exp(lk))
                return sum((run_scores(E, sig, kk, m) - y) ** 2 for E, y, m, _ in oos)
            k_oos = float(np.exp(minimize_scalar(sse_oos, bounds=(LOGK[0], LOGK[-1]), method="bounded").x))
            oos_rows = [[i_, y, run_scores(E, sig, np.full(p, k_oos), m), sum(m)] for E, y, m, i_ in oos]
        # ---- bands
        bsus, bseq = [], []
        for an in anchors:
            b = an["band"]
            if b is None:
                continue
            bsus.append((an["E"], b["sustained"]))
            bseq.append((an["E"], b["sequence"]))
        band_fit = None
        if len(bsus) >= 2 and any(sus_mask) and not all(sus_mask):
            ks, _ = fit_scalar_k([x[0] for x in bsus], np.array([x[1] for x in bsus]), sig, p, sus_mask)
            nm = [not m for m in sus_mask]
            kq, _ = fit_scalar_k([x[0] for x in bseq], np.array([x[1] for x in bseq]), sig, p, nm)
            band_fit = {
                "k_sustained_band_on_hold_runs": ks, "k_sequence_band_on_other_runs": kq,
                "hold_runs": [r.exp for r, m in zip(runs, sus_mask) if m],
                "other_runs": [r.exp for r, m in zip(runs, sus_mask) if not m],
                "rows": [[an["id"], an["band"]["sustained"], run_scores(an["E"], sig, kv, sus_mask),
                          an["band"]["sequence"], run_scores(an["E"], sig, kv, nm)]
                         for an in anchors if an["band"] is not None]}
        # ---- noise sanity
        sig_org = sig * k1
        ratio = sig_org / np.maximum(noise, 1e-12)
        # per-observable score of the best anchor at k1 (which observable dominates the loss)
        best = anchors[int(np.argmax(pubs))]
        po_best_k = per_obs_scores(best["E"], sig, kv)
        po_best_1 = per_obs_scores(best["E"], sig, np.ones(p))
        po_best_f = per_obs_scores(best["E"], sig, np.maximum(sig_org, 2 * noise) / sig)
        rec = {
            "observables": list(spec.observables),
            "k": k1,
            "k_per_obs": [float(x) for x in kpo] if ident else None,
            "k_per_obs_unidentified_fit": [float(x) for x in kpo],
            "sigma": [float(x) for x in np.maximum(sig_org, 2 * noise)],
            "sigma_unfloored": [float(x) for x in sig_org],
            "sigma_per_obs_fit": [float(x) for x in sig * kpo] if ident else None,
            "sigma_proxy": [float(x) for x in sig],
            "noise": [float(x) for x in noise],
            "sigma_over_noise": [float(x) for x in ratio],
            "flag_sigma_below_2x_noise": [spec.observables[j] for j in range(p) if ratio[j] < 2],
            "anchors": [["=".join(an["ids"]), an["pub"], run_scores(an["E"], sig, kv)]
                        for an in all_anchors],
            "anchor_kinds": {an["id"]: an["kind"] for an in all_anchors},
            "sigma_floored": [float(x) for x in np.maximum(sig_org, 2 * noise)],
            "loss_share_best_anchor": {
                "at_sigma_proxy": [float(x) for x in (1 - po_best_1) / np.sum(1 - po_best_1)],
                "at_k": [float(x) for x in (1 - po_best_k) / np.sum(1 - po_best_k)],
                "at_sigma_floored": [float(x) for x in (1 - po_best_f) / np.sum(1 - po_best_f)]},
            "fit": {
                "n_anchors": n, "sse": sse1, "rmse": float(np.sqrt(sse1 / max(n, 1))), "r2": r2,
                "residuals": {an["id"]: float(o - an["pub"]) for an, o in zip(anchors, ours)},
                "k_range_sse_within_50pct": [k_lo, k_hi],
                "k_tick_pooled": k_pool,
                "loo_residuals": [float(x) for x in loo],
                "k_out_of_sample": k_oos,
                "oos_rows_[upload,public,ours,n_unseen_runs]": oos_rows,
                "loo_rmse": loo_rmse,
                "per_obs": {"sse": sse_po, "jacobian_cond": cond, "loo_rmse": loo_po_rmse,
                            "identifiable": ident},
                "excluded_from_fit": excluded, "k_with_excluded": k_with_l1,
            },
            "bands": band_fit,
            "per_obs_score_best_anchor": {"anchor": best["id"], "at_k": [float(x) for x in po_best_k],
                                          "at_sigma_proxy": [float(x) for x in po_best_1]},
            "runs": [[r.exp, r.T, bool(m)] for r, m in zip(runs, sus_mask)],
        }
        # ---- alternative: eval_like schedules, best-scored upload as truth
        if a.alt and n >= 3:
            rng = np.random.default_rng(20260926)
            tests = []
            for ci, c in enumerate(CATS):
                for q in range(2):
                    tests.append((runs[(2 * ci + q) % len(runs)].y0, D.eval_like(spec, c, a.altT, rng)))
            ib = int(np.argmax(pubs))
            mods = {}
            for an in anchors:
                u = next(x for x in reg if x["id"] == an["id"])
                mods[an["id"]] = load_predict(u["dir"], sid, an["id"] + "_alt")[0]
            truth = [roll_any(mods[anchors[ib]["id"]], spec, y0, U) for y0, U in tests]
            Ea, pa = [], []
            for i, an in enumerate(anchors):
                if i == ib:
                    continue
                Ya = [roll_any(mods[an["id"]], spec, y0, U) for y0, U in tests]
                Ea.append([np.abs(x - y) for x, y in zip(Ya, truth)])
                pa.append(an["pub"])
            ka, ssea = fit_scalar_k(Ea, np.array(pa), sig, p)
            rec["alt_eval_like"] = {"truth": anchors[ib]["id"], "k": ka,
                                    "rmse": float(np.sqrt(ssea / len(pa))), "T": a.altT,
                                    "n_schedules": len(tests)}
        out[sid] = rec
        print(f"  k={k1:.4f} [{k_lo:.3f},{k_hi:.3f}] r2={r2:.3f} rmse={rec['fit']['rmse']:.4f} "
              f"k_pool={k_pool:.3f} k_oos={k_oos} ident={ident} kpo={np.round(kpo,3)} "
              f"band={None if band_fit is None else (round(band_fit['k_sustained_band_on_hold_runs'],3), round(band_fit['k_sequence_band_on_other_runs'],3))} "
              f"alt={rec.get('alt_eval_like',{}).get('k')}")
        for an in all_anchors:
            o = run_scores(an["E"], sig, kv)
            tag = " (excluded)" if an["id"] in excluded else ""
            print(f"    {'='.join(an['ids']):<22} {an['kind'][:40]:<40} pub {an['pub']:.4f} ours {o:.4f}{tag}")
        print("    loss share best @proxy", np.round(rec["loss_share_best_anchor"]["at_sigma_proxy"], 2),
              "@k", np.round(rec["loss_share_best_anchor"]["at_k"], 2),
              "@floored", np.round(rec["loss_share_best_anchor"]["at_sigma_floored"], 2),
              "k_with_l1", round(k_with_l1, 3), "loo", rec["fit"]["loo_rmse"], "loo_po", loo_po_rmse)
        print("    sigma/noise", np.round(ratio, 2), "flag", rec["flag_sigma_below_2x_noise"])
        outp.write_text(json.dumps(out, indent=1))
    print(f"done {time.time()-t0:.0f}s -> {outp}")


if __name__ == "__main__":
    main()
