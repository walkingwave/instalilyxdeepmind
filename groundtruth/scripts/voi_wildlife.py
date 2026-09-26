"""Value-of-information study for wildlife. Free: simulation only, no gateway calls.

1. Validate every candidate doc against its lab json 'insample' on the ledger runs.
2. Committee (validated docs + l0b_lin) on 40 test-shaped schedules (eval_like, 10 per category,
   T=4000): pairwise score matrix, disagreement by category / observable / control regime.
3. Anchor calibration: roll every uploaded wildlife model that has a public score on the same
   40 schedules; for each committee member taken as "truth" (and sigma multiplier), compare the
   implied anchor scores with the public ones.
4. Candidate experiments (100-300 ticks): expected value of sample information with members as
   truth hypotheses, plus a regime-weighted disagreement heuristic. Rank by value per credit.
5. Free model options: combinations of members (median, means, per-observable picks) scored
   against the calibrated truth weights and in-sample on our runs.

    python scripts/voi_wildlife.py [--quick]
Writes plans/voi_wildlife.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import systems as S, metric, design as D, select as SEL           # noqa: E402
from gtlab.models import common as C                                         # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs                         # noqa: E402
from gtlab.runtime.infer import rollout_from_blob, finalize, clip_vectors    # noqa: E402

warnings.filterwarnings("ignore")
SID = "wildlife"
CATS = ("sustained", "order", "recovery", "composition")
KMS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
DOCS = {                       # name -> doc file; v6 (min2) is a numerical duplicate of v7
    "v7": "wildlife_wildlife_min_v7_doc.json",
    "AB": "wildlife_wildlife_mech_AB_doc.json",
    "ABr2": "wildlife_wildlife_mech_ABJ_r2_doc.json",
    "AC": "wildlife_wildlife_mech_AC_doc.json",
    "ACr2": "wildlife_wildlife_mech_ACJ_r2_doc.json",
    "BC": "wildlife_wildlife_mech_BC_doc.json",
    "v1": "wildlife_wildlife_min_v1_doc.json", "v2": "wildlife_wildlife_min_v2_doc.json",
    "v3": "wildlife_wildlife_min_v3_doc.json", "v4": "wildlife_wildlife_min_v4_doc.json",
    "v5": "wildlife_wildlife_min_v5_doc.json", "min": "wildlife_wildlife_min_doc.json",
    "min_p3": "wildlife_wildlife_min_p3_doc.json", "v6": "wildlife_wildlife_min2_v6_doc.json",
}
ANCHORS = {                    # uploaded wildlife models with a public score
    "l0a (u001)": (None, 0.2227),
    "l0b relax": ("20260924-2232-20260924-relax", 0.2785),
    "l1 u003": ("20260924-2301-u003-screen", 0.4292),
    "l0b u004": ("20260924-2305-u004-test", 0.5550),
    "ode u008": ("20260925-1925-u008-ode", 0.6297),
    "ode v7 u010b": ("20260926-0135-u010b", 0.6355),
}
PUBLIC_V7 = {"overall": 0.6355, "sustained": 0.611, "sequence": 0.644}


# ----------------------------------------------------------------------------- helpers
def load_predict(folder, tag):
    p = ROOT / "submissions" / folder / SID / "predict.py"
    spec = importlib.util.spec_from_file_location(f"anchor_{tag}", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def roll_predict(mod, obs, ctrls, y0, U):
    init = {o: float(v) for o, v in zip(obs, y0)}
    iv = [{c: float(v) for c, v in zip(ctrls, row)} for row in U]
    out = mod.predict(init, iv, {"family": SID, "observables": obs})
    return np.array([[row[o] for o in obs] for row in out], float)


def seg_changes(U):
    ch = np.r_[True, np.any(np.abs(np.diff(U, axis=0)) > 1e-12, axis=1)]
    return ch


def regimes(spec, U, seen):
    """Per-tick label of the control state relative to what our runs have covered."""
    lo, hi = D.lo_hi(spec)
    rng_ = hi - lo
    rec = D.rec(spec)
    pul = D.pulse_level(spec, 1.0)
    Un = (U - lo) / rng_
    seen_n = (np.asarray(seen) - lo) / rng_
    nov = np.min(np.max(np.abs(Un[:, None, :] - seen_n[None, :, :]), axis=2), axis=1)
    # alpha per control (pulse-like if every control sits on the rec->pulse segment with alpha>=.7)
    a = (U - rec) / (pul - rec)
    on_seg = np.all(np.abs(a - a.mean(axis=1, keepdims=True)) < 0.35, axis=1)
    pulse_like = np.all(a >= 0.69, axis=1)
    is_rec = np.all(np.abs(U - rec) < 1e-9, axis=1)
    at_bound = np.any((np.abs(U - lo) < 1e-9) | (np.abs(U - hi) < 1e-9), axis=1)
    # time since last control change
    ch = seg_changes(U)
    since = np.zeros(len(U), int)
    for t in range(1, len(U)):
        since[t] = 0 if ch[t] else since[t - 1] + 1
    # was the previous (non-recovery) block a short pulse? (repeated short pulses)
    lab = np.empty(len(U), object)
    for t in range(len(U)):
        if is_rec[t]:
            lab[t] = "recovery"
        elif pulse_like[t]:
            lab[t] = "pulse_a>=.7"
        elif nov[t] < 0.05:
            lab[t] = "seen_level"
        elif on_seg[t]:
            lab[t] = "interior_on_rec-pulse_line"
        elif not at_bound[t]:
            lab[t] = "interior_off_line"
        else:
            lab[t] = "corner_or_edge"
    dw = np.where(since < 45, "<45", np.where(since < 200, "45-200", ">200"))
    return lab, dw, nov


def experiments(spec, rng):
    """Candidate purchases. Controls: [hunting_quota, habitat_protection, corridor_access]."""
    rec = D.rec(spec)
    pul = D.pulse_level(spec, 1.0)
    H = lambda u, n: D.hold(spec, np.asarray(u, float), n)

    def ptrain(n, L, gap, alpha, lead=10, tail=60):
        rows = [H(rec, lead)]
        for _ in range(n):
            rows += [H(D.pulse_level(spec, alpha), L), H(rec, gap)]
        rows += [H(rec, tail)]
        return np.concatenate(rows)

    E = {
        "E01_habitat05_hold": ("habitat 0.5 alone 150, recovery 50",
                               np.concatenate([H([0, .5, 0], 150), H(rec, 50)])),
        "E02_corridor_only": ("corridor 0.5 60, off 30, corridor 1 60, off 30",
                              np.concatenate([H([0, 1, .5], 60), H(rec, 30), H([0, 1, 1], 60), H(rec, 30)])),
        "E03_harvest_before_after_protect": (
            "brief: quota 6 with habitat 0.1 then habitat 1 (harvest before protection), rest, then "
            "habitat 1 then quota 6 at habitat 1 (after)",
            np.concatenate([H([6, .1, 0], 50), H([0, 1, 0], 50), H([0, .1, 0], 50), H([6, 1, 0], 50), H(rec, 30)])),
        "E04_habitat_recovery_corridor_closed_open": (
            "brief: habitat 0.1 then recovery with corridor closed, habitat 0.1 then habitat 1 with corridor open",
            np.concatenate([H([0, .1, 0], 60), H([0, 1, 0], 60), H([0, .1, 1], 60), H([0, 1, 1], 60)])),
        "E05_pulses_gap10": ("6 full pulses of 10 ticks, gaps 10, then recovery",
                             ptrain(6, 10, 10, 1.0, tail=70)),
        "E06_pulses_gap40": ("5 full pulses of 10 ticks, gaps 40, then recovery",
                             ptrain(5, 10, 40, 1.0, tail=0)),
        "E07_alpha05_hold": ("interior rec+0.5(pulse-rec) = (3.5, 0.55, 0.5) 150, recovery 50",
                             np.concatenate([H(D.pulse_level(spec, .5), 150), H(rec, 50)])),
        "E08_alpha07_train": ("5 pulses at alpha 0.7 of 20 ticks, gaps 30",
                              ptrain(5, 20, 30, 0.7, tail=0)),
        "E09_max_stress": ("corner (8, 0, 1) 100, recovery 100",
                           np.concatenate([H([8, 0, 1], 100), H(rec, 100)])),
        "E10_hunt_mid_alone": ("quota 3.5 at habitat 1 150, recovery 50",
                               np.concatenate([H([3.5, 1, 0], 150), H(rec, 50)])),
        "E11_long_pulse_hold": ("full pulse 250, recovery 50",
                                np.concatenate([H(pul, 250), H(rec, 50)])),
        "E12_recovery_like": ("test-shaped recovery train, T=300",
                              D.eval_like(spec, "recovery", 300, np.random.default_rng(11))),
        "E13_sustained_like": ("test-shaped sustained holds, T=300",
                               D.eval_like(spec, "sustained", 300, np.random.default_rng(12))),
        "E14_habitat05_then_pulses": ("habitat 0.5 100, then 3 full pulses 10 ticks gap 20, recovery",
                                      np.concatenate([H([0, .5, 0], 100), ptrain(3, 10, 20, 1.0, lead=0, tail=40)])),
        "E15_corridor_then_habitat05": (
            "combined: corridor 0.5 60, off 30, corridor 1 60, off 30, habitat 0.5 100, recovery 20",
            np.concatenate([H([0, 1, .5], 60), H(rec, 30), H([0, 1, 1], 60), H(rec, 30),
                            H([0, .5, 0], 100), H(rec, 20)])),
        "E16_pulse_then_long_recovery": ("full pulse 60, recovery 240 (settle and drift past 200 ticks)",
                                         np.concatenate([H(pul, 60), H(rec, 240)])),
    }
    return {k: (d, D.clip(spec, U)) for k, (d, U) in E.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="4 schedules per category")
    ap.add_argument("--T", type=int, default=4000)
    a = ap.parse_args()
    t0 = time.time()
    spec = S.get(SID)
    runs = load_runs(spec, Ledger(data_dir(SID, False), SID, False))
    sig = metric.sigma_proxy(runs)
    obs, ctrls = list(spec.observables), list(spec.controls)
    out = {"system": SID, "sigma_proxy": sig.tolist(), "runs": [[r.exp, r.T] for r in runs]}

    # 1. validation ------------------------------------------------------------------
    members, valid = {}, {}
    for name, f in DOCS.items():
        doc = json.loads((ROOT / "plans" / f).read_text())
        lab = json.loads((ROOT / "plans" / f.replace("_doc.json", ".json")).read_text())
        ins = lab.get("insample", {})
        lsig = np.array(lab.get("sigma_proxy", sig), float)
        rows, ok = {}, True
        for r in runs:
            try:
                Y = rollout_from_blob(doc["model"], r.y0, r.U, doc=doc)
                s = float(np.mean(metric.score_per_obs(Y, r.Y, lsig)))
            except Exception as e:
                s = f"ERR {type(e).__name__}"
            ref = ins.get(r.exp)
            ref = None if ref is None else float(np.mean(ref))
            rows[r.exp] = {"now": s, "lab": ref}
            if isinstance(s, str) or ref is None or abs(s - ref) > 1e-3:
                ok = False
        valid[name] = {"reproduces": ok, "runs": rows}
        if ok and name != "v6":
            members[name] = ("doc", doc)
    clip = C.soft_clip(spec, runs, margin=1.0)
    l0 = SEL.make_model("l0b", spec, clip, sig, cfg={"sq": False, "pairs": None})
    l0.fit(runs)
    members["l0b_lin"] = ("model", l0)
    out["validation"] = valid
    names = list(members)
    print("members:", names, f"{time.time()-t0:.0f}s")

    def roll(name, y0, U):
        kind, m = members[name]
        if kind == "doc":
            return rollout_from_blob(m["model"], y0, U, doc=m)
        return m.rollout(y0, U)

    # in-sample of every member (sigma_proxy of all runs)
    out["insample"] = {n: {r.exp: float(metric.score(roll(n, r.y0, r.U), r.Y, sig)) for r in runs}
                       for n in names}

    # 2. test schedules -----------------------------------------------------------------
    rng = np.random.default_rng(20260926)
    nper = 4 if a.quick else 10
    tests = []
    for c in CATS:
        for k in range(nper):
            tests.append((c, runs[k % len(runs)].y0, D.eval_like(spec, c, a.T, rng)))
    seen = np.unique(np.round(np.concatenate([r.U for r in runs]), 6), axis=0)
    P = {n: [] for n in names}
    for c, y0, U in tests:
        for n in names:
            P[n].append(roll(n, y0, U))
    print("committee rolled", f"{time.time()-t0:.0f}s")

    # anchors
    anc = {}
    for an, (folder, pub) in ANCHORS.items():
        try:
            if folder is None:
                Ya = [np.tile(y0, (len(U), 1)) for c, y0, U in tests]
            else:
                mod = load_predict(folder, an.replace(" ", "_").replace("(", "").replace(")", ""))
                Ya = [roll_predict(mod, obs, ctrls, y0, U) for c, y0, U in tests]
            flat = float(np.mean([np.mean(np.all(np.abs(Y - Y[0]) < 1e-12, axis=1)) for Y in Ya]))
            anc[an] = {"pub": pub, "Y": Ya, "frac_constant": flat}
        except Exception as e:
            print("anchor fail", an, e)
    print("anchors rolled", f"{time.time()-t0:.0f}s")

    # regime labels
    labs = [regimes(spec, U, seen) for c, y0, U in tests]
    cat_of = np.array([c for c, _, _ in tests])

    # pairwise score / disagreement ------------------------------------------------------
    def sc_tick(Yh, Yt, s=sig):
        return 1.0 / (1.0 + np.abs(Yh - Yt) / s[None, :])            # [T,p]

    K = len(names)
    Smat = {c: np.zeros((K, K)) for c in CATS + ("all",)}
    Dobs = {c: np.zeros(len(obs)) for c in CATS}
    npair = 0
    for i in range(K):
        for j in range(K):
            per = np.array([sc_tick(P[names[i]][e], P[names[j]][e]).mean() for e in range(len(tests))])
            for c in CATS:
                Smat[c][i, j] = per[cat_of == c].mean()
            Smat["all"][i, j] = per.mean()
    for i, j in combinations(range(K), 2):
        npair += 1
        for e, (c, _, _) in enumerate(tests):
            Dobs[c] += np.mean(np.abs(P[names[i]][e] - P[names[j]][e]) / sig, axis=0) / nper
    Dobs = {c: (v / npair).tolist() for c, v in Dobs.items()}
    out["members"] = names
    out["score_matrix"] = {c: np.round(m, 4).tolist() for c, m in Smat.items()}   # row=predictor, col=truth
    out["disagreement_by_cat_obs"] = Dobs

    # regime table: frequency and pairwise disagreement (all pairs, and v7 vs ABr2)
    reg = {}
    for e in range(len(tests)):
        lab, dw, nov = labs[e]
        dall = np.zeros(a.T)
        for i, j in combinations(range(K), 2):
            dall += np.mean(np.abs(P[names[i]][e] - P[names[j]][e]) / sig, axis=1)
        dall /= npair
        dv = np.mean(np.abs(P["v7"][e] - P["ABr2"][e]) / sig, axis=1) if "ABr2" in P else dall
        lv7 = 1 - np.mean(sc_tick(P["v7"][e], P["ABr2"][e]), axis=1) if "ABr2" in P else dall
        for key in set(zip(lab, dw)):
            msk = (lab == key[0]) & (dw == key[1])
            k = f"{key[0]} | dwell {key[1]}"
            r = reg.setdefault(k, {"ticks": 0, "d_all": 0.0, "d_v7_ABr2": 0.0, "loss_v7_vs_ABr2": 0.0,
                                   "by_cat": {c: 0 for c in CATS}})
            r["ticks"] += int(msk.sum())
            r["d_all"] += float(dall[msk].sum())
            r["d_v7_ABr2"] += float(dv[msk].sum())
            r["loss_v7_vs_ABr2"] += float(lv7[msk].sum())
            r["by_cat"][tests[e][0]] += int(msk.sum())
    tot = sum(r["ticks"] for r in reg.values())
    ltot = max(1e-9, sum(x["loss_v7_vs_ABr2"] for x in reg.values()))
    for k, r in reg.items():
        n = max(1, r["ticks"])
        r["freq"] = r["ticks"] / tot
        r["d_all"] /= n
        r["d_v7_ABr2"] /= n
        r["loss_share"] = r["loss_v7_vs_ABr2"] / ltot
        r["loss_v7_vs_ABr2"] /= n
    out["regimes"] = dict(sorted(reg.items(), key=lambda kv: -kv[1]["ticks"]))
    # overall novelty of the test distribution
    out["novel_tick_share"] = {
        "control_novelty>0.05": float(np.mean(np.concatenate([l[2] for l in labs]) > 0.05)),
        "dwell>200": float(np.mean(np.concatenate([l[1] for l in labs]) == ">200")),
        "max_dwell_in_our_runs": 200}

    # 3. anchor calibration ------------------------------------------------------------
    cal = {}
    for j, tn in enumerate(names):
        for km in KMS:
            s = sig * km
            pred = {an: float(np.mean([sc_tick(A["Y"][e], P[tn][e], s).mean() for e in range(len(tests))]))
                    for an, A in anc.items()}
            x = np.array([pred[an] for an in anc])
            y = np.array([anc[an]["pub"] for an in anc])
            cal[f"{tn}@{km}"] = {"truth": tn, "sig_mult": km, "pred": pred,
                                 "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                                 "corr": float(np.corrcoef(x, y)[0, 1])}
    out["anchor_calibration"] = dict(sorted(cal.items(), key=lambda kv: kv[1]["rmse"]))
    out["anchor_frac_constant"] = {an: A["frac_constant"] for an, A in anc.items()}
    # truth weights: best sigma multiplier per member, weight exp(-rmse^2 / (2 * 0.03^2))
    best = {}
    for v in cal.values():
        if v["truth"] not in best or v["rmse"] < best[v["truth"]]["rmse"]:
            best[v["truth"]] = v
    w = np.array([np.exp(-0.5 * (best[n]["rmse"] / 0.03) ** 2) for n in names])
    w = w / w.sum()
    out["truth_weights_anchor"] = dict(zip(names, np.round(w, 4).tolist()))
    out["truth_best_sigma_mult"] = {n: best[n]["sig_mult"] for n in names}
    # one global multiplier: minimise the weight-averaged rmse over members
    kstar = min(KMS, key=lambda km: sum(wi * cal[f"{n}@{km}"]["rmse"] for wi, n in zip(w, names)))
    out["sigma_mult_calibrated"] = kstar
    sigc = sig * kstar
    SmatC = {c: np.zeros((K, K)) for c in CATS + ("all",)}
    for i in range(K):
        for j in range(K):
            per = np.array([sc_tick(P[names[i]][e], P[names[j]][e], sigc).mean() for e in range(len(tests))])
            for c in CATS:
                SmatC[c][i, j] = per[cat_of == c].mean()
            SmatC["all"][i, j] = per.mean()
    out["score_matrix_calibrated"] = {c: np.round(m, 4).tolist() for c, m in SmatC.items()}
    # public v7 category scores implied by each truth at kstar
    out["v7_implied_by_truth"] = {n: {c: float(SmatC[c][names.index("v7"), j]) for c in CATS + ("all",)}
                                  for j, n in enumerate(names)}

    # 4. free model options --------------------------------------------------------------
    ode = [n for n in names if n != "l0b_lin"]
    combos = {n: [n] for n in names}
    combos.update({"median_all_ode": ode, "median_all": names, "mean_v7_ABr2": ["v7", "ABr2"],
                   "mean_v7_AB": ["v7", "AB"], "median_v7_AB_ABr2": ["v7", "AB", "ABr2"],
                   "mean_v7_ABr2_l0b": ["v7", "ABr2", "l0b_lin"]})

    def combo_roll(cname, y0, U, cache=None):
        ms = combos[cname]
        Ys = [cache[m] if cache is not None else roll(m, y0, U) for m in ms]
        return np.median(Ys, axis=0) if cname.startswith("median") else np.mean(Ys, axis=0)

    # per-observable picks: prey from one member, predators from another
    perobs = {"prey_ABr2+pred_v7": ("ABr2", "v7"), "prey_AB+pred_v7": ("AB", "v7"),
              "prey_mean(v7,ABr2)+pred_v7": ("mean_v7_ABr2", "v7")}
    free = {}
    uni = np.ones(K) / K
    for cname in list(combos) + list(perobs):
        per_truth = np.zeros((len(tests), K))
        cat_truth = {c: np.zeros(K) for c in CATS}
        for e, (c, y0, U) in enumerate(tests):
            cache = {n: P[n][e] for n in names}
            if cname in perobs:
                pa, pb = perobs[cname]
                Ya = combo_roll(pa, y0, U, cache) if pa in combos else cache[pa]
                Yb = cache[pb]
                Yh = Ya.copy()
                Yh[:, [1, 3]] = Yb[:, [1, 3]]
            else:
                Yh = combo_roll(cname, y0, U, cache)
            for j, tn in enumerate(names):
                per_truth[e, j] = sc_tick(Yh, P[tn][e], sigc).mean()
        ins = []
        for r in runs:
            cache = {n: roll(n, r.y0, r.U) for n in names}
            if cname in perobs:
                pa, pb = perobs[cname]
                Ya = combo_roll(pa, r.y0, r.U, cache) if pa in combos else cache[pa]
                Yh = Ya.copy()
                Yh[:, [1, 3]] = cache[pb][:, [1, 3]]
            else:
                Yh = combo_roll(cname, r.y0, r.U, cache)
            ins.append(metric.score(Yh, r.Y, sig))
        mt = per_truth.mean(axis=0)
        # leave-self-out uniform: exclude truths that are members of the combo
        mem = set(combos.get(cname, perobs.get(cname, ())))
        lso = [mt[j] for j, tn in enumerate(names) if tn not in mem]
        free[cname] = {"exp_score_uniform": float(mt @ uni), "exp_score_anchor_w": float(mt @ w),
                       "exp_score_others": float(np.mean(lso)) if lso else None,
                       "worst_truth": float(mt.min()), "insample_mean": float(np.mean(ins)),
                       "vs_truth": dict(zip(names, np.round(mt, 4).tolist()))}
    out["free_options"] = dict(sorted(free.items(), key=lambda kv: -kv[1]["exp_score_anchor_w"]))

    # oscillation check: last-quarter range on long holds, and spectral peak
    osc = {}
    for n in names:
        rr = []
        for e, (c, y0, U) in enumerate(tests):
            if c != "sustained":
                continue
            Y = P[n][e]
            ch = np.where(seg_changes(U))[0].tolist() + [len(U)]
            for s0, s1 in zip(ch[:-1], ch[1:]):
                if s1 - s0 >= 800:
                    seg = Y[s0 + (s1 - s0) // 2: s1]
                    rr.append(np.max((seg.max(0) - seg.min(0)) / sig))
        osc[n] = float(np.max(rr)) if rr else None
    out["oscillation_late_hold_range_sigma"] = osc

    # 5. experiments: EVSI -------------------------------------------------------------
    y0e = np.mean([r.y0 for r in runs], axis=0)
    EX = experiments(spec, np.random.default_rng(5))
    # effective noise = misfit of the better members on our data (per observable), and an
    # autocorrelation discount on the log-likelihood (residuals are serially correlated)
    res = []
    for n in ("v7", "ABr2", "AB"):
        if n in names:
            res.append(np.concatenate([roll(n, r.y0, r.U) - r.Y for r in runs]))
    s_eff = np.sqrt(np.mean(np.concatenate(res) ** 2, axis=0))
    kappa = 50.0
    out["evsi_setup"] = {"s_eff": s_eff.tolist(), "kappa_ticks": kappa,
                         "actions": "every member + combos (median/mean)", "truths": names}
    Utest = np.array([[free[a_]["vs_truth"][tn] for tn in names] for a_ in free])   # [A, K]
    actions = list(free)
    # regime weights for the heuristic: frequency x disagreement in the test
    rw = {k.split(" | ")[0]: 0.0 for k in reg}
    for k, r in reg.items():
        rw[k.split(" | ")[0]] += r["freq"] * r["d_all"]
    rsum = sum(rw.values())
    rw = {k: v / rsum for k, v in rw.items()}

    def evsi(Yex, prior, draws=40, seed=0):
        g = np.random.default_rng(seed)
        v0 = float(np.max(Utest @ prior))
        a0 = actions[int(np.argmax(Utest @ prior))]
        tot, picks = 0.0, {}
        for j in range(K):
            if prior[j] <= 0:
                continue
            acc = 0.0
            for d in range(draws):
                # correlated noise: blocks of kappa ticks share one draw
                T = Yex[j].shape[0]
                nb = int(np.ceil(T / kappa))
                eps = np.repeat(g.normal(size=(nb, len(obs))), int(kappa), axis=0)[:T] * s_eff
                y = Yex[j] + eps
                ll = np.array([-0.5 * np.sum(((y - Yex[i]) / s_eff) ** 2) / kappa for i in range(K)])
                post = prior * np.exp(ll - ll.max())
                post /= post.sum()
                ai = int(np.argmax(Utest @ post))
                picks[actions[ai]] = picks.get(actions[ai], 0) + prior[j] / draws
                acc += Utest[ai, j]
            tot += prior[j] * acc / draws
        return tot - float(Utest[actions.index(a0)] @ prior), picks, v0, a0

    exps = {}
    for eid, (desc, U) in EX.items():
        Yex = [roll(n, y0e, U) for n in names]
        dpair = np.zeros(len(U))
        for i, j in combinations(range(K), 2):
            dpair += np.mean(np.abs(Yex[i] - Yex[j]) / sig, axis=1)
        dpair /= npair
        lab, dw, nov = regimes(spec, U, seen)
        heur = float(sum(dpair[t] * rw.get(lab[t], 0.0) for t in range(len(U))))
        dv7 = float(np.mean(np.abs(Yex[names.index("v7")] - Yex[names.index("ABr2")]) / sig))
        # pairwise separation value: sum_{i<j} p_i p_j * stake_ij * sep_ij, with
        # stake_ij = test loss of confusing i and j, sep_ij = 1 - exp(-Delta_ij^2 / 8)
        pv = {"uniform": 0.0, "anchor": 0.0}
        sep_tab = {}
        for i, j in combinations(range(K), 2):
            d2 = float(np.sum(((Yex[i] - Yex[j]) / s_eff) ** 2) / kappa)
            sep = 1 - np.exp(-d2 / 8)
            Sa = SmatC["all"]
            stake = 0.5 * ((Sa[i, i] - Sa[j, i]) + (Sa[j, j] - Sa[i, j]))
            sep_tab[f"{names[i]}|{names[j]}"] = round(float(sep), 3)
            pv["uniform"] += uni[i] * uni[j] * stake * sep
            pv["anchor"] += w[i] * w[j] * stake * sep
        e_u, picks_u, v0u, a0u = evsi(Yex, uni)
        e_w, picks_w, v0w, a0w = evsi(Yex, w)
        exps[eid] = {"desc": desc, "ticks": int(len(U)), "mean_pair_disagreement": float(dpair.mean()),
                     "v7_vs_ABr2_disagreement": dv7, "heuristic_value": heur,
                     "heuristic_per_100": 100 * heur / len(U),
                     "pair_value_uniform": pv["uniform"], "pair_value_anchor": pv["anchor"],
                     "pair_value_uniform_per_100": 100 * pv["uniform"] / len(U),
                     "separation": sep_tab,
                     "evsi_uniform": e_u, "evsi_anchor": e_w,
                     "evsi_uniform_per_100": 100 * e_u / len(U), "evsi_anchor_per_100": 100 * e_w / len(U),
                     "post_picks_anchor": {k: round(v, 3) for k, v in picks_w.items()},
                     "regime_mix": {k: int(np.sum(lab == k)) for k in set(lab)},
                     "U": np.round(U, 6).tolist(),
                     "member_end_state": {n: np.round(Yex[i][-1], 2).tolist() for i, n in enumerate(names)}}
    out["prior_value"] = {"uniform": {"value": v0u, "action": a0u}, "anchor": {"value": v0w, "action": a0w}}
    out["experiments"] = dict(sorted(exps.items(), key=lambda kv: -kv[1]["pair_value_uniform_per_100"]))
    # free fixes checked in-sample at the calibrated sigma (our 3 runs): recovery equilibrium
    # shift (prey_south -3.3, predators -0.05 once a recovery hold is >60 ticks old) and the
    # pulse-level prey_north taken from ABr2
    sigc = sig * out["sigma_mult_calibrated"]
    fx = {k: [] for k in ("v7", "v7+eq", "v7+eq+pulse_ABr2", "ABr2", "ABr2+eq")}
    rec_, pul_ = D.rec(spec), D.pulse_level(spec, 1.0)
    for r in runs:
        Y7, Ya = roll("v7", r.y0, r.U), roll("ABr2", r.y0, r.U)
        ch = seg_changes(r.U)
        since = np.zeros(r.T, int)
        for t in range(1, r.T):
            since[t] = 0 if ch[t] else since[t - 1] + 1
        m = np.all(np.abs(r.U - rec_) < 1e-9, axis=1) & (since > 60)
        mp = np.all(np.abs(r.U - pul_) < 1e-9, axis=1)
        sh = np.zeros(len(obs))
        sh[1], sh[2], sh[3] = -0.05, -3.3, -0.05
        Ye = Y7.copy(); Ye[m] += sh
        Yp = Ye.copy(); Yp[mp, 0] = Ya[mp, 0]
        Yae = Ya.copy(); Yae[m] += sh
        for k, Z in (("v7", Y7), ("v7+eq", Ye), ("v7+eq+pulse_ABr2", Yp), ("ABr2", Ya), ("ABr2+eq", Yae)):
            fx[k].append(metric.score(Z, r.Y, sigc))
    out["free_fix_insample_calibrated"] = {k: float(np.mean(v)) for k, v in fx.items()}
    out["insample_calibrated"] = {n: float(np.mean([metric.score(roll(n, r.y0, r.U), r.Y, sigc) for r in runs]))
                                  for n in names}
    out["recommended"] = {
        "primary": {"exp": "E15_corridor_then_habitat05", "credits": 300,
                    "U": out["experiments"]["E15_corridor_then_habitat05"]["U"]},
        "fallback": {"exp": "E02_corridor_only", "credits": 180,
                     "U": out["experiments"]["E02_corridor_only"]["U"]},
        "controls": ctrls}
    print("free fixes (in-sample, calibrated sigma):", out["free_fix_insample_calibrated"])
    print("in-sample calibrated:", out["insample_calibrated"])
    out["seconds"] = time.time() - t0
    (ROOT / "plans" / "voi_wildlife.json").write_text(json.dumps(out, indent=1))

    # console summary
    print("\nvalidation:", {k: v["reproduces"] for k, v in valid.items()})
    print("insample:", {n: round(np.mean(list(v.values())), 3) for n, v in out["insample"].items()})
    print("\nscore matrix (row predictor, col truth), all:")
    print("        " + " ".join(f"{n[:7]:>7}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:7]:>7} " + " ".join(f"{Smat['all'][i, j]:7.3f}" for j in range(K)))
    for c in CATS:
        off = Smat[c][~np.eye(K, dtype=bool)].mean()
        print(f"{c:12s} mean off-diag score {off:.3f}  disagreement/obs {np.round(Dobs[c], 3)}")
    print("\nregimes (freq, d_all, d_v7_ABr2, loss share):")
    for k, r in list(out["regimes"].items())[:16]:
        print(f"  {k:45s} {r['freq']:.3f} {r['d_all']:.3f} {r['d_v7_ABr2']:.3f} {r['loss_share']:.3f}")
    print("novel:", out["novel_tick_share"])
    print("\nanchor calibration top 8:")
    for k, v in list(out["anchor_calibration"].items())[:8]:
        print(f"  {k:14s} rmse {v['rmse']:.3f} corr {v['corr']:.3f}", {a_: round(x, 3) for a_, x in v["pred"].items()})
    print("anchor constant frac:", out["anchor_frac_constant"])
    print("truth weights:", out["truth_weights_anchor"], "kstar", kstar)
    print("calibrated score matrix all:")
    for i, n in enumerate(names):
        print(f"{n[:7]:>7} " + " ".join(f"{SmatC['all'][i, j]:7.3f}" for j in range(K)))
    for c in CATS:
        print(f"{c:12s} calibrated off-diag {SmatC[c][~np.eye(K, dtype=bool)].mean():.3f}")
    print("v7 implied by truth:", {n: {c: round(x, 3) for c, x in v.items()} for n, v in out["v7_implied_by_truth"].items()})
    print("s_eff", s_eff if 's_eff' in dir() else None)
    print("\nfree options:")
    for k, v in out["free_options"].items():
        print(f"  {k:30s} anchorW {v['exp_score_anchor_w']:.4f} uni {v['exp_score_uniform']:.4f} "
              f"others {v['exp_score_others']} worst {v['worst_truth']:.3f} ins {v['insample_mean']:.4f}")
    print("oscillation:", osc)
    print("\nprior value:", out["prior_value"])
    print("experiments:")
    for k, v in out["experiments"].items():
        print(f"  {k:42s} T{v['ticks']:4d} d {v['mean_pair_disagreement']:.3f} v7-AB {v['v7_vs_ABr2_disagreement']:.3f} "
              f"heur/100 {v['heuristic_per_100']:.3f} PVu {1e3*v['pair_value_uniform']:.2f}e-3 PVu/100 {1e3*v['pair_value_uniform_per_100']:.2f}e-3 EVSIu/100 {1e3*v['evsi_uniform_per_100']:.2f}e-3 "
              f"EVSIw/100 {1e3*v['evsi_anchor_per_100']:.2f}e-3 picks {v['post_picks_anchor']}")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
