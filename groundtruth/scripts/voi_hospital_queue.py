"""Value-of-information study for hospital_queue (free, no credits).

    PYTHONPATH=. python scripts/voi_hospital_queue.py

1. validate each committee doc: roll it on every ledger run, compare per-observable scores with
   the "insample" block of its lab json (drop docs that do not reproduce)
2. committee (validated docs + l0b_lin on all runs) rolled on 40 eval_like schedules
   (10 per category, T=4000, fixed seeds); pairwise disagreement in sigma units and score-loss units,
   by category, observable and control regime
3. candidate experiments (100-300 ticks) rolled under every member; value = resolvable test
   disagreement, weighted by test-regime frequency; ranked per credit
4. free options: every member, the median, per-observable picks; scored by LOO (lab json),
   in-sample on the real runs, and expected score on the test distribution with the committee
   members as plausible truths
Writes plans/voi_hospital_queue.json.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import design as D, metric, select as SEL, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.runtime import infer

SYS = "hospital_queue"
DOCS = {
    "p3": ("plans/hospital_queue_p3_doc.json", "plans/hospital_queue_hospital_queue_min_p3.json"),
    "AB": ("plans/hospital_queue_hospital_queue_mech_AB_doc.json", "plans/hospital_queue_hospital_queue_mech_AB.json"),
    "AC": ("plans/hospital_queue_hospital_queue_mech_AC_doc.json", "plans/hospital_queue_hospital_queue_mech_AC.json"),
    "BC": ("plans/hospital_queue_hospital_queue_mech_BC_doc.json", "plans/hospital_queue_hospital_queue_mech_BC.json"),
    "min2v4": ("plans/hospital_queue_hospital_queue_min2_v4_doc.json", "plans/hospital_queue_hospital_queue_min2_v4.json"),
}
CATS = ("sustained", "order", "recovery", "composition")
AC_T = 10.0          # residual autocorrelation time (ticks) for effective sample counts
U_REC = None         # recovery action vector, set in main()


def loss(d, sig):
    """score loss 1 - 1/(1+|d|/sigma), per observable (last axis)."""
    return 1.0 - 1.0 / (1.0 + np.abs(d) / sig)


def regime(U):
    """Per-tick control regime label from the schedule (current controls + overtime history)."""
    s, e, dg, ur, ot, fu = U.T
    T = len(U)
    last_ot = np.full(T, -10 ** 6)
    k = -10 ** 6
    for t in range(T):
        if ot[t] >= 0.3:
            k = t
        last_ot[t] = k
    ots = np.where(ot >= 0.3, "on", np.where(np.arange(T) - last_ot <= 40, "post", "no"))
    # recovery-level ticks are labelled by the time since the last non-recovery tick
    isrec = np.all(np.isclose(U, U_REC[None, :]), axis=1)
    since, k = np.zeros(T), -10 ** 6
    for t in range(T):
        if not isrec[t]:
            k = t
        since[t] = t - k
    sb = np.where(s < 8, "lo", np.where(s < 15, "mid", "hi"))
    db = np.where(dg < 0.3, "lo", np.where(dg <= 0.6, "mid", "hi"))
    eb = np.where(e >= 5, "hi", "lo")
    fb = np.where(fu < 0.5, "lo", "hi")
    lab = np.array([f"S{a}|OT{b}|FU{c}|E{d}|D{f}" for a, b, c, d, f in zip(sb, ots, fb, eb, db)], dtype=object)
    rb = np.where(since > 10 ** 5, "fromreset", np.where(since < 50, "<50", np.where(since < 200, "50-200", ">=200")))
    lab[isrec] = np.array([f"REC|since{x}" for x in rb[isrec]], dtype=object)
    return lab.astype(str)


def main():
    t00 = time.time()
    spec = S.get(SYS)
    runs = load_runs(spec, Ledger(data_dir(SYS, False), SYS, False))
    sigma = metric.sigma_proxy(runs)
    ctrl = list(spec.controls)
    rec = D.rec(spec)
    global U_REC
    U_REC = rec
    pul = D.pulse_level(spec, 1.0)
    out = {"system": SYS, "sigma_proxy": sigma.tolist(), "runs": [(r.exp, r.T) for r in runs]}
    print("sigma", np.round(sigma, 3))

    # ---------------------------------------------------------------- 1. validation
    members, val = {}, {}
    for name, (dp, lp) in DOCS.items():
        doc = json.loads(Path(dp).read_text())
        lab = json.loads(Path(lp).read_text())
        rows, ok = {}, True
        for r in runs:
            Y = infer.rollout_from_blob(doc, r.y0, r.U, doc=doc)
            s = metric.score_per_obs(Y, r.Y, sigma)
            ref = lab["insample"].get(r.exp)
            dev = None if ref is None else float(np.max(np.abs(s - np.asarray(ref))))
            ok = ok and dev is not None and dev < 0.01
            rows[r.exp] = {"score": s.tolist(), "lab": ref, "max_dev": dev}
        val[name] = {"ok": ok, "runs": rows, "insample_mean": float(np.mean([np.mean(v["score"]) for v in rows.values()]))}
        print(f"validate {name:<7} ok={ok} dev={max(v['max_dev'] if v['max_dev'] is not None else 9 for v in rows.values()):.4f} "
              f"mean={val[name]['insample_mean']:.3f}")
        if ok:
            members[name] = ("doc", doc)
    with C.single_thread():
        clip = C.soft_clip(spec, runs, margin=1.0)
        l0 = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(runs)
    ins = [metric.score_per_obs(l0.rollout(r.y0, r.U), r.Y, sigma) for r in runs]
    val["l0b_lin"] = {"ok": True, "insample_mean": float(np.mean(ins)),
                      "runs": {r.exp: {"score": s.tolist()} for r, s in zip(runs, ins)}}
    members["l0b_lin"] = ("l0b", l0)
    out["validation"] = val
    names = list(members)
    print("committee", names)

    def roll(name, y0, U):
        kind, m = members[name]
        if kind == "doc":
            return infer.rollout_from_blob(m, y0, U, doc=m)
        return np.asarray(m.rollout(y0, U), float)

    # residual scale per member / observable (in-sample RMS) -> effective noise for resolvability
    resid = {n: np.sqrt(np.mean(np.concatenate([(roll(n, r.y0, r.U) - r.Y) ** 2 for r in runs]), axis=0)) for n in names}
    dnoise = np.concatenate([np.diff(r.Y, axis=0) for r in runs])
    noise = np.median(np.abs(dnoise - np.median(dnoise, 0)), 0) * 1.4826 / np.sqrt(2)
    eps = np.maximum(noise, np.median(np.stack([resid[n] for n in names if n != "l0b_lin"]), 0))
    out["noise"] = noise.tolist()
    out["resid_rms"] = {n: v.tolist() for n, v in resid.items()}
    out["eps"] = eps.tolist()
    print("noise", np.round(noise, 2), "eps (median ODE resid rms)", np.round(eps, 2))

    # ---------------------------------------------------------------- 2. test distribution
    y0s = [r.y0 for r in runs]
    pairs = list(itertools.combinations(range(len(names)), 2))
    ode_idx = [i for i, n in enumerate(names) if n != "l0b_lin"]
    ode_pairs = list(itertools.combinations(ode_idx, 2))
    preds = names + ["median_ode"]
    reg_stat = {}          # regime -> [ticks, sum loss (all pairs), sum loss (ode pairs), sum sigma-dis]
    cat = {}
    cross = {c: np.zeros((len(preds), len(names))) for c in CATS}   # predictor x truth, mean score
    cross_obs = {c: np.zeros((len(preds), len(names), 3)) for c in CATS}
    tests = []
    for ci, c in enumerate(CATS):
        acc = {"sig_all": [], "sig_ode": [], "loss_all": [], "loss_ode": []}
        for k in range(10):
            rng = np.random.default_rng([2026, ci, k])
            U = D.eval_like(spec, c, 4000, rng)
            y0 = y0s[k % len(y0s)]
            Ys = np.stack([roll(n, y0, U) for n in names])
            Ymed = np.median(Ys[ode_idx], axis=0)
            tests.append((c, k))
            dsig = np.stack([np.abs(Ys[i] - Ys[j]) / sigma for i, j in pairs])       # [P,T,p]
            ls = np.stack([loss(Ys[i] - Ys[j], sigma) for i, j in pairs])
            lo_ = np.stack([loss(Ys[i] - Ys[j], sigma) for i, j in ode_pairs])
            acc["sig_all"].append(dsig.mean((0, 1)))
            acc["sig_ode"].append(np.stack([np.abs(Ys[i] - Ys[j]) / sigma for i, j in ode_pairs]).mean((0, 1)))
            acc["loss_all"].append(ls.mean((0, 1)))
            acc["loss_ode"].append(lo_.mean((0, 1)))
            for a in range(len(preds)):
                P = Ymed if a == len(names) else Ys[a]
                for b in range(len(names)):
                    so = np.mean(1 / (1 + np.abs(P - Ys[b]) / sigma), 0)
                    cross_obs[c][a, b] += so / 10
                    cross[c][a, b] += so.mean() / 10
            R = regime(U)
            lt = lo_.mean(0).mean(1)          # ode-pair loss per tick (mean over obs)
            la = ls.mean(0).mean(1)
            ds = dsig.mean(0).mean(1)
            for lab_ in np.unique(R):
                msk = R == lab_
                st = reg_stat.setdefault(lab_, np.zeros(4))
                st += [msk.sum(), la[msk].sum(), lt[msk].sum(), ds[msk].sum()]
        cat[c] = {kk: np.mean(v, 0).tolist() for kk, v in acc.items()}
        print(f"test {c:<12} sigma-dis ode {np.round(cat[c]['sig_ode'], 3)} all {np.round(cat[c]['sig_all'], 3)} | "
              f"loss ode {np.round(cat[c]['loss_ode'], 3)} all {np.round(cat[c]['loss_all'], 3)}")
    out["test_disagreement"] = cat
    out["test_cross_score"] = {c: {"predictors": preds, "truths": names, "matrix": cross[c].tolist(),
                                   "per_obs": cross_obs[c].tolist()} for c in CATS}
    Ntot = sum(v[0] for v in reg_stat.values())
    regs = {k: {"freq": float(v[0] / Ntot), "loss_all": float(v[1] / v[0]), "loss_ode": float(v[2] / v[0]),
                "sig_all": float(v[3] / v[0]), "mass_ode": float(v[2] / Ntot)} for k, v in reg_stat.items()}
    out["test_regimes"] = dict(sorted(regs.items(), key=lambda kv: -kv[1]["mass_ode"]))
    total_mass = sum(v["mass_ode"] for v in regs.values())
    out["test_total_mass_ode"] = total_mass
    print(f"total test disagreement mass (ode pairs, score-loss units) {total_mass:.4f}; top regimes:")
    for k, v in list(out["test_regimes"].items())[:14]:
        print(f"   {k:<32} freq {v['freq']:.3f} loss_ode {v['loss_ode']:.3f} mass {v['mass_ode']:.4f} ({100 * v['mass_ode'] / total_mass:.1f}%)")

    # test disagreement per ode pair (score-loss units, mean over the 40 episodes, ticks, observables)
    pair_test = {}
    for i, j in ode_pairs:
        pair_test[(i, j)] = float(np.mean([1 - cross[c][i, j] for c in CATS]))
    pair_total = float(np.mean(list(pair_test.values())))
    out["test_pair_loss"] = {f"{names[i]}-{names[j]}": v for (i, j), v in pair_test.items()}

    # ---------------------------------------------------------------- 3. candidate experiments
    def H(lev, n):
        return D.hold(spec, lev, n)

    def lv(**kw):
        v = rec.copy()
        for kk, x in kw.items():
            v[ctrl.index(kk)] = x
        return v

    alpha5 = rec + 0.5 * (pul - rec)
    rng = np.random.default_rng(7)
    ex = {
        "E1_staff_dip10_noOT": ("staffing 20 -> 10 (60) -> 20, no overtime: orientation (B) without overtime/fatigue",
                                np.concatenate([H(rec, 20), H(lv(staffing=10), 60), H(rec, 100)])),
        "E2_staff_dip5_noOT": ("staffing 20 -> 5 (40) -> 20, no overtime: deeper dip, same question",
                               np.concatenate([H(rec, 20), H(lv(staffing=5), 40), H(rec, 100)])),
        "E3_ot_alone_hold": ("overtime 1 alone for 40 at recovery staffing, then 100 of recovery: fatigue after-effect / wait spike",
                             np.concatenate([H(rec, 10), H(lv(overtime=1), 40), H(rec, 100)])),
        "E4_ot_spacing_10v40": ("same staff-hours at staffing 10: 4 x 10-tick overtime pulses spaced 10, then 4 spaced 40 (brief test 1)",
                                np.concatenate([H(rec, 10)] + [np.concatenate([H(lv(overtime=1, staffing=10), 10), H(lv(staffing=10), 10)]) for _ in range(4)]
                                               + [H(rec, 30)] + [np.concatenate([H(lv(overtime=1, staffing=10), 10), H(lv(staffing=10), 40)]) for _ in range(4)])),
        "E5_diag_at_staff10": ("staffing 10 fixed, diagnostic 0.4 -> 0.1 -> 0.8 -> 0.4 (brief test 2)",
                               np.concatenate([H(lv(staffing=10), 40), H(lv(staffing=10, diagnostic_allocation=0.1), 50),
                                               H(lv(staffing=10, diagnostic_allocation=0.8), 50), H(lv(staffing=10), 40)])),
        "E6_burst_then_fu0": ("discharge burst (staffing 5 + elective 20 for 40, back to 20) with follow-up 0 afterwards (returns C, brief test 3)",
                              np.concatenate([H(lv(staffing=5, elective_scheduling=20), 40), H(lv(followup_capacity=0), 100), H(rec, 40)])),
        "E7_fu0_hold": ("follow-up 0 alone for 100 at recovery, then 50 recovery (returns C vs staff diversion)",
                        np.concatenate([H(rec, 10), H(lv(followup_capacity=0), 100), H(rec, 50)])),
        "E8_alpha05_hold": ("interior level alpha 0.5 of the pulse on every control for 100, then 60 recovery",
                            np.concatenate([H(alpha5, 100), H(rec, 60)])),
        "E9_pulse_train": ("recovery-shaped pulse train (alpha U(.7,1) per control, gaps 10-40), 200 ticks",
                           D.pulse_train(spec, rng, n=5, L=(5, 20), gaps=(10, 40), lead=10, T=200)),
        "E10_staff5_then_20": ("staffing 5 for 60 (no OT) then 20 for 100: orientation ramp on a rise not preceded by overtime",
                               np.concatenate([H(lv(staffing=5), 60), H(rec, 100)])),
        "E11_pulse_long_hold": ("full pulse held 150 then 100 recovery: sustained pulse regime + long recovery tail",
                                np.concatenate([H(pul, 150), H(rec, 100)])),
        "E13_pulse40_long_rec": ("full pulse 40 then recovery 200: long post-congestion tail of the reported wait (stranded cohort)",
                                 np.concatenate([H(pul, 40), H(rec, 200)])),
        "E14_congest_noOT_long_rec": ("staffing 5 + elective 20 (no overtime) for 40, then recovery 200: is the wait tail congestion or overtime?",
                                      np.concatenate([H(lv(staffing=5, elective_scheduling=20), 40), H(rec, 200)])),
        "E15_two_pulses_long_rec": ("pulse 30, recovery 60, pulse 30, recovery 160: cohort build-up over repeated pulses",
                                    np.concatenate([H(pul, 30), H(rec, 60), H(pul, 30), H(rec, 160)])),
        "E16_pulse40_rec260": ("full pulse 40 then recovery 260: sees the cohort-wait onset (~160-200 after congestion) and 60-100 ticks of its plateau",
                               np.concatenate([H(pul, 40), H(rec, 260)])),
        "E17_congest_noOT_rec260": ("staffing 5 + elective 20 (no overtime) 40, then recovery 260: same, congestion without overtime",
                                    np.concatenate([H(lv(staffing=5, elective_scheduling=20), 40), H(rec, 260)])),
        "E13b_pulse40_rec160": ("full pulse 40 then recovery 160 (shortened E13, fits beside a 100-tick run)",
                                np.concatenate([H(pul, 40), H(rec, 160)])),
        "E7b_fu0_short": ("follow-up 0 alone for 60 at recovery, then 40 recovery (shortened E7)",
                          np.concatenate([H(lv(followup_capacity=0), 60), H(rec, 40)])),
        "E12_ot_then_staff_split": ("staffing 5 + overtime 30, staffing 5 without overtime 30, then 20 for 100: fatigue and orientation on one rise",
                                    np.concatenate([H(lv(staffing=5, overtime=1), 30), H(lv(staffing=5), 30), H(rec, 100)])),
    }
    y0e = np.median(np.stack(y0s), 0)
    out["exp_y0"] = y0e.tolist()
    exps = {}
    for eid, (note, U) in ex.items():
        U = D.clip(spec, U)
        Ys = np.stack([roll(n, y0e, U) for n in names])
        R = regime(U)
        T = len(U)
        d_ode = np.stack([np.abs(Ys[i] - Ys[j]) for i, j in ode_pairs])            # [P,T,p]
        # resolvability per regime: z^2 = sum_t mean_pairs max_obs (d/eps)^2 / AC_T
        z2t = np.mean(np.max((d_ode / eps) ** 2, axis=2), axis=0)
        lsig = np.mean(loss(d_ode, sigma), axis=(0, 2))
        val_simple = float(np.sum([regs.get(R[t], {"freq": 0.0})["freq"] * lsig[t] for t in range(T)]))
        v_res, touched = 0.0, {}
        for lab_ in np.unique(R):
            msk = R == lab_
            z2 = z2t[msk].sum() / AC_T
            p_res = 1 - np.exp(-z2 / 8.0)
            g = regs.get(lab_)
            touched[lab_] = {"ticks": int(msk.sum()), "p_resolve": float(p_res), "test_mass": g["mass_ode"] if g else 0.0}
            if g:
                v_res += g["mass_ode"] * p_res
        pair_sep = {f"{names[i]}-{names[j]}": float(np.mean(np.abs(Ys[i] - Ys[j]) / eps)) for i, j in ode_pairs}
        v_pair = 0.0
        for q, (i, j) in enumerate(ode_pairs):
            z2p = np.sum(np.max((d_ode[q] / eps) ** 2, axis=1)) / AC_T
            v_pair += pair_test[(i, j)] * (1 - np.exp(-z2p / 8.0)) / len(ode_pairs)
        ob_sep = (d_ode / eps).mean((0, 1))
        exps[eid] = {"note": note, "T": T, "cost": T, "value_resolvable": v_res, "value_frac": v_res / total_mass,
                     "value_simple": val_simple, "value_pair": v_pair, "value_pair_frac": v_pair / pair_total,
                     "per_credit_pair": v_pair / T, "per_credit_resolvable": v_res / T, "per_credit_simple": val_simple / T,
                     "sep_by_obs_eps": ob_sep.tolist(), "max_sep_by_obs_eps": (d_ode / eps).max((0, 1)).tolist(),
                     "pair_sep": pair_sep, "regimes": touched,
                     "U": [[float(x) for x in row] for row in U],
                     "pred": {n: np.round(Ys[i], 4).tolist() for i, n in enumerate(names)}}
        print(f"exp {eid:<26} T {T:>3} value {v_res:.4f} ({100 * v_res / total_mass:4.1f}%) per100 {100 * v_res / T:.4f} "
              f"pair {v_pair:.4f} ({100 * v_pair / pair_total:4.1f}%) pair/100 {100 * v_pair / T:.4f} sep/obs {np.round(ob_sep, 2)} max {np.round(exps[eid]['max_sep_by_obs_eps'], 1)}")
    order = sorted(exps, key=lambda k: -exps[k]["per_credit_resolvable"])
    out["experiments"] = exps
    out["experiment_rank_per_credit"] = order
    out["experiment_rank_total"] = sorted(exps, key=lambda k: -exps[k]["value_resolvable"])
    out["experiment_rank_pair_per_credit"] = sorted(exps, key=lambda k: -exps[k]["per_credit_pair"])
    combos = []
    for a, b in itertools.combinations(list(exps) + [None], 2):
        ids = [x for x in (a, b) if x]
        cost = sum(exps[x]["T"] for x in ids)
        if cost > 300:
            continue
        v = 0.0
        for lab_, g in regs.items():
            p = [exps[x]["regimes"].get(lab_, {}).get("p_resolve", 0.0) for x in ids]
            v += g["mass_ode"] * (1 - np.prod([1 - q for q in p]))
        combos.append({"ids": ids, "cost": cost, "value": v, "value_frac": v / total_mass})
    combos.sort(key=lambda d: -d["value"])
    out["combos_le_300"] = combos[:10]
    print("best combos <= 300:")
    for d in combos[:6]:
        print("  ", d)

    out["recommended"] = {
        "primary": {"id": "E16_pulse40_rec260", "cost": exps["E16_pulse40_rec260"]["T"],
                    "U": exps["E16_pulse40_rec260"]["U"], "controls": ctrl,
                    "why": "largest resolvable share of test disagreement within 300 credits; the post-congestion "
                           "wait tail (stranded cohort) dominates the test disagreement"},
        "alternative": {"id": "E17_congest_noOT_rec260", "cost": exps["E17_congest_noOT_rec260"]["T"],
                        "U": exps["E17_congest_noOT_rec260"]["U"], "controls": ctrl,
                        "why": "same question without overtime: splits p3 from the cohort family more cleanly, "
                               "less separation inside the cohort family"},
        "not_now": ["E1/E2/E10 staffing steps (fatigue vs orientation): <6% of test disagreement",
                    "E4 overtime spacing: 5.5% at 320 credits", "E5 diagnostic at fixed staffing: 0%"],
    }

    # ---------------------------------------------------------------- 4. free options
    labs = {n: json.loads(Path(DOCS[n][1]).read_text()) for n in names if n in DOCS}
    loo = {}
    for n, lab in labs.items():
        L = lab["loo"] if isinstance(lab["loo"], list) else eval(lab["loo"])
        loo[n] = np.mean([x["ode"] for x in L], 0)
        loo["l0b_lin"] = np.mean([x["l0b_lin"] for x in L], 0)
        loo["persistence"] = np.mean([x["persistence"] for x in L], 0)
    out["loo_per_obs"] = {k: v.tolist() for k, v in loo.items()}
    med_ins = []
    for r in runs:
        Ys = np.stack([roll(n, r.y0, r.U) for n in names])
        med_ins.append(metric.score_per_obs(np.median(Ys[ode_idx], 0), r.Y, sigma))
    out["insample_median_ode"] = np.mean(med_ins, 0).tolist()
    ins_obs = {n: np.mean([val[n]["runs"][r.exp]["score"] for r in runs], 0) for n in names}
    out["insample_per_obs"] = {n: v.tolist() for n, v in ins_obs.items()}
    # expected test score per observable when the truth is one of the OTHER ode members (uniform)
    ets = {}
    for a, n in enumerate(preds):
        tr = [b for b in ode_idx if b != a]
        ets[n] = np.mean([np.mean([cross_obs[c][a, b] for b in tr], 0) for c in CATS], 0)
    out["expected_test_score_vs_other_ode"] = {n: v.tolist() for n, v in ets.items()}
    print("LOO per obs:", {k: np.round(v, 3).tolist() for k, v in loo.items()})
    print("in-sample per obs:", {k: np.round(v, 3).tolist() for k, v in ins_obs.items()}, "median", np.round(out["insample_median_ode"], 3))
    print("expected test score vs other ODE members:", {k: np.round(v, 3).tolist() for k, v in ets.items()})
    # direct evidence on the top test regime: recovery-level ticks after a non-recovery block, real runs
    post = {n: [] for n in names + ["median_ode"]}
    for r in runs:
        R = regime(r.U)
        msk = np.char.startswith(R, "REC|since") & ~(R == "REC|sincefromreset")
        if not msk.any():
            continue
        Ys = np.stack([roll(n, r.y0, r.U) for n in names])
        for a, n in enumerate(names + ["median_ode"]):
            P = np.median(Ys[ode_idx], 0) if n == "median_ode" else Ys[a]
            post[n].append((msk.sum(), metric.score_per_obs(P[msk], r.Y[msk], sigma)))
    out["post_congestion_recovery_score"] = {
        n: {"ticks": int(sum(t for t, _ in v)), "score": (np.sum([t * x for t, x in v], 0) / sum(t for t, _ in v)).tolist()}
        for n, v in post.items() if v}
    print("real-run recovery ticks after a disturbance:", {n: (d["ticks"], np.round(d["score"], 3).tolist())
                                                         for n, d in out["post_congestion_recovery_score"].items()})
    # free strategies vs each ODE member as truth (a strategy never uses the truth member itself)
    strategies = {"p3": None, "BC": None, "median_rest": None, "pick_wBC_qp3_dp3": None,
                  "pick_wMed_qp3_dp3": None, "pick_wMed_qMed_dp3": None}
    res = {k: {names[b]: np.zeros(3) for b in ode_idx} for k in strategies}
    for (c, k) in tests:
        U = D.eval_like(spec, c, 4000, np.random.default_rng([2026, CATS.index(c), k]))
        y0 = y0s[k % len(y0s)]
        Ys = {n: roll(n, y0, U) for n in names if n != "l0b_lin"}
        for b in Ys:
            rest = np.median(np.stack([Ys[n] for n in Ys if n != b]), 0)
            cand = {"p3": Ys["p3"], "BC": Ys["BC"], "median_rest": rest,
                    "pick_wBC_qp3_dp3": np.column_stack([Ys["BC"][:, 0], Ys["p3"][:, 1:]]),
                    "pick_wMed_qp3_dp3": np.column_stack([rest[:, 0], Ys["p3"][:, 1:]]),
                    "pick_wMed_qMed_dp3": np.column_stack([rest[:, :2], Ys["p3"][:, 2]])}
            for kk, P in cand.items():
                res[kk][b] += np.mean(1 / (1 + np.abs(P - Ys[b]) / sigma), 0) / len(tests)
    out["free_strategies_vs_truth"] = {k: {b: v.tolist() for b, v in d.items()} for k, d in res.items()}
    print("free strategies: expected score vs each ODE member as truth (a truth used by the strategy scores trivially high)")
    for kk, d in res.items():
        print(f"   {kk:<20}", {b: round(float(v.mean()), 3) for b, v in d.items()},
              "| mean over truths AB, AC, min2v4 (used by no strategy):",
              np.round(np.mean([d[b] for b in ("AB", "AC", "min2v4")], 0), 3).tolist(),
              round(float(np.mean([d[b].mean() for b in ("AB", "AC", "min2v4")])), 3))
    out["tests"] = [{"category": c, "k": k, "seed": [2026, CATS.index(c), k], "y0_run": k % len(y0s)} for c, k in tests]
    out["seconds"] = time.time() - t00
    Path("plans/voi_hospital_queue.json").write_text(json.dumps(out, indent=1))
    print(f"done {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
