"""Value of information for epidemic: where the committee disagrees on test-shaped schedules,
which short purchases would settle it, and which free model changes hedge it. Free (no credits).

    python scripts/voi_epidemic.py            -> plans/voi_epidemic.json (+ console tables)
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import systems as S, metric, design as D, select as SEL
from gtlab.models import common as C
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.runtime import infer as rt

warnings.filterwarnings("ignore")
SID = "epidemic"
CATS = ("sustained", "order", "recovery", "composition")
PLANS = Path("plans")
TOL = 0.01                  # max |score diff| per run/obs for a doc to count as reproduced
COMMITTEES = {"ode3": ["p3BC", "p3AB", "sirs_p3"], "top4": ["p3BC", "p3AB", "sirs_p3", "l0b_lin"],
              "AB_vs_BC": ["p3BC", "p3AB"]}
PRIMARY = "ode3"
RECOMMEND = {"buy": ["joint_a085_long_300"],
             "alternates": ["joint_pulse_long_300", "vax_only_200", "closure_vs_mask_240"]}


# ------------------------------------------------------------------ members
def load_docs():
    out = {}
    for p in sorted(PLANS.glob("epidemic_*_doc.json")):
        name = p.name[:-len("_doc.json")]
        lab = PLANS / f"{name}.json"
        short = name.replace("epidemic_epidemic_", "").replace("epidemic_", "")
        out[short] = {"doc": json.loads(p.read_text()), "lab": json.loads(lab.read_text()) if lab.exists() else {}}
    return out


def doc_roller(doc):
    return lambda y0, U: rt.rollout_from_blob(doc["model"], y0, U, doc=doc)


def validate(docs, runs, sigma):
    rep, keep = {}, {}
    for k, d in docs.items():
        ins = d["lab"].get("insample")
        lab_sig = np.asarray(d["lab"].get("sigma_proxy", sigma), float)
        row = {"has_lab": ins is not None}
        try:
            got, got_clip = {}, {}
            for r in runs:
                Yr = rt.rollout_from_blob(d["doc"]["model"], r.y0, r.U, doc=d["doc"], clip=False)
                Yc = rt.rollout_from_blob(d["doc"]["model"], r.y0, r.U, doc=d["doc"])
                got[r.exp] = metric.score_per_obs(Yr, r.Y, lab_sig).tolist()
                got_clip[r.exp] = metric.score_per_obs(Yc, r.Y, sigma).tolist()
            row["now"] = got
            row["now_clipped_mean"] = float(np.mean([np.mean(v) for v in got_clip.values()]))
            if ins is None:
                row["ok"] = False
                row["why"] = "no in-sample reference"
            else:
                diffs = [abs(a - b) for e, v in ins.items() if e in got for a, b in zip(v, got[e])]
                missing = [e for e in got if e not in ins]
                row["max_diff"] = float(max(diffs)) if diffs else None
                row["runs_in_lab"] = list(ins.keys())
                row["ok"] = bool(diffs) and max(diffs) <= TOL
                row["why"] = "reproduced" if row["ok"] else f"max diff {max(diffs):.3f}" if diffs else "no overlap"
                if missing:
                    row["why"] += f"; not fitted on {missing}"
                row["fitted_on_all"] = not missing
            row["lab"] = ins
        except Exception as e:
            row.update(ok=False, why=f"ERR {type(e).__name__}: {e}")
        rep[k] = row
        if row["ok"]:
            keep[k] = doc_roller(d["doc"])
    return rep, keep


def fit_l0b(spec, runs, sigma):
    clip = C.soft_clip(spec, runs, margin=1.0)
    m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None})
    m.fit(runs)
    return m


# ------------------------------------------------------------------ regimes
def regimes(U, spec):
    """Per-tick label: which controls are on (C/M/V), hold age bin, interior level flag,
    and the order flags the brief asks about (vaccination after / before a restriction)."""
    lo, hi = D.lo_hi(spec)
    Un = (np.asarray(U, float) - lo) / (hi - lo)
    T = len(Un)
    on = Un > 0.05
    labels, flags = [], []
    age = 0
    last_R = -10 ** 9          # last tick with restriction (closure or mask) on
    last_V = -10 ** 9
    for t in range(T):
        age = age + 1 if t > 0 and np.allclose(Un[t], Un[t - 1]) else 0
        combo = "".join(c for c, f in zip("CMV", on[t]) if f) or "rec"
        ab = "0-50" if age < 50 else "50-150" if age < 150 else "150+"
        interior = bool(np.any(on[t] & (Un[t] < 0.6)))
        R = bool(on[t, 0] or on[t, 1])
        V = bool(on[t, 2])
        fl = []
        if V and not R and t - last_R <= 300:
            fl.append("V_after_R")
        if R and not V and t - last_V <= 300:
            fl.append("R_after_V")
        if not R and not V and t - last_R <= 150:
            fl.append("release")
        if R:
            last_R = t
        if V:
            last_V = t
        labels.append(f"{combo}|{ab}|{'int' if interior else 'hi'}")
        flags.append(fl)
    return labels, flags


# ------------------------------------------------------------------ disagreement
def pair_d(P, sigma):
    """P [M,T,p] -> per-tick mean pairwise |Pi-Pj|/sigma, per observable [T,p]."""
    M = P.shape[0]
    acc = np.zeros(P.shape[1:])
    n = 0
    for i, j in itertools.combinations(range(M), 2):
        acc += np.abs(P[i] - P[j]) / sigma[None, :]
        n += 1
    return acc / max(n, 1)


def loss_units(d):
    """expected score lost if the truth sits d sigma away: 1 - 1/(1+d)."""
    return 1.0 - 1.0 / (1.0 + d)


# ------------------------------------------------------------------ candidates
def candidates(spec):
    r = D.rec(spec)
    pul = D.pulse_level(spec, 1.0)

    def u(c=0.0, m=0.0, v=0.0):
        return D.clip(spec, np.array([c, m, v * 0.003]))

    H = lambda lev, T: D.hold(spec, lev, T)
    cat = lambda *xs: D.clip(spec, np.concatenate(xs))
    rng = np.random.default_rng(7)
    a7 = D.pulse_level(spec, 0.7)
    pt = [H(r, 15)]
    for L, g in [(12, 25), (25, 45), (8, 20), (30, 60), (15, 35)]:
        pt += [H(a7, L), H(r, g)]
    ptrain = cat(*pt)[:300]
    cands = {
        "vax_only_200": H(u(v=1), 200),
        "R_pulse_then_vax_200": cat(H(u(.85, .85), 40), H(u(v=1), 160)),
        "vax_then_R_pulse_200": cat(H(u(v=1), 100), H(u(.85, .85), 40), H(r, 60)),
        "closure_vs_mask_240": cat(H(u(c=1), 60), H(r, 60), H(u(m=1), 60), H(r, 60)),
        "mask_vs_closure_240": cat(H(u(m=1), 60), H(r, 60), H(u(c=1), 60), H(r, 60)),
        "interior_a05_hold_200": H(D.pulse_level(spec, 0.5), 200),
        "pulse_train_a07_300": ptrain,
        "closure_long_250": H(u(c=1), 250),
        "mask_long_250": H(u(m=1), 250),
        "joint_pulse_long_250": H(pul, 250),
        "closure_plus_vax_200": H(u(c=1, v=1), 200),
        "recovery_free_250": H(r, 250),
        "joint_then_release_200": cat(H(pul, 80), H(r, 120)),
        "joint_a085_long_300": H(D.pulse_level(spec, 0.85), 300),
        "joint_pulse_long_300": H(pul, 300),
        "joint_a085_250_release_50": cat(H(D.pulse_level(spec, 0.85), 250), H(r, 50)),
        "closure_plus_vax_300": H(u(c=1, v=1), 300),
    }
    return cands


def committee_study(spec, runs, sigma, members, names, top, obs, verbose=True):
    if True:
        # -------- 1. test distribution
        y0s = [r.y0 for r in runs]
        tests = []
        for ci, cat in enumerate(CATS):
            for i in range(10):
                rng = np.random.default_rng(1000 * ci + i)
                U = D.eval_like(spec, cat, 4000, rng)
                tests.append((cat, i, U, y0s[int(rng.integers(len(y0s)))]))
        lab_tick_count = {}
        per_cat = {c: {"d": [], "d_obs": [], "loss": []} for c in CATS}
        pair_cat = {c: {} for c in CATS}
        lab_d = {}               # label -> [sum d, n]
        flag_d = {}
        all_cat = {c: [] for c in CATS}
        dev_from_med = {c: {k: [] for k in names} for c in CATS}
        for cat, i, U, y0 in tests:
            P = {k: members[k](y0, U) for k in names}
            Pt = np.stack([P[k] for k in top])
            d = pair_d(Pt, sigma)                          # [T,p]
            dt = d.mean(axis=1)
            per_cat[cat]["d"].append(float(dt.mean()))
            per_cat[cat]["d_obs"].append(d.mean(axis=0))
            per_cat[cat]["loss"].append(float(loss_units(d).mean()))
            for a, b in itertools.combinations(names, 2):
                pair_cat[cat].setdefault(f"{a}~{b}", []).append(float(np.mean(np.abs(P[a] - P[b]) / sigma)))
            Pall = np.stack([P[k] for k in names])
            med = np.median(np.stack([P[k] for k in top]), axis=0)
            for k in names:
                dev_from_med[cat][k].append(float(np.mean(np.abs(P[k] - med) / sigma)))
            labels, flags = regimes(U, spec)
            for t, (lb, fl) in enumerate(zip(labels, flags)):
                lab_tick_count[lb] = lab_tick_count.get(lb, 0) + 1
                s = lab_d.setdefault(lb, [0.0, 0, np.zeros(2)])
                s[0] += dt[t]; s[1] += 1; s[2] += d[t]
                for f in fl:
                    s2 = flag_d.setdefault(f, [0.0, 0])
                    s2[0] += dt[t]; s2[1] += 1
        Ntest = sum(lab_tick_count.values())
        share = {k: v / Ntest for k, v in lab_tick_count.items()}
        cat_tab = {}
        for c in CATS:
            cat_tab[c] = {"d_sigma": float(np.mean(per_cat[c]["d"])),
                          "d_sigma_per_obs": dict(zip(obs, np.mean(per_cat[c]["d_obs"], axis=0).round(3).tolist())),
                          "score_at_risk": float(np.mean(per_cat[c]["loss"])),
                          "pairs": {k: float(np.mean(v)) for k, v in pair_cat[c].items()},
                          "dev_from_top_median": {k: float(np.mean(v)) for k, v in dev_from_med[c].items()}}
            (print if verbose else (lambda *a, **k: None))(f"[{c:<11}] top-pairwise d={cat_tab[c]['d_sigma']:.2f} sigma  per obs {cat_tab[c]['d_sigma_per_obs']}  "
                  f"score at risk {cat_tab[c]['score_at_risk']:.3f}")
        reg_tab = sorted(({"label": k, "share": share[k], "d_sigma": v[0] / v[1],
                           "d_cases": float(v[2][0] / v[1]), "d_hosp": float(v[2][1] / v[1]),
                           "contrib": share[k] * v[0] / v[1]} for k, v in lab_d.items()), key=lambda z: -z["contrib"])
        flag_tab = {k: {"share": v[1] / Ntest, "d_sigma": v[0] / v[1]} for k, v in flag_d.items()}
        (print if verbose else (lambda *a, **k: None))("regimes by contribution (share x d):")
        for z in reg_tab[:14]:
            (print if verbose else (lambda *a, **k: None))(f"  {z['label']:<18} share {z['share']:.3f} d {z['d_sigma']:.2f} (cases {z['d_cases']:.2f} hosp {z['d_hosp']:.2f}) contrib {z['contrib']:.3f}")
        (print if verbose else (lambda *a, **k: None))("flags:", {k: (round(v['share'], 3), round(v['d_sigma'], 2)) for k, v in flag_tab.items()})

        # -------- 2. candidate experiments
        combo_share = {}
        for k, v in share.items():
            cb = k.split("|")[0]
            combo_share[cb] = combo_share.get(cb, 0) + v
        lab_mean_d = {z["label"]: z["d_sigma"] for z in reg_tab}
        mean_test_d = float(np.mean([cat_tab[c]["d_sigma"] for c in CATS]))
        cands = candidates(spec)
        exp_tab = []
        for name, U in cands.items():
            ds, dobs = [], []
            for y0 in y0s:
                Pt = np.stack([members[k](y0, U) for k in top])
                d = pair_d(Pt, sigma)
                ds.append(d.mean(axis=1)); dobs.append(d.mean(axis=0))
            dt = np.mean(ds, axis=0)
            labels, _ = regimes(U, spec)
            w = np.array([share.get(lb, 0.25 * combo_share.get(lb.split("|")[0], 0.0)) for lb in labels])
            # relevance: how much of the test lives in this regime, relative to a uniform spread over the
            # labels the test uses; times how uncertain the test is in that regime (relative to mean)
            rel = w * len(share)
            unc = np.array([lab_mean_d.get(lb, mean_test_d) for lb in labels]) / mean_test_d
            val = float(np.sum(dt * rel * unc))
            exp_tab.append({"name": name, "credits": int(len(U)), "d_sigma": float(dt.mean()),
                            "d_per_obs": dict(zip(obs, np.mean(dobs, axis=0).round(3).tolist())),
                            "test_share_covered": float(sum(share.get(lb, 0) for lb in set(labels))),
                            "value": val, "value_per_credit": val / len(U), "U": np.asarray(U).round(6).tolist()})
        exp_tab.sort(key=lambda z: -z["value_per_credit"])
        (print if verbose else (lambda *a, **k: None))("experiments (value per credit):")
        for z in exp_tab:
            (print if verbose else (lambda *a, **k: None))(f"  {z['name']:<24} {z['credits']:>4} cr  d {z['d_sigma']:.2f}  covers {z['test_share_covered']:.3f}  "
                  f"value {z['value']:.1f}  per credit {z['value_per_credit']:.3f}")

        return cat_tab, reg_tab, flag_tab, exp_tab


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    spec = S.get(SID)
    runs = load_runs(spec, Ledger(data_dir(SID, False), SID, False))
    sigma = metric.sigma_proxy(runs)
    obs = spec.observables
    print(f"runs {[(r.exp, r.T) for r in runs]} sigma {np.round(sigma, 2)}")

    docs = load_docs()
    vrep, members = validate(docs, runs, sigma)
    for k, v in vrep.items():
        print(f"  doc {k:<12} ok={v['ok']!s:<5} {v.get('why')}  clipped in-sample {v.get('now_clipped_mean', float('nan')):.3f}")
    l0b = fit_l0b(spec, runs, sigma)
    members["l0b_lin"] = lambda y0, U: l0b.rollout(y0, U)
    names = list(members)
    print(f"members {names}")
    cache = {}

    def cached(k, f):
        def g(y0, U):
            key = (k, hash(np.asarray(y0).tobytes()), hash(np.asarray(U).tobytes()), np.asarray(U).shape)
            if key not in cache:
                cache[key] = f(y0, U)
            return cache[key]
        return g
    members = {k: cached(k, f) for k, f in members.items()}

    res = {}
    for cname, top in COMMITTEES.items():
        top = [k for k in top if k in members]
        print(f"===== committee {cname}: {top}")
        res[cname] = committee_study(spec, runs, sigma, members, names, top, obs)
    cat_tab, reg_tab, flag_tab, exp_tab = res[PRIMARY]
    top = COMMITTEES["top4"]
    # -------- 3. free options: in-sample per-run per-obs, pairs (mean), 3-median, straddle
    ins = {}
    preds = {k: {r.exp: members[k](r.y0, r.U) for r in runs} for k in names}
    def sc(Pd):
        return np.mean([metric.score_per_obs(Pd[r.exp], r.Y, sigma) for r in runs], axis=0)
    for k in names:
        ins[k] = sc(preds[k]).round(4).tolist()
    pair_rows = []
    for a, b in itertools.combinations(top, 2):
        mid = {e: 0.5 * (preds[a][e] + preds[b][e]) for e in preds[a]}
        strad = np.mean([np.mean((preds[a][r.exp] - r.Y) * (preds[b][r.exp] - r.Y) < 0, axis=0) for r in runs], axis=0)
        sm = sc(mid)
        pair_rows.append({"pair": f"{a}+{b}", "mean_of_two": sm.round(4).tolist(), "straddle": strad.round(3).tolist(),
                          "best_single": np.maximum(ins[a], ins[b]).round(4).tolist(),
                          "gain_vs_best_single": (sm - np.maximum(ins[a], ins[b])).round(4).tolist()})
    med3 = {}
    for trio in itertools.combinations(top, 3):
        m = {e: np.median(np.stack([preds[k][e] for k in trio]), axis=0) for e in preds[trio[0]]}
        med3["+".join(trio)] = sc(m).round(4).tolist()
    # l0b_lin leave-one-run-out (cheap) so the linear member is not flattered in-sample
    loo_l0b = []
    for i in range(len(runs)):
        tr = [r for j, r in enumerate(runs) if j != i]
        m = fit_l0b(spec, tr, sigma)
        loo_l0b.append(metric.score_per_obs(m.rollout(runs[i].y0, runs[i].U), runs[i].Y, sigma).tolist())
    # Bayes pick on the test distribution: truth is one of the plausible members (uniform prior);
    # expected score of each candidate predictor per category and observable. Free-change evidence.
    truths = [k for k in ("p3BC", "p3AB", "sirs_p3", "AB", "BC") if k in members]
    opts = {k: None for k in ("p3BC", "p3AB", "sirs_p3", "l0b_lin")}
    opts.update({"median(p3BC,p3AB,sirs_p3)": ("median", ["p3BC", "p3AB", "sirs_p3"]),
                 "mean(p3BC,p3AB)": ("mean", ["p3BC", "p3AB"]),
                 "mean(p3BC,p3AB,sirs_p3)": ("mean", ["p3BC", "p3AB", "sirs_p3"]),
                 "median(p3BC,p3AB,l0b_lin)": ("median", ["p3BC", "p3AB", "l0b_lin"])})
    bayes = {c: {o: np.zeros(len(obs)) for o in opts} for c in CATS}
    y0s = [r.y0 for r in runs]
    for ci, cat in enumerate(CATS):
        for i in range(10):
            rng = np.random.default_rng(1000 * ci + i)
            U = D.eval_like(spec, cat, 4000, rng)
            y0 = y0s[int(rng.integers(len(y0s)))]
            P = {k: members[k](y0, U) for k in set(truths) | {"l0b_lin"}}
            for o, how in opts.items():
                if how is None:
                    Q = P[o]
                else:
                    St = np.stack([P[k] for k in how[1]])
                    Q = np.median(St, axis=0) if how[0] == "median" else St.mean(axis=0)
                bayes[cat][o] += np.mean([metric.score_per_obs(Q, P[t], sigma) for t in truths], axis=0) / 10
    bayes_tab = {c: {o: v.round(4).tolist() for o, v in bayes[c].items()} for c in CATS}
    bayes_mean = {o: float(np.mean([np.mean(bayes[c][o]) for c in CATS])) for o in opts}
    print("Bayes pick (truth uniform over", truths, "):")
    for o, v in sorted(bayes_mean.items(), key=lambda z: -z[1]):
        print(f"  {o:<28} {v:.4f}  " + " ".join(f"{c[:4]}={np.round(bayes[c][o], 3).tolist()}" for c in CATS))
    free = {"bayes_truths": truths, "bayes_by_category": bayes_tab, "bayes_mean": bayes_mean,
            "insample_per_obs": ins, "pairs": pair_rows, "median3": med3,
            "l0b_lin_loo_per_obs": np.mean(loo_l0b, axis=0).round(4).tolist(),
            "ode_loo_from_labs": {k: docs[k]["lab"].get("loo_mean_ode") for k in docs if k in members}}
    print("in-sample per obs:", ins)
    for z in pair_rows:
        print("  pair", z)
    print("  median3", med3)
    print("  l0b_lin LOO", free["l0b_lin_loo_per_obs"])

    out = {"system": SID, "runs": [(r.exp, r.T) for r in runs], "sigma": sigma.tolist(),
           "members": names, "committees": COMMITTEES, "primary": PRIMARY, "validation": vrep,
           "by_committee": {k: {"categories": v[0], "regimes": v[1], "flags": v[2], "experiments": v[3]} for k, v in res.items()}, "free": free,
           "recommended": {**RECOMMEND, "credits": int(sum(len(candidates(spec)[k]) for k in RECOMMEND["buy"])),
                           "schedules": {k: np.asarray(candidates(spec)[k]).round(6).tolist()
                                         for k in RECOMMEND["buy"] + RECOMMEND["alternates"]},
                           "controls": list(spec.controls)},
           "seconds": time.time() - t0}
    (PLANS / "voi_epidemic.json").write_text(json.dumps(out, indent=1, default=lambda x: np.asarray(x).tolist()))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
