"""Value-of-information study for social_contagion. Free: simulation only, no gateway calls.

1. Validate every committee doc (plans/social_contagion_*_doc.json) against its lab json's
   in-sample per-observable scores on the ledger runs; drop docs that no longer reproduce.
2. Committee disagreement on 40 eval_like schedules (10 per category, T=4000), in sigma_proxy
   units, broken down by control regime and by time since the last control change.
3. Candidate experiments (100-300 ticks): member disagreement weighted by how much of the test
   disagreement sits in the same control regime; value per credit.
4. Free model options: every candidate predictor scored against every committee member taken as
   the truth on the test schedules (a robust expected score).

    python scripts/voi_social_contagion.py            -> plans/voi_social_contagion.json
"""
import glob
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
from gtlab.runtime.infer import rollout_from_blob

warnings.filterwarnings("ignore")
SYS = "social_contagion"
CATS = ("sustained", "order", "recovery", "composition")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plans" / "voi_social_contagion.json"
T_TEST = 4000
N_PER_CAT = 10
TOP = ["BC2", "AB2", "min_v2", "l0b_lin"]          # distinct structures used for disagreement


# ------------------------------------------------------------------ regimes
def bins(U):
    """Per tick regime label: seeding / incentive / bridge level classes."""
    s, c, b = U[:, 0], U[:, 1], U[:, 2]
    sb = np.where(s <= 1e-9, "s0", np.where(s < 4, "sL", np.where(s < 8, "sM", "sH")))
    cb = np.where(c <= 1e-9, "c0", np.where(c < 0.8, "cL", np.where(c < 1.6, "cM", "cH")))
    bb = np.where(b <= 1e-9, "b0", np.where(b < 0.4, "bL", np.where(b < 0.7, "bM", "bH")))
    return np.array([f"{x}{y}{z}" for x, y, z in zip(sb, cb, bb)])


def dwell(U):
    """Ticks since the last control change (0 at a change)."""
    d = np.zeros(len(U), int)
    for t in range(1, len(U)):
        d[t] = 0 if np.any(U[t] != U[t - 1]) else d[t - 1] + 1
    return d


def dwell_bin(d):
    return np.where(d < 50, "<50", np.where(d < 200, "50-200", np.where(d < 1000, "200-1000", ">=1000")))


def coarse(label):
    """Collapse a regime label to which controls are on and whether any is interior."""
    s, c, b = label[:2], label[2:4], label[4:6]
    on = [n for n, v in (("seed", s), ("inc", c), ("bridge", b)) if v[1] != "0"]
    name = "+".join(on) if on else "recovery"
    interior = any(v[1] in "LM" for v in (s, c, b) if v[1] != "0")
    return name + (" (interior)" if interior else "")


def seq_labels(U):
    """coarse() per tick; recovery ticks carry the regime they follow ('recovery after X')."""
    labs = [coarse(l) for l in bins(U)]
    prev = None
    out = []
    for l in labs:
        if l == "recovery":
            out.append("recovery (start)" if prev is None else f"recovery after {prev}")
        else:
            prev = l
            out.append(l)
    return np.array(out)


# ------------------------------------------------------------------ members
def load_members(spec, runs, sigma):
    members, validation = {}, {}
    for f in sorted(glob.glob(str(ROOT / "plans" / f"{SYS}_{SYS}_*_doc.json"))):
        tag = Path(f).name[len(f"{SYS}_{SYS}_"):-len("_doc.json")].replace("mech_", "")
        lab = Path(f.replace("_doc.json", ".json"))
        doc = json.loads(Path(f).read_text())
        rep = json.loads(lab.read_text()) if lab.exists() else {}
        ins = rep.get("insample", {})
        diffs, now = [], {}
        for r in runs:
            try:
                Y = rollout_from_blob(doc["model"], r.y0, r.U, doc=doc)
                s = metric.score_per_obs(Y, r.Y, sigma)
            except Exception as e:
                s = np.full(2, np.nan)
            now[r.exp] = [round(float(v), 4) for v in s]
            if r.exp in ins:
                diffs.append(float(np.max(np.abs(np.asarray(ins[r.exp]) - s))))
            else:
                diffs.append(np.nan)
        ok = bool(ins) and all(np.isfinite(diffs)) and max(diffs) < 0.01
        validation[tag] = {"lab_insample": ins, "now": now, "max_abs_diff": [None if not np.isfinite(d) else round(d, 4) for d in diffs],
                           "valid": ok, "mech": rep.get("mech"), "loo_mean": rep.get("loo_mean_ode"),
                           "loo": rep.get("loo")}
        if ok:
            members[tag] = (lambda y0, U, doc=doc: rollout_from_blob(doc["model"], y0, U, doc=doc))
    # C-plausible probe: BC2 with k_C raised to the largest value whose in-sample mean score stays
    # within 0.01 of BC2 (no refit). Represents "bridge effect not yet visible in the data".
    bc2 = ROOT / "plans" / f"{SYS}_{SYS}_mech_BC2_doc.json"
    if "BC2" in members and bc2.exists():
        base = json.loads(bc2.read_text())
        def ins_mean(doc):
            return float(np.mean([metric.score_per_obs(rollout_from_blob(doc["model"], r.y0, r.U, doc=doc), r.Y, sigma).mean()
                                  for r in runs]))
        s0 = ins_mean(base)
        best, scan = None, []
        for kc in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            dc = json.loads(json.dumps(base))
            dc["model"]["theta"][14] = kc
            sm = ins_mean(dc)
            scan.append([kc, round(sm, 4)])
            if sm >= s0 - 0.01:
                best = dc
        validation["BC2_kC"] = {"valid": best is not None, "insample_BC2": round(s0, 4), "kC_scan": scan,
                                "kC": None if best is None else best["model"]["theta"][14]}
        if best is not None:
            members["BC2_kC"] = (lambda y0, U, doc=best: rollout_from_blob(doc["model"], y0, U, doc=doc))
    # l0b_lin on all runs, clip margin 1
    clip = C.soft_clip(spec, runs, margin=1.0)
    m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(runs)
    members["l0b_lin"] = lambda y0, U, m=m: np.asarray(m.rollout(y0, U), float)
    ins = {r.exp: [round(float(v), 4) for v in metric.score_per_obs(members["l0b_lin"](r.y0, r.U), r.Y, sigma)] for r in runs}
    validation["l0b_lin"] = {"now": ins, "valid": True}
    # current public model (u010b): median(min v2, l0b_lin) as packaged
    pub = sorted(glob.glob(str(ROOT / "submissions" / "*u010b*" / SYS / "model.json")))
    if pub:
        pdoc = json.loads(Path(pub[-1]).read_text())
        members["public"] = lambda y0, U, d=pdoc: rollout_from_blob(d["model"], y0, U, doc=d)
        validation["public"] = {"now": {r.exp: [round(float(v), 4) for v in metric.score_per_obs(members["public"](r.y0, r.U), r.Y, sigma)] for r in runs},
                                "valid": True, "source": str(Path(pub[-1]).relative_to(ROOT))}
    return members, validation


# ------------------------------------------------------------------ experiments
def candidates(spec):
    z = np.zeros(3)

    def seg(*parts):
        return D.clip(spec, np.concatenate([D.hold(spec, lev, n) for lev, n in parts]))

    P = D.pulse_level(spec, 1.0)                               # [9, 2, 0.6]
    rng = np.random.default_rng(11)
    E = {
        "E01_interior_alpha05_200": ("hold 0.5*pulse = [4.5,1,0.3] 150 then stop 50",
                                     seg((D.pulse_level(spec, 0.5), 150), (z, 50))),
        "E02_seed5_only_250": ("seeding 5 alone 200, stop 50", seg(([5, 0, 0], 200), (z, 50))),
        "E03_inc08_only_200": ("incentive 0.8 alone 150, stop 50", seg(([0, 0.8, 0], 150), (z, 50))),
        "E04_inc_before_seed_250": ("incentive 2 for 60, then seeding 9 + incentive 2 for 90, stop 100",
                                    seg(([0, 2, 0], 60), ([9, 2, 0], 90), (z, 100))),
        "E05_inc_after_seed_250": ("seeding 9 for 90, then seeding 9 + incentive 2 for 60, stop 100",
                                   seg(([9, 0, 0], 90), ([9, 2, 0], 60), (z, 100))),
        "E06_bridge1_only_200": ("bridge 1.0 alone 150, stop 50", seg(([0, 0, 1], 150), (z, 50))),
        "E07_seed9_long_queue_300": ("seeding 9 alone 200 (queue build, credibility A), stop 100",
                                     seg(([9, 0, 0], 200), (z, 100))),
        "E08_seed9_bridge1_200": ("local vs bridge: seeding 9 + bridge 1.0 for 150, stop 50",
                                  seg(([9, 0, 1], 150), (z, 50))),
        "E09_inc_stepdown_300": ("seeding 9 + incentive 2 for 100, incentive 0.8 (seeding 9) 100, stop 100",
                                 seg(([9, 2, 0], 100), ([9, 0.8, 0], 100), (z, 100))),
        "E10_interior_joint_300": ("seeding 5 + incentive 1 + bridge 0.5 for 200, stop 100",
                                   seg(([5, 1, 0.5], 200), (z, 100))),
        "E11_recovery_train_300": ("eval-style pulse train (alpha U(.7,1)), 300 ticks",
                                   D.eval_like(spec, "recovery", 300, rng)),
        "E12_sustained_multilevel_300": ("three 100-tick holds: mid, uniform, pulse(alpha)",
                                         seg((D.level(spec, "mid", rng), 100), (D.level(spec, "uniform", rng), 100),
                                             (D.level(spec, "pulse", rng), 100))),
        "E13_seed9_inc2_long_300": ("seeding 9 + incentive 2 (no bridge) 200, stop 100",
                                    seg(([9, 2, 0], 200), (z, 100))),
    }
    return E


# ------------------------------------------------------------------ main
def pair_dis(Ys, names, sig):
    """Per tick mean over pairs and observables of |Yi - Yj| / sigma; also per observable."""
    out = []
    for a, b in itertools.combinations(names, 2):
        out.append(np.abs(Ys[a] - Ys[b]) / sig[None, :])
    X = np.stack(out)                        # [pairs, T, p]
    return X.mean(axis=(0, 2)), X.mean(axis=0)


def main():
    t0 = time.time()
    spec = S.get(SYS)
    runs = load_runs(spec, Ledger(data_dir(SYS, False), SYS, False))
    sigma = metric.sigma_proxy(runs)
    Yall = np.concatenate([r.Y for r in runs])
    d = np.concatenate([np.diff(r.Y, axis=0) for r in runs])
    noise = np.median(np.abs(d - np.median(d, axis=0)), axis=0) * 1.4826 / np.sqrt(2)
    print("runs", [(r.exp, r.T) for r in runs], "sigma", np.round(sigma, 2), "noise", np.round(noise, 2))
    members, val = load_members(spec, runs, sigma)
    for k, v in val.items():
        print(f"  {k:<8} valid={v['valid']} diff={v.get('max_abs_diff')} now={v.get('now', v)}")
    names = list(members)
    top = [n for n in TOP if n in members]
    print("members", names, "top", top)

    # ---------------- test schedules
    inits = [r.y0 for r in runs]
    tests = []
    for ci, cat in enumerate(CATS):
        for k in range(N_PER_CAT):
            rng = np.random.default_rng(1000 * ci + k)
            U = D.eval_like(spec, cat, T_TEST, rng)
            tests.append((cat, k, inits[(ci + k) % len(inits)], U))
    rolls = []                                   # list of dict name -> Y
    for cat, k, y0, U in tests:
        rolls.append({n: np.asarray(members[n](y0, U), float) for n in names})
    print(f"rolled test set [{time.time() - t0:.0f}s]")

    # ---------------- disagreement by category / regime / dwell
    res = {"sigma_proxy": sigma.tolist(), "noise": noise.tolist(), "validation": val, "members": names, "top": top}
    by_cat, reg_mass, reg_ticks, dw_cat = {}, {}, {}, {}
    pair_tab = {}
    for (cat, k, y0, U), Ys in zip(tests, rolls):
        Dt, Dobs = pair_dis(Ys, top, sigma)
        by_cat.setdefault(cat, []).append([float(Dt.mean()), *Dobs.mean(0).tolist(),
                                           float(np.mean(1 - 1 / (1 + Dt)))])
        labs = seq_labels(U)
        for l in np.unique(labs):
            msk = labs == l
            reg_mass[l] = reg_mass.get(l, 0.0) + float(Dt[msk].sum())
            reg_ticks[l] = reg_ticks.get(l, 0) + int(msk.sum())
        db = dwell_bin(dwell(U))
        for l in np.unique(db):
            key = (cat, l)
            m_ = db == l
            a = dw_cat.setdefault(key, [0.0, 0])
            a[0] += float(Dt[m_].sum()); a[1] += int(m_.sum())
        for a_, b_ in itertools.combinations(names, 2):
            pair_tab.setdefault(cat, {}).setdefault(f"{a_}|{b_}", []).append(
                float(np.mean(np.abs(Ys[a_] - Ys[b_]) / sigma[None, :])))
    tot_mass = sum(reg_mass.values())
    tot_ticks = sum(reg_ticks.values())
    res["disagreement_by_category"] = {
        c: dict(zip(["mean_sigma", "adopters_a", "adopters_b", "score_gap"], np.mean(v, axis=0).round(4).tolist()))
        for c, v in by_cat.items()}
    res["disagreement_by_regime"] = sorted(
        [{"regime": l, "tick_share": round(reg_ticks[l] / tot_ticks, 4), "mass_share": round(reg_mass[l] / tot_mass, 4),
          "mean_sigma": round(reg_mass[l] / reg_ticks[l], 4)} for l in reg_mass], key=lambda x: -x["mass_share"])
    res["disagreement_by_dwell"] = {f"{c}|{l}": {"ticks": v[1], "mean_sigma": round(v[0] / max(1, v[1]), 4)}
                                    for (c, l), v in sorted(dw_cat.items())}
    for q, pr in (("A_vs_C", ("AB2", "BC2")), ("C_on", ("BC2", "BC2_kC")), ("public_vs_BC2", ("public", "BC2"))):
        if all(x in names for x in pr):
            res.setdefault("question_by_category", {})[q] = {c: round(float(np.mean(pair_tab[c][f"{pr[0]}|{pr[1]}"] if f"{pr[0]}|{pr[1]}" in pair_tab[c] else pair_tab[c][f"{pr[1]}|{pr[0]}"])), 4) for c in CATS}
    print("questions by category:", res.get("question_by_category"))
    res["pairwise_by_category"] = {c: {p: round(float(np.mean(v)), 3) for p, v in d_.items()} for c, d_ in pair_tab.items()}
    print(json.dumps(res["disagreement_by_category"], indent=1))
    for r in res["disagreement_by_regime"][:12]:
        print("  ", r)

    # ---------------- cross-scores: predictor P vs member-as-truth M (sigma_proxy)
    preds = dict((n, None) for n in names)
    combos = {"med(BC2,AB2,min_v2)": ["BC2", "AB2", "min_v2"], "med(BC2,AB2,l0b_lin)": ["BC2", "AB2", "l0b_lin"],
              "mean(BC2,AB2)": ["BC2", "AB2"], "med(BC2,AB2,min_v2,l0b_lin)": ["BC2", "AB2", "min_v2", "l0b_lin"],
              "mean(BC2,l0b_lin)": ["BC2", "l0b_lin"]}
    combos = {k: v for k, v in combos.items() if all(x in members for x in v)}
    truths = [n for n in ["BC2", "AB2", "BC", "AB", "min_v2", "l0b_lin"] if n in members]
    cross = {}
    for (cat, k, y0, U), Ys in zip(tests, rolls):
        P = dict(Ys)
        for cname, mem in combos.items():
            P[cname] = np.median(np.stack([Ys[x] for x in mem]), axis=0)
        # per-observable pick: BC2 for a, public for b (example of a per-obs option)
        if "BC2" in Ys and "public" in Ys:
            P["perobs(a:BC2,b:public)"] = np.stack([Ys["BC2"][:, 0], Ys["public"][:, 1]], 1)
            P["perobs(a:BC2,b:med3)"] = np.stack([Ys["BC2"][:, 0], P.get("med(BC2,AB2,min_v2)", Ys["BC2"])[:, 1]], 1)
        for pn, Yp in P.items():
            for tn in truths:
                s = metric.score_per_obs(Yp, Ys[tn], sigma)
                cross.setdefault(pn, {}).setdefault(tn, {}).setdefault(cat, []).append(s.tolist())
    xs = {}
    for pn, d_ in cross.items():
        xs[pn] = {}
        for tn, dc in d_.items():
            xs[pn][tn] = {c: np.mean(v, axis=0).round(4).tolist() for c, v in dc.items()}
            xs[pn][tn]["all"] = float(np.mean([np.mean(v) for v in dc.values()]))
        # robust: mean over the ODE truths (BC2, AB2) and worst case
        ode_t = [t for t in ("BC2", "AB2") if t in d_]
        xs[pn]["robust_mean_BC2_AB2"] = float(np.mean([xs[pn][t]["all"] for t in ode_t])) if ode_t else None
        xs[pn]["worst_over_truths"] = float(min(xs[pn][t]["all"] for t in truths if t != pn)) if len(truths) > 1 else None
    res["cross_scores"] = xs
    print("cross (robust mean over BC2/AB2 truths, worst):")
    for pn in sorted(xs, key=lambda p: -(xs[p]["robust_mean_BC2_AB2"] or 0)):
        print(f"   {pn:<30} {xs[pn]['robust_mean_BC2_AB2']:.4f}  worst {xs[pn]['worst_over_truths']:.4f}  "
              + " ".join(f"{c[:4]} {np.mean([xs[pn][t][c] for t in ('BC2','AB2') if t in xs[pn]]):.3f}" for c in CATS))

    # ---------------- experiments
    w_reg = {l: reg_mass[l] / tot_mass for l in reg_mass}
    y0e = np.median(np.stack(inits), axis=0)
    E = candidates(spec)
    exps = []
    for eid, (desc, U) in E.items():
        U = np.asarray(U, float)
        Ys = {n: np.asarray(members[n](y0e, U), float) for n in names}
        Dt, Dobs = pair_dis(Ys, top, sigma)
        Dt_ode, _ = pair_dis(Ys, [x for x in ("BC2", "AB2") if x in Ys], sigma)
        qs = {}
        for qn, pr in (("A_vs_C(AB2|BC2)", ("AB2", "BC2")), ("C_on(BC2|BC2_kC)", ("BC2", "BC2_kC")),
                       ("public|BC2", ("public", "BC2"))):
            if all(x in Ys for x in pr):
                dq = np.mean(np.abs(Ys[pr[0]] - Ys[pr[1]]) / sigma[None, :], axis=1)
                qs[qn] = {"mean_sigma": round(float(dq.mean()), 4), "max_sigma": round(float(dq.max()), 4),
                          "noise_units_mean": round(float(np.mean(np.abs(Ys[pr[0]] - Ys[pr[1]]) / noise[None, :])), 2)}
        Dn = np.stack([np.abs(Ys[a] - Ys[b]) / noise[None, :] for a, b in itertools.combinations(top, 2)]).mean(axis=(0, 2))
        labs = seq_labels(U)
        w = np.array([w_reg.get(l, 0.0) for l in labs])
        # relevance-weighted information: disagreement at each tick x share of test disagreement
        # in that regime, normalised so a regime holding all test disagreement has weight 1
        value = float(np.sum(Dt * w))
        exps.append({"id": eid, "desc": desc, "credits": int(len(U)), "mean_dis_sigma": round(float(Dt.mean()), 4),
                     "max_dis_sigma": round(float(Dt.max()), 4), "mean_dis_BC2_AB2": round(float(Dt_ode.mean()), 4),
                     "mean_dis_noise_units": round(float(Dn.mean()), 3),
                     "dis_obs": Dobs.mean(0).round(4).tolist(), "value": round(value, 3),
                     "value_per_credit": round(value / len(U), 5),
                     "questions": qs, "regimes": sorted(set(labs.tolist())), "U": U.round(4).tolist()})
    exps.sort(key=lambda e: -e["value_per_credit"])
    res["experiments"] = exps
    print("experiments (value per credit):")
    for e in exps:
        print(f"   {e['id']:<30} cr {e['credits']:>3} dis {e['mean_dis_sigma']:.3f} (ode {e['mean_dis_BC2_AB2']:.3f}, "
              f"noise-units {e['mean_dis_noise_units']:.1f}) q {e['questions']} value {e['value']:.2f} vpc {e['value_per_credit']:.4f}")
    res["y0_experiments"] = y0e.tolist()
    ex = {e["id"]: e for e in exps}
    res["recommended"] = {
        "free_first": "replace public median(min_v2, l0b_lin) by median(BC2, AB2, l0b_lin); upload and read the bands",
        "buy": [{"id": k, "credits": ex[k]["credits"], "U": ex[k]["U"]} for k in ("E12_sustained_multilevel_300",) if k in ex],
        "alternative_cheaper": [{"id": k, "credits": ex[k]["credits"]} for k in ("E01_interior_alpha05_200",) if k in ex],
        "mechanism_only_if_wanted": [{"id": k, "credits": ex[k]["credits"]} for k in ("E09_inc_stepdown_300", "E06_bridge1_only_200") if k in ex],
    }
    res["test_seeds"] = "np.random.default_rng(1000*cat_index + k), k=0..9; y0 cycles the observed initials"
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
