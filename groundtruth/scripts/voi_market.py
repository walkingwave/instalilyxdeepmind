"""Value-of-information study for market (free: simulation only, no credits).

    python scripts/voi_market.py [--thin 10] [--mc 200]

1. validate each committee doc against its lab json's in-sample scores (raw core.rollout on every run)
2. committee disagreement on 40 eval_like schedules (10 per category, T=4000, seeds 0..39)
3. candidate experiments: Bayesian preposterior value = expected test score after picking the
   posterior-best predictor, minus the prior-best score; hypotheses = committee members
4. free options: per-observable picks / medians scored against every hypothesis
Writes plans/voi_market.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import design as D, metric, select as SEL, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.models import common as C
from gtlab.ode import core
from gtlab.runtime.infer import ode_modules, rollout_from_blob

DOCS = {"gate_s1": "plans/market_market_gate_s1_doc.json", "min_p3": "plans/market_market_min_p3_doc.json",
        "AC2": "plans/market_market_mech_AC2_doc.json", "AB2": "plans/market_market_mech_AB2_doc.json",
        "BC2": "plans/market_market_mech_BC2_doc.json"}
PUBLIC = "submissions/20260926-0135-u010b/market/model.json"
RECOMMENDED = {
    "primary": {"name": "E2 rate0.1 hold300", "credits": 300,
                "why": "highest preposterior value (+0.134) and posterior on truth (0.85); shows 94% of the mechanism models' long-hold fall, so the floor question is answered by data, not extrapolation"},
    "cheaper": {"name": "E1 rate0.1 hold200", "credits": 200, "why": "+0.125, 80% of the fall visible; keeps 100 credits"},
    "alternative": {"name": "E11 rate0.1 200 -> 0 100", "credits": 300,
                    "why": "+0.130; trades 100 hold ticks for the recovery from a deep fall (largest zero-control disagreement share)"},
    "free_change": "median(gate_s1, AC2, l0b_lin) replaces median(min_p3, l0b_lin): +0.031 expected, +0.010 worst case over hypotheses",
}
CATS = ("sustained", "order", "recovery", "composition")
TAX_C = 0.047


def lab_json(p):
    return p.replace("_doc.json", ".json")


class DocMember:
    def __init__(self, doc):
        self.doc = doc

    def roll(self, y0, U):
        return rollout_from_blob(self.doc["model"], y0, U, doc=self.doc)


class L0bMember:
    def __init__(self, spec, runs, sigma):
        clip = C.soft_clip(spec, runs, margin=1.0)
        self.m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(runs)

    def roll(self, y0, U):
        return np.asarray(self.m.rollout(np.asarray(y0, float), np.asarray(U, float)), float)


class Median:
    def __init__(self, members):
        self.members = members

    def roll(self, y0, U):
        return np.median(np.stack([m.roll(y0, U) for m in self.members]), axis=0)


class PerObs:
    def __init__(self, members):          # one member per observable
        self.members = members

    def roll(self, y0, U):
        cache = {}
        out = []
        for j, m in enumerate(self.members):
            if id(m) not in cache:
                cache[id(m)] = m.roll(y0, U)
            out.append(cache[id(m)][:, j])
        return np.stack(out, axis=1)


def validate(spec, runs, sigma):
    rep = {}
    for name, p in DOCS.items():
        doc = json.loads(Path(p).read_text())
        lab = json.loads(Path(lab_json(p)).read_text())
        blob = doc["model"]
        _, fam = ode_modules(blob["family"])
        th = core.theta_dict(fam, [float(v) for v in blob["theta"]])
        diffs, clipped = [], []
        for r in runs:
            Y = core.rollout(fam, r.y0, r.U, th, frozenset(blob["mech"]), n_sub=int(blob.get("n_sub", 4)))
            s = metric.score_per_obs(Y, r.Y, sigma)
            want = np.asarray(lab["insample"][r.exp])
            diffs.append(float(np.max(np.abs(s - want))))
            Yc = rollout_from_blob(blob, r.y0, r.U, doc=doc)
            clipped.append(metric.score_per_obs(Yc, r.Y, sigma).tolist())
        ok = max(diffs) < 1e-3
        rep[name] = {"max_abs_diff": max(diffs), "reproduces": ok, "insample_clipped": clipped,
                     "insample_mean": float(np.mean(clipped))}
        print(f"validate {name:<8} max|d|={max(diffs):.2e} ok={ok} clipped in-sample mean {np.mean(clipped):.3f}")
    return rep


def test_set(spec, runs, n_per=10, T=4000):
    sets = []
    y0s = [r.y0 for r in runs]
    for ci, cat in enumerate(CATS):
        for k in range(n_per):
            seed = ci * n_per + k
            rng = np.random.default_rng(1000 + seed)
            U = D.eval_like(spec, cat, T, rng)
            sets.append({"cat": cat, "seed": seed, "U": U, "y0": y0s[seed % len(y0s)]})
    return sets


def regime_labels(U):
    """Per tick: rate band x tax side x hold age (ticks since the controls last changed)."""
    r, x = U[:, 0], U[:, 1]
    rb = np.where(r < 0.01, "r0", np.where(r < 0.07, "rmid", "rhigh"))
    xb = np.where(x >= TAX_C, "xfrz", np.where(x < 0.01, "x0", "xmid"))
    age = np.zeros(len(U), int)
    for t in range(1, len(U)):
        age[t] = 0 if np.any(U[t] != U[t - 1]) else age[t - 1] + 1
    ab = np.where(age < 60, "a<60", np.where(age < 300, "a60-300", "a>=300"))
    return rb, xb, ab


def score_mat(A, B, sigma):
    return 1.0 / (1.0 + np.abs(A - B) / sigma[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thin", type=int, default=25, help="use every k-th tick in the likelihood (autocorrelated misfit)")
    ap.add_argument("--mc", type=int, default=300)
    a = ap.parse_args()
    spec = S.get("market")
    runs = load_runs(spec, Ledger(data_dir("market", False), "market", False))
    sigma = metric.sigma_proxy(runs)
    print("runs", [(r.exp, r.T) for r in runs], "sigma", np.round(sigma, 2))
    out = {"sigma_proxy": sigma.tolist(), "runs": [(r.exp, r.T) for r in runs]}

    # ---- 1. validation
    out["validation"] = validate(spec, runs, sigma)
    hyp = {n: DocMember(json.loads(Path(DOCS[n]).read_text())) for n in DOCS if out["validation"][n]["reproduces"]}
    with C.single_thread():
        l0b = L0bMember(spec, runs, sigma)
    hyp["l0b_lin"] = l0b
    pub_doc = json.loads(Path(PUBLIC).read_text())
    public = DocMember(pub_doc)
    names = list(hyp)
    # l0b in-sample for reference
    out["validation"]["l0b_lin"] = {"insample_mean": float(np.mean([metric.score_per_obs(l0b.roll(r.y0, r.U), r.Y, sigma)
                                                                     for r in runs]))}
    out["validation"]["public_u010b"] = {"insample_mean": float(np.mean([metric.score_per_obs(public.roll(r.y0, r.U), r.Y, sigma)
                                                                          for r in runs]))}
    print("in-sample l0b_lin", round(out["validation"]["l0b_lin"]["insample_mean"], 3),
          "public", round(out["validation"]["public_u010b"]["insample_mean"], 3))

    # ---- predictors (free options)
    preds = dict(hyp)
    preds["public_u010b"] = public
    preds["med(gate,l0b)"] = Median([hyp["gate_s1"], l0b]) if "gate_s1" in hyp else None
    odes = [hyp[n] for n in names if n != "l0b_lin"]
    preds["med(all ODE)"] = Median(odes)
    preds["med(all)"] = Median(odes + [l0b])
    if "AC2" in hyp:
        preds["med(gate,AC2,l0b)"] = Median([hyp["gate_s1"], hyp["AC2"], l0b])
        preds["med(gate,AC2,AB2)"] = Median([hyp["gate_s1"], hyp["AC2"], hyp["AB2"]])
    preds = {k: v for k, v in preds.items() if v is not None}

    # ---- 2. test distribution
    tests = test_set(spec, runs)
    Yt = {n: [] for n in preds}
    for tcase in tests:
        for n, m in preds.items():
            Yt[n].append(m.roll(tcase["y0"], tcase["U"]))
    cat_idx = {c: [i for i, t in enumerate(tests) if t["cat"] == c] for c in CATS}
    # regime composition
    lab = [regime_labels(t["U"]) for t in tests]
    comp = {}
    for c in CATS:
        rb = np.concatenate([lab[i][0] for i in cat_idx[c]])
        xb = np.concatenate([lab[i][1] for i in cat_idx[c]])
        ab = np.concatenate([lab[i][2] for i in cat_idx[c]])
        comp[c] = {"rhigh_xfree": float(np.mean((rb == "rhigh") & (xb != "xfrz"))),
                   "rhigh_xfree_age>=300": float(np.mean((rb == "rhigh") & (xb != "xfrz") & (ab == "a>=300"))),
                   "rmid_xfree_age>=300": float(np.mean((rb == "rmid") & (xb != "xfrz") & (ab == "a>=300"))),
                   "tax_frozen": float(np.mean(xb == "xfrz")),
                   "r0_x0": float(np.mean((rb == "r0") & (xb == "x0"))),
                   "age>=300": float(np.mean(ab == "a>=300"))}
    out["test_composition"] = comp
    print("\ntest regime fractions per category")
    for c in CATS:
        print(f"  {c:<12}", {k: round(v, 3) for k, v in comp[c].items()})

    # pairwise disagreement among hypotheses (sigma units), per category and observable
    dis = {}
    for c in CATS:
        per = []
        for i, j in itertools.combinations(names, 2):
            d = np.mean([np.mean(np.abs(Yt[i][k] - Yt[j][k]), axis=0) / sigma for k in cat_idx[c]], axis=0)
            per.append(d)
        dis[c] = {"mean_pairwise_sigma": np.mean(per, axis=0).tolist(), "max_pair_sigma": np.max(per, axis=0).tolist()}
    out["disagreement_by_category"] = dis
    print("\nmean pairwise disagreement |dy|/sigma (price, volume, depth)")
    for c in CATS:
        print(f"  {c:<12} mean {np.round(dis[c]['mean_pairwise_sigma'], 2)}  max-pair {np.round(dis[c]['max_pair_sigma'], 2)}")
    # disagreement by regime (all categories pooled), price only + all obs
    reg = {}
    pooled_lab = np.concatenate([np.char.add(np.char.add(np.char.add(lab[k][0], "|"), np.char.add(lab[k][1], "|")), lab[k][2])
                                 for k in range(len(tests))])
    pair_abs = []
    for i, j in itertools.combinations(names, 2):
        pair_abs.append(np.concatenate([np.abs(Yt[i][k] - Yt[j][k]) / sigma for k in range(len(tests))]))
    pair_abs = np.mean(pair_abs, axis=0)       # [N, p] mean pairwise
    for g in np.unique(pooled_lab):
        msk = pooled_lab == g
        reg[str(g)] = {"frac": float(msk.mean()), "dis": pair_abs[msk].mean(0).tolist(),
                       "share_of_total_dis": float(pair_abs[msk].sum() / pair_abs.sum())}
    out["disagreement_by_regime"] = dict(sorted(reg.items(), key=lambda kv: -kv[1]["share_of_total_dis"]))
    print("\nregime (rate|tax|age): tick share, mean pairwise dis (P,V,D), share of total disagreement")
    for g, v in list(out["disagreement_by_regime"].items())[:12]:
        print(f"  {g:<22} {v['frac']:.3f}  {np.round(v['dis'], 2)}  {v['share_of_total_dis']:.3f}")

    # score matrix: predictor i scored against hypothesis j as truth, per category
    Smat = {}
    for c in CATS:
        Smat[c] = {i: {j: metric.score_per_obs(np.concatenate([Yt[i][k] for k in cat_idx[c]]),
                                               np.concatenate([Yt[j][k] for k in cat_idx[c]]), sigma).tolist()
                       for j in names} for i in preds}
    Sall = {i: {j: float(np.mean([np.mean(Smat[c][i][j]) for c in CATS])) for j in names} for i in preds}
    Sobs = {i: {j: np.mean([Smat[c][i][j] for c in CATS], axis=0) for j in names} for i in preds}
    out["score_matrix_overall"] = Sall
    print("\nscore of predictor (row) if hypothesis (col) is the truth, mean of 4 categories")
    print("  " + " " * 20 + " ".join(f"{j:>8}" for j in names) + "   mean    min")
    for i in preds:
        row = [Sall[i][j] for j in names]
        print(f"  {i:<20}" + " ".join(f"{v:8.3f}" for v in row) + f"  {np.mean(row):.3f}  {np.min(row):.3f}")

    # per-observable best picks (maximise mean over hypotheses, per observable)
    base_names = [n for n in preds]
    best_obs = []
    for jo in range(3):
        vals = {i: float(np.mean([Sobs[i][h][jo] for h in names])) for i in base_names}
        best_obs.append(max(vals, key=vals.get))
    po = PerObs([preds[n] for n in best_obs])
    out["perobs_pick"] = best_obs
    po_scores = {h: float(np.mean([Sobs[best_obs[jo]][h][jo] for jo in range(3)])) for h in names}
    Sall["perobs_pick"] = po_scores
    print("  perobs_pick", best_obs, "mean", round(np.mean(list(po_scores.values())), 3),
          "min", round(min(po_scores.values()), 3))
    # also per-category per-observable table for the free-change section
    out["score_by_obs_mean_over_hyp"] = {i: np.mean([Sobs[i][h] for h in names], axis=0).tolist() for i in base_names}

    # ---- 3. candidate experiments
    y0_exp = [r.y0 for r in runs]
    Z = lambda n, r=0.0, x=0.0: np.tile([r, x], (n, 1))
    rng = np.random.default_rng(7)
    ptrain = D.pulse_train(spec, rng, n=6, L=(10, 25), gaps=(5, 20), lead=10, T=200)
    cands = {
        "E1 rate0.1 hold200": Z(200, 0.1),
        "E2 rate0.1 hold300": Z(300, 0.1),
        "E3 rate0.1 hold120": Z(120, 0.1),
        "E4 rate0.07 hold200": Z(200, 0.07),
        "E5 rate0.1 150 -> 0 100 (reversal)": np.vstack([Z(150, 0.1), Z(100)]),
        "E6 rate0.1+tax0.044 100 -> rate0.1+tax0.049 100": np.vstack([Z(100, 0.1, 0.044), Z(100, 0.1, 0.049)]),
        "E7 alternating rate/tax 25-tick x8": np.vstack([Z(25, 0.1, 0) if k % 2 == 0 else Z(25, 0, 0.05) for k in range(8)]),
        "E8 interior (0.05,0.025) hold200": Z(200, 0.05, 0.025),
        "E9 rate pulses 6x short gaps 200": ptrain,
        "E10 pulse-ref alpha0.85 (0.085,0.0425) hold250": Z(250, 0.085, 0.0425),
        "E11 rate0.1 200 -> 0 100": np.vstack([Z(200, 0.1), Z(100)]),
        "E12 rate0.1+tax0.03 hold200": Z(200, 0.1, 0.03),
        "E13 rate0.1+tax0.044 60 -> rate0.1+tax0.049 60": np.vstack([Z(60, 0.1, 0.044), Z(60, 0.1, 0.049)]),
        "E14 rate0.1 hold150": Z(150, 0.1),
        "E15 rate0.1 120 -> 0 80 (short reversal)": np.vstack([Z(120, 0.1), Z(80)]),
    }
    # observation noise per obs: white-noise estimate + model misfit (median member in-sample RMS)
    dd = np.concatenate([np.diff(r.Y, axis=0) for r in runs])
    noise = np.median(np.abs(dd - np.median(dd, 0)), 0) * 1.4826 / np.sqrt(2)
    mis = []
    for n in names:
        res = np.concatenate([hyp[n].roll(r.y0, r.U) - r.Y for r in runs])
        mis.append(np.sqrt(np.mean(res ** 2, 0)))
    misfit = np.median(mis, axis=0)
    sd = np.sqrt(noise ** 2 + misfit ** 2)
    out["likelihood_sd"] = {"noise": noise.tolist(), "misfit": misfit.tolist(), "sd": sd.tolist(), "thin": a.thin}
    print("\nlikelihood sd per obs", np.round(sd, 2), "(noise", np.round(noise, 2), "misfit", np.round(misfit, 2), ")")

    H = len(names)
    Smat_h = np.array([[Sall[i][j] for j in names] for i in names])     # predictors = hypotheses
    prior_val = Smat_h.mean(1).max()
    prior_pick = names[int(Smat_h.mean(1).argmax())]
    pub_val = float(np.mean([Sall["public_u010b"][j] for j in names]))
    print(f"prior-best predictor {prior_pick} expected {prior_val:.3f}; public expected {pub_val:.3f}")
    rng_mc = np.random.default_rng(11)
    exps = {}
    store_Ys = {}
    for en, U in cands.items():
        Ys = np.stack([np.stack([hyp[n].roll(y0, U) for y0 in y0_exp]) for n in names])   # [H, n_y0, T, p]
        Ys = Ys[:, :, ::a.thin, :]
        # expected post-experiment value by Monte Carlo over truth k, y0 and noise
        vals = []
        post_right = []
        for _ in range(a.mc):
            k = rng_mc.integers(H)
            q = rng_mc.integers(len(y0_exp))
            y = Ys[k, q] + rng_mc.normal(0, 1, Ys[k, q].shape) * sd
            ll = -0.5 * np.sum(((Ys[:, q] - y[None]) / sd) ** 2, axis=(1, 2))
            w = np.exp(ll - ll.max()); w /= w.sum()
            pick = int(np.argmax(Smat_h @ w))
            vals.append(Smat_h[pick, k])
            post_right.append(w[k])
        voi = float(np.mean(vals) - prior_val)
        # simple spread: mean pairwise |dy|/sigma on the experiment
        spread = np.mean([np.mean(np.abs(Ys[i] - Ys[j]) / sigma) for i, j in itertools.combinations(range(H), 2)])
        sep = np.mean([np.mean(np.abs(Ys[i] - Ys[j]) / sd) for i, j in itertools.combinations(range(H), 2)])
        # decision-weighted pair resolution: sum_ij (1-S_ij) * P(resolve ij) / sum_ij (1-S_ij),
        # P(resolve) = 1 - exp(-z^2/8), z^2 = mean over y0 of sum_t ((Yi-Yj)/sd)^2 on thinned ticks
        num = den = 0.0
        for i, j in itertools.combinations(range(H), 2):
            z2 = float(np.mean(np.sum(((Ys[i] - Ys[j]) / sd) ** 2, axis=(1, 2))))
            wgt = 1.0 - 0.5 * (Smat_h[i, j] + Smat_h[j, i])
            num += wgt * (1 - np.exp(-z2 / 8)); den += wgt
        resolve = num / den
        # settle fraction: share of each member's 4000-tick price move (under this schedule's final
        # controls held on) that is already visible at the end of the experiment
        Ulong = np.vstack([U, np.tile(U[-1], (4000 - len(U), 1))])
        sf = {}
        for i, n in enumerate(names):
            fr = []
            for y0 in y0_exp:
                YL = hyp[n].roll(y0, Ulong)[:, 0]
                tot = YL[-1] - y0[0]
                fr.append(float((YL[len(U) - 1] - y0[0]) / tot) if abs(tot) > 1.0 else 1.0)
            sf[n] = float(np.mean(fr))
        store_Ys[en] = Ys
        cost = len(U)
        exps[en] = {"ticks": cost, "voi": voi, "resolve": float(resolve), "resolve_per_100": 100 * float(resolve) / cost,
                    "settle_frac": sf, "voi_per_100": 100 * voi / cost, "mean_post_on_truth": float(np.mean(post_right)),
                    "spread_sigma": float(spread), "sep_sd": float(sep),
                    "price_end": {n: float(np.mean(Ys[i, :, -1, 0])) for i, n in enumerate(names)},
                    "U": U.tolist()}
        print(f"  {en:<48} T={cost:3d} VOI {voi:+.4f} per100 {100 * voi / cost:+.4f} "
              f"P(truth) {np.mean(post_right):.2f} resolve {resolve:.3f} spread {spread:.2f} "
              f"settle(mech) {np.mean([sf[n] for n in names if n in ('AC2', 'AB2', 'BC2')]):.2f}")
    # pairs of experiments (separate resets) within 320 ticks: product likelihood, same MC
    combos = {}
    keys = list(cands)
    for e1, e2 in itertools.combinations(keys, 2):
        cost = len(cands[e1]) + len(cands[e2])
        if cost > 320:
            continue
        vals = []
        for _ in range(a.mc):
            k = rng_mc.integers(H)
            ll = np.zeros(H)
            for en in (e1, e2):
                Ys = store_Ys[en]
                q = rng_mc.integers(len(y0_exp))
                y = Ys[k, q] + rng_mc.normal(0, 1, Ys[k, q].shape) * sd
                ll += -0.5 * np.sum(((Ys[:, q] - y[None]) / sd) ** 2, axis=(1, 2))
            w = np.exp(ll - ll.max()); w /= w.sum()
            vals.append(Smat_h[int(np.argmax(Smat_h @ w)), k])
        combos[f"{e1.split()[0]}+{e2.split()[0]}"] = {"ticks": cost, "voi": float(np.mean(vals) - prior_val)}
    out["experiment_pairs"] = dict(sorted(combos.items(), key=lambda kv: -kv[1]["voi"])[:10])
    print("best pairs within 320 ticks", {k: (v["ticks"], round(v["voi"], 4)) for k, v in out["experiment_pairs"].items()})
    y0m = np.mean([r.y0 for r in runs], 0)
    ends = {}
    for lab_, u in [("r0,x0", [0, 0]), ("r0.1,x0", [0.1, 0]), ("r0.07,x0", [0.07, 0]), ("r0.05,x0.025", [0.05, 0.025]),
                    ("r0.085,x0.0425", [0.085, 0.0425]), ("r0,x0.05", [0, 0.05]), ("r0.1,x0.05", [0.1, 0.05])]:
        UL = np.tile(u, (4000, 1))
        ends[lab_] = {n: np.round(hyp[n].roll(y0m, UL)[[50, 120, 200, 300, 1000, 3999], 0], 1).tolist() for n in names}
    out["long_hold_price_t50_120_200_300_1000_3999"] = ends
    out["prior"] = {"best_predictor": prior_pick, "expected": float(prior_val), "public_expected": pub_val}
    # sequential: best pairs within 320 ticks
    out["experiments"] = dict(sorted(exps.items(), key=lambda kv: -kv[1]["voi_per_100"]))
    out["hypotheses"] = names
    out["recommended"] = {k: dict(v, U=exps[v["name"]]["U"]) if isinstance(v, dict) else v for k, v in RECOMMENDED.items()}
    out["tests"] = [{"cat": t["cat"], "seed": t["seed"]} for t in tests]
    Path("plans/voi_market.json").write_text(json.dumps(out, indent=1, default=lambda o: np.asarray(o).tolist()))
    print("wrote plans/voi_market.json")


if __name__ == "__main__":
    main()
