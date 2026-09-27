"""Round-2 value of information across the five weakest systems (free: simulation only, no credits).

    python scripts/voi_round2.py --system market [--n-per 10 --mc 400 --thin 20]   # one system -> scratch npz/json
    python scripts/voi_round2.py --portfolio                                         # combine -> plans/voi_round2.json

Per system:
1. committee = current public model + strongest alternatives; each doc must reproduce its lab json's
   in-sample scores (lab sigma, runs the lab saw) within 0.01, or match the shipped predictor it came from.
2. 40 eval_like test schedules (10 per category, T=4000), y0 cycled over the ledger initials; score matrix
   S[i, j] = test score of predictor i if member j were the truth, at the organizer's sigma.
3. candidate runs: preposterior value = E_k,noise[ S[pick(w_post), k] ] - max_i sum_j w_j S[i, j], where the
   truth k ~ prior w, the run is simulated under k with noise sd = sqrt(noise^2 + misfit^2) on every
   `thin`-th tick, and w_post is the likelihood update. Sets of runs use the product likelihood.
4. portfolio: greedy by marginal value per credit across systems, reserve 150 per system.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gtlab import design as D, metric, systems as S
from gtlab.ledger import Ledger, data_dir, load_runs
from gtlab.ode import core
from gtlab.runtime.infer import ode_modules, rollout_from_blob

SCR = Path(r"C:\Users\DYLANH~1\AppData\Local\Temp\claude\scratch\voi2")
CATS = ("sustained", "order", "recovery", "composition")
BAL = {"social_contagion": 489, "market": 660, "hospital_queue": 930, "epidemic": 889, "wildlife": 889}
PUBLIC_NOW = {"social_contagion": 0.626, "market": 0.627, "hospital_queue": 0.672, "epidemic": 0.687, "wildlife": 0.724}
RESERVE = 150
P = "plans/"
# committee: name -> (doc path, public overall score of that model if it was scored, reference for docs w/o lab json)
COMMITTEE = {
    "market": {"x4p": (P + "market_x4p_doc.json", 0.627, "submissions/20260927-0233-u013/market/model.json"),
               "x4": (P + "market_market_x4_x7_doc.json", None, None),
               "x3": (P + "market_market_x3_x7_doc.json", None, None),
               "v9g": (P + "market_market_v9g_v9cal_doc.json", 0.588, None),
               "v8t2": (P + "market_market_v8t2_v8_doc.json", None, None)},
    "social_contagion": {k: (P + f"social_contagion_social_contagion_{k}_xr5_doc.json", s, None)
                         for k, s in (("x7", 0.626), ("x2", None), ("x6", None), ("x8", None), ("v8o", None))},
    "epidemic": {"x11_AC": (P + "epidemic_epidemic_x11_AC_doc.json", 0.687, None),
                 "x10_AC": (P + "epidemic_epidemic_x10_AC_doc.json", None, None),
                 "x7_AB": (P + "epidemic_epidemic_x7_AB_doc.json", None, None),
                 "x8_AC": (P + "epidemic_epidemic_x8_AC_doc.json", None, None),
                 "v8g_gAB": (P + "epidemic_epidemic_v8g_gAB_doc.json", None, None)},
    "wildlife": {"x13": (P + "wildlife_wildlife_x13_a_doc.json", 0.724, None),
                 "x6": (P + "wildlife_wildlife_x6_a_doc.json", None, None),
                 "x10": (P + "wildlife_wildlife_x10_a_doc.json", None, None),
                 "x16": (P + "wildlife_wildlife_x16_a_doc.json", None, None),
                 "v8h": (P + "wildlife_wildlife_v8h_v8h_loo_doc.json", 0.697, None)},
    "hospital_queue": {"p3": (P + "hospital_queue_p3_doc.json", 0.690, "submissions/20260926-0135-u010b/hospital_queue/model.json"),
                       "v9g": (P + "hospital_queue_hospital_queue_v9g_v9gs15_doc.json", 0.672, None),
                       "v9d": (P + "hospital_queue_hospital_queue_v9d_v9ds15_doc.json", None, None),
                       "v8f": (P + "hospital_queue_hospital_queue_v8f_best_doc.json", 0.679, None),
                       "min2_v4": (P + "hospital_queue_hospital_queue_min2_v4_doc.json", None, None)},
}
PUBLIC_MEMBER = {"market": "x4p", "social_contagion": "x7", "epidemic": "x11_AC", "wildlife": "x13", "hospital_queue": "v9g"}
TAU_W = 0.02          # prior weight exp((public - best public) / TAU_W); unscored members exp(-0.5)
UNSCORED_W = float(np.exp(-0.5))
# hospital: p3 has the best sustained band (0.677) and best overall (0.690): it anchors the prior


def hold(v, n):
    return np.tile(np.asarray(v, float), (int(n), 1))


def candidates(sysname, spec, rng):
    """name -> (U, y0 mode, note). y0 mode 'runs' cycles the ledger initials; 'high' = mean initial x 1.2."""
    rec = np.array([spec.recovery[c] for c in spec.controls], float)
    pul = np.array([spec.pulse[c] for c in spec.controls], float)
    exam = D.eval_like(spec, "mixed", 400, np.random.default_rng(2027))
    c = {}
    if sysname == "social_contagion":        # seeding, incentive, bridge
        c["S1 zero hold 330 (proposed)"] = (hold([0, 0, 0], 330), "runs")
        c["S1b zero hold 200"] = (hold([0, 0, 0], 200), "runs")
        c["S2 seeding (4,0,0) 150"] = (hold([4, 0, 0], 150), "runs")
        c["S3 seeding (4,0,0) 150 -> +incentive (4,1.5,0) 150"] = (np.vstack([hold([4, 0, 0], 150), hold([4, 1.5, 0], 150)]), "runs")
        c["S4 bridge (5,1,0) 100 -> (5,1,0.8) 100"] = (np.vstack([hold([5, 1, 0], 100), hold([5, 1, 0.8], 100)]), "runs")
        c["S5 zero hold 330 from high start"] = (hold([0, 0, 0], 330), "high")
        c["S6 exam mixed 330"] = (D.eval_like(spec, "mixed", 330, np.random.default_rng(2027)), "runs")
    elif sysname == "market":                # rate, tax
        c["M1 zero hold 350 high start (proposed)"] = (hold([0, 0], 350), "high")
        c["M1b zero hold 350 typical start"] = (hold([0, 0], 350), "runs")
        c["M1c zero hold 200 high start"] = (hold([0, 0], 200), "high")
        c["M2 mid-rate (0.035,0) 200"] = (hold([0.035, 0], 200), "runs")
        c["M3 tax step (0,0.043) 100 -> (0,0.045) 100"] = (np.vstack([hold([0, 0.043], 100), hold([0, 0.045], 100)]), "runs")
        c["M4 frozen hold (0,0.05) 300"] = (hold([0, 0.05], 300), "runs")
        c["M5 exam mixed 400"] = (exam, "runs")
    elif sysname == "hospital_queue":        # staffing, elective, diag, urgent, overtime, followup
        c["H1 exam mixed 400 (proposed)"] = (exam, "runs")
        st = rec.copy(); st[0] = 8.0
        c["H2 staffing step 20->8, no overtime, 150"] = (hold(st, 150), "runs")
        c["H3 pulse 40 + recovery 260"] = (np.vstack([hold(pul, 40), hold(rec, 260)]), "runs")
        ot = st.copy(); ot[4] = 1.0
        c["H4 overtime hold (staffing 8, overtime 1) 300"] = (hold(ot, 300), "runs")
        c["H5 exam mixed 250"] = (D.eval_like(spec, "mixed", 250, np.random.default_rng(2027)), "runs")
    elif sysname == "epidemic":              # closure, mask, vaccination
        c["E1 mask-only (0,0.9,0) hold 350 (proposed)"] = (hold([0, 0.9, 0], 350), "runs")
        c["E1b mask-only (0,0.9,0) hold 200"] = (hold([0, 0.9, 0], 200), "runs")
        c["E2 pulse-release (0.75,0.75,.0022) 60 -> free 200"] = (np.vstack([hold([0.75, 0.75, 0.0022], 60), hold([0, 0, 0], 200)]), "runs")
        c["E3 mask 1 100 -> vaccination .003 200"] = (np.vstack([hold([0, 1, 0], 100), hold([0, 0, 0.003], 200)]), "runs")
        c["E4 exam mixed 400"] = (exam, "runs")
    elif sysname == "wildlife":              # hunting, habitat, corridor
        c["W1 hold (4,1,0) 300 (proposed)"] = (hold([4, 1, 0], 300), "runs")
        c["W1b hold (4,1,0) 200"] = (hold([4, 1, 0], 200), "runs")
        c["W2 habitat 0.05 100 -> +hunting 4 100"] = (np.vstack([hold([0, 0.05, 0], 100), hold([4, 0.05, 0], 100)]), "runs")
        c["W3 repeated pulse (30 on / 70 rec) x2"] = (np.vstack([hold(pul, 30), hold(rec, 70), hold(pul, 30), hold(rec, 70)]), "runs")
        c["W4 habitat 0.5 no corridor (0,0.5,0) 200"] = (hold([0, 0.5, 0], 200), "runs")
        c["W5 exam mixed 400"] = (exam, "runs")
    # the test's dominant uncertain regime: every control away from recovery, held 150+ ticks (uniform / mid levels)
    tag = sysname[0].upper()
    c[f"{tag}J1 joint mid-level hold 300"] = (hold(D.level(spec, "mid", rng), 300), "runs")
    c[f"{tag}J2 joint uniform-level hold 300"] = (hold(D.level(spec, "uniform", np.random.default_rng(31)), 300), "runs")
    c[f"{tag}J3 joint mid-level hold 200"] = (hold(D.level(spec, "mid", rng), 200), "runs")
    return {k: (D.clip(spec, U), m) for k, (U, m) in c.items()}


def validate(name, path, ref, runs, spec):
    doc = json.loads(Path(path).read_text())
    lab_p = Path(path.replace("_doc.json", ".json"))
    if lab_p.exists() and ref is None:
        lab = json.loads(lab_p.read_text())
        sig = np.asarray(lab["sigma_proxy"], float)
        devs = {}
        for r in runs:
            want = lab["insample"].get(r.exp)
            if want is None:
                continue
            s_clip = metric.score_per_obs(rollout_from_blob(doc["model"], r.y0, r.U, doc=doc), r.Y, sig)
            d = float(np.max(np.abs(s_clip - np.asarray(want))))
            if d >= 0.01 and doc["model"].get("family"):
                blob = doc["model"]
                _, fam = ode_modules(blob["family"])
                th = core.theta_dict(fam, [float(v) for v in blob["theta"]])
                Y = core.rollout(fam, r.y0, r.U, th, frozenset(blob["mech"]), n_sub=int(blob.get("n_sub", 4)))
                d = min(d, float(np.max(np.abs(metric.score_per_obs(Y, r.Y, sig) - np.asarray(want)))))
            devs[r.exp] = d
        ok = bool(devs) and max(devs.values()) < 0.01
        return doc, {"ref": str(lab_p), "runs_checked": len(devs), "max_dev": max(devs.values()) if devs else None, "ok": ok}
    ref_doc = json.loads(Path(ref).read_text())
    dev = 0.0
    for r in runs:
        a = rollout_from_blob(doc["model"], r.y0, r.U, doc=doc)
        b = rollout_from_blob(ref_doc["model"] if "model" in ref_doc else ref_doc, r.y0, r.U, doc=ref_doc)
        dev = max(dev, float(np.max(np.abs(a - b))))
    return doc, {"ref": ref, "runs_checked": len(runs), "max_abs_dy_vs_shipped": dev, "ok": dev < 1e-6}


def regime(U, rec):
    on = np.abs(U - rec[None]) > 1e-9
    age = np.zeros(len(U), int)
    for t in range(1, len(U)):
        age[t] = 0 if np.any(U[t] != U[t - 1]) else age[t - 1] + 1
    ab = np.where(age < 50, "0-50", np.where(age < 150, "50-150", "150+"))
    tag = np.array(["".join("1" if x else "0" for x in row) for row in on])
    return np.char.add(np.char.add(tag, "|"), ab)


def run_system(sysname, a):
    t0 = time.time()
    spec = S.get(sysname)
    runs = load_runs(spec, Ledger(data_dir(sysname, False), sysname, False))
    cal = json.loads(Path("plans/sigma_calibrated.json").read_text())[sysname]
    sigma = np.asarray(cal["sigma"], float)
    noise = np.asarray(cal["noise"], float)
    out = {"system": sysname, "sigma_org": sigma.tolist(), "runs": [(r.exp, r.T) for r in runs], "balance": BAL[sysname]}
    members, val = {}, {}
    for name, (path, pub, ref) in COMMITTEE[sysname].items():
        doc, v = validate(name, path, ref, runs, spec)
        v["insample_org"] = float(np.mean([metric.score_per_obs(rollout_from_blob(doc["model"], r.y0, r.U, doc=doc), r.Y, sigma)
                                           for r in runs]))
        v["public"] = pub
        val[name] = v
        print(f"[{sysname}] validate {name:<8} ok={v['ok']} {v} ", flush=True)
        if v["ok"]:
            members[name] = doc
    out["validation"] = val
    names = list(members)
    H = len(names)
    best_pub = max(val[n]["public"] for n in names if val[n]["public"] is not None)
    w = np.array([np.exp((val[n]["public"] - best_pub) / TAU_W) if val[n]["public"] is not None else UNSCORED_W for n in names])
    w = w / w.sum()
    out["members"], out["prior_w"] = names, w.tolist()
    roll = lambda n, y0, U: rollout_from_blob(members[n]["model"], y0, U, doc=members[n])

    # ---- test distribution
    y0s = [r.y0 for r in runs]
    rec = np.array([spec.recovery[c] for c in spec.controls], float)
    tests = []
    for ci, cat in enumerate(CATS):
        for k in range(a.n_per):
            seed = ci * a.n_per + k
            tests.append({"cat": cat, "U": D.eval_like(spec, cat, 4000, np.random.default_rng(1000 + seed)), "y0": y0s[seed % len(y0s)]})
    Yt = np.stack([np.stack([roll(n, t["y0"], t["U"]) for t in tests]) for n in names])       # [H, N, T, p]
    Ymed = np.median(Yt, axis=0)
    # score matrix per category (mean over tests of that cat, mean over obs)
    def smat(Ypred):   # Ypred [N, T, p] -> [4, H]
        sc = 1.0 / (1.0 + np.abs(Ypred[None] - Yt) / sigma)      # [H, N, T, p]
        per_test = sc.mean(axis=(2, 3))                             # [H, N]
        return np.array([[per_test[j, [i for i, t in enumerate(tests) if t["cat"] == c]].mean() for j in range(H)] for c in CATS])
    Scat = np.stack([smat(Yt[i]) for i in range(H)])               # [H(pred), 4, H(truth)]
    Smed = smat(Ymed)                                              # [4, H]
    Spred = np.concatenate([Scat.mean(1), Smed.mean(0)[None]])     # [H+1, H]
    pred_names = names + ["median"]
    out["score_matrix"] = {pn: dict(zip(names, Spred[i].tolist())) for i, pn in enumerate(pred_names)}
    exp_now = Spred @ w
    pub_i = names.index(PUBLIC_MEMBER[sysname])
    out["prior"] = {"expected_by_predictor": dict(zip(pred_names, exp_now.tolist())), "best": pred_names[int(np.argmax(exp_now))],
                    "best_expected": float(exp_now.max()), "public_expected": float(exp_now[pub_i]),
                    "evpi_member": float(w @ np.array([Spred[:H].max(0)[j] for j in range(H)]) - exp_now.max())}
    # disagreement by category / regime (pairwise mean |dy|/sigma)
    pairs = list(itertools.combinations(range(H), 2))
    dcat = {}
    for ci, c in enumerate(CATS):
        idx = [i for i, t in enumerate(tests) if t["cat"] == c]
        d = np.mean([np.mean(np.abs(Yt[i][idx] - Yt[j][idx]) / sigma, axis=(0, 1)) for i, j in pairs], axis=0)
        # score at risk for the public member: 1 - mean_j w_j S[pub, j] within category
        dcat[c] = {"pairwise_sigma_by_obs": d.tolist(), "mean": float(d.mean()),
                   "public_expected": float(Scat[pub_i, ci] @ w)}
    out["disagreement_by_category"] = dcat
    labs = np.concatenate([regime(t["U"], rec) for t in tests])
    dall = np.mean([np.mean(np.abs(Yt[i] - Yt[j]) / sigma, axis=2).reshape(-1) for i, j in pairs], axis=0)
    reg = {}
    for g in np.unique(labs):
        m = labs == g
        reg[str(g)] = {"share": float(m.mean()), "d": float(dall[m].mean()), "contribution": float(dall[m].sum() / dall.sum())}
    reg_all = reg
    out["disagreement_by_regime"] = dict(sorted(reg.items(), key=lambda kv: -kv[1]["contribution"])[:10])
    out["regime_key"] = "controls away from recovery (" + ",".join(spec.controls) + ") | hold age"
    print(f"[{sysname}] tests rolled {time.time() - t0:.0f}s; prior best {out['prior']['best']} "
          f"{exp_now.max():.4f}; public {exp_now[pub_i]:.4f}", flush=True)

    # ---- likelihood sd: noise + median member misfit on the ledger runs
    mis = np.median([np.sqrt(np.mean(np.concatenate([roll(n, r.y0, r.U) - r.Y for r in runs]) ** 2, 0)) for n in names], axis=0)
    sd = np.sqrt(noise ** 2 + mis ** 2)
    out["likelihood_sd"] = {"noise": noise.tolist(), "misfit": mis.tolist(), "sd": sd.tolist(), "thin": a.thin}

    # ---- candidates: simulate every member on every candidate from the start set
    cands = candidates(sysname, spec, np.random.default_rng(5))
    ymean = np.mean(y0s, 0)
    cstore, cinfo = {}, {}
    for cn, (U, mode) in cands.items():
        starts = [ymean * 1.2] if mode == "high" else y0s
        Ys = np.stack([np.stack([roll(n, y0, U) for y0 in starts]) for n in names])[:, :, ::a.thin, :]   # [H, S, T', p]
        cstore[cn] = Ys
        spread = float(np.mean([np.mean(np.abs(Ys[i] - Ys[j]) / sigma) for i, j in pairs]))
        dt = np.mean([np.mean(np.abs(Ys[i] - Ys[j]) / sigma, axis=(0, 2)) for i, j in pairs], axis=0)   # [T']
        rl = regime(U, rec)[::a.thin]
        wt = np.array([reg_all.get(str(g), {"share": 0.0, "d": 0.0})["share"] * reg_all.get(str(g), {"d": 0.0})["d"] for g in rl])
        v_reg = float(np.sum(dt * wt) * a.thin / (dall.mean() + 1e-12))
        cinfo[cn] = {"credits": int(len(U)), "start": mode, "spread_sigma": spread, "regime_value": v_reg,
                     "regime_value_per_100": 100 * v_reg / len(U), "U": U.tolist()}
    np.savez_compressed(SCR / f"{sysname}_cands.npz", **{f"c{i}": cstore[k] for i, k in enumerate(cands)})
    out["cand_names"] = list(cands)
    out["Spred"] = Spred.tolist()
    out["candidates"] = cinfo
    # single-run values
    rng = np.random.default_rng(11)
    for cn in cands:
        v, post = voi_set(Spred, w, sd, [cstore[cn]], a.mc, rng, off=mis)
        cinfo[cn].update({"value": v, "value_per_100": 100 * v / cinfo[cn]["credits"], "post_on_truth": post})
        print(f"[{sysname}]  {cn:<55} T={cinfo[cn]['credits']:3d} value {v:+.4f} per100 {100 * v / cinfo[cn]['credits']:+.4f} "
              f"P(truth) {post:.2f} spread {cinfo[cn]['spread_sigma']:.2f} regime/100 {cinfo[cn]['regime_value_per_100']:.3f}", flush=True)
    out["seconds"] = time.time() - t0
    (SCR / f"{sysname}.json").write_text(json.dumps(out))
    print(f"[{sysname}] done {out['seconds']:.0f}s", flush=True)


def voi_set(Spred, w, sd, Ylist, mc, rng, off=0.0):
    """Expected test score after observing every run in Ylist (product likelihood), minus the prior best.
    Truth k ~ w; each run's start drawn from its start set; noise N(0, sd) on the thinned ticks."""
    H = Spred.shape[1]
    base = float((Spred @ w).max())
    vals, post = [], []
    for _ in range(mc):
        k = rng.choice(H, p=w)
        ll = np.log(np.maximum(w, 1e-300))
        for Ys in Ylist:
            q = rng.integers(Ys.shape[1])
            # truth is not a member: the run carries a per-run level offset (misfit scale) on top of white noise
            y = Ys[k, q] + rng.normal(0, 1, Ys[k, q].shape[1:]) * off + rng.normal(0, 1, Ys[k, q].shape) * sd
            ll = ll - 0.5 * np.sum(((Ys[:, q] - y[None]) / sd) ** 2, axis=(1, 2))
        wp = np.exp(ll - ll.max()); wp /= wp.sum()
        vals.append(Spred[int(np.argmax(Spred @ wp)), k])
        post.append(wp[k])
    return float(np.mean(vals) - base), float(np.mean(post))


def portfolio(a):
    systems = list(BAL)
    R = {s: json.loads((SCR / f"{s}.json").read_text()) for s in systems}
    Ys = {s: dict(zip(R[s]["cand_names"], [np.load(SCR / f"{s}_cands.npz")[f"c{i}"] for i in range(len(R[s]["cand_names"]))]))
          for s in systems}
    cache = {}

    def val(s, subset):
        key = (s, tuple(sorted(subset)))
        if key not in cache:
            if not subset:
                cache[key] = 0.0
            else:
                rng = np.random.default_rng(1234)     # common random numbers across subsets
                cache[key], _ = voi_set(np.asarray(R[s]["Spred"]), np.asarray(R[s]["prior_w"]),
                                        np.asarray(R[s]["likelihood_sd"]["sd"]), [Ys[s][c] for c in key[1]], a.mc, rng,
                                        off=np.asarray(R[s]["likelihood_sd"]["misfit"]))
        return cache[key]

    def cost(s, c):
        return R[s]["candidates"][c]["credits"]

    out = {"reserve": RESERVE, "balances": BAL, "systems": {}, "budgets": {}}
    for s in systems:
        out["systems"][s] = {k: R[s][k] for k in ("validation", "members", "prior_w", "prior", "disagreement_by_category",
                                                    "disagreement_by_regime", "regime_key", "likelihood_sd", "sigma_org", "runs")}
        out["systems"][s]["candidates"] = {c: {k: v for k, v in d.items() if k != "U"} for c, d in R[s]["candidates"].items()}
    # variants of the same candidate family (e.g. 200 vs 350 of the same hold) are exclusive
    fam = lambda c: c.split()[0].rstrip("bc").replace("J3", "J1")
    # proposed five
    proposed = {"social_contagion": "S1 zero hold 330 (proposed)", "market": "M1 zero hold 350 high start (proposed)",
                "hospital_queue": "H1 exam mixed 400 (proposed)", "wildlife": "W1 hold (4,1,0) 300 (proposed)",
                "epidemic": "E1 mask-only (0,0.9,0) hold 350 (proposed)"}
    pv = {s: val(s, [c]) for s, c in proposed.items()}
    out["proposed"] = {"runs": proposed, "credits": sum(cost(s, c) for s, c in proposed.items()),
                       "value_by_system": pv, "mean_score_gain": sum(pv.values()) / 10}
    # exact portfolio: per system every feasible subset (<= 3 runs, one per candidate family, within
    # balance - reserve), then a knapsack over systems in 10-credit steps
    opts = {}
    for s in systems:
        cap = BAL[s] - RESERVE
        cs = R[s]["cand_names"]
        lst = [((), 0, 0.0)]
        for k in (1, 2, 3):
            for sub in itertools.combinations(cs, k):
                if len({fam(c) for c in sub}) < k:
                    continue
                cc = sum(cost(s, c) for c in sub)
                if cc <= cap:
                    lst.append((sub, cc, val(s, list(sub))))
        opts[s] = lst
        print(f"{s}: {len(lst)} feasible sets; best single {max(lst, key=lambda x: x[2] if len(x[0]) == 1 else -9)[:3]}")
    for B in (1000, 1730, 1750, 2500):
        nb = B // 10
        best = {0: (0.0, {})}                      # credits/10 -> (value, choice)
        for s in systems:
            nxt = {}
            for used, (v0, ch) in best.items():
                for sub, cc, v in opts[s]:
                    u = used + int(np.ceil(cc / 10))
                    if u > nb:
                        continue
                    if u not in nxt or v0 + v > nxt[u][0]:
                        nxt[u] = (v0 + v, dict(ch, **{s: sub}))
            best = nxt
        # smallest spend within 0.002 of the best value (spending more for MC noise is holding in disguise)
        vmax = max(v for v, _ in best.values())
        u_pick = min(u for u, (v, _) in best.items() if v >= vmax - 0.002)
        v, ch = best[u_pick]
        spent = sum(cost(s, c) for s in ch for c in ch[s])
        tot = {s: val(s, list(ch.get(s, ()))) for s in systems}
        out["budgets"][str(B)] = {"credits_used": spent, "picks": {s: list(ch.get(s, ())) for s in systems},
                                  "credits_by_system": {s: sum(cost(s, c) for c in ch.get(s, ())) for s in systems},
                                  "value_by_system": tot, "value_committee_sum": float(sum(tot.values())),
                                  "value_at_full_budget": float(vmax),
                                  "mean_score_gain_committee": sum(tot.values()) / 10,
                                  "schedules": {f"{s}: {c}": R[s]["candidates"][c]["U"] for s in systems for c in ch.get(s, ())}}
        print(f"budget {B}: used {spent}, committee value {sum(tot.values()):.4f} (full-budget max {vmax:.4f}), mean gain {sum(tot.values()) / 10:+.4f}")
        for s in systems:
            print(f"   {s:<17} {tot[s]:+.4f} {list(ch.get(s, ()))}")
    # same credits as the proposed five, optimal
    # recommendation: the 1,000 portfolio with epidemic's 300-tick E3 in place of E1b (+100 credits, +0.038)
    rec_runs = {"social_contagion": ["SJ2 joint uniform-level hold 300"], "market": ["MJ3 joint mid-level hold 200"],
                "hospital_queue": ["H3 pulse 40 + recovery 260"], "epidemic": ["E3 mask 1 100 -> vaccination .003 200"],
                "wildlife": []}
    rv = {s: val(s, rec_runs[s]) for s in systems}
    out["recommended"] = {"runs": rec_runs, "credits": sum(cost(s, c) for s in systems for c in rec_runs[s]),
                          "value_by_system": rv, "mean_score_gain_committee": sum(rv.values()) / 10,
                          "transfer_factor": 0.25, "mean_score_gain_expected": 0.25 * sum(rv.values()) / 10,
                          "balances_after": {s: BAL[s] - sum(cost(s, c) for c in rec_runs[s]) for s in systems},
                          "schedules": {s: {c: R[s]["candidates"][c]["U"] for c in rec_runs[s]} for s in systems},
                          "starts": {s: {c: R[s]["candidates"][c]["start"] for c in rec_runs[s]} for s in systems}}
    out["proposed"]["mean_score_gain_expected"] = 0.25 * out["proposed"]["mean_score_gain"]
    print("recommended:", out["recommended"]["credits"], "credits", {k: round(v, 4) for k, v in rv.items()},
          "committee mean gain", round(sum(rv.values()) / 10, 4))
    # all-candidate U for materializing
    out["schedules_all"] = {s: {c: R[s]["candidates"][c]["U"] for c in R[s]["cand_names"]} for s in systems}
    print("proposed five:", out["proposed"]["credits"], "credits, mean gain", round(out["proposed"]["mean_score_gain"], 4),
          {k: round(v, 4) for k, v in pv.items()})
    Path("plans/voi_round2.json").write_text(json.dumps(out, indent=1))
    print("wrote plans/voi_round2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--portfolio", action="store_true")
    ap.add_argument("--n-per", type=int, default=10)
    ap.add_argument("--mc", type=int, default=800)
    ap.add_argument("--thin", type=int, default=40)
    ap.add_argument("--min-per-credit", type=float, default=2e-6)
    a = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    if a.portfolio:
        portfolio(a)
    else:
        run_system(a.system, a)


if __name__ == "__main__":
    main()
