"""Per-observable model zoo selection on held-out evidence (no credits).

    python scripts/zoo_select.py validate            # docs reproduce their lab in-sample; shipped u013 reproduces
    python scripts/zoo_select.py loo [--systems a,b] [--workers 14] [--budget 60] [--nfev 30]
    python scripts/zoo_select.py select              # per-observable choice -> plans/zoo_select.json
    python scripts/zoo_select.py assemble            # plans/zoo_<system>_doc.json + checks
    python scripts/zoo_select.py report              # plans/zoo_select.md

Held-out protocol (same for every member, so the table is comparable across families):
  sigma = calibrated organizer sigma (plans/sigma_calibrated.json, 1.0x);
  fold k = leave run k out; ODE members refit with fit_ode (cauchy, ONE start at the member's
  full-data theta, early_T None, max_nfev --nfev, time budget --budget); l0b_lin refit per fold
  (clip margin 1); every member prediction finalized with the current public doc's clip vectors;
  combinations: member alone, or mean of a pair (= per-tick median of two); post rules of the
  public doc applied after combining (as the runtime does).
Acceptance per observable: best option beats the public option by > 0.01 (mean over folds)
AND wins on a strict majority of folds. Forecast public gain = 0.6 x held-out gain.
Predictions are cached in the scratch dir (ZOO_SCRATCH env or the default below).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import importlib
import importlib.util
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, metric, select as SEL, systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402
from gtlab.runtime import infer as rt  # noqa: E402

SCRATCH = Path(os.environ.get("ZOO_SCRATCH", r"C:\Users\DYLANH~1\AppData\Local\Temp\claude\scratch\zoo"))
PUB_DIR = ROOT / "submissions" / "20260927-0233-u013"
PLANS = ROOT / "plans"
SYSTEMS = ["ad_auction", "epidemic", "hospital_queue", "market", "power_grid", "reservoir",
           "social_contagion", "supply_chain", "traffic", "wildlife"]
EXAM = ("p5.testlike", "p6.exam")          # test-shaped runs, reported separately
NOFIT = {"p6.exam"}                         # never used for any fit: exam only
MARGIN, SHRINK = 0.01, 0.6

# shortlist: label -> doc file under plans/ ("PUB" = ODE member of the shipped u013 doc)
SHORT = {
    "ad_auction": {"pub_v8b": "PUB", "v8c": "ad_auction_ad_auction_v8c_loo_doc.json",
                   "min": "ad_auction_ad_auction_min_v8ref_doc.json", "s1": "ad_auction_ad_auction_s1_cal4_doc.json"},
    "epidemic": {"pub_x11AC": "PUB", "x7AC": "epidemic_epidemic_x7_AC_doc.json", "x10AC": "epidemic_epidemic_x10_AC_doc.json",
                 "x8AC": "epidemic_epidemic_x8_AC_doc.json"},
    "hospital_queue": {"pub_v9g": "PUB", "v9w": "hospital_queue_hospital_queue_v9_v9w_doc.json",
                       "v8b": "hospital_queue_hospital_queue_v8b_loo_doc.json", "min": "hospital_queue_hospital_queue_min_p3_doc.json"},
    "market": {"pub_x4": "PUB", "x3": "market_market_x3_x7_doc.json", "v9g": "market_market_v9g_x7ref_doc.json",
               "v8t2": "market_market_v8t2_v8_doc.json"},
    "power_grid": {"pub_v9c": "PUB", "v9": "power_grid_power_grid_v9_v9_doc.json",
                   "v8k": "power_grid_power_grid_v8k_v8k_lab_doc.json", "min": "power_grid_power_grid_min_v8ref_doc.json"},
    "reservoir": {"pub_v8": "PUB", "min2": "reservoir_reservoir_min2_doc.json", "min": "reservoir_reservoir_min_doc.json",
                  "s1": "reservoir_reservoir_s1_cal4_doc.json"},
    "social_contagion": {"pub_x7": "PUB", "x2": "social_contagion_social_contagion_x2_xr5_doc.json",
                         "x8": "social_contagion_social_contagion_x8_xr5_doc.json", "v8o": "social_contagion_social_contagion_v8o_xr5_doc.json"},
    "supply_chain": {"pub_v8b": "PUB", "v8c": "supply_chain_supply_chain_v8c_a_doc.json",
                     "min": "supply_chain_supply_chain_min_v7b_doc.json", "min2": "supply_chain_supply_chain_min2_v7b_doc.json"},
    "traffic": {"pub_v8d": "PUB", "v8c": "traffic_traffic_v8c_v8c_doc.json",
                "min": "traffic_traffic_min_p3_doc.json", "s1": "traffic_traffic_s1_cal4_doc.json"},
    "wildlife": {"pub_x13": "PUB", "x6": "wildlife_wildlife_x6_a_doc.json", "x5": "wildlife_wildlife_x5_b_doc.json",
                 "v8i": "wildlife_wildlife_v8i_v8i_loo_doc.json"},
}


# ----------------------------------------------------------------------------- helpers
def pub_doc(system):
    return json.loads((PUB_DIR / system / "model.json").read_text())


def _find_ode(blob):
    if blob["kind"] == "ode":
        return blob
    for m in blob.get("members", []) + ([blob["member"]] if "member" in blob else []):
        b = _find_ode(m)
        if b is not None:
            return b
    return None


def pub_option(system, labels):
    """Public model per observable as an option tuple over member labels."""
    p = len(S.get(system).observables)
    pub = [l for l in labels if l.startswith("pub_")][0]
    if system == "market":                      # price = mean(x4, l0b_lin); volume, depth = x4
        return [(pub, "l0b"), (pub,), (pub,)]
    return [(pub,)] * p


def member_blob(system, label):
    src = SHORT[system][label]
    if src == "PUB":
        return copy.deepcopy(_find_ode(pub_doc(system)["model"]))
    return json.loads((PLANS / src).read_text())["model"]


def lab_path(system, label):
    src = SHORT[system][label]
    return None if src == "PUB" else PLANS / src.replace("_doc.json", ".json")


def get_runs(system):
    spec = S.get(system)
    return spec, load_runs(spec, Ledger(data_dir(system, False), system, False))


def cal_sigma(system):
    return np.asarray(json.loads((PLANS / "sigma_calibrated.json").read_text())[system]["sigma"], float)


def ode_roll(blob, y0, U):
    mod = importlib.import_module(f"gtlab.ode.{blob['family']}")
    return core.rollout(mod, np.asarray(y0, float), U, core.theta_dict(mod, blob["theta"]), frozenset(blob["mech"]),
                        n_sub=int(blob.get("n_sub", 4)))


def split_runs(runs):
    """(fit runs, exam runs): exam runs (NOFIT) are never used for fitting, only scored."""
    return [r for r in runs if r.exp not in NOFIT], [r for r in runs if r.exp in NOFIT]


def needs_full(system, label, runs):
    """True if the member's doc was not fitted on exactly the current fit runs."""
    lp = lab_path(system, label)
    if lp is None:
        return False
    lab = json.loads(lp.read_text())
    return sorted(e for e, _ in lab["runs"]) != sorted(r.exp for r in split_runs(runs)[0])


# ----------------------------------------------------------------------------- validate
def cmd_validate(a):
    out = {}
    for s in SYSTEMS:
        spec, runs = get_runs(s)
        byexp = {r.exp: r for r in runs}
        out[s] = {}
        for label, src in SHORT[s].items():
            blob = member_blob(s, label)
            if src == "PUB":
                # shipped predict.py (inlined modules) vs dev rollout of the same doc
                spec_ = importlib.util.spec_from_file_location(f"pub_{s}", PUB_DIR / s / "predict.py")
                mod = importlib.util.module_from_spec(spec_)
                cwd = os.getcwd()
                os.chdir(PUB_DIR / s)
                try:
                    spec_.loader.exec_module(mod)
                    r = runs[-1]
                    init = dict(zip(spec.observables, map(float, r.y0)))
                    iv = [dict(zip(spec.controls, map(float, u))) for u in r.U]
                    shipped = np.array([[d[o] for o in spec.observables] for d in mod.predict(init, iv, {"family": s})])
                finally:
                    os.chdir(cwd)
                d = pub_doc(s)
                dev = rt.rollout_from_blob(d, r.y0, r.U, doc=d)
                err = float(np.max(np.abs(shipped - dev) / cal_sigma(s)))
                out[s][label] = {"check": "shipped predict.py vs dev rollout", "max_err_sigma": err, "ok": err < 1e-6}
            else:
                lab = json.loads(lab_path(s, label).read_text())
                sig = np.asarray(lab["sigma_proxy"], float)
                diffs = []
                ok = True
                for e, T in lab["runs"]:
                    if e not in byexp or byexp[e].T != T:
                        ok = False
                        continue
                    r = byexp[e]
                    sc = metric.score_per_obs(ode_roll(blob, r.y0, r.U), r.Y, sig)
                    diffs.append(float(np.max(np.abs(sc - np.asarray(lab["insample"][e])))))
                theta_same = bool(np.allclose(blob["theta"], lab["theta_vec"]))
                mx = max(diffs) if diffs else float("nan")
                out[s][label] = {"check": "doc reproduces lab in-sample", "max_abs_diff": mx, "theta_same": bool(theta_same),
                                 "lab_runs": len(lab["runs"]), "cur_runs": len(runs), "ok": bool(ok and mx < 2e-3)}
            print(f"{s:<17} {label:<10} {out[s][label]}", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "validate.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- loo
def _job(system, label, held, budget, nfev):
    cold = label.endswith("@cold")                     # cold start: PARAMS init + 4 LHS starts (lab protocol)
    label = label.replace("@cold", "")
    """held = exp name of the held-out fit run, or "FULL" (fit on every fit run, predict the exam runs).
    Exam runs (NOFIT) are never in a training set."""
    spec, runs = get_runs(system)
    sigma = cal_sigma(system)
    fit_runs, exams = split_runs(runs)
    tr = [r for r in fit_runs if r.exp != held]
    targets = exams if held == "FULL" else [r for r in fit_runs if r.exp == held]
    t0 = time.time()
    pay = {"train": [r.exp for r in tr]}
    with C.single_thread():
        if label == "l0b":
            clip = C.soft_clip(spec, tr, margin=1.0)
            m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(tr)
            if held == "FULL":
                pay["blob"] = m.export()
            pay["Y"] = {r.exp: m.rollout(r.y0, r.U).tolist() for r in targets}
        else:
            blob = member_blob(system, label)
            mod = importlib.import_module(f"gtlab.ode.{blob['family']}")
            mech = frozenset(blob["mech"])
            n_sub = int(blob.get("n_sub", getattr(mod, "N_SUB", 2)))
            if cold:
                th, info = F.fit_ode(mod, tr, sigma, mech, n_starts=4, max_nfev=40, n_sub=n_sub, early_T=None,
                                     time_budget=budget, theta0=None, spread=0.5)
                pay["refit"] = True
                pay["cost"] = float(info["cost"])
            elif held == "FULL" and not needs_full(system, label, runs):
                th = np.asarray(blob["theta"], float)          # shipped / doc theta, already fit on exactly these runs
                pay["refit"] = False
            else:
                th, info = F.fit_ode(mod, tr, sigma, mech, n_starts=1, max_nfev=nfev, n_sub=n_sub, early_T=None,
                                     time_budget=budget, theta0=blob["theta"], spread=0.0)
                pay["refit"] = True
                pay["cost"] = float(info["cost"])
            pay["theta"] = [float(v) for v in th]
            pay["Y"] = {r.exp: np.asarray(core.rollout(mod, r.y0, r.U, core.theta_dict(mod, th), mech, n_sub=n_sub)).tolist()
                        for r in targets}
    pay["sec"] = time.time() - t0
    return (system, label, held), pay


def cmd_loo(a):
    systems = a.systems.split(",") if a.systems else SYSTEMS
    jobs = []
    for s in systems:
        _, runs = get_runs(s)
        fit_runs, _ = split_runs(runs)
        (SCRATCH / s).mkdir(parents=True, exist_ok=True)
        labels = a.labels.split(",") if a.labels else list(SHORT[s]) + ["l0b"]
        for label in labels:
            for held in [r.exp for r in fit_runs] + ([] if "@cold" in label else ["FULL"]):
                f = SCRATCH / s / f"{label}__{held}.json"
                if not f.exists() or a.force:
                    jobs.append((s, label, held))
    jobs.sort(key=lambda j: j[2] != "FULL")
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
    t0 = time.time()
    def done(n, s, l, h, pay):
        (SCRATCH / s / f"{l}__{h}.json").write_text(json.dumps(pay))
        print(f"[{n + 1}/{len(jobs)} {time.time() - t0:.0f}s] {s} {l} held {h} "
              f"{'ERR ' + pay['error'] if 'error' in pay else '%.0fs' % pay['sec']}", flush=True)
    if a.workers <= 1:                                 # one process, serial (shared machine)
        for n, (s, l, h) in enumerate(jobs):
            try:
                _, pay = _job(s, l, h, a.budget, a.nfev)
            except Exception as e:                     # record failures, keep going
                pay = {"error": repr(e)}
            done(n, s, l, h, pay)
        return
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_job, s, l, h, a.budget, a.nfev): (s, l, h) for s, l, h in jobs}
        for n, fu in enumerate(as_completed(futs)):
            s, l, h = futs[fu]
            try:
                _, pay = fu.result()
            except Exception as e:
                pay = {"error": repr(e)}
            done(n, s, l, h, pay)


# ----------------------------------------------------------------------------- select
def _preds(system, runs, labels, doc):
    """{label: {exp: finalized held-out Y}}: every fit run from its LOO fold, every exam run from the FULL fit."""
    lo, hi = rt.clip_vectors(doc)
    fit_runs, _ = split_runs(runs)
    P = {}
    for l in labels:
        P[l] = {}
        for held in [r.exp for r in fit_runs] + ["FULL"]:
            pay = json.loads((SCRATCH / system / f"{l}__{held}.json").read_text())
            if "error" in pay:
                raise RuntimeError(f"{system} {l} held {held}: {pay['error']}")
            for e, Y in pay["Y"].items():
                r = next(x for x in runs if x.exp == e)
                P[l][e] = rt.finalize(np.asarray(Y, float), r.y0, lo, hi)
    return P


def options(labels):
    return [(l,) for l in labels] + list(itertools.combinations(labels, 2))


def opt_name(o):
    return o[0] if len(o) == 1 else f"mean({o[0]},{o[1]})"


def combo(P, o, e):
    return np.mean([P[l][e] for l in o], axis=0)


def cmd_select(a):
    res = {}
    for s in SYSTEMS:
        spec, runs = get_runs(s)
        fit_runs, exams = split_runs(runs)
        doc = pub_doc(s)
        sigma = cal_sigma(s)
        labels = list(SHORT[s]) + ["l0b"]
        P = _preds(s, runs, labels, doc)
        opts = options(labels)
        nf, p = len(fit_runs), len(spec.observables)

        def sc(o, r):
            return metric.score_per_obs(rt.apply_post(doc, combo(P, o, r.exp), r.U, r.y0), r.Y, sigma)
        Sc = {o: np.array([sc(o, r) for r in fit_runs]) for o in opts}              # [folds, p]
        Ex = {o: np.array([sc(o, r) for r in exams]).reshape(len(exams), p) for o in opts}
        base = pub_option(s, labels)
        chosen, rows = [], []
        for j, obs in enumerate(spec.observables):
            b = Sc[base[j]][:, j]
            best = max(opts, key=lambda o: Sc[o][:, j].mean())
            sb = Sc[best][:, j]
            wins = int(np.sum(sb > b + 1e-12))
            # clean evidence: exam runs + fit runs no member doc had seen when it was fitted (p6.*)
            clean_f = [k for k, r in enumerate(fit_runs) if r.exp.startswith("p6.")]
            cd = list(Ex[best][:, j] - Ex[base[j]][:, j]) + [Sc[best][k, j] - Sc[base[j]][k, j] for k in clean_f]
            exam_d = float(np.mean(cd)) if cd else None
            accept = (sb.mean() - b.mean() > MARGIN) and wins * 2 > nf
            veto = accept and exam_d is not None and exam_d < -MARGIN          # clean runs clearly disagree
            pick = best if (accept and not veto) else base[j]
            chosen.append(pick)
            rows.append({"obs": obs, "public": opt_name(base[j]), "public_score": float(b.mean()),
                         "best": opt_name(best), "best_score": float(sb.mean()), "wins": wins, "folds": nf,
                         "exam_delta": exam_d, "accept": bool(accept), "exam_veto": bool(veto), "pick": opt_name(pick),
                         "table": {opt_name(o): float(Sc[o][:, j].mean()) for o in opts},
                         "exam_table": {opt_name(o): float(Ex[o][:, j].mean()) for o in opts} if exams else None})
        pick_scores = np.array([Sc[chosen[j]][:, j] for j in range(p)]).T      # [folds, p]
        base_scores = np.array([Sc[base[j]][:, j] for j in range(p)]).T
        gain = float(pick_scores.mean() - base_scores.mean())
        special = {}
        for k, r in enumerate(fit_runs):
            if r.exp in EXAM or r.exp.startswith("p6."):          # test-shaped fold / run no member doc has seen
                special[r.exp] = {"kind": "LOO fold", "public": base_scores[k].tolist(), "pick": pick_scores[k].tolist(),
                                  "gain": float(pick_scores[k].mean() - base_scores[k].mean()),
                                  "per_member": {l: Sc[(l,)][k].tolist() for l in labels}}
        for i, r in enumerate(exams):
            pe = np.array([Ex[chosen[j]][i, j] for j in range(p)])
            be = np.array([Ex[base[j]][i, j] for j in range(p)])
            special[r.exp] = {"kind": "exam (never fitted; members fitted on all fit runs)", "public": be.tolist(),
                              "pick": pe.tolist(), "gain": float(pe.mean() - be.mean()),
                              "per_member": {l: Ex[(l,)][i].tolist() for l in labels},
                              "best_options": sorted(((opt_name(o), float(Ex[o][i].mean())) for o in opts),
                                                     key=lambda t: -t[1])[:5]}
        res[s] = {"runs": [r.exp for r in runs], "fit_runs": [r.exp for r in fit_runs], "exam_runs": [r.exp for r in exams],
                  "members": labels, "short": SHORT[s],
                  "per_obs": rows, "map": [opt_name(o) for o in chosen], "chosen": [list(o) for o in chosen],
                  "heldout_public": float(base_scores.mean()), "heldout_pick": float(pick_scores.mean()),
                  "heldout_gain": gain, "forecast_gain": SHRINK * gain, "changed": any(c != b for c, b in zip(chosen, base)),
                  "per_fold": {"public": base_scores.mean(1).tolist(), "pick": pick_scores.mean(1).tolist()},
                  "member_mean": {l: Sc[(l,)].mean(0).tolist() for l in labels}, "special": special}
        print(f"== {s}: public {base_scores.mean():.3f} pick {pick_scores.mean():.3f} gain {gain:+.4f} map {res[s]['map']}")
        for r in rows:
            ed = "" if r["exam_delta"] is None else f" examD {r['exam_delta']:+.3f}{' VETO' if r['exam_veto'] else ''}"
            print(f"   {r['obs']:<18} public {r['public']:<22} {r['public_score']:.3f} | best {r['best']:<24} "
                  f"{r['best_score']:.3f} wins {r['wins']}/{r['folds']}{ed} -> {r['pick']}")
        for e, x in special.items():
            print(f"   {e} ({x['kind'][:4]}): public {np.mean(x['public']):.3f} pick {np.mean(x['pick']):.3f} | members "
                  + " ".join(f"{l} {np.mean(v):.3f}" for l, v in x["per_member"].items()))
    old = json.loads((PLANS / "zoo_select.json").read_text()) if (PLANS / "zoo_select.json").exists() else {}
    for s in res:
        if s in old and "assembled" in old[s] and old[s].get("map") == res[s]["map"]:
            res[s]["assembled"] = old[s]["assembled"]
    (PLANS / "zoo_select.json").write_text(json.dumps(res, indent=1))
    print("total held-out gain (mean over 10):", np.mean([res[s]["heldout_gain"] for s in res]),
          "forecast:", np.mean([res[s]["forecast_gain"] for s in res]))


# ----------------------------------------------------------------------------- assemble
def full_blob(system, label, runs):
    """Member fitted on every fit run (never on an exam run)."""
    pay = json.loads((SCRATCH / system / f"{label}__FULL.json").read_text())
    if label == "l0b":
        return pay["blob"]
    blob = copy.deepcopy(member_blob(system, label))
    blob["theta"] = pay["theta"]
    return blob


def cmd_assemble(a):
    res = json.loads((PLANS / "zoo_select.json").read_text())
    for s in SYSTEMS:
        spec, runs = get_runs(s)
        r = res[s]
        doc = pub_doc(s)
        chosen = [tuple(c) for c in r["chosen"]]
        if not r["changed"]:
            zdoc = copy.deepcopy(doc)
            zdoc.setdefault("info", {})["model_id"] = "keep current (u013)"
        else:
            used = sorted({l for o in chosen for l in o}, key=lambda l: (list(SHORT[s]) + ["l0b"]).index(l))
            blobs = {l: full_blob(s, l, runs) for l in used}
            members, idx = [], {}
            for o in dict.fromkeys(chosen):
                idx[o] = len(members)
                members.append(blobs[o[0]] if len(o) == 1 else {"kind": "ensemble", "members": [blobs[l] for l in o]})
            zdoc = copy.deepcopy(doc)
            zdoc["model"] = {"kind": "perobs", "members": members, "map": [idx[o] for o in chosen]}
            zdoc["info"] = {"model_id": "zoo perobs: " + ", ".join(f"{ob}={opt_name(o)}" for ob, o in zip(spec.observables, chosen)),
                            "members": {l: (b.get("family", b["kind"]) + ":" + "".join(b.get("mech", ""))) for l, b in blobs.items()},
                            "n_runs": len(runs)}
        # checks: runtime reproduces the manual combination in-sample; eval-like rollouts
        lo, hi = rt.clip_vectors(zdoc)
        chk = {"max_rel_err_vs_manual": 0.0, "insample": {}}
        for rr in runs:
            Yz = rt.rollout_from_blob(zdoc, rr.y0, rr.U, doc=zdoc)
            if r["changed"]:
                M = {}
                for o in chosen:
                    for l in o:
                        if l not in M:
                            b = full_blob(s, l, runs)
                            Yl = ode_roll(b, rr.y0, rr.U) if b["kind"] == "ode" else rt.rollout_from_blob(b, rr.y0, rr.U, doc=zdoc, clip=False)
                            M[l] = rt.finalize(Yl, rr.y0, lo, hi)
                Ym = np.stack([np.mean([M[l] for l in o], axis=0)[:, j] for j, o in enumerate(chosen)], axis=1)
                Ym = rt.apply_post(zdoc, rt.finalize(Ym, rr.y0, lo, hi), rr.U, rr.y0)
                chk["max_rel_err_vs_manual"] = max(chk["max_rel_err_vs_manual"], float(np.max(np.abs(Ym - Yz) / cal_sigma(s))))
            chk["insample"][rr.exp] = metric.score_per_obs(Yz, rr.Y, cal_sigma(s)).tolist()
        Yall = np.concatenate([rr.Y for rr in runs])
        ymin, ymax = Yall.min(0), Yall.max(0)
        rg = ymax - ymin
        rng = np.random.default_rng(0)
        chk["eval"] = {}
        for cat in ("sustained", "order", "recovery", "composition"):
            U = D.eval_like(spec, cat, 4000, rng)
            t0 = time.time()
            Y = rt.rollout_from_blob(zdoc, runs[0].y0, U, doc=zdoc)
            dt = time.time() - t0
            chk["eval"][cat] = {"finite": bool(np.all(np.isfinite(Y))), "seconds": dt,
                                "frac_outside": float(np.mean((Y > ymax + 0.05 * rg) | (Y < ymin - 0.05 * rg)))}
        chk["ok"] = (all(v["finite"] and v["seconds"] < 1.5 for v in chk["eval"].values())
                     and chk["max_rel_err_vs_manual"] < 1e-9)
        out = PLANS / f"zoo_{s}_doc.json"
        out.write_text(json.dumps(zdoc, indent=1))
        r["assembled"] = {"doc": str(out.relative_to(ROOT)).replace("\\", "/"), "checks": chk}
        print(f"{s:<17} changed={r['changed']} ok={chk['ok']} err={chk['max_rel_err_vs_manual']:.1e} "
              + " ".join(f"{c}:{v['seconds']:.2f}s/{v['frac_outside']:.2f}" for c, v in chk["eval"].items()), flush=True)
    (PLANS / "zoo_select.json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- report
def cmd_report(a):
    res = json.loads((PLANS / "zoo_select.json").read_text())
    reg = {}
    for line in (PUB_DIR / "scores.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            reg[parts[0]] = float(parts[-1])
    L = ["# Per-observable zoo selection (held-out evidence only)", "",
         "We took every model family we already have per system, refitted a shortlist leave-one-run-out on all "
         "current runs, and chose per observable among each member alone and the mean of each pair "
         "(the runtime's two-member median). Everything below is held-out, scored at the calibrated organizer "
         "sigma (plans/sigma_calibrated.json, 1.0x). No credits were used.", "",
         "## Protocol", "",
         "- Folds: leave one run out over all current runs (including the round-2 runs bought Sep 27 19:09).",
         "- ODE members: `fit_ode` robust cauchy, one start at the member's full-data theta, no early phase, "
         "`max_nfev` 10, 12 s per fold (run: `loo --workers 1 --budget 12 --nfev 10`). No direct score polishing.",
         "- l0b_lin: `make_model('l0b', cfg={'sq': False, 'pairs': None})`, clip margin 1, refitted per fold.",
         "- Every member prediction is finalized with the current public doc's clip vectors; public post rules "
         "(ad_auction spend cap, hospital_queue queue cap) are applied after combining.",
         r"- Score per observable: $\frac{1}{T}\sum_t 1/(1+|\hat y_t-y_t|/\sigma)$, mean over folds.",
         f"- Accept a change on an observable only if the best option beats the public option by more than "
         f"{MARGIN} (mean over folds) AND wins on a strict majority of folds.",
         "- Exam runs `p6.exam` are never used in any fit; they are scored with members fitted on all other runs. "
         "Clean-run veto: a change is dropped if, averaged over the exam runs and the `p6.*` folds (runs no member doc "
         f"had seen), it loses more than {MARGIN} to the public option.",
         "- Cold-start check: the warm-start folds start from a theta fitted on all runs, so they leak the held-out run. "
         "Where time allowed we refitted the competing members from the family's default init (4 starts, spread 0.5, "
         "40 evaluations, 45 s) and reverted every changed observable whose cold-start gain is not positive.",
         f"- Forecast public gain = {SHRINK} x held-out gain of the assembled combination over the public family.",
         "- Validation: each shortlisted doc reproduces its lab in-sample scores exactly (max diff 0.0); "
         "each shipped u013 predict.py reproduces the dev rollout of its doc exactly.", "",
         "## Summary", "",
         "| system | public (u013) | held-out public | held-out pick | warm gain | cold gain | forecast (0.6 x min) | map | doc |",
         "|---|---|---|---|---|---|---|---|---|"]
    tot = []
    for s in SYSTEMS:
        r = res[s]
        doc = r.get("assembled", {}).get("doc", "") if r["changed"] else "keep current"
        cg = r.get("coldcheck", {}).get("gain")
        g = r["heldout_gain"] if (cg is None or not r["changed"]) else min(r["heldout_gain"], cg)
        r["forecast_conservative"] = SHRINK * g
        L.append(f"| {s} | {reg.get(s, float('nan')):.4f} | {r['heldout_public']:.3f} | {r['heldout_pick']:.3f} | "
                 f"{r['heldout_gain']:+.4f} | {'n/a' if cg is None else f'{cg:+.4f}'} | {SHRINK * g:+.4f} | {', '.join(r['map'])} | {doc} |")
        tot.append(SHRINK * g)
    L += ["", f"Forecast mean public gain over the 10 systems: **{np.mean(tot):+.4f}** (0.6 x the smaller of the warm and "
          "cold-start held-out gains of the final map; systems without a change count 0).", ""]
    for s in SYSTEMS:
        r = res[s]
        L += [f"## {s}", "", f"Runs ({len(r['runs'])}): {', '.join(r['runs'])}.", "",
              "Shortlist: " + ", ".join(f"`{k}` ({'u013 shipped' if v == 'PUB' else v})" for k, v in r["short"].items())
              + ", `l0b` (l0b_lin).", "",
              "Members alone, held-out per observable:", "",
              "| member | " + " | ".join(x["obs"] for x in r["per_obs"]) + " | mean |",
              "|---|" + "---|" * (len(r["per_obs"]) + 1)]
        for l in r["members"]:
            v = r["member_mean"][l]
            L.append(f"| {l} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.3f} |")
        L += ["", "Per observable decision (best of singles and pair means):", "",
              "| observable | public option | public | best option | best | wins | pick |", "|---|---|---|---|---|---|---|"]
        for x in r["per_obs"]:
            note = []
            if x.get("exam_delta") is not None:
                note.append(f"clean-run delta {x['exam_delta']:+.3f}" + (" VETO" if x.get("exam_veto") else ""))
            if x.get("cold_revert"):
                note.append("reverted by cold-start check")
            L.append(f"| {x['obs']} | {x['public']} | {x['public_score']:.3f} | {x['best']} | {x['best_score']:.3f} | "
                     f"{x['wins']}/{x['folds']} | **{x['pick']}** {'(' + '; '.join(note) + ')' if note else ''} |")
        L += ["", f"Held-out (warm-start folds): public {r['heldout_public']:.3f} -> pick {r['heldout_pick']:.3f} "
                  f"(gain {r['heldout_gain']:+.4f}, forecast public {r['forecast_gain']:+.4f})."]
        if not r.get("coldcheck", {}).get("reverted"):
            L.append("Per fold (public -> pick): " + ", ".join(f"{e} {p:.3f}->{q:.3f}" for e, p, q in
                                                              zip(r["fit_runs"], r["per_fold"]["public"], r["per_fold"]["pick"])) + ".")
        if r.get("coldcheck"):
            c = r["coldcheck"]
            L.append(f"Cold-start check (members refitted from the family's default init, 4 starts, as in the lab LOO): "
                     f"public {c['public']:.3f}, selection before reverts {c['pick']:.3f} ({c['gain']:+.4f}), per observable "
                     + ", ".join(f"{g:+.3f}" for g in c["per_obs_gain"]) + f"; fold wins {c['fold_wins']}/{c['folds']}; "
                     + ("reverted: " + ", ".join(c["reverted"]) if c["reverted"] else "nothing reverted") + ".")
        for e, x in r.get("special", {}).items():
            L.append(f"{e} ({x['kind']}): public {np.mean(x['public']):.3f}, selection before reverts {np.mean(x['pick']):.3f}; "
                     "members " + ", ".join(f"{l} {np.mean(v):.3f}" for l, v in x["per_member"].items()) + ".")
        if r["changed"] and r.get("assembled"):
            c = r["assembled"]["checks"]
            L.append(f"Assembled `{r['assembled']['doc']}`: runtime vs manual combination max err "
                     f"{c['max_rel_err_vs_manual']:.1e} sigma; 4,000-tick eval-like rollouts: " +
                     ", ".join(f"{k} finite={v['finite']} {v['seconds']:.2f}s outside={v['frac_outside']:.2f}" for k, v in c["eval"].items()) + ".")
        L.append("")
    (PLANS / "zoo_select.md").write_text(chr(10).join(L), encoding="utf-8")
    print("wrote plans/zoo_select.md")


def cmd_coldcheck(a):
    """Re-score the chosen map vs the public option with cold-start fold fits (members with @cold files);
    members without cold files keep their warm fold predictions (l0b has no warm start anyway)."""
    res = json.loads((PLANS / "zoo_select.json").read_text())
    for s in (a.systems.split(",") if a.systems else SYSTEMS):
        spec, runs = get_runs(s)
        fit_runs, _ = split_runs(runs)
        doc = pub_doc(s)
        sigma = cal_sigma(s)
        lo, hi = rt.clip_vectors(doc)
        labels = res[s]["members"]
        P = {}
        src = {}
        for l in labels:
            P[l] = {}
            cold = all((SCRATCH / s / f"{l}@cold__{r.exp}.json").exists() for r in fit_runs)
            src[l] = "cold" if cold else "warm"
            for r in fit_runs:
                pay = json.loads((SCRATCH / s / f"{l}{'@cold' if cold else ''}__{r.exp}.json").read_text())
                P[l][r.exp] = rt.finalize(np.asarray(pay["Y"][r.exp], float), r.y0, lo, hi)
        base = pub_option(s, labels)
        chosen = [tuple(c) for c in res[s]["chosen"]]
        def sc(o, r):
            return metric.score_per_obs(rt.apply_post(doc, combo(P, o, r.exp), r.U, r.y0), r.Y, sigma)
        B = np.array([[sc(base[j], r)[j] for j in range(len(base))] for r in fit_runs])
        Q = np.array([[sc(chosen[j], r)[j] for j in range(len(base))] for r in fit_runs])
        mem = {l: float(np.mean([sc((l,), r).mean() for r in fit_runs])) for l in labels}
        res[s]["coldcheck"] = {"sources": src, "public": float(B.mean()), "pick": float(Q.mean()),
                               "gain": float(Q.mean() - B.mean()), "per_obs_gain": (Q.mean(0) - B.mean(0)).tolist(),
                               "fold_wins": int(np.sum(Q.mean(1) > B.mean(1))), "folds": len(fit_runs), "member_mean": mem}
        # revert every changed observable whose cold-start held-out gain is not positive
        rev = [j for j in range(len(base)) if chosen[j] != base[j] and Q.mean(0)[j] - B.mean(0)[j] <= 0]
        for j in rev:
            chosen[j] = base[j]
            res[s]["per_obs"][j]["pick"] = opt_name(base[j])
            res[s]["per_obs"][j]["cold_revert"] = True
        if rev:
            rows = res[s]["per_obs"]
            res[s]["chosen"] = [list(o) for o in chosen]
            res[s]["map"] = [opt_name(o) for o in chosen]
            res[s]["changed"] = any(c != b for c, b in zip(chosen, base))
            res[s]["heldout_pick"] = float(np.mean([x["table"][x["pick"]] for x in rows]))
            res[s]["heldout_gain"] = res[s]["heldout_pick"] - res[s]["heldout_public"]
            res[s]["forecast_gain"] = SHRINK * res[s]["heldout_gain"]
            res[s].pop("assembled", None)
        res[s]["coldcheck"]["reverted"] = [spec.observables[j] for j in rev]
        print(f"{s}: cold-check public {B.mean():.3f} pick {Q.mean():.3f} gain {Q.mean() - B.mean():+.4f} "
              f"per-obs {np.round(Q.mean(0) - B.mean(0), 3)} fold wins {res[s]['coldcheck']['fold_wins']}/{len(fit_runs)} "
              f"members {{{', '.join(f'{k}:{v:.3f}({src[k][0]})' for k, v in mem.items())}}}")
    (PLANS / "zoo_select.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "loo", "select", "assemble", "report", "coldcheck"])
    ap.add_argument("--systems", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--nfev", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--labels", default=None, help="comma list of member labels; suffix @cold = cold-start refit")
    a = ap.parse_args()
    {"validate": cmd_validate, "loo": cmd_loo, "select": cmd_select, "assemble": cmd_assemble, "report": cmd_report, "coldcheck": cmd_coldcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
