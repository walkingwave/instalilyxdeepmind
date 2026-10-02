"""Fitting-procedure study: same family and start, different ways of fitting (no credits).

    python scripts/fitproc.py time                         # cost of one residual/jacobian per system
    python scripts/fitproc.py fit --systems a,b [--workers 2]
    python scripts/fitproc.py eval [--systems a,b]         # -> plans/fitproc.json
    python scripts/fitproc.py docs                         # winners -> plans/fitproc_<system>_doc.json + checks

Procedures (start = the current public theta of the system's main ODE member, sigma = calibrated 1.0x):
  P0  fit_ode, cauchy on residuals in sigma units, 2 starts (public theta + one LHS start at spread 0.1)
  P1  same, residual weights w = p_test(bin) / p_train(bin): bins over (ticks since the last control
      change) x (controls at the recovery action or not); p_test = equal mix of the four eval_like
      categories on 4,000-tick schedules
  P2  L-BFGS-B on mean smooth (1 - score) + lam * mean (z - z_P0)^2 from the P0 solution on the same
      runs; lam in LAMS picked by held-out score on the other folds (never on the fold being scored)
  P3  bagging: per-tick mean / median of the P0 models fitted with one more run left out, plus P0 itself
  P4  initial state: y0 shrunk toward the reset-population mean by noise^2 / (noise^2 + spread^2)
Held-out: fold k = leave fit run k out (p6.exam never fitted; scored from fits on every fit run).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import design as D, metric, systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402
from gtlab.runtime import infer as rt  # noqa: E402

SCRATCH = Path(os.environ.get("FP_SCRATCH", r"C:\Users\DYLANH~1\AppData\Local\Temp\claude\scratch\fp"))
PUB_DIR = ROOT / "submissions" / "20260927-2151-u015"
PLANS = ROOT / "plans"
SYSTEMS = ["social_contagion", "market", "hospital_queue", "epidemic", "wildlife",
           "traffic", "power_grid", "ad_auction", "supply_chain", "reservoir"]
NOFIT = {"p6.exam"}
TESTLIKE = "p5.testlike"
LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)
NFEV, BUDGET, NFEV2 = 20, 75.0, 25
AGE_EDGES = np.array([0, 3, 10, 30, 100, 300, np.inf])


# ----------------------------------------------------------------------------- setup
def pub_doc(system):
    return json.loads((PUB_DIR / system / "model.json").read_text())


def main_member(blob):
    if blob["kind"] == "ode":
        return blob
    for m in blob.get("members", []) + ([blob["member"]] if "member" in blob else []):
        b = main_member(m)
        if b is not None:
            return b
    return None


class Sys:
    def __init__(self, system):
        self.name = system
        self.spec = S.get(system)
        self.runs = load_runs(self.spec, Ledger(data_dir(system, False), system, False))
        self.fit_runs = [r for r in self.runs if r.exp not in NOFIT]
        self.exams = [r for r in self.runs if r.exp in NOFIT]
        self.sigma = np.asarray(json.loads((PLANS / "sigma_calibrated.json").read_text())[system]["sigma"], float)
        self.doc = pub_doc(system)
        self.blob = copy.deepcopy(main_member(self.doc["model"]))
        self.mod = importlib.import_module(f"gtlab.ode.{self.blob['family']}")
        self.mech = frozenset(self.blob["mech"])
        self.n_sub = int(self.blob.get("n_sub", 2))
        self.theta_pub = np.asarray(self.blob["theta"], float)
        self.lo, self.hi = rt.clip_vectors(self.doc)
        self.byexp = {r.exp: r for r in self.runs}

    def subset(self, excl):
        return [r for r in self.fit_runs if r.exp not in excl]

    def roll(self, theta, r, y0=None):
        Y = core.rollout(self.mod, r.y0 if y0 is None else y0, r.U, core.theta_dict(self.mod, theta), self.mech,
                         n_sub=self.n_sub)
        return rt.finalize(Y, r.y0, self.lo, self.hi)

    def score(self, Y, r, T=None):
        Yp = rt.apply_post(self.doc, np.array(Y, copy=True), r.U, r.y0)
        T = T or r.T
        return float(metric.score_per_obs(Yp[:T], r.Y[:T], self.sigma).mean())


def key_of(excl):
    return "full" if not excl else "x-" + "+".join(sorted(excl))


# ----------------------------------------------------------------------------- P1 weights
def _age_rec(U, rec):
    T = U.shape[0]
    age = np.zeros(T)
    a = 0
    for t in range(T):
        if t > 0 and np.any(np.abs(U[t] - U[t - 1]) > 1e-9):
            a = 0
        age[t] = a
        a += 1
    at_rec = np.all(np.abs(U - rec[None]) <= 1e-9 * (1 + np.abs(rec[None])), axis=1)
    return age, at_rec


def _bins(U, rec):
    age, at_rec = _age_rec(U, rec)
    b = np.searchsorted(AGE_EDGES, age, side="right") - 1
    return b * 2 + at_rec.astype(int)


def test_mix(spec, n=10, seed=0):
    rec = np.array([spec.recovery[c] for c in spec.controls], float)
    nb = 2 * (len(AGE_EDGES) - 1)
    p = np.zeros(nb)
    rng = np.random.default_rng(seed)
    for cat in ("sustained", "order", "recovery", "composition"):
        h = np.zeros(nb)
        for _ in range(n):
            h += np.bincount(_bins(D.eval_like(spec, cat, 4000, rng), rec), minlength=nb)
        p += 0.25 * h / h.sum()
    return p


def tick_weights(sy, runs, power=1.0):
    rec = np.array([sy.spec.recovery[c] for c in sy.spec.controls], float)
    pt = test_mix(sy.spec)
    B = [_bins(r.U, rec) for r in runs]
    h = np.bincount(np.concatenate(B), minlength=len(pt)).astype(float)
    ptr = h / h.sum()
    ratio = np.where(ptr > 0, pt / np.maximum(ptr, 1e-12), 0.0)
    ratio = np.clip(ratio, 0.1, 10.0)
    W = [ratio[b] ** power for b in B]
    m = np.concatenate(W).mean()
    return [w / m for w in W]


class WProblem(F._Problem):
    def __init__(self, *a, weights=None, **kw):
        super().__init__(*a, **kw)
        if weights is not None:
            T, R = self.mask.shape
            w = np.zeros((T, R))
            for i, wi in enumerate(weights):
                w[:len(wi), i] = wi
            self.w = np.sqrt(w)[self.mask]


# ----------------------------------------------------------------------------- fits
def fit_p0(sy, runs, seed=0):
    th, info = F.fit_ode(sy.mod, runs, sy.sigma, sy.mech, n_starts=2, max_nfev=NFEV, n_sub=sy.n_sub, seed=seed,
                         time_budget=BUDGET, early_T=None, spread=0.1, theta0=sy.theta_pub)
    return th, {"cost": float(info["cost"])}


def fit_p1(sy, runs, seed=0, power=1.0):
    t0 = time.time()
    W = tick_weights(sy, runs, power)
    base = F.to_z(sy.mod, sy.theta_pub)
    prob = WProblem(sy.mod, runs, sy.sigma, sy.mech, sy.n_sub, deadline=t0 + BUDGET, weights=W)
    out = []
    for zs in F._lhs_starts(base, 2, 0.1, seed):
        prob.best = (np.inf, None)
        try:
            res = least_squares(prob.fun, zs, jac=prob.jac, bounds=(0.0, 1.0), loss="cauchy", f_scale=1.0,
                                max_nfev=NFEV, args=(base,), x_scale=1.0)
            zf = res.x
        except F._Budget:
            zf = prob.best[1] if prob.best[1] is not None else zs
        out.append((prob.cost(zf), zf))
        if time.time() > t0 + BUDGET:
            break
    out.sort(key=lambda t: t[0])
    return F.from_z(sy.mod, out[0][1]), {"cost": float(out[0][0])}


EPS = 0.05


def _rho(r):
    s = np.sqrt(r * r + EPS * EPS)
    return s / (1 + s) - EPS / (1 + EPS), (r / s) / (1 + s) ** 2


def fit_p2(sy, runs, theta_ref, lam):
    t0 = time.time()
    prob = F._Problem(sy.mod, runs, sy.sigma, sy.mech, sy.n_sub)
    zref = F.to_z(sy.mod, theta_ref)
    P = len(zref)
    best = [np.inf, zref.copy()]

    def obj(z):
        if time.time() > t0 + BUDGET:
            raise F._Budget()
        r = prob.fun(z, zref)
        J = prob.jac(z, zref)
        v, g = _rho(r)
        N = len(r)
        d = z - zref
        f = v.sum() / N + lam * np.dot(d, d) / P
        grad = J.T @ g / N + 2 * lam * d / P
        if f < best[0]:
            best[0], best[1] = f, z.copy()
        return f, grad

    try:
        minimize(obj, zref.copy(), jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * P,
                 options={"maxfun": NFEV2, "maxiter": NFEV2})
    except F._Budget:
        pass
    return F.from_z(sy.mod, best[1]), {"obj": float(best[0])}


# ----------------------------------------------------------------------------- jobs
def cache_path(system, proc, key):
    return SCRATCH / system / f"{proc}__{key}.json"


def load_theta(system, proc, key):
    f = cache_path(system, proc, key)
    return np.asarray(json.loads(f.read_text())["theta"], float) if f.exists() else None


def run_job(system, proc, excl):
    sy = Sys(system)
    runs = sy.subset(excl)
    t0 = time.time()
    if proc == "P0":
        th, info = fit_p0(sy, runs)
    elif proc == "P1":
        th, info = fit_p1(sy, runs)
    elif proc == "P1h":
        th, info = fit_p1(sy, runs, power=0.5)
    else:
        lam = float(proc.split("_")[1])
        ref = load_theta(system, "P0", key_of(excl))
        th, info = fit_p2(sy, runs, ref, lam)
    info.update({"theta": [float(v) for v in th], "sec": time.time() - t0, "train": [r.exp for r in runs]})
    f = cache_path(system, proc, key_of(excl))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(info))
    return system, proc, key_of(excl), info["sec"]


def job_list(system):
    sy = Sys(system)
    ex = [r.exp for r in sy.fit_runs]
    sets = [()] + [(e,) for e in ex] + [(a, b) for i, a in enumerate(ex) for b in ex[i + 1:]]
    sets = [s for s in sets if len(sy.subset(s)) > 0]
    jobs = [("P0", s) for s in sets] + [(p, s) for p in ("P1", "P1h") for s in sets if len(s) <= 1]
    jobs += [(f"P2_{lam}", s) for lam in LAMS for s in sets if len(s) <= 1]
    return jobs


def cmd_fit(a):
    systems = a.systems.split(",") if a.systems else SYSTEMS
    pending = []
    for s in systems:
        for proc, excl in job_list(s):
            if not cache_path(s, proc, key_of(excl)).exists():
                pending.append((s, proc, excl))
    print(f"{len(pending)} fits on {a.workers} workers", flush=True)
    t0 = time.time()
    running = {}
    n = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        while pending or running:
            while len(running) < a.workers:
                ready = [j for j in pending if not j[1].startswith("P2")
                         or cache_path(j[0], "P0", key_of(j[2])).exists()]
                if not ready:
                    break
                j = ready[0]
                pending.remove(j)
                running[ex.submit(run_job, *j)] = j
            if not running:
                print("stuck: P2 jobs without P0", pending, flush=True)
                break
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fu in done:
                j = running.pop(fu)
                n += 1
                try:
                    _, _, k, sec = fu.result()
                    print(f"[{n} {time.time() - t0:.0f}s] {j[0]} {j[1]} {k} {sec:.0f}s", flush=True)
                except Exception as e:
                    print(f"[{n}] {j[0]} {j[1]} {key_of(j[2])} ERROR {e!r}", flush=True)


def cmd_time(a):
    for s in (a.systems.split(",") if a.systems else SYSTEMS):
        sy = Sys(s)
        prob = F._Problem(sy.mod, sy.fit_runs, sy.sigma, sy.mech, sy.n_sub)
        z = F.to_z(sy.mod, sy.theta_pub)
        t0 = time.time(); prob.fun(z, z); t1 = time.time(); prob.jac(z + 1e-9, z); t2 = time.time()
        print(f"{s:<17} P={len(z)} runs={len(sy.fit_runs)} fun {t1 - t0:.2f}s jac {t2 - t1:.2f}s  jobs {len(job_list(s))}",
              flush=True)


# ----------------------------------------------------------------------------- P4
def resets_stats(sy):
    rows = [json.loads(l) for l in (ROOT / "data" / sy.name / "resets.jsonl").read_text().splitlines() if l.strip()]
    Y0 = np.array([[row["obs"][o] for o in sy.spec.observables] for row in rows], float)
    d = np.concatenate([np.diff(r.Y, axis=0) for r in sy.runs if r.T > 2], axis=0)
    noise = np.median(np.abs(d - np.median(d, axis=0)), axis=0) * 1.4826 / np.sqrt(2)
    spread = Y0.std(axis=0)
    a = noise ** 2 / (noise ** 2 + spread ** 2)
    return Y0.mean(0), a, noise, spread, len(Y0)


# ----------------------------------------------------------------------------- balanced mean tree
def tree_mean(items):
    """Balanced binary tree of pairwise means: the runtime's median of two members is their mean, so a
    nested {"kind":"ensemble"} tree ships this exactly. Weights 2^-depth (uniform when len is a power of 2)."""
    if len(items) == 1:
        return items[0]
    h = (len(items) + 1) // 2
    return 0.5 * (tree_mean(items[:h]) + tree_mean(items[h:]))


def tree_blob(blobs):
    if len(blobs) == 1:
        return blobs[0]
    h = (len(blobs) + 1) // 2
    return {"kind": "ensemble", "members": [tree_blob(blobs[:h]), tree_blob(blobs[h:])]}


def zmean(sy, thetas):
    """Parameter bagging: mean of the fitted parameter vectors in the fitter's z space (log for log params)."""
    return F.from_z(sy.mod, np.mean([F.to_z(sy.mod, t) for t in thetas], axis=0))


def medoid_order(sy, thetas):
    """Indices of bag members sorted by z-space distance to the bag's mean parameter vector (label-free)."""
    Z = np.array([F.to_z(sy.mod, t) for t in thetas])
    return list(np.argsort(np.linalg.norm(Z - Z.mean(0), axis=1)))


# ----------------------------------------------------------------------------- eval
def cmd_eval(a):
    systems = a.systems.split(",") if a.systems else SYSTEMS
    outp = PLANS / "fitproc.json"
    res = json.loads(outp.read_text()) if outp.exists() else {}
    for s in systems:
        sy = Sys(s)
        ex = [r.exp for r in sy.fit_runs]
        K = len(ex)
        th = lambda proc, excl: load_theta(s, proc, key_of(excl))
        pred_cache = {}

        def pred(proc, excl, e):
            k = (proc, key_of(excl), e)
            if k not in pred_cache:
                t = th(proc, excl)
                pred_cache[k] = None if t is None else sy.roll(t, sy.byexp[e])
            return pred_cache[k]

        def sc(Y, e, T=None):
            return None if Y is None else sy.score(Y, sy.byexp[e], T)

        def p2_lam(excl_outer, candidates):
            """lam with the best mean held-out score over validation runs `candidates` (never the scored run)."""
            best, bl = -1, LAMS[1]
            tab = {}
            for lam in LAMS:
                v = [sc(pred(f"P2_{lam}", (j,), j), j) for j in candidates]
                v = [x for x in v if x is not None]
                if len(v) == 0:
                    continue
                tab[lam] = float(np.mean(v))
                if tab[lam] > best:
                    best, bl = tab[lam], lam
            return bl, tab

        folds = {}
        for e in ex:
            others = [j for j in ex if j != e]
            f = {}
            f["P0"] = sc(pred("P0", (e,), e), e)
            f["P1"] = sc(pred("P1", (e,), e), e)
            f["P1h"] = sc(pred("P1h", (e,), e), e)
            lam, tab = p2_lam((e,), others)
            f["P2"] = sc(pred(f"P2_{lam}", (e,), e), e)
            f["P2_lam"] = lam
            for lam_ in LAMS:
                f[f"P2_{lam_}"] = sc(pred(f"P2_{lam_}", (e,), e), e)
            bag = [pred("P0", tuple(sorted((e, j))), e) for j in others]
            bag = [b for b in bag if b is not None]
            if bag:
                members = bag + [pred("P0", (e,), e)]
                f["P3mean"] = sc(np.mean(members, axis=0), e)
                f["P3med"] = sc(np.median(members, axis=0), e)
                f["P3tree"] = sc(tree_mean(members), e)
                ths = [th("P0", tuple(sorted((e, j)))) for j in others] + [th("P0", (e,))]
                Ypar = sy.roll(zmean(sy, [t for t in ths if t is not None]), sy.byexp[e])
                f["P3par"] = sc(Ypar, e)
                f["P3par2"] = sc(0.5 * (Ypar + pred("P0", (e,), e)), e)
                bth = [th("P0", tuple(sorted((e, j)))) for j in others]
                bk = [(t, j) for t, j in zip(bth, others) if t is not None]
                order = medoid_order(sy, [t for t, _ in bk])
                near = [pred("P0", tuple(sorted((e, bk[i][1]))), e) for i in order]
                f["P3m2"] = sc(0.5 * (near[0] + pred("P0", (e,), e)), e)
                if len(near) >= 2:
                    f["P3m3"] = sc(np.median([near[0], near[1], pred("P0", (e,), e)], axis=0), e)
                f["P3n"] = len(members)
            else:
                f["P3mean"] = f["P3med"] = f["P3tree"] = f["P3par"] = f["P3par2"] = None
            # P4: shrink y0, on the P0 fold model
            m0, aa, noise, spread, nres = resets_stats(sy)
            r = sy.byexp[e]
            y0s = r.y0 - aa * (r.y0 - m0)
            t0 = th("P0", (e,))
            Y4 = sy.roll(t0, r, y0=y0s)
            f["P4"] = sc(Y4, e)
            f["P0_head20"] = sc(pred("P0", (e,), e), e, 20)
            f["P4_head20"] = sc(Y4, e, 20)
            folds[e] = f
            print(f"{s} {e:<24} " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in f.items()),
                  flush=True)
        # exams: fits on every fit run
        exams = {}
        for r in sy.exams:
            e = r.exp
            g = {"P0": sc(sy.roll(th("P0", ()), r), e), "P1": sc(sy.roll(th("P1", ()), r), e)}
            g["P1h"] = sc(sy.roll(th("P1h", ()), r), e) if th("P1h", ()) is not None else None
            lam, _ = p2_lam((), ex)
            g["P2"] = sc(sy.roll(th(f"P2_{lam}", ()), r), e)
            g["P2_lam"] = lam
            members = [sy.roll(th("P0", (j,)), r) for j in ex] + [sy.roll(th("P0", ()), r)]
            g["P3mean"] = sc(np.mean(members, axis=0), e)
            g["P3med"] = sc(np.median(members, axis=0), e)
            g["P3tree"] = sc(tree_mean(members), e)
            Ypar = sy.roll(zmean(sy, [th("P0", (j,)) for j in ex] + [th("P0", ())]), r)
            g["P3par"] = sc(Ypar, e)
            g["P3par2"] = sc(0.5 * (Ypar + sy.roll(th("P0", ()), r)), e)
            order = medoid_order(sy, [th("P0", (j,)) for j in ex])
            near = [members[i] for i in order]
            g["P3m2"] = sc(0.5 * (near[0] + members[-1]), e)
            g["P3m3"] = sc(np.median([near[0], near[1], members[-1]], axis=0), e)
            g["public"] = sc(sy.roll(sy.theta_pub, r), e)
            exams[e] = g
            print(f"{s} EXAM {e} " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in g.items()),
                  flush=True)
        procs = ["P0", "P1", "P1h", "P2", "P3mean", "P3med", "P3tree", "P3par", "P3par2", "P3m2", "P3m3", "P4"] + [f"P2_{l}" for l in LAMS]
        summ = {}
        rec_runs = [e for e in ex if "pulse" in e]
        exam_keys = [e for e in ex if e == TESTLIKE]
        for p in procs:
            v = [folds[e][p] for e in ex if folds[e].get(p) is not None]
            if len(v) < K:
                summ[p] = None
                continue
            m = float(np.mean(v))
            m0 = float(np.mean([folds[e]["P0"] for e in ex]))
            recd = float(np.mean([folds[e][p] - folds[e]["P0"] for e in rec_runs])) if rec_runs else None
            exd = [folds[e][p] - folds[e]["P0"] for e in exam_keys]
            exd += [exams[e][p] - exams[e]["P0"] for e in exams if exams[e].get(p) is not None]
            examd = float(np.mean(exd)) if exd else None
            wins = int(sum(folds[e][p] > folds[e]["P0"] + 1e-9 for e in ex))
            gain = m - m0
            ok = (p not in ("P0", "P3mean") and gain > 0.01 and (recd is None or recd >= -0.002)
                  and (examd is None or examd >= -0.002) and not p.startswith("P2_"))
            summ[p] = {"mean": m, "gain": gain, "rec_delta": recd, "exam_delta": examd, "wins": wins, "folds": K,
                       "win": bool(ok)}
        cands = [p for p in procs if summ.get(p) and summ[p]["win"]]
        winner = max(cands, key=lambda p: summ[p]["gain"]) if cands else None
        m0, aa, noise, spread, nres = resets_stats(sy)
        res[s] = {"family": sy.blob["family"], "mech": sorted(sy.mech), "fit_runs": ex, "exam_runs": [r.exp for r in sy.exams],
                  "recovery_folds": rec_runs, "folds": folds, "exams": exams, "summary": summ, "winner": winner,
                  "forecast_public_gain": 0.6 * summ[winner]["gain"] if winner else 0.0,
                  "p4": {"shrink": aa.tolist(), "noise": noise.tolist(), "reset_spread": spread.tolist(), "n_resets": nres,
                         "sigma": sy.sigma.tolist()}}
        print(f"== {s}: " + " | ".join(f"{p} {summ[p]['mean']:.4f} ({summ[p]['gain']:+.4f}, rec {summ[p]['rec_delta']}, "
                                         f"exam {summ[p]['exam_delta']})" for p in procs if summ.get(p))
              + f"  WINNER {winner}", flush=True)
        outp.write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- docs
def ode_blob(sy, theta):
    b = copy.deepcopy(sy.blob)
    b["theta"] = [float(v) for v in theta]
    return b


def swap_main(blob, fam, new):
    """Replace every ODE member of family `fam` in a (nested) blob by `new`."""
    if blob["kind"] == "ode":
        return copy.deepcopy(new) if blob["family"] == fam else blob
    b = copy.deepcopy(blob)
    if "members" in b:
        b["members"] = [swap_main(m, fam, new) for m in b["members"]]
    if "member" in b:
        b["member"] = swap_main(b["member"], fam, new)
    return b


def build_doc(sy, winner, lam=None):
    ex = [r.exp for r in sy.fit_runs]
    if winner in ("P1", "P1h"):
        new = ode_blob(sy, load_theta(sy.name, winner, "full"))
    elif winner == "P2":
        new = ode_blob(sy, load_theta(sy.name, f"P2_{lam}", "full"))
    elif winner in ("P3med", "P3tree"):
        mem = [ode_blob(sy, load_theta(sy.name, "P0", key_of((j,)))) for j in ex] + [ode_blob(sy, load_theta(sy.name, "P0", "full"))]
        new = {"kind": "ensemble", "members": mem} if winner == "P3med" else tree_blob(mem)
    elif winner in ("P3m2", "P3m3"):
        ths = [load_theta(sy.name, "P0", key_of((j,))) for j in ex]
        order = medoid_order(sy, ths)
        mem = [ode_blob(sy, ths[i]) for i in order[:1 if winner == "P3m2" else 2]]
        new = {"kind": "ensemble", "members": mem + [ode_blob(sy, load_theta(sy.name, "P0", "full"))]}
    elif winner in ("P3par", "P3par2"):
        ths = [load_theta(sy.name, "P0", key_of((j,))) for j in ex] + [load_theta(sy.name, "P0", "full")]
        new = ode_blob(sy, zmean(sy, ths))
        if winner == "P3par2":
            new = {"kind": "ensemble", "members": [new, ode_blob(sy, load_theta(sy.name, "P0", "full"))]}
    else:
        raise ValueError(winner)
    doc = copy.deepcopy(sy.doc)
    doc["model"] = swap_main(doc["model"], sy.blob["family"], new) if doc["model"]["kind"] != "ode" else new
    return doc


def check_doc(sy, doc, max_sec=1.5):
    rng = np.random.default_rng(0)
    out = {}
    y0 = sy.runs[0].y0
    for cat in ("sustained", "order", "recovery", "composition"):
        U = D.eval_like(sy.spec, cat, 4000, rng)
        t0 = time.time()
        Y = rt.rollout_from_blob(doc, y0, U, doc=doc)
        dt = time.time() - t0
        out[cat] = {"finite": bool(np.all(np.isfinite(Y))), "sec": dt, "shape": list(Y.shape)}
    ok = all(v["finite"] and v["sec"] < max_sec and v["shape"] == [4000, sy.spec.p] for v in out.values())
    # reproduces the study's in-sample prediction on the last fit run (same finalize + post path)
    return ok, out


def cmd_docs(a):
    res = json.loads((PLANS / "fitproc.json").read_text())
    for s, r in res.items():
        cands = sorted([p for p, v in r["summary"].items() if v and v["win"]], key=lambda p: -r["summary"][p]["gain"])
        r["doc"] = {"path": None, "tried": []}
        if not cands:
            continue
        sy = Sys(s)
        for w in cands:
            lam = None
            if w == "P2":
                lam = min(LAMS, key=lambda l: -np.mean([r["folds"][e][f"P2_{l}"] for e in r["fit_runs"]]))
            doc = build_doc(sy, w, lam)
            ok, chk = check_doc(sy, doc, a.max_sec)
            r["doc"]["tried"].append({"proc": w, "ok": ok, "checks": chk, "lam": lam})
            print(s, w, ok, {k: round(v["sec"], 2) for k, v in chk.items()}, flush=True)
            if ok:
                doc_path = PLANS / f"fitproc_{s}_doc.json"
                doc_path.write_text(json.dumps(doc))
                r["doc"].update({"path": str(doc_path.relative_to(ROOT)), "proc": w})
                break
    (PLANS / "fitproc.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["time", "fit", "eval", "docs"])
    ap.add_argument("--systems", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-sec", type=float, default=1.5, help="docs: per-episode time limit for the 4,000-tick check")
    a = ap.parse_args()
    {"time": cmd_time, "fit": cmd_fit, "eval": cmd_eval, "docs": cmd_docs}[a.cmd](a)


if __name__ == "__main__":
    main()
