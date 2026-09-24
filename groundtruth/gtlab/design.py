"""Schedule generators and default experiment plans (PLAN §4).

Every generator returns U[T, m] in physical units, ordered as spec.controls, clipped to bounds.
All randomness comes from an explicit numpy Generator.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CATEGORIES = ("sustained", "order", "recovery", "composition")


# --------------------------------------------------------------------------- levels
def lo_hi(spec):
    return np.array(spec.lo(), float), np.array(spec.hi(), float)


def clip(spec, U):
    lo, hi = lo_hi(spec)
    return np.clip(np.asarray(U, float), lo, hi)


def rec(spec):
    return clip(spec, np.array([spec.recovery[c] for c in spec.controls], float))


def pulse_level(spec, alpha=1.0):
    """recovery + alpha*(pulse - recovery); alpha scalar or per-control vector."""
    r = rec(spec)
    p = np.array([spec.pulse[c] for c in spec.controls], float)
    return clip(spec, r + np.asarray(alpha, float) * (p - r))


def draw_alpha(spec, rng):
    return rng.uniform(0.7, 1.0, size=spec.m)


def level(spec, kind, rng):
    lo, hi = lo_hi(spec)
    if kind == "recovery":
        return rec(spec)
    if kind == "pulse":
        return pulse_level(spec, draw_alpha(spec, rng))
    if kind == "mid":
        return (lo + hi) / 2
    if kind == "uniform":
        return rng.uniform(lo, hi)
    if kind == "corner":
        return np.where(rng.random(spec.m) < 0.5, lo, hi)
    raise ValueError(kind)


def hold(spec, lev, T):
    return clip(spec, np.tile(np.asarray(lev, float), (int(T), 1)))


def _loguni(rng, a, b, size=None):
    return np.exp(rng.uniform(math.log(a), math.log(b), size=size))


# --------------------------------------------------------------------------- generators
def pulse_train(spec, rng, n=None, L=(3, 40), gaps=(5, 300), lead=0, tail=0, T=None):
    """Recovery baseline with n pulses; pulse = rec + alpha*(pulse-rec), alpha ~ U(.7,1) per
    control, redrawn for each pulse. If T is given the gaps/lengths are shrunk to fit exactly."""
    n = int(n if n is not None else rng.integers(8, 13))
    lens = rng.integers(L[0], L[1] + 1, size=n)
    gps = _loguni(rng, gaps[0], gaps[1], size=n)            # gap before each pulse (first after lead)
    if T is not None:
        avail = int(T) - lead - tail
        if avail < n * 2:
            raise ValueError("T too small for pulse train")
        min_gap = 2
        # shrink pulse lengths first if they alone do not leave room for the minimum gaps
        while lens.sum() + n * min_gap > avail:
            lens = np.maximum(1, (lens * 0.8).astype(int))
        room = avail - lens.sum()
        g = np.maximum(min_gap, gps * room / gps.sum())
        g = np.floor(g).astype(int)
        g = np.maximum(g, min_gap)
        while g.sum() > room:
            g[np.argmax(g)] -= 1
        extra = room - g.sum()
        gps = g
        tail = tail + extra
    else:
        gps = np.round(gps).astype(int)
    r = rec(spec)
    rows = [hold(spec, r, lead)] if lead else []
    for i in range(n):
        rows.append(hold(spec, r, gps[i]))
        rows.append(hold(spec, pulse_level(spec, draw_alpha(spec, rng)), lens[i]))
    if tail:
        rows.append(hold(spec, r, tail))
    return clip(spec, np.concatenate(rows, axis=0))


def order_pair(spec, rng, dwell=20, lead=5, blocks=None):
    """Two schedules with the same blocks in reversed order (ABC vs CBA). For two blocks this
    is the A->B vs B->A swap. Blocks default to: pulse, a random interior level, a corner."""
    if blocks is None:
        blocks = [pulse_level(spec, draw_alpha(spec, rng)), level(spec, "uniform", rng),
                  level(spec, "corner", rng)]
    blocks = [clip(spec, b) for b in blocks]
    r = rec(spec)

    def build(seq):
        rows = [hold(spec, r, lead)] if lead else []
        rows += [hold(spec, b, dwell) for b in seq]
        return np.concatenate(rows, axis=0)

    return build(blocks), build(blocks[::-1])


def single_vs_joint(spec, rng=None, base=None, dwell=None, T=None):
    """From base (default recovery): each control alone to its pulse level (dwell D, back to
    base for D), then all jointly, then cumulative pairwise in a different (reversed) order."""
    base = rec(spec) if base is None else clip(spec, base)
    m = spec.m
    D = int(dwell if dwell is not None else max(10, 150 // (m + 1)))
    if T is not None:
        # shrink the dwell until the whole design (incl. the joint + pairwise part) fits in T
        size = lambda d: (m + 1) * (d + max(3, d // 2)) + (m - 1) * max(3, d // 2)
        while D > 2 and size(D) > T:
            D -= 1
    pul = pulse_level(spec, 1.0)
    rows = []
    for j in range(m):
        u = base.copy()
        u[j] = pul[j]
        rows += [hold(spec, u, D), hold(spec, base, max(3, D // 2))]
    rows += [hold(spec, pul, D), hold(spec, base, max(3, D // 2))]
    order = list(range(m))[::-1]
    for a, b in zip(order[:-1], order[1:]):
        u = base.copy()
        u[a] = pul[a]
        u[b] = pul[b]
        rows += [hold(spec, u, max(3, D // 2))]
    U = np.concatenate(rows, axis=0)
    return fit_length(spec, U, T) if T is not None else clip(spec, U)


def multilevel(spec, rng, T, dwell=(3, 120)):
    """General-coverage excitation: 50% interior uniform, 30% corners, 20% recovery/pulse."""
    rows, t = [], 0
    while t < T:
        d = int(min(T - t, max(1, round(_loguni(rng, *dwell)))))
        x = rng.random()
        if x < 0.5:
            lev = level(spec, "uniform", rng)
        elif x < 0.8:
            lev = level(spec, "corner", rng)
        else:
            lev = level(spec, "recovery" if rng.random() < 0.5 else "pulse", rng)
        rows.append(hold(spec, lev, d))
        t += d
    return clip(spec, np.concatenate(rows, axis=0)[:T])


def fit_length(spec, U, T):
    """Truncate, or pad with the recovery action, to exactly T rows."""
    U = np.asarray(U, float)
    if len(U) >= T:
        return clip(spec, U[:T])
    return clip(spec, np.concatenate([U, hold(spec, rec(spec), T - len(U))], axis=0))


def eval_like(spec, category, T, rng):
    """Schedules in the style of the four scored categories (plus 'mixed' = all four)."""
    T = int(T)
    if category == "sustained":
        rows, t = [], 0
        while t < T:
            d = int(min(T - t, max(10, rng.uniform(0.25, 1.0) * T)))
            kind = rng.choice(["recovery", "pulse", "uniform", "corner", "mid"])
            rows.append(hold(spec, level(spec, kind, rng), d))
            t += d
        return clip(spec, np.concatenate(rows)[:T])
    if category == "order":
        nb = int(rng.integers(2, 5))
        blocks = [level(spec, rng.choice(["pulse", "uniform", "corner"]), rng) for _ in range(nb)]
        perm = rng.permutation(nb)
        seq = [blocks[i] for i in perm] + [blocks[i] for i in perm[::-1]]
        dw = max(1, T // (len(seq) + 1))
        U = np.concatenate([hold(spec, rec(spec), dw)] + [hold(spec, b, dw) for b in seq])
        return fit_length(spec, U, T)
    if category == "recovery":
        n = int(max(1, min(12, T // 60)))
        gaps = (5, max(6, min(300, T // max(1, n))))
        return pulse_train(spec, rng, n=n, L=(3, max(3, min(40, T // (4 * n)))), gaps=gaps,
                           lead=min(20, T // 10), T=T)
    if category == "composition":
        dw = max(2, T // (2 * spec.m + 4))
        U = single_vs_joint(spec, rng, base=None, dwell=dw)
        if rng.random() < 0.5:           # shuffle the single-control blocks' order too
            U = U[::-1].copy()
        return fit_length(spec, U, T)
    if category == "mixed":
        parts = np.diff(np.round(np.linspace(0, T, 5)).astype(int))
        cats = ["sustained", "order", "recovery", "composition"]
        return clip(spec, np.concatenate([eval_like(spec, c, int(p), rng) for c, p in zip(cats, parts)]))
    raise ValueError(f"unknown category {category}")


# --------------------------------------------------------------------------- reset shop
def shop_choice(candidates, bought=(), mode="spread", target=None, scale=None):
    """Pick an index among candidate initial observations (pure function).

    spread: maximize the minimum normalized distance to already-bought initials (if none were
            bought, pick the candidate closest to the candidates' median: a typical start).
    match : minimize distance to `target` (e.g. the partner run of an order pair).
    Normalization is on log1p(|y|)*sign(y), divided by `scale` (default: pooled std, floored).
    """
    C = np.atleast_2d(np.asarray(candidates, float))
    if len(C) == 1:
        return 0
    B = np.asarray(bought, float).reshape(-1, C.shape[1]) if len(bought) else np.zeros((0, C.shape[1]))
    f = lambda x: np.sign(x) * np.log1p(np.abs(x))
    Cz, Bz = f(C), f(B)
    if scale is None:
        pool = np.concatenate([Cz, Bz], axis=0)
        scale = np.maximum(pool.std(axis=0), 1e-6)
    scale = np.asarray(scale, float)
    if mode == "match":
        if target is None:
            raise ValueError("match mode needs target")
        d = np.linalg.norm((Cz - f(np.asarray(target, float))) / scale, axis=1)
        return int(np.argmin(d))
    if len(Bz) == 0:
        d = np.linalg.norm((Cz - np.median(Cz, axis=0)) / scale, axis=1)
        return int(np.argmin(d))
    d = np.linalg.norm((Cz[:, None, :] - Bz[None, :, :]) / scale, axis=2).min(axis=1)
    return int(np.argmax(d))


# --------------------------------------------------------------------------- plans
PHASES = ("p1", "p2", "val", "reserve")


def _exp(spec, eid, U, category, shop=None, note=""):
    U = clip(spec, U)
    return {"exp_id": eid, "run_idx": 0, "T": int(len(U)), "category": category,
            "shop": shop or {"n": 5, "mode": "spread"}, "note": note,
            "U": [[float(v) for v in row] for row in U]}


def default_experiments(spec, phase, seed=0):
    """Default experiments for one phase (§4). p2 is a placeholder to be replaced after p1."""
    # stable across processes (no Python hash randomization)
    rng = np.random.default_rng([int(seed), PHASES.index(phase) if phase in PHASES else 99,
                                 sum(map(ord, spec.id))])
    r = rec(spec)
    long_T = 700 if spec.id == "reservoir" else 450
    ex = []
    if phase == "p1":
        ex.append(_exp(spec, "p1.hold_rec", hold(spec, r, 120), "sustained",
                       note="reset transient, recovery equilibrium, noise from tail"))
        ex.append(_exp(spec, "p1.hold_pulse",
                       np.concatenate([hold(spec, pulse_level(spec, 1.0), 60), hold(spec, r, 60)]),
                       "sustained", note="step on / step off"))
        ex.append(_exp(spec, "p1.long_train",
                       pulse_train(spec, rng, n=int(rng.integers(8, 13)), lead=40, tail=100, T=long_T),
                       "recovery", note="slow modes + history"))
        ex.append(_exp(spec, "p1.compose", single_vs_joint(spec, rng, T=180), "composition"))
        a, b = order_pair(spec, rng, dwell=20, lead=5)
        ex.append(_exp(spec, "p1.order_AB", a, "order"))
        ex.append(_exp(spec, "p1.order_BA", b, "order",
                       shop={"n": 8, "mode": "match", "match_exp": "p1.order_AB"}))
    elif phase == "p2":
        rev_T = 50 if spec.id == "reservoir" else 75
        p = pulse_level(spec, 1.0)
        mid = level(spec, "mid", rng)
        h = rev_T // 3
        ex.append(_exp(spec, "p2.mech_rev", np.concatenate(
            [hold(spec, r, 5), hold(spec, p, h), hold(spec, mid, h), hold(spec, r, rev_T - 5 - 2 * h)]),
            "order", note="PLACEHOLDER: pulse -> reversal; replace with the brief's experiment (§7)"))
        ex.append(_exp(spec, "p2.mech_hold", np.concatenate(
            [hold(spec, r, 5), hold(spec, p, 2 * h), hold(spec, r, rev_T - 5 - 2 * h)]),
            "order", shop={"n": 8, "mode": "match", "match_exp": "p2.mech_rev"},
            note="PLACEHOLDER: pulse -> hold"))
        ml_T = 150 if spec.id == "reservoir" else 200
        ex.append(_exp(spec, "p2.multilevel", multilevel(spec, rng, ml_T), "sustained"))
        if spec.id != "reservoir":
            ex.append(_exp(spec, "p2.fix", eval_like(spec, "recovery", 150, rng), "recovery",
                           note="PLACEHOLDER: target worst category after p1"))
    elif phase == "val":
        for i in (1, 2):
            ex.append(_exp(spec, f"val.val{i}", eval_like(spec, "mixed", 125, rng), "mixed",
                           note="held out; never trained on until final refit"))
    elif phase == "reserve":
        pass                       # filled by hand on Sep 27
    else:
        raise ValueError(phase)
    return ex


def experiments_from_file(spec, path, phase):
    """Custom experiments (p2/reserve): JSON list, or {system_id: list}. exp ids get the
    phase prefix; schedules are validated (shape) and clipped to bounds."""
    doc = json.loads(Path(path).read_text())
    if isinstance(doc, dict):
        doc = doc.get(spec.id, [])
    out = []
    for e in doc:
        U = np.asarray(e["U"], float)
        if U.ndim != 2 or U.shape[1] != spec.m or not np.all(np.isfinite(U)):
            raise ValueError(f"{e.get('exp_id')}: U must be finite [T, {spec.m}] in {spec.controls} order")
        eid = e["exp_id"] if e["exp_id"].startswith(phase + ".") else f"{phase}.{e['exp_id']}"
        out.append(_exp(spec, eid, U, e.get("category", "custom"), shop=e.get("shop"), note=e.get("note", "")))
    if len({e["exp_id"] for e in out}) != len(out):
        raise ValueError("duplicate exp_id")
    return out


def _hash_obj(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def phase_plan(spec, phase, seed=0, experiments=None):
    ex = experiments if experiments is not None else default_experiments(spec, phase, seed)
    for e in ex:
        e["hash"] = _hash_obj({"exp_id": e["exp_id"], "U": e["U"]})
    body = {"system": spec.id, "phase": phase, "seed": int(seed), "experiments": ex,
            "steps": int(sum(e["T"] for e in ex))}
    body["hash"] = _hash_obj({"experiments": [e["hash"] for e in ex]})
    return body


def plan_path(directory) -> Path:
    return Path(directory) / "plan.json"


def load_plan_file(directory) -> dict:
    p = plan_path(directory)
    return json.loads(p.read_text()) if p.exists() else {"phases": {}}


def materialize(spec, directory, phase, seed=0, experiments=None, force=False, ledger=None):
    """Write the phase into plan.json once. Existing phases are never silently replaced: with
    force=True it is replaced only if the ledger has no rows for any of its experiments."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    doc = load_plan_file(directory)
    doc.setdefault("system", spec.id)
    doc.setdefault("phases", {})
    new = phase_plan(spec, phase, seed, experiments)
    old = doc["phases"].get(phase)
    if old is not None and not force:
        return old
    if old is not None and force and ledger is not None:
        touched = [e["exp_id"] for e in old["experiments"]
                   if ledger.run_state(e["exp_id"], e.get("run_idx", 0))["reset"] is not None]
        new_ids = {e["exp_id"]: e["hash"] for e in new["experiments"]}
        changed = [x for x in touched
                   if new_ids.get(x) != next(e["hash"] for e in old["experiments"] if e["exp_id"] == x)]
        if changed:
            raise ValueError(f"experiments already started, change needs new exp_ids: {changed}")
    doc["phases"][phase] = new
    tmp = plan_path(directory).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(plan_path(directory))
    return new
