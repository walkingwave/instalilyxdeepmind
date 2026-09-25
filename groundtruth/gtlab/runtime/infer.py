"""Numpy-only inference runtime (ships in every submission folder as gt_infer.py).

Only numpy + stdlib. No scipy, no gtlab imports. The dev models in gtlab/models/* reuse the
transform / feature helpers below so that training and inference share one definition, and
every model kind has a parity test (dev rollout == rollout_from_blob).

model.json ("doc") layout:
    {"format": 1, "system": str, "observables": [...], "controls": [...],
     "bounds": {ctrl: [lo, hi]}, "recovery": {ctrl: v},
     "hard_lo": [p], "hard_hi": [p or null], "clip_lo": [p or null], "clip_hi": [p or null],
     "model": <blob>, "info": {...}}
blob kinds: l0a, l0b, l1, l2, ode, ensemble, perobs.
Arrays may be stored in an optional model.npz next to model.json and referenced as "npz:<key>".
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

FORMAT = 1
_VCLAMP = 60.0


# ----------------------------------------------------------------------------- loading
def _resolve(obj, arrays):
    if isinstance(obj, str) and obj.startswith("npz:") and arrays is not None:
        return np.asarray(arrays[obj[4:]])
    if isinstance(obj, dict):
        return {k: _resolve(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, arrays) for v in obj]
    return obj


def load_doc(folder):
    folder = Path(folder)
    doc = json.loads((folder / "model.json").read_text())
    npz = folder / "model.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        doc = _resolve(doc, arrays)
    if int(doc.get("format", 1)) != FORMAT:
        raise ValueError("unsupported model.json format")
    return doc


def _arr(x, dtype=float):
    return np.asarray(x, dtype=dtype)


def _bound_vec(vals, default):
    return np.array([default if v is None else float(v) for v in vals], dtype=float)


def clip_vectors(doc):
    """Final clip range = intersection of hard bounds and soft (data) range."""
    p = len(doc["observables"])
    hlo = _bound_vec(doc.get("hard_lo", [0.0] * p), -np.inf)
    hhi = _bound_vec(doc.get("hard_hi", [None] * p), np.inf)
    clo = _bound_vec(doc.get("clip_lo", [None] * p), -np.inf)
    chi = _bound_vec(doc.get("clip_hi", [None] * p), np.inf)
    lo = np.maximum(hlo, clo)
    hi = np.minimum(hhi, chi)
    hi = np.maximum(hi, lo)
    return lo, hi


def finalize(Y, y0, lo, hi):
    """Clip to [lo, hi]; any nonfinite entry is replaced by clipped persistence."""
    Y = np.array(Y, dtype=float, copy=True)
    y0 = np.asarray(y0, dtype=float)
    fill_default = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0))
    ypers = np.where(np.isfinite(y0), y0, fill_default)
    ypers = np.minimum(np.maximum(ypers, lo), hi)
    bad = ~np.isfinite(Y)
    if bad.any():
        Y[bad] = np.broadcast_to(ypers, Y.shape)[bad]
    np.clip(Y, lo, hi, out=Y)
    return Y


# ----------------------------------------------------------------------------- features
def norm_u(doc, U):
    """Physical U[T, m] (doc['controls'] order) -> [0, 1]."""
    lo = np.array([doc["bounds"][c][0] for c in doc["controls"]], dtype=float)
    hi = np.array([doc["bounds"][c][1] for c in doc["controls"]], dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (np.asarray(U, dtype=float) - lo) / span


def delay_u(Un, delays):
    """Column i uses u[t - d_i]; ticks before the start repeat u[0]."""
    if not delays or not any(int(d) for d in delays):
        return Un
    T = Un.shape[0]
    out = np.empty_like(Un)
    for i, d in enumerate(delays):
        d = int(d)
        idx = np.maximum(np.arange(T) - d, 0)
        out[:, i] = Un[idx, i]
    return out


def phi(Un, ps):
    """Feature map phi(u~): [u, u^2 (opt), pairwise products (listed), 1 (opt)]."""
    Un = np.asarray(Un, dtype=float)
    cols = [Un]
    if ps.get("sq", True):
        cols.append(Un * Un)
    for i, j in ps.get("pairs", []):
        cols.append((Un[:, int(i)] * Un[:, int(j)])[:, None])
    if ps.get("const", True):
        cols.append(np.ones((Un.shape[0], 1)))
    return np.concatenate(cols, axis=1)


def g_fwd(Y, tr):
    """Physical -> standardized transformed space. Y [..., p]."""
    Y = np.asarray(Y, dtype=float)
    kinds, s = tr["kind"], _arr(tr["s"])
    mu, sd = _arr(tr["mu"]), _arr(tr["sd"])
    eps = float(tr.get("eps", 1e-3))
    G = np.empty_like(Y)
    for j, k in enumerate(kinds):
        y = Y[..., j]
        if k == "log1p":
            G[..., j] = np.log1p(np.maximum(y, 0.0) / s[j])
        elif k == "logit":
            yc = np.clip(y, 0.0, 1.0)
            G[..., j] = np.log((yc + eps) / (1.0 - yc + eps))
        else:
            G[..., j] = y / s[j]
    return (G - mu) / sd


def g_inv(V, tr):
    V = np.clip(np.asarray(V, dtype=float), -_VCLAMP, _VCLAMP)
    kinds, s = tr["kind"], _arr(tr["s"])
    mu, sd = _arr(tr["mu"]), _arr(tr["sd"])
    eps = float(tr.get("eps", 1e-3))
    G = mu + sd * V
    Y = np.empty_like(G)
    for j, k in enumerate(kinds):
        g = G[..., j]
        if k == "log1p":
            Y[..., j] = s[j] * np.expm1(np.minimum(g, 700.0))
        elif k == "logit":
            sg = 0.5 * (1.0 + np.tanh(0.5 * g))
            Y[..., j] = sg * (1.0 + 2.0 * eps) - eps
        else:
            Y[..., j] = s[j] * g
    return Y


# ----------------------------------------------------------------------------- rollouts
def _roll_l0a(blob, doc, y0, U, ctx):
    return np.tile(np.asarray(y0, dtype=float), (U.shape[0], 1))


def _roll_l0b(blob, doc, y0, U, ctx):
    tr = blob["tr"]
    F = phi(delay_u(norm_u(doc, U), blob.get("delays")), blob["phi"])
    Q = F @ _arr(blob["W"])                     # [T, p] equilibrium targets
    a = _arr(blob["a"])
    oma = 1.0 - a
    v = g_fwd(y0, tr)
    V = np.empty_like(Q)
    for t in range(Q.shape[0]):
        v = a * v + oma * Q[t]
        V[t] = v
    return g_inv(V, tr)


def _roll_l1(blob, doc, y0, U, ctx):
    tr = blob["tr"]
    Un = delay_u(norm_u(doc, U), blob.get("delays"))
    F = phi(Un, blob["phi"])
    W = _arr(blob["W"])                          # [K, F, p]
    a = _arr(blob["a"])                          # [K, p]
    E = _arr(blob["E"])                          # [K, p, p]
    b = _arr(blob["b"])                          # [K, p]
    c = _arr(blob["c"])                          # [p]
    K = a.shape[0]
    Q = np.einsum("tf,kfp->tkp", F, W)           # [T, K, p]
    v0 = g_fwd(y0, tr)
    z = np.einsum("kpq,q->kp", E, v0) + b        # [K, p]
    hist = blob.get("hist")
    T = Q.shape[0]
    if hist:
        ah = _arr(hist["ah"])                    # [p]
        gam = _arr(hist["gamma"])                # [K, p]
        d = np.sqrt(np.mean((norm_u(doc, U) - _arr(hist["urec"])) ** 2, axis=1))  # [T]
    oma = 1.0 - a
    V = np.empty((T, a.shape[1]))
    h = np.zeros(a.shape[1])
    for t in range(T):
        if hist:
            h = ah * h + (1.0 - ah) * d[t]
            z = a * z + oma * (1.0 + gam * h) * Q[t]
        else:
            z = a * z + oma * Q[t]
        V[t] = c + z.sum(axis=0)
    return g_inv(V, tr)


def _roll_l2(blob, doc, y0, U, ctx):
    tr = blob["tr"]
    F = phi(delay_u(norm_u(doc, U), blob.get("delays")), blob["phi"])
    A, B, C = _arr(blob["A"]), _arr(blob["B"]), _arr(blob["C"])
    D, c = _arr(blob["D"]), _arr(blob["c"])
    BF = F @ B.T                                 # [T, n]
    DF = F @ D.T + c                             # [T, p]
    if blob.get("x0_from_y0"):
        x = np.linalg.pinv(C) @ (g_fwd(y0, tr) - DF[0])
    else:
        E, e0 = _arr(blob["E"]), _arr(blob["e0"])
        x = E @ g_fwd(y0, tr) + e0
    T = F.shape[0]
    X = np.empty((T, x.size))
    for t in range(T):
        x = A @ x + BF[t]
        X[t] = x
    return g_inv(X @ C.T + DF, tr)


_MOD_CACHE = {}


def _load_file_module(name, path):
    key = (name, str(path))
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD_CACHE[key] = mod
    return mod


def ode_modules(family, base_dir=None, tag=""):
    """Locate gt_ode_core.py / gt_ode_<family>.py next to this file (submission layout),
    or gtlab/ode/{core,<family>}.py (dev layout)."""
    here = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    cands_core = [here / "gt_ode_core.py", here.parent / "ode" / "core.py"]
    cands_fam = [here / f"gt_ode_{family}.py", here.parent / "ode" / f"{family}.py"]
    core_p = next((p for p in cands_core if p.exists()), None)
    fam_p = next((p for p in cands_fam if p.exists()), None)
    if core_p is None or fam_p is None:
        raise FileNotFoundError(f"ode modules for {family} not found near {here}")
    tag = tag or family
    core = _load_file_module(f"gt_{tag}_ode_core", core_p)
    fam = _load_file_module(f"gt_{tag}_ode_{family}", fam_p)
    return core, fam


def _roll_ode(blob, doc, y0, U, ctx):
    core, fam = ode_modules(blob["family"], ctx.get("base_dir"), ctx.get("tag", ""))
    th = core.theta_dict(fam, [float(v) for v in blob["theta"]])
    return core.rollout(fam, np.asarray(y0, float), U, th, frozenset(blob["mech"]),
                        n_sub=int(blob.get("n_sub", 4)))


def _roll_ensemble(blob, doc, y0, U, ctx):
    lo, hi = ctx["lo"], ctx["hi"]
    outs = [finalize(_dispatch(m, doc, y0, U, ctx), y0, lo, hi) for m in blob["members"]]
    return np.median(np.stack(outs, axis=0), axis=0)


def _roll_perobs(blob, doc, y0, U, ctx):
    lo, hi = ctx["lo"], ctx["hi"]
    idx = [int(i) for i in blob["map"]]
    outs = {}
    Y = np.empty((U.shape[0], len(idx)))
    for j, i in enumerate(idx):
        if i not in outs:
            outs[i] = finalize(_dispatch(blob["members"][i], doc, y0, U, ctx), y0, lo, hi)
        Y[:, j] = outs[i][:, j]
    return Y


def _roll_blend(blob, doc, y0, U, ctx):
    """Shrink a member toward persistence: y = y0 + lam * (member - y0).
    lam may be a scalar or a per-observable list. lam=0 -> persistence, lam=1 -> member."""
    lo, hi = ctx["lo"], ctx["hi"]
    M = finalize(_dispatch(blob["member"], doc, y0, U, ctx), y0, lo, hi)
    P = np.broadcast_to(np.asarray(y0, float)[None, :], M.shape)
    lam = np.asarray(blob.get("lam", 1.0), float)
    return P + lam * (M - P)


_KINDS = {"l0a": _roll_l0a, "l0b": _roll_l0b, "l1": _roll_l1, "l2": _roll_l2, "ode": _roll_ode,
          "ensemble": _roll_ensemble, "perobs": _roll_perobs, "blend": _roll_blend}


def _dispatch(blob, doc, y0, U, ctx):
    return _KINDS[blob["kind"]](blob, doc, y0, U, ctx)


def rollout_from_blob(blob, y0, U, doc=None, base_dir=None, tag="", clip=True):
    """Open-loop rollout. `blob` may be a full doc (has 'model') or a model blob with `doc`.
    y0 [p] physical (doc['observables'] order); U [T, m] physical (doc['controls'] order).
    Returns clipped, finite Y [T, p]."""
    if doc is None:
        doc = blob
    if "model" in blob and "observables" in blob:
        blob = blob["model"]
    y0 = np.asarray(y0, dtype=float)
    U = np.atleast_2d(np.asarray(U, dtype=float))
    lo, hi = clip_vectors(doc)
    ctx = {"base_dir": base_dir, "tag": tag, "lo": lo, "hi": hi}
    Y = _dispatch(blob, doc, y0, U, ctx)
    Y = np.asarray(Y, dtype=float).reshape(U.shape[0], len(doc["observables"]))
    if not clip:
        return Y
    Y = finalize(Y, y0, lo, hi)
    return apply_post(doc, Y, U, y0)


def apply_post(doc, Y, U, y0=None):
    """Physical constraints that hold exactly, applied after the model (data-driven rules in
    doc['post']): le_control: obs <= control value at the same tick (e.g. spend <= budget_cap)."""
    rules = doc.get("post") or []
    if not rules:
        return Y
    obs, ctrls = list(doc["observables"]), list(doc["controls"])
    for r in rules:
        if r.get("type") == "le_control" and r.get("obs") in obs and r.get("control") in ctrls:
            j, k = obs.index(r["obs"]), ctrls.index(r["control"])
            cap = U[:, k] * float(r.get("scale", 1.0))
            if r.get("lag"):                     # observable may lag the control by one tick
                cap = np.maximum(cap, np.concatenate([cap[:1], cap[:-1]]))
            Y[:, j] = np.minimum(Y[:, j], cap)
        elif r.get("type") == "integrate" and r.get("obs") in obs:
            # obs_t = clip(obs_{t-1} + sum_i coef_i * Y[t, src_i] + bias, lo, hi): a capped stock
            j = obs.index(r["obs"])
            flow = np.full(Y.shape[0], float(r.get("bias", 0.0)))
            for name, cf in zip(r.get("src", []), r.get("coef", [])):
                if name in obs:
                    flow = flow + float(cf) * Y[:, obs.index(name)]
            lo_, hi_ = float(r.get("lo", -np.inf)), float(r.get("hi", np.inf))
            prev = float(y0[j]) if y0 is not None and np.isfinite(y0[j]) else float(Y[0, j])
            out = np.empty(Y.shape[0])
            for t in range(Y.shape[0]):
                prev = min(max(prev + flow[t], lo_), hi_)
                out[t] = prev
            Y[:, j] = out
    return Y


# ----------------------------------------------------------------------------- episode API
def _num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def build_inputs(doc, initial, interventions):
    obs, ctrls = doc["observables"], doc["controls"]
    lo, hi = clip_vectors(doc)
    y0 = np.array([_num(initial.get(o), np.nan) for o in obs], dtype=float)
    rec = doc.get("recovery", {})
    prev = [_num(rec.get(c), 0.5 * (doc["bounds"][c][0] + doc["bounds"][c][1])) for c in ctrls]
    U = np.empty((len(interventions), len(ctrls)))
    for t, a in enumerate(interventions):
        for i, c in enumerate(ctrls):
            v = _num(a.get(c), prev[i])
            b0, b1 = doc["bounds"][c]
            v = min(max(v, float(b0)), float(b1))
            U[t, i] = v
            prev[i] = v
    return y0, U


def predict_episode(doc, initial, interventions, context=None, base_dir=None, tag=""):
    """Full predict() body: returns list of dicts of python floats, keys = initial's keys."""
    if context is not None:
        fam = context.get("family")
        if fam is not None and fam != doc.get("system"):
            raise ValueError("model.json belongs to another system")
    y0, U = build_inputs(doc, initial, interventions)
    if not np.all(np.isfinite(y0)):
        raise ValueError("nonfinite initial observation")
    Y = rollout_from_blob(doc, y0, U, doc=doc, base_dir=base_dir, tag=tag)
    obs = doc["observables"]
    names = list(initial.keys())
    cols = []
    for n in names:
        if n in obs:
            cols.append(Y[:, obs.index(n)].tolist())
        else:
            v = _num(initial.get(n), 0.0)
            cols.append([v] * U.shape[0])
    return [dict(zip(names, row)) for row in zip(*cols)] if names else [{} for _ in range(U.shape[0])]
