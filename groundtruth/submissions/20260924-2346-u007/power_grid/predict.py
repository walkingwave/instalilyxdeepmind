import json
import math
from pathlib import Path
import numpy as np
FORMAT = 1
_VCLAMP = 60.0

def _resolve(obj, arrays):
    if isinstance(obj, str) and obj.startswith('npz:') and (arrays is not None):
        return np.asarray(arrays[obj[4:]])
    if isinstance(obj, dict):
        return {k: _resolve(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, arrays) for v in obj]
    return obj

def load_doc(folder):
    folder = Path(folder)
    doc = json.loads((folder / 'model.json').read_text())
    npz = folder / 'model.npz'
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        doc = _resolve(doc, arrays)
    if int(doc.get('format', 1)) != FORMAT:
        raise ValueError('unsupported model.json format')
    return doc

def _arr(x, dtype=float):
    return np.asarray(x, dtype=dtype)

def _bound_vec(vals, default):
    return np.array([default if v is None else float(v) for v in vals], dtype=float)

def clip_vectors(doc):
    p = len(doc['observables'])
    hlo = _bound_vec(doc.get('hard_lo', [0.0] * p), -np.inf)
    hhi = _bound_vec(doc.get('hard_hi', [None] * p), np.inf)
    clo = _bound_vec(doc.get('clip_lo', [None] * p), -np.inf)
    chi = _bound_vec(doc.get('clip_hi', [None] * p), np.inf)
    lo = np.maximum(hlo, clo)
    hi = np.minimum(hhi, chi)
    hi = np.maximum(hi, lo)
    return (lo, hi)

def finalize(Y, y0, lo, hi):
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

def norm_u(doc, U):
    lo = np.array([doc['bounds'][c][0] for c in doc['controls']], dtype=float)
    hi = np.array([doc['bounds'][c][1] for c in doc['controls']], dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (np.asarray(U, dtype=float) - lo) / span

def delay_u(Un, delays):
    if not delays or not any((int(d) for d in delays)):
        return Un
    T = Un.shape[0]
    out = np.empty_like(Un)
    for i, d in enumerate(delays):
        d = int(d)
        idx = np.maximum(np.arange(T) - d, 0)
        out[:, i] = Un[idx, i]
    return out

def phi(Un, ps):
    Un = np.asarray(Un, dtype=float)
    cols = [Un]
    if ps.get('sq', True):
        cols.append(Un * Un)
    for i, j in ps.get('pairs', []):
        cols.append((Un[:, int(i)] * Un[:, int(j)])[:, None])
    if ps.get('const', True):
        cols.append(np.ones((Un.shape[0], 1)))
    return np.concatenate(cols, axis=1)

def g_fwd(Y, tr):
    Y = np.asarray(Y, dtype=float)
    kinds, s = (tr['kind'], _arr(tr['s']))
    mu, sd = (_arr(tr['mu']), _arr(tr['sd']))
    eps = float(tr.get('eps', 0.001))
    G = np.empty_like(Y)
    for j, k in enumerate(kinds):
        y = Y[..., j]
        if k == 'log1p':
            G[..., j] = np.log1p(np.maximum(y, 0.0) / s[j])
        elif k == 'logit':
            yc = np.clip(y, 0.0, 1.0)
            G[..., j] = np.log((yc + eps) / (1.0 - yc + eps))
        else:
            G[..., j] = y / s[j]
    return (G - mu) / sd

def g_inv(V, tr):
    V = np.clip(np.asarray(V, dtype=float), -_VCLAMP, _VCLAMP)
    kinds, s = (tr['kind'], _arr(tr['s']))
    mu, sd = (_arr(tr['mu']), _arr(tr['sd']))
    eps = float(tr.get('eps', 0.001))
    G = mu + sd * V
    Y = np.empty_like(G)
    for j, k in enumerate(kinds):
        g = G[..., j]
        if k == 'log1p':
            Y[..., j] = s[j] * np.expm1(np.minimum(g, 700.0))
        elif k == 'logit':
            sg = 0.5 * (1.0 + np.tanh(0.5 * g))
            Y[..., j] = sg * (1.0 + 2.0 * eps) - eps
        else:
            Y[..., j] = s[j] * g
    return Y

def _roll_l0b(blob, doc, y0, U, ctx):
    tr = blob['tr']
    F = phi(delay_u(norm_u(doc, U), blob.get('delays')), blob['phi'])
    Q = F @ _arr(blob['W'])
    if blob.get('q_lo') is not None:
        Q = np.clip(Q, _arr(blob['q_lo'])[None, :], _arr(blob['q_hi'])[None, :])
    a = _arr(blob['a'])
    oma = 1.0 - a
    v = g_fwd(y0, tr)
    V = np.empty_like(Q)
    for t in range(Q.shape[0]):
        v = a * v + oma * Q[t]
        V[t] = v
    return g_inv(V, tr)
_KINDS = {'l0b': _roll_l0b}

def _dispatch(blob, doc, y0, U, ctx):
    return _KINDS[blob['kind']](blob, doc, y0, U, ctx)

def rollout_from_blob(blob, y0, U, doc=None, base_dir=None, tag='', clip=True):
    if doc is None:
        doc = blob
    if 'model' in blob and 'observables' in blob:
        blob = blob['model']
    y0 = np.asarray(y0, dtype=float)
    U = np.atleast_2d(np.asarray(U, dtype=float))
    lo, hi = clip_vectors(doc)
    ctx = {'base_dir': base_dir, 'tag': tag, 'lo': lo, 'hi': hi}
    Y = _dispatch(blob, doc, y0, U, ctx)
    Y = np.asarray(Y, dtype=float).reshape(U.shape[0], len(doc['observables']))
    if not clip:
        return Y
    Y = finalize(Y, y0, lo, hi)
    return apply_post(doc, Y, U, y0)

def apply_post(doc, Y, U, y0=None):
    rules = doc.get('post') or []
    if not rules:
        return Y
    obs, ctrls = (list(doc['observables']), list(doc['controls']))
    for r in rules:
        if r.get('type') == 'le_control' and r.get('obs') in obs and (r.get('control') in ctrls):
            j, k = (obs.index(r['obs']), ctrls.index(r['control']))
            cap = U[:, k] * float(r.get('scale', 1.0))
            if r.get('lag'):
                cap = np.maximum(cap, np.concatenate([cap[:1], cap[:-1]]))
            Y[:, j] = np.minimum(Y[:, j], cap)
        elif r.get('type') == 'exo_harmonic' and r.get('obs') in obs:
            j = obs.index(r['obs'])
            t = np.arange(Y.shape[0], dtype=float)
            cf = [float(v) for v in r.get('coef', [])]
            P = float(r['period'])
            out = np.full(Y.shape[0], cf[0] if cf else 0.0)
            for h in range((len(cf) - 1) // 2):
                w = 2.0 * np.pi * (h + 1) * t / P
                out = out + cf[1 + 2 * h] * np.sin(w) + cf[2 + 2 * h] * np.cos(w)
            Y[:, j] = out
        elif r.get('type') == 'integrate' and r.get('obs') in obs:
            j = obs.index(r['obs'])
            flow = np.full(Y.shape[0], float(r.get('bias', 0.0)))
            for name, cf in zip(r.get('src', []), r.get('coef', [])):
                if name in obs:
                    flow = flow + float(cf) * Y[:, obs.index(name)]
            lo_, hi_ = (float(r.get('lo', -np.inf)), float(r.get('hi', np.inf)))
            prev = float(y0[j]) if y0 is not None and np.isfinite(y0[j]) else float(Y[0, j])
            out = np.empty(Y.shape[0])
            for t in range(Y.shape[0]):
                prev = min(max(prev + flow[t], lo_), hi_)
                out[t] = prev
            Y[:, j] = out
    return Y

def _num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default

def build_inputs(doc, initial, interventions):
    obs, ctrls = (doc['observables'], doc['controls'])
    lo, hi = clip_vectors(doc)
    y0 = np.array([_num(initial.get(o), np.nan) for o in obs], dtype=float)
    rec = doc.get('recovery', {})
    prev = [_num(rec.get(c), 0.5 * (doc['bounds'][c][0] + doc['bounds'][c][1])) for c in ctrls]
    U = np.empty((len(interventions), len(ctrls)))
    for t, a in enumerate(interventions):
        for i, c in enumerate(ctrls):
            v = _num(a.get(c), prev[i])
            b0, b1 = doc['bounds'][c]
            v = min(max(v, float(b0)), float(b1))
            U[t, i] = v
            prev[i] = v
    return (y0, U)

def predict_episode(doc, initial, interventions, context=None, base_dir=None, tag=''):
    if context is not None:
        fam = context.get('family')
        if fam is not None and fam != doc.get('system'):
            raise ValueError('model.json belongs to another system')
    y0, U = build_inputs(doc, initial, interventions)
    if not np.all(np.isfinite(y0)):
        raise ValueError('nonfinite initial observation')
    Y = rollout_from_blob(doc, y0, U, doc=doc, base_dir=base_dir, tag=tag)
    obs = doc['observables']
    names = list(initial.keys())
    cols = []
    for n in names:
        if n in obs:
            cols.append(Y[:, obs.index(n)].tolist())
        else:
            v = _num(initial.get(n), 0.0)
            cols.append([v] * U.shape[0])
    return [dict(zip(names, row)) for row in zip(*cols)] if names else [{} for _ in range(U.shape[0])]
SYSTEM = 'power_grid'
HARD = {'load': [0.0, None], 'frequency': [0.0, None], 'renewable_share': [0.0, 1.0]}
_HERE = Path(__file__).resolve().parent
_DOC = {}

def _get_doc():
    if 'doc' not in _DOC:
        _DOC['doc'] = load_doc(_HERE)
    return _DOC['doc']

def _fnum(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0

def _persistence(initial, n, names):
    row = {}
    for k in names:
        v = _fnum(initial.get(k)) if isinstance(initial, dict) else 0.0
        lo, hi = HARD.get(k, (0.0, None))
        if lo is not None and v < lo:
            v = float(lo)
        if hi is not None and v > hi:
            v = float(hi)
        row[k] = v
    return [dict(row) for _ in range(n)]

def _names(initial, context):
    names = list(initial.keys()) if isinstance(initial, dict) else []
    try:
        for k in context.get('observables', []) or []:
            if k not in names:
                names.append(k)
    except Exception:
        pass
    return names

def _valid(out, n, names):
    if not isinstance(out, list) or len(out) != n:
        return False
    for row in out:
        if not isinstance(row, dict) or len(row) != len(names):
            return False
        for k in names:
            v = row.get(k)
            if not isinstance(v, float) or not math.isfinite(v):
                return False
    return True

def predict(initial, interventions, context):
    try:
        n = len(interventions)
    except Exception:
        n = 4000
    try:
        names = _names(initial, context)
    except Exception:
        names = []
    try:
        doc = _get_doc()
        init = {k: initial.get(k, 0.0) for k in names}
        out = predict_episode(doc, init, interventions, context, base_dir=_HERE, tag=SYSTEM)
        if _valid(out, n, names):
            return out
    except Exception:
        pass
    return _persistence(initial, n, names)
