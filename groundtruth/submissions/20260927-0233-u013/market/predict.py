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

def _roll_ode(blob, doc, y0, U, ctx):
    core, fam = ode_modules(blob['family'], ctx.get('base_dir'), ctx.get('tag', ''))
    th = core.theta_dict(fam, [float(v) for v in blob['theta']])
    return core.rollout(fam, np.asarray(y0, float), U, th, frozenset(blob['mech']), n_sub=int(blob.get('n_sub', 4)))

def _roll_ensemble(blob, doc, y0, U, ctx):
    lo, hi = (ctx['lo'], ctx['hi'])
    outs = [finalize(_dispatch(m, doc, y0, U, ctx), y0, lo, hi) for m in blob['members']]
    return np.median(np.stack(outs, axis=0), axis=0)

def _roll_perobs(blob, doc, y0, U, ctx):
    lo, hi = (ctx['lo'], ctx['hi'])
    idx = [int(i) for i in blob['map']]
    outs = {}
    Y = np.empty((U.shape[0], len(idx)))
    for j, i in enumerate(idx):
        if i not in outs:
            outs[i] = finalize(_dispatch(blob['members'][i], doc, y0, U, ctx), y0, lo, hi)
        Y[:, j] = outs[i][:, j]
    return Y
_KINDS = {'l0b': _roll_l0b, 'ode': _roll_ode, 'ensemble': _roll_ensemble, 'perobs': _roll_perobs}

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
        elif r.get('type') == 'le_const' and r.get('obs') in obs:
            Y[:, obs.index(r['obs'])] = np.minimum(Y[:, obs.index(r['obs'])], float(r['value']))
        elif r.get('type') == 'ge_const' and r.get('obs') in obs:
            Y[:, obs.index(r['obs'])] = np.maximum(Y[:, obs.index(r['obs'])], float(r['value']))
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

def _ode_core():
    import numpy as np

    def theta_dict(mod, vec=None):
        names = [p[0] for p in mod.PARAMS]
        if vec is None:
            vec = [p[1] for p in mod.PARAMS]
        return dict(zip(names, (float(v) for v in vec)))

    def _clip(mod, x):
        lo = getattr(mod, 'STATE_LO', None)
        hi = getattr(mod, 'STATE_HI', None)
        x = np.maximum(x, 0.0 if lo is None else lo)
        if hi is not None:
            x = np.minimum(x, hi)
        return x

    def rollout(mod, y0, U, th, mech, n_sub=4, return_states=False):
        y0 = np.asarray(y0, dtype=float)
        U = np.asarray(U, dtype=float)
        mech = frozenset(mech)
        x = _clip(mod, np.asarray(mod.x0(y0, th, mech), dtype=float))
        T = U.shape[0]
        Y = np.empty((T, len(mod.OBS)))
        X = np.empty((T, x.size)) if return_states else None
        dt = 1.0 / n_sub
        f = mod.f
        for t in range(T):
            u = U[t]
            for _ in range(n_sub):
                k1 = f(x, u, th, mech)
                k2 = f(_clip(mod, x + 0.5 * dt * k1), u, th, mech)
                k3 = f(_clip(mod, x + 0.5 * dt * k2), u, th, mech)
                k4 = f(_clip(mod, x + dt * k3), u, th, mech)
                x = _clip(mod, x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=0.0)
            Y[t] = mod.h(x, u, th, mech)
            if return_states:
                X[t] = x
        return (Y, X) if return_states else Y

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7

def _ode_market_x4():
    import math
    import numpy as np
    FAMILY = 'market_x4'
    OBS = ['price', 'volume', 'depth']
    CTRL = ['interest_rate', 'transaction_tax']
    MECHS = {'A': 'settlement tie-up (burst inventory H)', 'B': 'risk capacity loss R', 'C': 'momentum (price follower)'}
    N_SUB = 2
    STATE = ['B', 'P', 'v', 'C', 'D', 'R', 'H', 'Q', 'W', 'G', 'A', 'S']
    STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    STATE_HI = np.array([10000.0, 10000.0, 20.0, 1.0, 10000.0, 0.9, 200.0, 10000.0, 1.0, 200.0, 10000.0, 1.0])
    TAU_B = 2.7
    TAU_R = 20.0
    W_X = 0.0005
    C_G = 0.0184
    V0 = 1.77
    C_W = 1.139
    TAU_W = 70.71
    TAU_G = 3.0
    W_R = 0.004
    TAU_D = 8.0
    C_H = 0.0759
    TAU_H1 = 9.57
    A_H = 38.2
    W_RR = 0.7
    PARAMS = [('c_p', 1.1276, 0.01, 10.0, True), ('d0', 102.0482, 60.0, 130.0, False), ('m1', 0.5741, 0.0, 0.95, False), ('x_d', 0.008, 0.002, 0.035, False), ('w_d', 0.005, 0.0005, 0.02, True), ('m2', 0.4941, 0.0, 0.95, False), ('k_R', 0.8822, 0.0, 2.0, False), ('x_R', 0.0309, 0.0, 0.05, False), ('g_l', 0.02, 0.0, 0.3, False), ('x_c', 0.0446, 0.0426, 0.0458, False), ('p_full', 91.0, 89.0, 99.0, False), ('tau_Q', 312.9812, 10.0, 2000.0, True), ('p_lo', 74.5827, 55.0, 90.0, False), ('rho', 0.0737, 0.001, 0.5, True), ('c0', 0.1648, 0.001, 2.0, True), ('a_r', 1.5663, 0.001, 2.0, True), ('n_r', 3.9288, 0.5, 10.0, False), ('tau_A', 12.2581, 1.0, 100.0, True), ('tau_Au', 15.1205, 2.0, 300.0, True), ('tau_s', 1.5, 1.0, 100.0, True), ('b0', 0.036, 0.01, 3.0, True), ('kap', 0.1983, 0.005, 2.0, True), ('s_v', 0.4509, 0.05, 5.0, True), ('tau_v', 8.7092, 0.5, 60.0, True)]

    class _PY:
        max = max
        min = min

        @staticmethod
        def exp(z):
            return math.exp(z if z < 50.0 else 50.0)

        @staticmethod
        def sqrt(z):
            return math.sqrt(z) if z > 0.0 else 0.0

        @staticmethod
        def pw(a, b):
            return (a if a > 1e-12 else 1e-12) ** b

    class _NP:
        max = np.maximum
        min = np.minimum

        @staticmethod
        def exp(z):
            return np.exp(np.minimum(z, 50.0))

        @staticmethod
        def sqrt(z):
            return np.sqrt(np.maximum(z, 0.0))

        @staticmethod
        def pw(a, b):
            return np.power(np.maximum(a, 1e-12), b)

    def _unpack(x, u):
        if np.ndim(x) == 1:
            return (_PY, x.tolist(), np.asarray(u, dtype=float).tolist())
        return (_NP, [x[..., i] for i in range(x.shape[-1])], [u[..., j] for j in range(u.shape[-1])])

    def _pack(M, vals):
        if M is _PY:
            return np.array(vals, dtype=float)
        return np.stack(np.broadcast_arrays(*vals), axis=-1).astype(float)

    def _yunpack(y0):
        if np.ndim(y0) == 1:
            return (_PY, np.asarray(y0, dtype=float).tolist())
        return (_NP, [y0[..., j] for j in range(y0.shape[-1])])

    def _gate(M, tax, th):
        return 1.0 / (1.0 + M.exp((tax - th['x_c']) / W_X))

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        rate, tax = uu
        B, P, v, C, D, R, H, Q, W, G, A, S = s
        dB = -B / TAU_B
        g = _gate(M, tax, th)
        rr = M.pw(rate / 0.1, th['n_r'])
        dC = th['rho'] * g * (C + th['c0']) * (1.0 - C) - th['a_r'] * rr * (1.0 - C + th['b0'])
        ea = th['p_lo'] + (Q - th['p_lo']) * C - A
        su = 0.5 * (1.0 + ea / M.sqrt(ea * ea + 1.0))
        dA = ea * ((1.0 - su) / th['tau_A'] + su / th['tau_Au'])
        dS = ((1.0 - g) * rate * 10.0 - S) / th['tau_s']
        dQ = (th['p_full'] - Q) / th['tau_Q']
        zv = th['kap'] * (A - P) / th['s_v']
        zv = M.min(M.max(zv, -20.0), 20.0)
        ez = M.exp(2.0 * zv)
        dv = ((th['g_l'] + (1.0 - th['g_l']) * g) * th['s_v'] * (ez - 1.0) / (ez + 1.0) - v) / th['tau_v']
        dP = v
        fall = 0.5 * (M.sqrt(dP * dP + 0.01) - dP)
        sR = 1.0 / (1.0 + M.exp((tax - th['x_R']) / W_R))
        dR = (th['k_R'] * fall * (1.0 - W_RR + W_RR * rr / (1.0 + rr)) * sR - R) / TAU_R
        dG = C_H * B - G / TAU_G
        dH = G / TAU_G - H * M.exp(-A_H * tax) / TAU_H1
        sd = 1.0 / (1.0 + M.exp(-(tax - th['x_d']) / th['w_d']))
        Dstar = th['d0'] * (1.0 - th['m1'] * sd) * (1.0 - th['m2'] * S)
        dD = (Dstar - D) / TAU_D
        return _pack(M, [dB, dP, dv, dC, dD, dR, dH, dQ, -W / TAU_W, dG, dA, dS])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        B, P, v, C, D, R, H, Q, W, G, A, S = s
        rate, tax = uu
        g = _gate(M, tax, th)
        gap = A - P
        vol = V0 + B + C_G * M.sqrt(gap * gap + 0.25) + th['c_p'] * g * M.sqrt(v * v + 0.0001) + C_W * W * g * (rate * 10.0 + tax * 20.0)
        dep = D * (1.0 - R) + H
        return _pack(M, [P, vol, dep])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        P = M.max(y[0], 0.0)
        B = M.max(y[1] - V0 - 0.5 * C_G, 0.0)
        D = M.max(y[2], 0.0)
        z = 0.0 * P
        return _pack(M, [B, P, z, z + 1.0, D, z, z, P, z + 1.0, z, P, z])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7
_ODE_BUILD = {'market_x4': _ode_market_x4}
_ODE_MODS = {}

def ode_modules(family, base_dir=None, tag=''):
    if family not in _ODE_MODS:
        _ODE_MODS[family] = (_ode_core(), _ODE_BUILD[family]())
    return _ODE_MODS[family]
SYSTEM = 'market'
HARD = {'price': [0.0, None], 'volume': [0.0, None], 'depth': [0.0, None]}
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
