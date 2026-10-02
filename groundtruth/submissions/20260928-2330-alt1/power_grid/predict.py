import json
import math
from pathlib import Path
import numpy as np
FORMAT = 1

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

def _roll_ode(blob, doc, y0, U, ctx):
    core, fam = ode_modules(blob['family'], ctx.get('base_dir'), ctx.get('tag', ''))
    th = core.theta_dict(fam, [float(v) for v in blob['theta']])
    return core.rollout(fam, np.asarray(y0, float), U, th, frozenset(blob['mech']), n_sub=int(blob.get('n_sub', 4)))

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
_KINDS = {'ode': _roll_ode, 'perobs': _roll_perobs}

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

def _ode_power_grid_pg9():
    import math
    import numpy as np
    FAMILY = 'power_grid_pg9'
    OBS = ['load', 'frequency', 'renewable_share']
    CTRL = ['price_signal', 'reserve_dispatch', 'charging_allowance', 'interconnector']
    MECHS = {'A': 'thermostat synchronisation (always on)', 'B': 'governor + reserve stock (always on)', 'C': 'unused'}
    N_SUB = 1
    M = 32
    K = 2
    X_LO, X_HI = (0.3, 0.95)
    DX = (X_HI - X_LO) / M
    XC = X_LO + DX * (np.arange(M) + 0.5)
    XE = X_LO + DX * np.arange(1, M)
    NP_ = K * M
    REST = ['z', 'Rs', 'fdev', 'sh', 'G', 'E', 'Th']
    STATE = [f'off{c}_{i}' for c in range(K) for i in range(M)] + [f'on{c}_{i}' for c in range(K) for i in range(M)] + REST
    _NS = len(STATE)
    STATE_LO = np.concatenate([np.zeros(2 * NP_), np.array([-80.0, 0.0, -2.5, 0.0, -80.0, 0.0, 0.0])])
    STATE_HI = np.concatenate([np.ones(2 * NP_), np.array([80.0, 150.0, 2.5, 1.0, 80.0, 1.0, 260.0])])
    IZ = 2 * NP_
    A_R = 0.6
    A_SH = 1.5
    A_TH = 0.12
    A_F = 1.0
    G_LIM = 40.0
    K_C = 80.0
    SMIN_W = 3.0
    RELAX = 300
    PARAMS = [('base', 50.9, 20.0, 100.0, False), ('P', 137.6, 40.0, 250.0, False), ('s08', 0.58, 0.35, 0.75, False), ('s1', 0.1275, 0.0, 0.5, False), ('s2', 0.008, -0.3, 0.3, False), ('w', 0.236, 0.08, 0.3, False), ('tau1', 53.2, 25.0, 400.0, True), ('tau2', 31.2, 25.0, 300.0, True), ('phi', 0.34, 0.0, 1.0, False), ('r', 0.2, 0.05, 1.2, True), ('az', 0.42, 0.02, 3.0, True), ('zk', 0.106, 0.0, 1.5, False), ('kf', 0.0384, 0.005, 0.3, True), ('gb', -9.21, -40.0, 40.0, False), ('a_g', 0.0554, 0.005, 1.0, True), ('kg', 4.59, 0.0, 200.0, False), ('c_r', 1.09, 0.0, 2.0, False), ('cap', 304.0, 20.0, 20000.0, True), ('p_ch', 86.1, 0.0, 150.0, False), ('s0', 0.365, 0.05, 0.9, False), ('k_th', 0.25, -2.0, 2.0, False), ('f_hi', 1.89, 0.5, 3.5, False), ('beta', 0.503, 0.0, 1.0, False), ('g_i', 13.7, 0.0, 80.0, False), ('i0', 0.01, 0.0, 1.0, False), ('sh_i0', 0.432, 0.0, 1.0, False), ('p_l', 21.9, 0.0, 250.0, False), ('p_r', 82.2, 0.0, 250.0, False)]

    def _p(th, n):
        return np.asarray(th[n], dtype=float)

    def _col(v):
        return np.asarray(v, dtype=float)[..., None, None]

    def _sig(z):
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def _band(price, th):
        dp = price - 0.8
        return _p(th, 's08') + _p(th, 's1') * dp + _p(th, 's2') * dp * dp

    def _pop_deriv(off, on, S, th):
        tau = np.stack(np.broadcast_arrays(_p(th, 'tau1'), _p(th, 'tau2')), axis=-1)[..., None]
        w = _col(_p(th, 'w'))
        Sc = _col(S)
        r = _col(_p(th, 'r'))
        eps = 0.5 * DX
        up = _sig((XC - (Sc + 0.5 * w)) / eps)
        lo = _sig((Sc - 0.5 * w - XC) / eps)
        Fo = (1.0 - XE) / tau * off[..., :-1] / DX
        Fn = XE / tau * on[..., 1:] / DX
        zo = np.zeros(Fo.shape[:-1] + (1,))
        d_off = np.concatenate([zo, Fo], axis=-1) - np.concatenate([Fo, zo], axis=-1)
        d_on = np.concatenate([Fn, zo], axis=-1) - np.concatenate([zo, Fn], axis=-1)
        sw = r * (up * off - lo * on)
        return (d_off - sw, d_on + sw)

    def _split(x):
        off = x[..., :NP_].reshape(x.shape[:-1] + (K, M))
        on = x[..., NP_:2 * NP_].reshape(x.shape[:-1] + (K, M))
        return (off, on)

    def _load(x, th):
        on = x[..., NP_:2 * NP_]
        return _p(th, 'base') + _p(th, 'P') * on.sum(axis=-1) + x[..., IZ]

    def _dref(th):
        return _p(th, 'base') + _p(th, 'P') * (1.0 - _p(th, 's08'))
    _POP_KEYS = ('tau1', 'tau2', 'w', 'r', 's08', 's1', 's2')
    _REST_KEYS = ('base', 'P', 's08', 'az', 'c_r', 'beta', 'i0', 'p_l', 'p_r', 'p_ch', 'cap', 'gb', 'g_i', 'kf', 'f_hi', 'kg', 'a_g', 's0', 'sh_i0', 'k_th')
    _A_CACHE = {}
    _TH_CACHE = [None, None, None]

    def _pop_matrix(price, pkey, th):
        key = (price,) + pkey
        A = _A_CACHE.get(key)
        if A is None:
            if len(_A_CACHE) > 4096:
                _A_CACHE.clear()
            E = np.eye(2 * NP_)
            off, on = _split(E)
            d_off, d_on = _pop_deriv(off, on, _band(np.full(2 * NP_, price), th), th)
            A = np.concatenate([d_off.reshape(2 * NP_, NP_), d_on.reshape(2 * NP_, NP_)], axis=-1).T
            ones = np.concatenate([np.zeros(NP_), np.ones(NP_)])[None]
            A = np.ascontiguousarray(np.concatenate([A, ones], axis=0))
            _A_CACHE[key] = A
        return A

    def _th_consts(th):
        if _TH_CACHE[0] is not th:
            _TH_CACHE[0] = th
            _TH_CACHE[1] = tuple((float(th[n]) for n in _POP_KEYS))
            _TH_CACHE[2] = tuple((float(th[n]) for n in _REST_KEYS))
        return (_TH_CACHE[1], _TH_CACHE[2])

    def _f_fast(x, u, th):
        price, res, chg, ic = u.tolist() if hasattr(u, 'tolist') else [float(v) for v in u]
        pkey, c = _th_consts(th)
        base, P, s08, az, c_r, beta, i0, p_l, p_r, p_ch, cap, gb, g_i, kf, f_hi, kg, a_g, s0, sh_i0, k_th = c
        A = _pop_matrix(price, pkey, th)
        dpop = A @ x[:2 * NP_]
        z, Rs, fdev, sh, G, E, Th = x[IZ:].tolist()
        load = base + P * float(dpop[-1]) + z
        dz = -az * z
        dRs = A_R * (res - Rs)
        Dref = base + P * (1.0 - s08)
        draw = c_r * beta * Rs * E
        icf = i0 + (1.0 - i0) * ic
        Rreq = c_r * (1.0 - beta) * Rs + draw
        Pcap = p_l + p_r * ic
        dd = Rreq - Pcap
        Reff = 0.5 * (Rreq + Pcap - math.sqrt(dd * dd + SMIN_W * SMIN_W))
        Pch = p_ch * chg * (1.0 - E)
        dE = (Pch - draw) / cap
        bal = Dref + gb + g_i * (ic - 1.0) + G + Reff - Pch * icf - load
        ft = kf * bal
        ft = min(ft, f_hi * math.tanh(ft / f_hi))
        dfdev = A_F * (ft - fdev)
        gt = max(min(-kg * fdev, G_LIM), -G_LIM)
        dG = a_g * (gt - G)
        dTh = A_TH * (load - Th)
        q = res / K_C
        sh_t = s0 * (sh_i0 + (1.0 - sh_i0) * ic) / (1.0 + q * q) * (1.0 - k_th * (Th - Dref) / 100.0)
        dsh = A_SH * (sh_t - sh)
        out = np.empty(_NS)
        out[:IZ] = dpop[:IZ]
        out[IZ:] = (dz, dRs, dfdev, dsh, dG, dE, dTh)
        return out

    def f(x, u, th, mech):
        if np.ndim(x) == 1 and np.ndim(th['tau1']) == 0:
            return _f_fast(x, u, th)
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        price, res, chg, ic = (u[..., 0], u[..., 1], u[..., 2], u[..., 3])
        off, on = _split(x)
        S = _band(price, th)
        d_off, d_on = _pop_deriv(off, on, S, th)
        z, Rs, fdev, sh, G, E, Th = (x[..., IZ + i] for i in range(7))
        g = lambda n: _p(th, n)
        load = _load(x, th)
        dz = -g('az') * z
        dRs = A_R * (res - Rs)
        Dref = _dref(th)
        draw = g('c_r') * g('beta') * Rs * E
        icf = g('i0') + (1.0 - g('i0')) * ic
        Rreq = g('c_r') * (1.0 - g('beta')) * Rs + draw
        Pcap = g('p_l') + g('p_r') * ic
        dd = Rreq - Pcap
        Reff = 0.5 * (Rreq + Pcap - np.sqrt(dd * dd + SMIN_W * SMIN_W))
        Pch = g('p_ch') * chg * (1.0 - E)
        dE = (Pch - draw) / g('cap')
        bal = Dref + g('gb') + g('g_i') * (ic - 1.0) + G + Reff - Pch * icf - load
        ft = g('kf') * bal
        fh = g('f_hi')
        ft = np.minimum(ft, fh * np.tanh(ft / fh))
        dfdev = A_F * (ft - fdev)
        gt = np.clip(-g('kg') * fdev, -G_LIM, G_LIM)
        dG = g('a_g') * (gt - G)
        dTh = A_TH * (load - Th)
        q = res / K_C
        sh_t = g('s0') * (g('sh_i0') + (1.0 - g('sh_i0')) * ic) / (1.0 + q * q) * (1.0 - g('k_th') * (Th - Dref) / 100.0)
        dsh = A_SH * (sh_t - sh)
        rest = np.stack(np.broadcast_arrays(dz, dRs, dfdev, dsh, dG, dE, dTh), axis=-1)
        lead = rest.shape[:-1]
        return np.concatenate([d_off.reshape(lead + (NP_,)), d_on.reshape(lead + (NP_,)), rest], axis=-1)

    def h(x, u, th, mech):
        x = np.asarray(x, dtype=float)
        load = _load(x, th)
        return np.stack(np.broadcast_arrays(load, 50.0 + x[..., IZ + 2], x[..., IZ + 3]), axis=-1)

    def _stationary(th, lead):
        S = np.broadcast_to(_band(np.asarray(0.8), th), lead)
        w = _col(_p(th, 'w'))
        Sc = _col(S)
        tau = np.stack(np.broadcast_arrays(_p(th, 'tau1'), _p(th, 'tau2')), axis=-1)[..., None]
        band = ((XC > Sc - 0.5 * w) & (XC < Sc + 0.5 * w)).astype(float)
        o = band * tau / (1.0 - XC)
        n = band * tau / XC
        tot = o.sum(-1, keepdims=True) + n.sum(-1, keepdims=True) + 1e-12
        phi = _p(th, 'phi')
        wts = np.stack(np.broadcast_arrays(phi, 1.0 - phi), axis=-1)[..., None]
        off = np.broadcast_to(o / tot * wts, lead + (K, M)).copy()
        on = np.broadcast_to(n / tot * wts, lead + (K, M)).copy()
        dt = 0.5
        for _ in range(2 * RELAX):
            a, b = _pop_deriv(off, on, S, th)
            off = np.maximum(off + dt * a, 0.0)
            on = np.maximum(on + dt * b, 0.0)
        return (off, on)

    def x0(y0, th, mech):
        y0 = np.asarray(y0, dtype=float)
        lead = y0.shape[:-1]
        off, on = _stationary(th, lead)
        lref = _p(th, 'base') + _p(th, 'P') * on.sum(axis=(-2, -1))
        z = _p(th, 'zk') * (y0[..., 0] - lref)
        fdev = np.clip(y0[..., 1] - 50.0, -2.5, 2.5)
        sh = np.clip(y0[..., 2], 0.0, 1.0)
        zero = 0.0 * fdev
        Th = lref + z
        rest = np.stack(np.broadcast_arrays(z, zero, fdev, sh, zero, zero + 1.0, Th), axis=-1)
        return np.concatenate([off.reshape(lead + (NP_,)), on.reshape(lead + (NP_,)), rest], axis=-1)

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7

def _ode_power_grid_w5():
    import math
    import numpy as np
    FAMILY = 'power_grid_w5'
    OBS = ['load', 'frequency', 'renewable_share']
    CTRL = ['price_signal', 'reserve_dispatch', 'charging_allowance', 'interconnector']
    MECHS = {'A': 'thermostat synchronisation (always on)', 'B': 'governor + reserve stock (always on)', 'C': 'unused'}
    N_SUB = 2
    STATE = ['Ds', 'x1', 'x2', 'Rs', 'fdev', 'sh', 'G', 'E', 'Th']
    _NS = len(STATE)
    STATE_LO = np.array([0.0, -32.0, -60.0, 0.0, -2.5, 0.0, -80.0, 0.0, 0.0])
    STATE_HI = np.array([260.0, 32.0, 60.0, 150.0, 2.5, 1.0, 80.0, 1.0, 260.0])
    A_S = 1.0
    A_R = 0.6
    A_SH = 1.5
    A_TH = 0.12
    A_F = 1.0
    G_LIM = 40.0
    K_C = 80.0
    DSAT = 3.0
    IC0 = 0.15
    SH_IC0 = 0.5
    SMIN_W = 3.0
    PARAMS = [('d0', 126.35363, 60.0, 250.0, False), ('d1', 20.89758, 0.0, 80.0, False), ('w', 0.09957, 0.03, 0.4, True), ('zeta', 0.29103, 0.02, 1.0, True), ('kick', 0.53744, 0.0, 1.0, False), ('kf', 0.04227, 0.005, 0.3, True), ('gb', -5.49506, -40.0, 40.0, False), ('a_g', 0.01826, 0.005, 1.0, True), ('kg', 10.63558, 0.0, 200.0, False), ('c_r', 1.35666, 0.0, 2.0, False), ('cap', 321.88525, 20.0, 20000.0, True), ('p_ch', 76.08126, 0.0, 150.0, False), ('s0', 0.35706, 0.05, 0.9, False), ('k_th', 0.3632, -2.0, 2.0, False), ('f_hi', 1.73317, 0.5, 3.5, False), ('beta', 0.5514, 0.0, 1.0, False), ('d2', 0.0, -20.0, 20.0, False), ('g_i', 10.0, 0.0, 80.0, False), ('i0', 0.15, 0.0, 1.0, False), ('sh_i0', 0.5, 0.0, 1.0, False), ('p_l', 35.0, 0.0, 250.0, False), ('p_r', 50.0, 0.0, 250.0, False)]

    class _PY:
        max = max
        min = min

        @staticmethod
        def tanh(z):
            return math.tanh(z)

    class _NP:
        max = np.maximum
        min = np.minimum

        @staticmethod
        def tanh(z):
            return np.tanh(z)

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

    def _demand(th, price):
        dp = price - 0.8
        return th['d0'] - th['d1'] * price + th['d2'] * dp * dp

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        price, res, chg, ic = uu
        Ds, x1, x2, Rs, fdev, sh, G, E, Th = s
        D = _demand(th, price)
        gap = D - Ds
        load = Ds + x1
        w = th['w']
        dDs = A_S * gap
        dx1 = x2
        dx2 = -2.0 * th['zeta'] * w * x2 - w * w * x1 + th['kick'] * A_S * DSAT * M.tanh(gap / DSAT)
        dRs = A_R * (res - Rs)
        Dref = _demand(th, 0.8)
        draw = th['c_r'] * th['beta'] * Rs * E
        icf = th['i0'] + (1.0 - th['i0']) * ic
        Rreq = th['c_r'] * (1.0 - th['beta']) * Rs + draw
        Pcap = th['p_l'] + th['p_r'] * ic
        dd = Rreq - Pcap
        Reff = 0.5 * (Rreq + Pcap - (dd * dd + SMIN_W * SMIN_W) ** 0.5)
        Pch = th['p_ch'] * chg * (1.0 - E)
        dE = (Pch - draw) / th['cap']
        bal = Dref + th['gb'] + th['g_i'] * (ic - 1.0) + G + Reff - Pch * icf - load
        ft = th['kf'] * bal
        fh = th['f_hi']
        ft = M.min(ft, fh * M.tanh(ft / fh))
        dfdev = A_F * (ft - fdev)
        gl = G_LIM
        gt = M.max(M.min(-th['kg'] * fdev, gl), -gl)
        dG = th['a_g'] * (gt - G)
        dTh = A_TH * (load - Th)
        q = res / K_C
        sh_t = th['s0'] * (th['sh_i0'] + (1.0 - th['sh_i0']) * ic) / (1.0 + q * q) * (1.0 - th['k_th'] * (Th - Dref) / 100.0)
        dsh = A_SH * (sh_t - sh)
        return _pack(M, [dDs, dx1, dx2, dRs, dfdev, dsh, dG, dE, dTh])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        Ds, x1, x2, Rs, fdev, sh, G, E, Th = s
        return _pack(M, [Ds + x1, 50.0 + fdev, sh])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        Ds = M.max(y[0], 0.0)
        fdev = M.max(M.min(y[1] - 50.0, 2.5), -2.5)
        sh = M.max(M.min(y[2], 1.0), 0.0)
        z = 0.0 * Ds
        return _pack(M, [Ds, z, z, z, fdev, sh, z, z + 1.0, Ds])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7
_ODE_BUILD = {'power_grid_pg9': _ode_power_grid_pg9, 'power_grid_w5': _ode_power_grid_w5}
_ODE_MODS = {}

def ode_modules(family, base_dir=None, tag=''):
    if family not in _ODE_MODS:
        _ODE_MODS[family] = (_ode_core(), _ODE_BUILD[family]())
    return _ODE_MODS[family]
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
