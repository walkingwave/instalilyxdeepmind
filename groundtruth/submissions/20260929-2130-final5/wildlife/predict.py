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

def _roll_ode(blob, doc, y0, U, ctx):
    core, fam = ode_modules(blob['family'], ctx.get('base_dir'), ctx.get('tag', ''))
    th = core.theta_dict(fam, [float(v) for v in blob['theta']])
    return core.rollout(fam, np.asarray(y0, float), U, th, frozenset(blob['mech']), n_sub=int(blob.get('n_sub', 4)))

def _roll_ensemble(blob, doc, y0, U, ctx):
    lo, hi = (ctx['lo'], ctx['hi'])
    outs = [finalize(_dispatch(m, doc, y0, U, ctx), y0, lo, hi) for m in blob['members']]
    return np.median(np.stack(outs, axis=0), axis=0)
_KINDS = {'ode': _roll_ode, 'ensemble': _roll_ensemble}

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

def _ode_wildlife_tmp():
    import math
    import numpy as np
    FAMILY = 'wildlife_tmp'
    FLAGS = ['pd']
    OBS = ['prey_north', 'predator_north', 'prey_south', 'predator_south']
    CTRL = ['hunting_quota', 'habitat_protection', 'corridor_access']
    MECHS = {'A': 'juvenile condition: nursery food competition caps recruitment', 'B': 'finite food renewal: grazing depletes a slowly renewing resource', 'C': 'settlement competition: arrivals settle less where the destination is crowded'}
    N_SUB = 2
    D0 = 0.7
    STATE = ['PN', 'RN', 'QN', 'PS', 'RS', 'QS', 'TNS', 'TSN', 'VNS', 'VSN', 'GN', 'GS', 'SN', 'SS']
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([10000.0, 1.0, 1000.0, 10000.0, 1.0, 1000.0, 10000.0, 10000.0, 1000.0, 1000.0, 100.0, 100.0, 1.0, 1.0])
    PARAMS = [('r', 0.8808542347385102, 0.05, 3.0, True), ('ks', 0.7975559776432426, 0.3, 1.5, False), ('H', 1.3679210929831758, 0.01, 50.0, True), ('Ph', 25.10751258551428, 0.5, 500.0, True), ('em', 0.022415886831882822, 0.0005, 0.5, True), ('emS', 0.03480665493839356, 0.0005, 0.5, True), ('tau', 88.06126139556702, 0.5, 150.0, True), ('sv', 0.9009507418792193, 0.0, 1.0, False), ('cJ', 898.4621785824578, 5.0, 5000.0, True), ('w', 0.025934196891793958, 0.002, 1.5, True), ('kR', 2.3528179848824946e-05, 1e-06, 0.02, True), ('bh', 0.40625352293732203, 0.0, 1.0, False), ('dhN', 0.047178310123975935, 0.0, 1.0, False), ('dhS', 0.030439676088799085, 0.0, 1.0, False), ('q0', 1.7679310288109504, 0.0, 10.0, False), ('q1', 1.128539203225617, 0.0, 10.0, False), ('cq', 0.10132402353824638, 0.002, 2.0, True), ('tz', 19.622238239382696, 0.5, 300.0, True), ('emq', 0.029743134949109507, 0.0005, 0.5, True), ('tauq', 18.1073226714383, 0.5, 150.0, True), ('svq', 0.8056096865981848, 0.0, 1.0, False), ('Kc', 200.0, 2.0, 100000.0, True), ('Kq', 20.0, 0.2, 10000.0, True), ('ap', 0.5, 0.001, 20.0, True), ('ts', 30.0, 1.0, 500.0, True), ('xh', 0.5, 0.0, 1.0, False), ('s0', 0.5, 0.0, 1.0, False)]

    class _PY:
        max = max
        min = min

    class _NP:
        max = np.maximum
        min = np.minimum

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

    def _region(M, th, mech, P, R, Q, G, Sh, sc, hunt, hab, cor, dh, em, arrP, arrQ):
        B = 'B' in mech
        F = R if B else 1.0
        b = th['r'] * F * P
        rec = b / (1.0 + P / (th['cJ'] * sc)) if 'A' in mech else b
        P2 = P * P
        Ph = th['Ph']
        harv = th['H'] * hunt * P2 / (P2 + Ph * Ph)
        if 'sh' in FLAGS:
            harv = harv * (1.0 - th['xh'] * Sh)
        dS = (hab - Sh) / th['ts']
        Z = P / (P + 100.0 * sc)
        setP = 1.0
        setQ = 1.0
        if 'C' in mech:
            setP = 1.0 / (1.0 + P / (th['Kc'] * sc))
            if 'cq' in FLAGS:
                setQ = 1.0 / (1.0 + Q / (th['Kq'] * sc))
        dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th['sv'] * setP * arrP
        if 'pd' in FLAGS:
            dP = dP - th['ap'] * Q * Z
        if B:
            dR = th['w'] * (1.0 - th['bh'] * (1.0 - hab)) * (1.0 - R) - th['kR'] * (P / sc) * R
        else:
            dR = 1.0 - R
        dG = (Z - G) / th['tz']
        outQ = th['emq'] * cor * Q
        dQ = th['cq'] * (th['q0'] + th['q1'] * G - Q) * Q / (Q + 4.0) - outQ + th['svq'] * setQ * arrQ
        return (dP, dR, dQ, dG, outQ, dS)

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        hunt, hab, cor = uu
        PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS, SN, SS = s
        k = 1.0 / th['tau']
        kq = 1.0 / th['tauq']
        dPN, dRN, dQN, dGN, oQN, dSN = _region(M, th, mech, PN, RN, QN, GN, SN, 1.0, hunt, hab, cor, th['dhN'], th['em'], k * TSN, kq * VSN)
        dPS, dRS, dQS, dGS, oQS, dSS = _region(M, th, mech, PS, RS, QS, GS, SS, th['ks'], hunt, hab, cor, th['dhS'], th['emS'], k * TNS, kq * VNS)
        dTNS = th['em'] * cor * PN - k * TNS
        dTSN = th['emS'] * cor * PS - k * TSN
        dVNS = oQN - kq * VNS
        dVSN = oQS - kq * VSN
        return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS, dSN, dSS])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        return _pack(M, [s[0], s[2], s[3], s[5]])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        PN = M.max(y[0], 0.0)
        QN = M.max(y[1], 0.01)
        PS = M.max(y[2], 0.0)
        QS = M.max(y[3], 0.01)
        z = 0.0 * PN
        GN = PN / (PN + 100.0)
        GS = PS / (PS + 100.0 * th['ks'])
        S0 = th['s0'] + z
        return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS, S0, S0])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7

def _ode_wildlife_tmq():
    import math
    import numpy as np
    FAMILY = 'wildlife_tmq'
    FLAGS = ['cq']
    OBS = ['prey_north', 'predator_north', 'prey_south', 'predator_south']
    CTRL = ['hunting_quota', 'habitat_protection', 'corridor_access']
    MECHS = {'A': 'juvenile condition: nursery food competition caps recruitment', 'B': 'finite food renewal: grazing depletes a slowly renewing resource', 'C': 'settlement competition: arrivals settle less where the destination is crowded'}
    N_SUB = 2
    D0 = 0.7
    STATE = ['PN', 'RN', 'QN', 'PS', 'RS', 'QS', 'TNS', 'TSN', 'VNS', 'VSN', 'GN', 'GS', 'SN', 'SS']
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([10000.0, 1.0, 1000.0, 10000.0, 1.0, 1000.0, 10000.0, 10000.0, 1000.0, 1000.0, 100.0, 100.0, 1.0, 1.0])
    PARAMS = [('r', 0.8808542347385102, 0.05, 3.0, True), ('ks', 0.7975559776432426, 0.3, 1.5, False), ('H', 1.3679210929831758, 0.01, 50.0, True), ('Ph', 25.10751258551428, 0.5, 500.0, True), ('em', 0.022415886831882822, 0.0005, 0.5, True), ('emS', 0.03480665493839356, 0.0005, 0.5, True), ('tau', 88.06126139556702, 0.5, 150.0, True), ('sv', 0.9009507418792193, 0.0, 1.0, False), ('cJ', 898.4621785824578, 5.0, 5000.0, True), ('w', 0.025934196891793958, 0.002, 1.5, True), ('kR', 2.3528179848824946e-05, 1e-06, 0.02, True), ('bh', 0.40625352293732203, 0.0, 1.0, False), ('dhN', 0.047178310123975935, 0.0, 1.0, False), ('dhS', 0.030439676088799085, 0.0, 1.0, False), ('q0', 1.7679310288109504, 0.0, 10.0, False), ('q1', 1.128539203225617, 0.0, 10.0, False), ('cq', 0.10132402353824638, 0.002, 2.0, True), ('tz', 19.622238239382696, 0.5, 300.0, True), ('emq', 0.029743134949109507, 0.0005, 0.5, True), ('tauq', 18.1073226714383, 0.5, 150.0, True), ('svq', 0.8056096865981848, 0.0, 1.0, False), ('Kc', 200.0, 2.0, 100000.0, True), ('Kq', 20.0, 0.2, 10000.0, True), ('ap', 0.5, 0.001, 20.0, True), ('ts', 30.0, 1.0, 500.0, True), ('xh', 0.5, 0.0, 1.0, False), ('s0', 0.5, 0.0, 1.0, False)]

    class _PY:
        max = max
        min = min

    class _NP:
        max = np.maximum
        min = np.minimum

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

    def _region(M, th, mech, P, R, Q, G, Sh, sc, hunt, hab, cor, dh, em, arrP, arrQ):
        B = 'B' in mech
        F = R if B else 1.0
        b = th['r'] * F * P
        rec = b / (1.0 + P / (th['cJ'] * sc)) if 'A' in mech else b
        P2 = P * P
        Ph = th['Ph']
        harv = th['H'] * hunt * P2 / (P2 + Ph * Ph)
        if 'sh' in FLAGS:
            harv = harv * (1.0 - th['xh'] * Sh)
        dS = (hab - Sh) / th['ts']
        Z = P / (P + 100.0 * sc)
        setP = 1.0
        setQ = 1.0
        if 'C' in mech:
            setP = 1.0 / (1.0 + P / (th['Kc'] * sc))
            if 'cq' in FLAGS:
                setQ = 1.0 / (1.0 + Q / (th['Kq'] * sc))
        dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th['sv'] * setP * arrP
        if 'pd' in FLAGS:
            dP = dP - th['ap'] * Q * Z
        if B:
            dR = th['w'] * (1.0 - th['bh'] * (1.0 - hab)) * (1.0 - R) - th['kR'] * (P / sc) * R
        else:
            dR = 1.0 - R
        dG = (Z - G) / th['tz']
        outQ = th['emq'] * cor * Q
        dQ = th['cq'] * (th['q0'] + th['q1'] * G - Q) * Q / (Q + 4.0) - outQ + th['svq'] * setQ * arrQ
        return (dP, dR, dQ, dG, outQ, dS)

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        hunt, hab, cor = uu
        PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS, SN, SS = s
        k = 1.0 / th['tau']
        kq = 1.0 / th['tauq']
        dPN, dRN, dQN, dGN, oQN, dSN = _region(M, th, mech, PN, RN, QN, GN, SN, 1.0, hunt, hab, cor, th['dhN'], th['em'], k * TSN, kq * VSN)
        dPS, dRS, dQS, dGS, oQS, dSS = _region(M, th, mech, PS, RS, QS, GS, SS, th['ks'], hunt, hab, cor, th['dhS'], th['emS'], k * TNS, kq * VNS)
        dTNS = th['em'] * cor * PN - k * TNS
        dTSN = th['emS'] * cor * PS - k * TSN
        dVNS = oQN - kq * VNS
        dVSN = oQS - kq * VSN
        return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS, dSN, dSS])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        return _pack(M, [s[0], s[2], s[3], s[5]])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        PN = M.max(y[0], 0.0)
        QN = M.max(y[1], 0.01)
        PS = M.max(y[2], 0.0)
        QS = M.max(y[3], 0.01)
        z = 0.0 * PN
        GN = PN / (PN + 100.0)
        GS = PS / (PS + 100.0 * th['ks'])
        S0 = th['s0'] + z
        return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS, S0, S0])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7

def _ode_wildlife_tms():
    import math
    import numpy as np
    FAMILY = 'wildlife_tms'
    FLAGS = ['sh']
    OBS = ['prey_north', 'predator_north', 'prey_south', 'predator_south']
    CTRL = ['hunting_quota', 'habitat_protection', 'corridor_access']
    MECHS = {'A': 'juvenile condition: nursery food competition caps recruitment', 'B': 'finite food renewal: grazing depletes a slowly renewing resource', 'C': 'settlement competition: arrivals settle less where the destination is crowded'}
    N_SUB = 2
    D0 = 0.7
    STATE = ['PN', 'RN', 'QN', 'PS', 'RS', 'QS', 'TNS', 'TSN', 'VNS', 'VSN', 'GN', 'GS', 'SN', 'SS']
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([10000.0, 1.0, 1000.0, 10000.0, 1.0, 1000.0, 10000.0, 10000.0, 1000.0, 1000.0, 100.0, 100.0, 1.0, 1.0])
    PARAMS = [('r', 0.8808542347385102, 0.05, 3.0, True), ('ks', 0.7975559776432426, 0.3, 1.5, False), ('H', 1.3679210929831758, 0.01, 50.0, True), ('Ph', 25.10751258551428, 0.5, 500.0, True), ('em', 0.022415886831882822, 0.0005, 0.5, True), ('emS', 0.03480665493839356, 0.0005, 0.5, True), ('tau', 88.06126139556702, 0.5, 150.0, True), ('sv', 0.9009507418792193, 0.0, 1.0, False), ('cJ', 898.4621785824578, 5.0, 5000.0, True), ('w', 0.025934196891793958, 0.002, 1.5, True), ('kR', 2.3528179848824946e-05, 1e-06, 0.02, True), ('bh', 0.40625352293732203, 0.0, 1.0, False), ('dhN', 0.047178310123975935, 0.0, 1.0, False), ('dhS', 0.030439676088799085, 0.0, 1.0, False), ('q0', 1.7679310288109504, 0.0, 10.0, False), ('q1', 1.128539203225617, 0.0, 10.0, False), ('cq', 0.10132402353824638, 0.002, 2.0, True), ('tz', 19.622238239382696, 0.5, 300.0, True), ('emq', 0.029743134949109507, 0.0005, 0.5, True), ('tauq', 18.1073226714383, 0.5, 150.0, True), ('svq', 0.8056096865981848, 0.0, 1.0, False), ('Kc', 200.0, 2.0, 100000.0, True), ('Kq', 20.0, 0.2, 10000.0, True), ('ap', 0.5, 0.001, 20.0, True), ('ts', 30.0, 1.0, 500.0, True), ('xh', 0.5, 0.0, 1.0, False), ('s0', 0.5, 0.0, 1.0, False)]

    class _PY:
        max = max
        min = min

    class _NP:
        max = np.maximum
        min = np.minimum

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

    def _region(M, th, mech, P, R, Q, G, Sh, sc, hunt, hab, cor, dh, em, arrP, arrQ):
        B = 'B' in mech
        F = R if B else 1.0
        b = th['r'] * F * P
        rec = b / (1.0 + P / (th['cJ'] * sc)) if 'A' in mech else b
        P2 = P * P
        Ph = th['Ph']
        harv = th['H'] * hunt * P2 / (P2 + Ph * Ph)
        if 'sh' in FLAGS:
            harv = harv * (1.0 - th['xh'] * Sh)
        dS = (hab - Sh) / th['ts']
        Z = P / (P + 100.0 * sc)
        setP = 1.0
        setQ = 1.0
        if 'C' in mech:
            setP = 1.0 / (1.0 + P / (th['Kc'] * sc))
            if 'cq' in FLAGS:
                setQ = 1.0 / (1.0 + Q / (th['Kq'] * sc))
        dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th['sv'] * setP * arrP
        if 'pd' in FLAGS:
            dP = dP - th['ap'] * Q * Z
        if B:
            dR = th['w'] * (1.0 - th['bh'] * (1.0 - hab)) * (1.0 - R) - th['kR'] * (P / sc) * R
        else:
            dR = 1.0 - R
        dG = (Z - G) / th['tz']
        outQ = th['emq'] * cor * Q
        dQ = th['cq'] * (th['q0'] + th['q1'] * G - Q) * Q / (Q + 4.0) - outQ + th['svq'] * setQ * arrQ
        return (dP, dR, dQ, dG, outQ, dS)

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        hunt, hab, cor = uu
        PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS, SN, SS = s
        k = 1.0 / th['tau']
        kq = 1.0 / th['tauq']
        dPN, dRN, dQN, dGN, oQN, dSN = _region(M, th, mech, PN, RN, QN, GN, SN, 1.0, hunt, hab, cor, th['dhN'], th['em'], k * TSN, kq * VSN)
        dPS, dRS, dQS, dGS, oQS, dSS = _region(M, th, mech, PS, RS, QS, GS, SS, th['ks'], hunt, hab, cor, th['dhS'], th['emS'], k * TNS, kq * VNS)
        dTNS = th['em'] * cor * PN - k * TNS
        dTSN = th['emS'] * cor * PS - k * TSN
        dVNS = oQN - kq * VNS
        dVSN = oQS - kq * VSN
        return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS, dSN, dSS])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        return _pack(M, [s[0], s[2], s[3], s[5]])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        PN = M.max(y[0], 0.0)
        QN = M.max(y[1], 0.01)
        PS = M.max(y[2], 0.0)
        QS = M.max(y[3], 0.01)
        z = 0.0 * PN
        GN = PN / (PN + 100.0)
        GS = PS / (PS + 100.0 * th['ks'])
        S0 = th['s0'] + z
        return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS, S0, S0])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7

def _ode_wildlife_v8h():
    import math
    import numpy as np
    FAMILY = 'wildlife_v8h'
    OBS = ['prey_north', 'predator_north', 'prey_south', 'predator_south']
    CTRL = ['hunting_quota', 'habitat_protection', 'corridor_access']
    MECHS = {'A': 'juvenile condition: nursery food competition caps recruitment', 'B': 'finite food renewal: grazing depletes a slowly renewing resource', 'C': 'settlement competition (not modelled in this family)'}
    N_SUB = 2
    D0 = 0.7
    STATE = ['PN', 'RN', 'QN', 'PS', 'RS', 'QS', 'TNS', 'TSN', 'VNS', 'VSN', 'GN', 'GS']
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([10000.0, 1.0, 1000.0, 10000.0, 1.0, 1000.0, 10000.0, 10000.0, 1000.0, 1000.0, 100.0, 100.0])
    PARAMS = [('r', 0.75, 0.05, 3.0, True), ('ks', 0.8, 0.3, 1.5, False), ('H', 2.0, 0.01, 50.0, True), ('Ph', 20.0, 0.5, 500.0, True), ('em', 0.03, 0.0005, 0.5, True), ('emS', 0.05, 0.0005, 0.5, True), ('tau', 10.0, 0.5, 150.0, True), ('sv', 0.8, 0.0, 1.0, False), ('cJ', 60.0, 5.0, 5000.0, True), ('w', 0.033, 0.002, 1.5, True), ('kR', 8e-05, 1e-06, 0.02, True), ('bh', 0.5, 0.0, 1.0, False), ('dhN', 0.05, 0.0, 1.0, False), ('dhS', 0.03, 0.0, 1.0, False), ('q0', 1.6, 0.0, 10.0, False), ('q1', 1.4, 0.0, 10.0, False), ('cq', 0.1, 0.002, 2.0, True), ('tz', 20.0, 0.5, 300.0, True)]

    class _PY:
        max = max
        min = min

    class _NP:
        max = np.maximum
        min = np.minimum

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

    def _region(th, mech, P, R, Q, G, sc, hunt, hab, cor, dh, em, arrP, arrQ):
        B = 'B' in mech
        F = R if B else 1.0
        b = th['r'] * F * P
        rec = b / (1.0 + P / (th['cJ'] * sc)) if 'A' in mech else b
        P2 = P * P
        harv = th['H'] * hunt * P2 / (P2 + th['Ph'] ** 2)
        dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th['sv'] * arrP
        if B:
            dR = th['w'] * (1.0 - th['bh'] * (1.0 - hab)) * (1.0 - R) - th['kR'] * (P / sc) * R
        else:
            dR = 1.0 - R
        Z = P / (P + 100.0 * sc)
        dG = (Z - G) / th['tz']
        dQ = th['cq'] * (th['q0'] + th['q1'] * G - Q) * Q / (Q + 4.0) - th['em'] * cor * Q + th['sv'] * arrQ
        return (dP, dR, dQ, dG)

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        hunt, hab, cor = uu
        PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
        k = 1.0 / th['tau']
        dPN, dRN, dQN, dGN = _region(th, mech, PN, RN, QN, GN, 1.0, hunt, hab, cor, th['dhN'], th['em'], k * TSN, k * VSN)
        dPS, dRS, dQS, dGS = _region(th, mech, PS, RS, QS, GS, th['ks'], hunt, hab, cor, th['dhS'], th['emS'], k * TNS, k * VNS)
        dTNS = th['em'] * cor * PN - k * TNS
        dTSN = th['emS'] * cor * PS - k * TSN
        dVNS = th['em'] * cor * QN - k * VNS
        dVSN = th['em'] * cor * QS - k * VSN
        return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        return _pack(M, [s[0], s[2], s[3], s[5]])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        PN = M.max(y[0], 0.0)
        QN = M.max(y[1], 0.01)
        PS = M.max(y[2], 0.0)
        QS = M.max(y[3], 0.01)
        z = 0.0 * PN
        GN = PN / (PN + 100.0)
        GS = PS / (PS + 100.0 * th['ks'])
        return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7

def _ode_wildlife_wild9m():
    import math
    import numpy as np
    FAMILY = 'wildlife_wild9m'
    FLAGS = ['gx', 'pt']
    OBS = ['prey_north', 'predator_north', 'prey_south', 'predator_south']
    CTRL = ['hunting_quota', 'habitat_protection', 'corridor_access']
    MECHS = {'A': 'juvenile condition: nursery food competition caps recruitment', 'B': 'finite food renewal: grazing depletes a slowly renewing resource', 'C': 'settlement competition (not modelled in this family)'}
    N_SUB = 2
    D0 = 0.7
    STATE = ['PN', 'RN', 'QN', 'PS', 'RS', 'QS', 'TNS', 'TSN', 'VNS', 'VSN', 'GN', 'GS']
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([10000.0, 1.0, 1000.0, 10000.0, 1.0, 1000.0, 10000.0, 10000.0, 1000.0, 1000.0, 100.0, 100.0])
    PARAMS = [('r', 0.879476323218783, 0.05, 3.0, True), ('ks', 0.7991713725708542, 0.3, 1.5, False), ('H', 1.3567344001368091, 0.01, 50.0, True), ('Ph', 24.944829387699713, 0.5, 500.0, True), ('em', 0.02593781706810343, 0.0005, 0.5, True), ('emS', 0.04033527068702279, 0.0005, 0.5, True), ('tau', 26.39036409931354, 0.5, 150.0, True), ('sv', 0.7800573507586477, 0.0, 1.0, False), ('cJ', 900.1504294955678, 5.0, 5000.0, True), ('w', 0.026202108245787555, 0.002, 1.5, True), ('kR', 2.3675571495756507e-05, 1e-06, 0.02, True), ('bh', 0.3462105711884271, 0.0, 1.0, False), ('dhN', 0.043905550461308965, 0.0, 1.0, False), ('dhS', 0.025175861577830756, 0.0, 1.0, False), ('q0', 1.753304368495746, 0.0, 10.0, False), ('q1', 1.1579882592866027, 0.0, 10.0, False), ('cq', 0.10086003859133522, 0.002, 2.0, True), ('tz', 27.25352098237159, 0.5, 300.0, True), ('xgN', 0.1, 0.0, 10.0, False), ('xgS', 0.1, 0.0, 10.0, False), ('Pt', 2.0, 0.1, 500.0, True)]

    class _PY:
        max = max
        min = min

    class _NP:
        max = np.maximum
        min = np.minimum

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

    def _region(M, th, mech, P, R, Q, G, sc, south, hunt, hab, cor, dh, em, arrP, arrQ):
        B = 'B' in mech
        F = R if B else 1.0
        b = th['r'] * F * P
        rec = b / (1.0 + P / (th['cJ'] * sc)) if 'A' in mech else b
        Ph = th['Ph']
        if 'phr' in FLAGS:
            Ph = Ph * (1.0 + (th['hPS'] if south else th['hPN']) * hab)
        if 'phs' in FLAGS:
            Ph = Ph * sc
        Pe = M.max(P - th['Pr'] * sc, 0.0) if 'ref' in FLAGS else P
        Hq = th['H'] * (th['xS'] if 'hsx' in FLAGS and south else 1.0)
        P2 = Pe * Pe
        harv = Hq * hunt * P2 / (P2 + Ph * Ph)
        dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th['sv'] * arrP
        bh = th['bhS'] if 'bhr' in FLAGS and south else th['bh']
        if B:
            dR = th['w'] * (1.0 - bh * (1.0 - hab)) * (1.0 - R) - th['kR'] * (P / sc) * R
        else:
            dR = 1.0 - R
        Pz = P * (1.0 + (th['xgS'] if south else th['xgN']) * (1.0 - hab)) if 'gx' in FLAGS else P
        Z = Pz / (Pz + 100.0 * sc)
        dG = (Z - G) / th['tz']
        qt = th['q0'] + th['q1'] * G - (th['xq'] * cor if 'qx' in FLAGS else 0.0)
        eq = th['em']
        if 'pt' in FLAGS:
            eq = eq * P / (P + th['Pt'] * sc)
        if 'ptq' in FLAGS:
            eq = eq * Q / (Q + th['Qt'])
        outQ = eq * cor * Q
        dQ = th['cq'] * (qt - Q) * Q / (Q + 4.0) - outQ + th['sv'] * arrQ
        return (dP, dR, dQ, dG, outQ)

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        hunt, hab, cor = uu
        PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
        k = 1.0 / th['tau']
        dPN, dRN, dQN, dGN, oQN = _region(M, th, mech, PN, RN, QN, GN, 1.0, False, hunt, hab, cor, th['dhN'], th['em'], k * TSN, k * VSN)
        dPS, dRS, dQS, dGS, oQS = _region(M, th, mech, PS, RS, QS, GS, th['ks'], True, hunt, hab, cor, th['dhS'], th['emS'], k * TNS, k * VNS)
        dTNS = th['em'] * cor * PN - k * TNS
        dTSN = th['emS'] * cor * PS - k * TSN
        dVNS = oQN - k * VNS
        dVSN = oQS - k * VSN
        return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        return _pack(M, [s[0], s[2], s[3], s[5]])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        PN = M.max(y[0], 0.0)
        QN = M.max(y[1], 0.01)
        PS = M.max(y[2], 0.0)
        QS = M.max(y[3], 0.01)
        z = 0.0 * PN
        GN = PN / (PN + 100.0)
        GS = PS / (PS + 100.0 * th['ks'])
        return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7
_ODE_BUILD = {'wildlife_tmp': _ode_wildlife_tmp, 'wildlife_tmq': _ode_wildlife_tmq, 'wildlife_tms': _ode_wildlife_tms, 'wildlife_v8h': _ode_wildlife_v8h, 'wildlife_wild9m': _ode_wildlife_wild9m}
_ODE_MODS = {}

def ode_modules(family, base_dir=None, tag=''):
    if family not in _ODE_MODS:
        _ODE_MODS[family] = (_ode_core(), _ODE_BUILD[family]())
    return _ODE_MODS[family]
SYSTEM = 'wildlife'
HARD = {'prey_north': [0.0, None], 'predator_north': [0.0, None], 'prey_south': [0.0, None], 'predator_south': [0.0, None]}
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
