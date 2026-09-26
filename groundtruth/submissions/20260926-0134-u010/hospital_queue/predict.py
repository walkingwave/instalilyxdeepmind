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
_KINDS = {'ode': _roll_ode}

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

def _ode_hospital_queue_min():
    import math
    import numpy as np
    FAMILY = 'hospital_queue_min'
    OBS = ['wait_time', 'queue', 'discharges']
    CTRL = ['staffing', 'elective_scheduling', 'diagnostic_allocation', 'urgent_priority', 'overtime', 'followup_capacity']
    MECHS = {'A': 'stranded cohort (always on)', 'B': 'finite chairs/beds (always on)', 'C': 'unused'}
    N_SUB = 2
    STATE = ['W', 'A1', 'A2', 'T', 'wt', 'C', 'a']
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 10000.0, 333.0, 3000.0])
    Q_W = 6.0
    Q_E_W = 20.0
    K_DRAIN = 3.0
    TAU_A = 1.77
    D_H = 0.04
    Q_CAP = 325.0
    TAU_Z = 150.0
    C_H = 5.0
    W_WIDTH = 5.0
    WT_H = 30.0
    PARAMS = [('lam0', 11.5, 5.0, 25.0, False), ('c_a', 0.7, 0.2, 6.0, True), ('c_t', 1.9, 0.3, 10.0, True), ('chairs', 30.0, 5.0, 300.0, True), ('beds', 15.0, 5.0, 300.0, True), ('ot_gain', 0.45, 0.0, 1.5, False), ('k_l', 0.02, 0.001, 0.5, True), ('tau_w', 10.0, 1.0, 60.0, True), ('q_e', 250.0, 120.0, 330.0, False), ('k_s', 0.0006, 1e-05, 0.01, True), ('g_z', 2.0, 0.3, 6.0, False), ('w_th', 20.0, 2.0, 40.0, True)]

    class _PY:
        max = max
        min = min

        @staticmethod
        def exp(z):
            return math.exp(z if z < 50.0 else 50.0)

        @staticmethod
        def sqrt(z):
            return math.sqrt(z) if z > 0.0 else 0.0

    class _NP:
        max = np.maximum
        min = np.minimum

        @staticmethod
        def exp(z):
            return np.exp(np.minimum(z, 50.0))

        @staticmethod
        def sqrt(z):
            return np.sqrt(np.maximum(z, 0.0))

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

    def _flows(M, s, uu, th):
        W, A1, A2, T, wt, C, a = s
        S, E, D, U, O, Fu = uu
        eff = S * (1.0 + th['ot_gain'] * O)
        r_a = th['c_a'] * eff * D / (D + D_H) + 1e-06
        r_t = th['c_t'] * eff * (1.0 - D) + 1e-06
        free_c = M.max(th['chairs'] - A1 - A2, 0.0)
        free_b = M.max(th['beds'] - T, 0.0)
        f1 = M.min(M.min(r_a, K_DRAIN * W), K_DRAIN * free_c)
        ka = 2.0 / TAU_A
        f2a = ka * A1
        f2b = M.min(ka * A2, K_DRAIN * free_b)
        f3 = M.min(r_t, K_DRAIN * T)
        return (f1, f2a, f2b, f3)

    def _target(M, s, f1, th):
        W, A1, A2, T, wt, C, a = s
        sw = 1.0 / (1.0 + M.exp((W - th['w_th']) / W_WIDTH))
        return W / (f1 + 1.0) + a * (C * C / (C * C + C_H * C_H)) * sw

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        W, A1, A2, T, wt, C, a = s
        S, E, D, U, O, Fu = uu
        f1, f2a, f2b, f3 = _flows(M, s, uu, th)
        q = W + A1 + A2 + T
        gate = 1.0 / (1.0 + M.exp((q - Q_CAP) / Q_W))
        gate_e = 1.0 / (1.0 + M.exp((q - th['q_e']) / Q_E_W))
        arr = th['lam0'] * gate + E * gate_e
        leave = th['k_l'] * U * W
        strand = th['k_s'] * W * wt / (wt + WT_H)
        dW = arr - f1 - leave - strand
        dA1 = f1 - f2a
        dA2 = f2a - f2b
        dT = f2b - f3
        dwt = (_target(M, s, f1, th) - wt) / th['tau_w']
        dC = strand - C / TAU_Z
        da = th['g_z'] - a * strand / (C + 1.0) - a / TAU_Z
        return _pack(M, [dW, dA1, dA2, dT, dwt, dC, da])

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        f1, f2a, f2b, f3 = _flows(M, s, uu, th)
        W, A1, A2, T, wt, C, a = s
        return _pack(M, [wt, W + A1 + A2 + T, f3])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        wt = M.max(y[0], 0.0)
        W = M.max(y[1], 0.0)
        z = 0.0 * W
        return _pack(M, [W, z, z, z, wt, z, z])

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7
_ODE_BUILD = {'hospital_queue_min': _ode_hospital_queue_min}
_ODE_MODS = {}

def ode_modules(family, base_dir=None, tag=''):
    if family not in _ODE_MODS:
        _ODE_MODS[family] = (_ode_core(), _ODE_BUILD[family]())
    return _ODE_MODS[family]
SYSTEM = 'hospital_queue'
HARD = {'wait_time': [0.0, None], 'queue': [0.0, None], 'discharges': [0.0, None]}
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
