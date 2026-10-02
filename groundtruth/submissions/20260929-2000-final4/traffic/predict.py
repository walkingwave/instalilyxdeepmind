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

def _ode_traffic_hi4():
    import math
    import numpy as np
    FAMILY = 'traffic_hi4'
    SPEED = 'hill'
    KX = 1
    KTR = 4
    OBS = ['flow_a', 'flow_b', 'speed_a', 'speed_b']
    CTRL = ['signal_timing', 'lane_closure', 'toll', 'ramp_metering', 'freight_priority', 'clearance_effort']
    MECHS = {'A': 'route learning (admitted split learns toward the faster route)', 'B': 'crew fatigue / switching cost (lagged, tiring clearance crew)', 'C': 'persistent spillback fronts (not modelled in this family)'}
    N_SUB = 2
    STATE = ['p1a', 'p2a', 'p3a', 'p1b', 'p2b', 'p3b', 'sa', 'sb', 'ma', 'mb', 'q', 'ce', 'fat', 'ja', 'jb'] + ['xa%d' % k for k in range(KX - 1)] + ['xb%d' % k for k in range(KX - 1)] + ['ta%d' % k for k in range(KTR)] + ['tb%d' % k for k in range(KTR)]
    _NS = len(STATE)
    STATE_LO = np.zeros(_NS)
    STATE_HI = np.array([100000.0] * 6 + [200.0, 200.0, 100000.0, 100000.0, 1.0, 1.0, 1.0, 200.0, 200.0] + [100000.0] * (2 * KX - 2 + 2 * KTR))
    BASE = ['dem0', 'w_sig', 'r_pipe', 'junc', 'cap', 'L_a', 'L_b', 'clr', 'v_free', 'n_ref', 'rho', 'n_max', 'gam', 'w1', 'r_m', 'kap', 'k_toll', 'n_l']
    MECH_PARAMS = {'A': ['beta_L', 'tau_L'], 'B': ['tau_sw', 'k_F', 'tau_F', 'ce0'], 'C': []}
    EXTRA = ['w_j', 'r_J', 'f0', 'cl_a', 'cl_b', 'n_v', 'r_q', 'cn_a', 'cn_b', 'D_tr']
    PARAMS = [('dem0', 46.927, 5.0, 300.0, True), ('w_sig', 0.72055, 0.0, 1.5, False), ('r_pipe', 0.24254, 0.05, 1.5, True), ('junc', 35.343, 3.0, 600.0, True), ('cap', 16.733, 3.0, 300.0, True), ('L_a', 1.6836, 0.3, 50.0, True), ('L_b', 1.2262, 0.3, 50.0, True), ('clr', 0.38409, 0.0, 3.0, False), ('v_free', 48.714, 30.0, 70.0, False), ('n_ref', 318.99, 10.0, 5000.0, True), ('rho', 0.29726, 0.02, 1.5, True), ('n_max', 1069.7, 50.0, 20000.0, True), ('gam', 1.0587, 0.2, 4.0, False), ('w1', 1.7512, 0.0, 8.0, False), ('r_m', 0.24796, 0.03, 3.0, True), ('kap', 0.66022, 0.0, 2.0, False), ('k_toll', 0.196, -0.5, 1.0, False), ('n_l', 1.3642, 1.0, 12.0, False), ('beta_L', 0.26168, 0.0, 20.0, False), ('tau_L', 6.5424, 1.0, 5000.0, True), ('tau_sw', 0.71574, 0.2, 300.0, True), ('k_F', 0.58697, 0.0, 1.0, False), ('tau_F', 76.067, 5.0, 5000.0, True), ('ce0', 0.072468, 0.0, 1.0, False), ('ps', 4.0, 1.0, 40.0, True), ('w_j', 0.0, 0.0, 1.0, False), ('r_J', 0.3, 0.01, 3.0, True), ('f0', 1.0, 0.05, 50.0, True), ('cl_a', 0.0, 0.0, 1.5, False), ('cl_b', 0.0, 0.0, 1.5, False), ('n_v', 2.0, 0.5, 6.0, False), ('r_q', 1.0, 0.2, 20.0, True), ('cn_a', 0.0, 0.0, 1.3, False), ('cn_b', 0.0, 0.0, 1.3, False), ('D_tr', 4.0, 0.5, 20.0, True)]

    def free_for(mech):
        out = list(BASE)
        for m in sorted(mech):
            out += MECH_PARAMS[m]
        return out

    class _PY:
        max = max
        min = min

        @staticmethod
        def exp(z):
            return math.exp(z if z < 50.0 else 50.0)

        @staticmethod
        def pow(a, b):
            return math.pow(a, b) if a > 0.0 else 0.0

    class _NP:
        max = np.maximum
        min = np.minimum

        @staticmethod
        def exp(z):
            return np.exp(np.minimum(z, 50.0))

        @staticmethod
        def pow(a, b):
            return np.power(np.maximum(a, 0.0), b)

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

    def _smin(M, x, c, ps):
        q = M.min(x / c, 30.0)
        return c * q / M.pow(1.0 + M.pow(q, ps), 1.0 / ps)

    def _exits(M, s, uu, th, mech):
        sig, lane, toll, ramp, frt, clr = uu
        p3a, p3b = (s[2], s[5])
        r = th['r_pipe']
        if 'B' in mech:
            ce, fat = (s[11], s[12])
            boost = 1.0 + th['clr'] * ce * (1.0 - th['k_F'] * fat)
        else:
            boost = 1.0 + th['clr'] * clr
        cap_a = th['cap'] * M.max(1.0 - M.pow(lane / th['L_a'], th['n_l']), 0.0) * boost + 1e-06
        cap_b = th['cap'] * M.max(1.0 - M.pow(lane / th['L_b'], th['n_l']), 0.0) * boost + 1e-06
        rq = r * th['r_q']
        return (_smin(M, rq * p3a, cap_a, th['ps']), _smin(M, rq * p3b, cap_b, th['ps']))

    def _hill(M, m, v, th):
        return v / (1.0 + M.pow(m / th['n_ref'], th['gam']))

    def f(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        sig, lane, toll, ramp, frt, clr = uu
        p1a, p2a, p3a, p1b, p2b, p3b, sa, sb, ma, mb, q, ce, fat, ja, jb = s[:15]
        xa = s[15:15 + KX - 1]
        xb = s[15 + KX - 1:15 + 2 * (KX - 1)]
        r = th['r_pipe']
        n_a = p1a + p2a + p3a
        n_b = p1b + p2b + p3b
        for k in range(KX - 1):
            n_a = n_a + xa[k]
            n_b = n_b + xb[k]
        ta = s[15 + 2 * (KX - 1):15 + 2 * (KX - 1) + KTR]
        tb = s[15 + 2 * (KX - 1) + KTR:15 + 2 * (KX - 1) + 2 * KTR]
        for k in range(KTR):
            n_a = n_a + ta[k]
            n_b = n_b + tb[k]
        a_tot = th['dem0'] * ramp * M.exp(-th['k_toll'] * (toll - 2.5))
        qa = q if 'A' in mech else 0.5
        nm_a = th['n_max'] * M.max(1.0 - th['cn_a'] * lane, 0.02)
        nm_b = th['n_max'] * M.max(1.0 - th['cn_b'] * lane, 0.02)
        a_a = a_tot * qa * M.max(1.0 - n_a / nm_a, 0.0)
        a_b = a_tot * (1.0 - qa) * M.max(1.0 - n_b / nm_b, 0.0)
        dta = []
        dtb = []
        ktr = KTR / th['D_tr']
        for k in range(KTR):
            dta.append(a_a - ktr * ta[k])
            dtb.append(a_b - ktr * tb[k])
            a_a = ktr * ta[k]
            a_b = ktr * tb[k]
        share = M.min(M.max(0.5 + th['w_sig'] * (sig - 0.5), 0.02), 0.98)
        jc = th['junc']
        rq = r * th['r_q']
        j_a = _smin(M, rq * p1a, jc * share + 1e-06, th['ps'])
        j_b = _smin(M, rq * p1b, jc * (1.0 - share) + 1e-06, th['ps'])
        out_a, out_b = _exits(M, s, uu, th, mech)
        kr = KX * r
        dxa = []
        dxb = []
        ina, inb = (j_a, j_b)
        for k in range(KX - 1):
            dxa.append(ina - kr * xa[k])
            dxb.append(inb - kr * xb[k])
            ina = kr * xa[k]
            inb = kr * xb[k]
        dp1a = a_a - j_a
        dp2a = ina - kr * p2a
        dp3a = kr * p2a - out_a
        dp1b = a_b - j_b
        dp2b = inb - kr * p2b
        dp3b = kr * p2b - out_b
        w1 = th['w1']
        rm = th['r_m']
        tra = 0.0 * p1a
        trb = 0.0 * p1a
        for k in range(KTR):
            tra = tra + ta[k]
            trb = trb + tb[k]
        occ_a = w1 * p1a * M.pow(0.5 / share, th['kap']) + (n_a - p1a - tra)
        occ_b = w1 * p1b * M.pow(0.5 / (1.0 - share), th['kap']) + (n_b - p1b - trb)
        dma = rm * (occ_a - ma)
        dmb = rm * (occ_b - mb)
        va = th['v_free'] * M.max(1.0 - th['cl_a'] * M.pow(lane, th['n_v']), 0.05)
        vb = th['v_free'] * M.max(1.0 - th['cl_b'] * M.pow(lane, th['n_v']), 0.05)
        tgt_a = _hill(M, ma, va, th)
        tgt_b = _hill(M, mb, vb, th)
        dja = th['r_J'] * out_a / (out_a + th['f0']) * (tgt_a - ja)
        djb = th['r_J'] * out_b / (out_b + th['f0']) * (tgt_b - jb)
        wj = th['w_j']
        dsa = th['rho'] * ((1.0 - wj) * tgt_a + wj * ja - sa)
        dsb = th['rho'] * ((1.0 - wj) * tgt_b + wj * jb - sb)
        z = 0.0 * p1a
        if 'A' in mech:
            qs = 1.0 / (1.0 + M.exp(-th['beta_L'] * (sa - sb) / 10.0))
            dq = (qs - q) / th['tau_L']
        else:
            dq = z
        if 'B' in mech:
            dce = (clr - ce) / th['tau_sw']
            dfat = (ce - fat) / th['tau_F']
        else:
            dce = z
            dfat = z
        return _pack(M, [dp1a, dp2a, dp3a, dp1b, dp2b, dp3b, dsa, dsb, dma, dmb, dq, dce, dfat, dja, djb] + dxa + dxb + dta + dtb)

    def h(x, u, th, mech):
        M, s, uu = _unpack(x, u)
        out_a, out_b = _exits(M, s, uu, th, mech)
        return _pack(M, [out_a, out_b, s[6], s[7]])

    def x0(y0, th, mech):
        M, y = _yunpack(y0)
        z = 0.0 * y[0]
        sa = M.min(M.max(y[2], 1.0), 80.0)
        sb = M.min(M.max(y[3], 1.0), 80.0)
        ce = z + th['ce0']
        v0 = z + th['v_free']
        return _pack(M, [z, z, z, z, z, z, sa, sb, z, z, z + 0.5, ce, z, v0, v0] + [z] * (2 * KX - 2 + 2 * KTR))

    class _Namespace_q7:
        pass
    _obj_q7 = _Namespace_q7()
    _obj_q7.__dict__.update(locals())
    return _obj_q7
_ODE_BUILD = {'traffic_hi4': _ode_traffic_hi4}
_ODE_MODS = {}

def ode_modules(family, base_dir=None, tag=''):
    if family not in _ODE_MODS:
        _ODE_MODS[family] = (_ode_core(), _ODE_BUILD[family]())
    return _ODE_MODS[family]
SYSTEM = 'traffic'
HARD = {'flow_a': [0.0, None], 'flow_b': [0.0, None], 'speed_a': [0.0, None], 'speed_b': [0.0, None]}
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
