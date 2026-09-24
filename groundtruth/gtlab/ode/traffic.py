"""Grey-box ODE: traffic. NUMPY + math only (ships verbatim in submissions).

Structure (two routes a, b; light L and heavy H vehicles; pcu = passenger-car units)
- Arrivals: demand split by class; toll changes the class mix (deters light and heavy
  differently); ramp metering scales admitted demand (1 = arrivals stopped); route split
  phi (fixed, or learned under mechanism A).
- Finite approach buffers reject excess arrivals (smooth gate); waiting drivers divert.
- Shared junction: signal timing divides crossing admissions between routes; freight
  priority chooses which class is served; crossing vehicles occupy the junction for tau_j
  and cannot leave until exit space opens (blocking => one route obstructs the other).
- Exits: service capacity reduced by lane closure (unequally per route) and boosted by
  clearance effort, which takes crew from junction operation.
- flow = exit service; speed = vfree / (1 + k*occupancy) * (1 - slowdown * heavy share).
Mechanisms: A route learning (split drifts toward the faster route), B crew fatigue /
switching cost (clearance effectiveness decays with use), C persistent spillback fronts
(hysteretic junction capacity loss when an exit buffer fills, slow recovery).
"""
import math

import numpy as np

FAMILY = "traffic"
OBS = ["flow_a", "flow_b", "speed_a", "speed_b"]
CTRL = ["signal_timing", "lane_closure", "toll", "ramp_metering", "freight_priority",
        "clearance_effort"]
MECHS = {
    "A": "route learning: route split drifts toward the faster route",
    "B": "crew fatigue/switching cost: clearance effectiveness decays with sustained use",
    "C": "persistent spillback fronts: junction capacity drops once exits back up, recovers slowly",
}
N_SUB = 2

STATE = ["qLa", "qHa", "qLb", "qHb", "Ja", "Jb", "Xa", "Xb", "split", "crew_fat", "spill_a", "spill_b"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e5)
STATE_HI[8] = 1.0
STATE_HI[9] = 1.0
STATE_HI[10] = 1.0
STATE_HI[11] = 1.0

PARAMS = [
    ("dem", 4.0, 0.05, 200.0, True),
    ("heavy", 0.2, 0.01, 0.8, False),
    ("toll_l", 0.4, 0.0, 1.0, False),
    ("toll_h", 0.1, 0.0, 1.0, False),
    ("split0", 0.5, 0.05, 0.95, False),
    ("qcap", 50.0, 2.0, 5000.0, True),
    ("divert", 0.01, 0.0, 0.5, False),
    ("mu_j", 6.0, 0.1, 200.0, True),
    ("jcap", 20.0, 1.0, 1000.0, True),
    ("tau_j", 2.0, 0.5, 50.0, True),
    ("pcu_h", 2.0, 1.0, 5.0, False),
    ("xcap", 20.0, 1.0, 1000.0, True),
    ("mu_x", 3.0, 0.1, 200.0, True),
    ("lc_a", 0.9, 0.0, 1.0, False),
    ("lc_b", 0.4, 0.0, 1.0, False),
    ("ce_j", 0.3, 0.0, 0.95, False),
    ("ce_x", 0.5, 0.0, 3.0, False),
    ("vfree_a", 60.0, 2.0, 300.0, True),
    ("vfree_b", 50.0, 2.0, 300.0, True),
    ("k_occ", 0.03, 1e-4, 2.0, True),
    ("hv_slow", 0.3, 0.0, 0.9, False),
    ("tau_rl", 100.0, 5.0, 3000.0, True),   # A
    ("k_rl", 3.0, 0.0, 40.0, False),        # A
    ("tau_cf", 50.0, 2.0, 3000.0, True),    # B
    ("cf_k", 0.5, 0.0, 0.95, False),        # B
    ("spill_th", 0.8, 0.2, 1.0, False),     # C
    ("spill_k", 0.5, 0.0, 0.95, False),     # C
    ("tau_sp", 200.0, 5.0, 5000.0, True),   # C
]

# ---- numeric helpers (duplicated in every family file: files ship standalone) ----
class _PY:
    max = max
    min = min
    tanh = math.tanh

    @staticmethod
    def exp(z):
        return math.exp(z if z < 50.0 else 50.0)

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0

    @staticmethod
    def log(z):
        return math.log(z) if z > 1e-300 else -690.0


class _NP:
    max = np.maximum
    min = np.minimum
    tanh = np.tanh

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))

    @staticmethod
    def log(z):
        return np.log(np.maximum(z, 1e-300))


def _smin(M, a, b, e):
    return 0.5 * (a + b - M.sqrt((a - b) * (a - b) + e * e))


def _pos(M, a, e):
    return 0.5 * (a + M.sqrt(a * a + e * e))


def _unpack(x, u):
    if np.ndim(x) == 1:
        return _PY, x.tolist(), np.asarray(u, dtype=float).tolist()
    return _NP, [x[..., i] for i in range(x.shape[-1])], [u[..., j] for j in range(u.shape[-1])]


def _pack(M, vals):
    if M is _PY:
        return np.array(vals, dtype=float)
    return np.stack(np.broadcast_arrays(*vals), axis=-1).astype(float)


def _sig(M, z):
    return 1.0 / (1.0 + M.exp(-z))


def _yunpack(y0):
    if np.ndim(y0) == 1:
        return _PY, np.asarray(y0, dtype=float).tolist()
    return _NP, [y0[..., j] for j in range(y0.shape[-1])]


# ---- model ----
def _alg(M, s, uu, th, mech):
    sig_t, lc, toll, ramp, fp, ce = uu
    qLa, qHa, qLb, qHb, Ja, Jb, Xa, Xb, phi, cf, spa, spb = s
    pcu = th["pcu_h"]
    ceff = ce * (1.0 - th["cf_k"] * cf) if "B" in mech else ce
    # exits
    boost = 1.0 + th["ce_x"] * ceff
    ex_a = _smin(M, Xa, th["mu_x"] * (1.0 - th["lc_a"] * lc) * boost, 0.01)
    ex_b = _smin(M, Xb, th["mu_x"] * (1.0 - th["lc_b"] * lc) * boost, 0.01)
    ex_a = _pos(M, ex_a, 1e-4)
    ex_b = _pos(M, ex_b, 1e-4)
    # junction completions, blocked by exit space
    tj = th["tau_j"]
    xc = th["xcap"]
    out_a = _smin(M, Ja / tj, _pos(M, xc - Xa, 0.05) + ex_a, 0.01)
    out_b = _smin(M, Jb / tj, _pos(M, xc - Xb, 0.05) + ex_b, 0.01)
    # junction admissions
    jfree = _pos(M, th["jcap"] - Ja - Jb, 0.05)
    jf = jfree / (jfree + 1.0)
    capj = th["mu_j"] * (1.0 - th["ce_j"] * ceff) * jf
    cap_a = capj * sig_t
    cap_b = capj * (1.0 - sig_t)
    if "C" in mech:
        cap_a = cap_a * (1.0 - th["spill_k"] * spa)
        cap_b = cap_b * (1.0 - th["spill_k"] * spb)
    Qa = qLa + pcu * qHa
    Qb = qLb + pcu * qHb
    adm_a = _pos(M, _smin(M, Qa, cap_a, 0.01), 1e-4)
    adm_b = _pos(M, _smin(M, Qb, cap_b, 0.01), 1e-4)
    wHa = fp * pcu * qHa
    wLa = (1.0 - fp) * qLa
    hs_a = wHa / (wHa + wLa + 1e-6)
    wHb = fp * pcu * qHb
    wLb = (1.0 - fp) * qLb
    hs_b = wHb / (wHb + wLb + 1e-6)
    # speeds
    occ_a = Qa + Ja + Xa
    occ_b = Qb + Jb + Xb
    hsh_a = pcu * qHa / (Qa + 1.0)
    hsh_b = pcu * qHb / (Qb + 1.0)
    v_a = th["vfree_a"] / (1.0 + th["k_occ"] * occ_a) * (1.0 - th["hv_slow"] * hsh_a)
    v_b = th["vfree_b"] / (1.0 + th["k_occ"] * occ_b) * (1.0 - th["hv_slow"] * hsh_b)
    return ex_a, ex_b, out_a, out_b, adm_a, adm_b, hs_a, hs_b, v_a, v_b, Qa, Qb


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    sig_t, lc, toll, ramp, fp, ce = uu
    qLa, qHa, qLb, qHb, Ja, Jb, Xa, Xb, phi, cf, spa, spb = s
    ex_a, ex_b, out_a, out_b, adm_a, adm_b, hs_a, hs_b, v_a, v_b, Qa, Qb = _alg(M, s, uu, th, mech)
    pcu = th["pcu_h"]
    tf = toll / 5.0
    open_ = 1.0 - ramp
    dL = th["dem"] * (1.0 - th["heavy"]) * (1.0 - th["toll_l"] * tf) * open_
    dH = th["dem"] * th["heavy"] * (1.0 - th["toll_h"] * tf) * open_
    ph = phi if "A" in mech else th["split0"]
    qc = th["qcap"]
    ga = _pos(M, 1.0 - Qa / qc, 0.01)
    gb = _pos(M, 1.0 - Qb / qc, 0.01)
    dv = th["divert"]
    d = [0.0] * _NS
    d[0] = dL * ph * ga - adm_a * (1.0 - hs_a) - dv * qLa
    d[1] = dH * ph * ga - adm_a * hs_a / pcu - dv * qHa
    d[2] = dL * (1.0 - ph) * gb - adm_b * (1.0 - hs_b) - dv * qLb
    d[3] = dH * (1.0 - ph) * gb - adm_b * hs_b / pcu - dv * qHb
    d[4] = adm_a - out_a
    d[5] = adm_b - out_b
    d[6] = out_a - ex_a
    d[7] = out_b - ex_b
    if "A" in mech:
        s0 = th["split0"]
        lg0 = M.log(s0 / (1.0 - s0))
        tgt = _sig(M, lg0 + th["k_rl"] * (v_a / th["vfree_a"] - v_b / th["vfree_b"]))
        d[8] = (tgt - phi) / th["tau_rl"]
    if "B" in mech:
        d[9] = (ce - cf) / th["tau_cf"]
    if "C" in mech:
        ts = th["spill_th"]
        xc = th["xcap"]
        ta = _sig(M, 20.0 * (Xa / xc - ts))
        tb = _sig(M, 20.0 * (Xb / xc - ts))
        slow = 1.0 / th["tau_sp"]
        ra = slow + (0.2 - slow) * _sig(M, 20.0 * (ta - spa))
        rb = slow + (0.2 - slow) * _sig(M, 20.0 * (tb - spb))
        d[10] = ra * (ta - spa)
        d[11] = rb * (tb - spb)
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    ex_a, ex_b, out_a, out_b, adm_a, adm_b, hs_a, hs_b, v_a, v_b, Qa, Qb = _alg(M, s, uu, th, mech)
    return _pack(M, [ex_a, ex_b, _pos(M, v_a, 1e-3), _pos(M, v_b, 1e-3)])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    # roads and crossings start empty
    return _pack(M, [z, z, z, z, z, z, z, z, th["split0"] + z, z, z, z])
