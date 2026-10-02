"""Grey-box ODE: social_contagion canon3 (21 parameters). NUMPY + math only.

Literal reading of the brief on top of the z20 base (see social_contagion_z20.py):
  AUD: relationship-led / incentive-led / deliberative audiences with a fixed mix per community.
A credibility: dK = (1 - K)/50 - k_A K (promised waiting)/N.  B incentive expectations: dE = (c - E)/tau_E.
C cross-community relationships: dR = (s beta/10 - R)/15 carry the introductions.
"""
import math

import numpy as np

FAMILY = "social_contagion_canon3"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility erodes with promised waiting cohorts",
         "B": "incentive expectations: incentive-led members leave when incentive falls below memory",
         "C": "cross-community relationships built by introductions carry word of mouth"}
N_SUB = 2
WF = False
SYM = False
AUD = True

_PER = ["L", "M", "X", "D", "W1", "W2", "W3", "V1", "V2", "V3", "K"]
STATE = [s + "a" for s in _PER] + [s + "b" for s in _PER] + ["E", "R"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array(([1e4] * 10 + [1.0]) * 2 + [2.0, 1.0])

R_MAX = 1.5
PHI_HALF = 0.7
TAU_K = 50.0
TAU_R = 15.0

PARAMS = [
    ("N_a", 360.0, 60.0, 3000.0, True),
    ("N_b", 350.0, 40.0, 3000.0, True),
    ("s_a", 0.003, 1e-4, 1.0, True),
    ("s_b", 0.0008, 1e-5, 1.0, True),
    ("q", 0.015, 1e-3, 1.5, True),
    ("o", 0.001, 1e-5, 0.1, True),
    ("tau_on", 9.0, 1.0, 60.0, True),
    ("c_L", 0.012, 1e-4, 0.5, True),
    ("k_X", 0.12, 0.005, 1.5, True),
    ("m0", 0.33, 0.0, 0.9, False),
    ("k_conv", 0.3, 0.0, 0.7, False),
    ("k_A", 0.1, 0.0, 3.0, False),
    ("k_B", 0.045, 0.0, 0.7, False),
    ("tau_E", 28.0, 1.0, 500.0, True),
    ("tau_D", 13.0, 1.0, 1000.0, True),
    ("w_rel_a", 0.2, 0.0, 0.9, False),
    ("w_rel_b", 0.4, 0.0, 0.9, False),
    ("w_inc_a", 0.5, 0.0, 1.0, False),
    ("w_inc_b", 0.2, 0.0, 1.0, False),
    ("s_xb", 0.001, 1e-6, 1.0, True),
    ("tau_on_b", 9.0, 1.0, 60.0, True),
]


class _PY:
    max = max
    min = min

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))


def _pos(M, a, e):
    return 0.5 * (a + M.sqrt(a * a + e * e))


def _smin(M, a, b, e):
    return 0.5 * (a + b - M.sqrt((a - b) * (a - b) + e * e))


def _unpack(x, u):
    if np.ndim(x) == 1:
        return _PY, x.tolist(), np.asarray(u, dtype=float).tolist()
    return _NP, [x[..., i] for i in range(x.shape[-1])], [u[..., j] for j in range(u.shape[-1])]


def _pack(M, vals):
    if M is _PY:
        return np.array(vals, dtype=float)
    return np.stack(np.broadcast_arrays(*vals), axis=-1).astype(float)


def _yunpack(y0):
    if np.ndim(y0) == 1:
        return _PY, np.asarray(y0, dtype=float).tolist()
    return _NP, [y0[..., j] for j in range(y0.shape[-1])]


def _community(M, s, N, raw, th, inc, phi, chM, mech, k, gate, give_up):
    L, Mm, X, D, W1, W2, W3, V1, V2, V3, K = s
    A = L + Mm + X
    W = W1 + W2 + W3 + V1 + V2 + V3
    pool = _pos(M, N - A - W - D, 1.0)
    if "A" in mech:
        raw = raw * K
    r = raw / (1.0 + raw / R_MAX)
    new = r * pool
    conv = th["k_conv"] * inc * L
    outL = th["c_L"] * L
    outM = chM * Mm
    outX = th["k_X"] * X
    dW1 = (1.0 - phi) * new - k * W1
    dW2 = k * (W1 - W2)
    dW3 = k * W2 - k * gate * W3 - give_up * W3
    dV1 = phi * new - k * V1
    dV2 = k * (V1 - V2)
    dV3 = k * V2 - k * gate * V3 - give_up * V3
    dL = k * gate * W3 - outL - conv
    dM = k * gate * V3 + conv - outM
    dX = -outX
    dD = outL + outM + outX + give_up * (W3 + V3) - D / th["tau_D"]
    if "A" in mech:
        dK = (1.0 - K) / TAU_K - th["k_A"] * K * (V1 + V2 + V3) / N
    else:
        dK = 0.0 * K
    return [dL, dM, dX, dD, dW1, dW2, dW3, dV1, dV2, dV3, dK]


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    n = len(_PER)
    sa, sb, E, R = s[:n], s[n:2 * n], s[2 * n], s[2 * n + 1]
    chM = th["c_L"] + 0.0 * E
    dE = 0.0 * E
    if "B" in mech:
        chM = th["c_L"] + th["k_B"] * _pos(M, E - inc, 0.05)
        dE = (inc - E) / th["tau_E"]
    phi = inc / (inc + PHI_HALF)
    loc = seed * _pos(M, 1.0 - br, 0.01)
    Aa = sa[0] + sa[1] + sa[2]
    Ab = sb[0] + sb[1] + sb[2]
    Na = th["N_a"]
    Nb = th["N_b"]
    if "C" in mech:
        dR = (seed * br / 10.0 - R) / TAU_R
        eff = 10.0 * R
    else:
        dR = 0.0 * R
        eff = seed * br
    if SYM:
        xa = th["s_xb"] * 2.0 * eff * Ab / Nb
        xb = th["s_xb"] * 2.0 * eff * Aa / Na
    else:
        xa = 0.0 * eff
        xb = th["s_xb"] * eff
    wa = th["q"] * Aa / Na
    wb = th["q"] * Ab / Nb
    if AUD:
        # relationship-led: word of mouth + introductions; incentive-led: outreach x offer; deliberative: outreach
        ra_ = th["w_rel_a"]
        rb_ = th["w_rel_b"]
        ia_ = th["w_inc_a"] * (1.0 - ra_)
        ib_ = th["w_inc_b"] * (1.0 - rb_)
        g3 = 3.0 * phi
        raw_a = ra_ * 3.0 * (wa + xa) + (th["s_a"] * loc + th["o"]) * ((1.0 - ra_ - ia_) + ia_ * 2.0 * g3)
        raw_b = rb_ * 3.0 * (wb + xb) + (th["s_b"] * loc + th["o"]) * ((1.0 - rb_ - ib_) + ib_ * 2.0 * g3)
    else:
        raw_a = (th["s_a"] * loc + xa + wa + th["o"]) * (1.0 + th["k_inc"] * phi)
        raw_b = (th["s_b"] * loc + xb + wb + th["o"]) * (1.0 + th["k_inc_b"] * phi)
    ka = 3.0 / th["tau_on"]
    kb = 3.0 / th["tau_on_b"]
    if WF:
        dem = ka * (sa[6] + sa[9]) + kb * (sb[6] + sb[9])
        cap = th["kap"] * _pos(M, th["A_cap"] - Aa - Ab, 1.0)
        d2 = dem * dem
        c2 = cap * cap + 1e-9
        gate = cap / M.sqrt(M.sqrt(d2 * d2 + c2 * c2))
        gu = 1.0 / th["tau_w"]
    else:
        gate = 1.0 + 0.0 * Aa
        gu = 0.0
    da = _community(M, sa, Na, raw_a, th, inc, phi, chM, mech, ka, gate, gu)
    db = _community(M, sb, Nb, raw_b, th, inc, phi, chM, mech, kb, gate, gu)
    return _pack(M, da + db + [dE, dR])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    n = len(_PER)
    return _pack(M, [s[0] + s[1] + s[2], s[n] + s[n + 1] + s[n + 2]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    z = 0.0 * a + 0.0 * m0
    one = z + 1.0
    return _pack(M, [(1.0 - m0) * a, z, m0 * a, z, z, z, z, z, z, z, one,
                     (1.0 - m0) * b, z, m0 * b, z, z, z, z, z, z, z, one, z, z])
