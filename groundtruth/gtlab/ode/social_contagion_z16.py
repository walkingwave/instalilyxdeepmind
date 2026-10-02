"""Grey-box ODE: social_contagion z16 (18 parameters) = y13 with the relationship memory fixed at 15 ticks. NUMPY + math only.

x7 (see social_contagion_x7.py for the base equations) with:
  audiences mix differently per community: incentive-led share of interest (1 + k_inc_i phi(c)), k_inc_a = k_inc, k_inc_b free;
  bridge introductions recruit in b: effort_b = s (s_b (1 - beta) + s_xb beta);
  C cross-community relationships: introductions act through relationships R that build and fade with tau_R,
    dR = (s beta/10 - R)/tau_R, b effort from introductions s_xb 10 R (without C: s_xb s beta at once);
  relationship memory tau_R fixed at 15 ticks (y13's free tau_R went to 28-53 and let introductions keep
    recruiting in b for 30-50 ticks after a campaign; the recovery run shows b collapsing instead);
  no workforce ceiling (every fit parked it at its bound).
A credibility: dK = (1 - K)/50 - k_A K (promised waiting)/N.  B incentive expectations: dE = (c - E)/tau_E.
"""
import math

import numpy as np

FAMILY = "social_contagion_z16"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility erodes with promised waiting cohorts",
         "B": "incentive expectations: incentive-led members leave when incentive falls below memory",
         "C": "cross-community relationships built by introductions carry word of mouth"}
N_SUB = 2

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
    ("k_inc", 0.8, 0.0, 3.0, False),
    ("k_inc_b", 0.8, 0.0, 3.0, False),
    ("s_xb", 0.001, 1e-6, 1.0, True),
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


def _community(M, s, N, drive, th, inc, phi, boost, chM, mech, g):
    L, Mm, X, D, W1, W2, W3, V1, V2, V3, K = s
    A = L + Mm + X
    W = W1 + W2 + W3 + V1 + V2 + V3
    pool = _pos(M, N - A - W - D, 1.0)
    raw = (drive + th["q"] * A / N + th["o"]) * boost
    if "A" in mech:
        raw = raw * K
    r = raw / (1.0 + raw / R_MAX)
    new = r * pool
    k = 3.0 * g / th["tau_on"]
    conv = th["k_conv"] * inc * L
    outL = th["c_L"] * L
    outM = chM * Mm
    outX = th["k_X"] * X
    dW1 = (1.0 - phi) * new - k * W1
    dW2 = k * (W1 - W2)
    dW3 = k * (W2 - W3)
    dV1 = phi * new - k * V1
    dV2 = k * (V1 - V2)
    dV3 = k * (V2 - V3)
    dL = k * W3 - outL - conv
    dM = k * V3 + conv - outM
    dX = -outX
    dD = outL + outM + outX - D / th["tau_D"]
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
    boost = 1.0 + th["k_inc"] * phi
    boost_b = 1.0 + th["k_inc_b"] * phi
    loc = seed * _pos(M, 1.0 - br, 0.01)
    Aa = sa[0] + sa[1] + sa[2]
    Ab = sb[0] + sb[1] + sb[2]
    dra = th["s_a"] * loc
    drb = th["s_b"] * loc
    dR = 0.0 * R
    if "C" in mech:
        drb = drb + th["s_xb"] * 10.0 * R
        dR = (seed * br / 10.0 - R) / TAU_R
    else:
        drb = drb + th["s_xb"] * seed * br
    g = 1.0
    da = _community(M, sa, th["N_a"], dra, th, inc, phi, boost, chM, mech, g)
    db = _community(M, sb, th["N_b"], drb, th, inc, phi, boost_b, chM, mech, g)
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
