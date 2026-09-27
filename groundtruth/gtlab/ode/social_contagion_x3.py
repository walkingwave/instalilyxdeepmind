"""Grey-box ODE: social_contagion x3 (rebuild from the brief, bridge as introductions). NUMPY + math only.

Per community i in {a, b} (j = the other community):
  pool P_i = (N_i - A_i - W_i - D_i)_+, A_i = L_i + M_i + X_i observed, W_i = waiting cohorts.
  Seeding effort s is split by bridge beta: local s(1 - beta), introductions s beta.
  rho_i = K_i [s_i s (1 - beta) + k_Ii s beta A_j/N_j + q A_i/N_i + x_q R A_j/N_j + o],
  r_i = rho_i/(1 + rho_i/1.5). Introductions reach people through the other community's members.
  A share phi(c) = c/(c + 0.7) of new interest carries a paid promise. Onboarding: Erlang-3 chain,
  mean tau_on, for unpromised (W1..W3) and promised (V1..V3) cohorts; unpromised -> L, promised -> M.
  Paid offer converts L -> M at k_conv c. Churn: L at c_L, X (initial incentive-led members, no
  promise) at k_X, M at c_L + k_B (E - c)_+. Leavers -> D, back to the pool at 1/tau_D.
A credibility: dK = (1 - K)/50 - k_A K (promised waiting)/N.
B incentive expectations: dE = (c - E)/tau_E.
C cross-community relationships: dR = (s beta/10 - R)/tau_R; cross word of mouth x_q R A_j/N_j.
"""
import math

import numpy as np

FAMILY = "social_contagion_x3"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility erodes with promised waiting cohorts",
         "B": "incentive expectations: incentive-led members leave when incentive falls below memory",
         "C": "cross-community relationships built by introductions, with memory"}
N_SUB = 2

_PER = ["L", "M", "X", "D", "W1", "W2", "W3", "V1", "V2", "V3", "K"]
STATE = [s + "a" for s in _PER] + [s + "b" for s in _PER] + ["E", "R"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array(([1e4] * 10 + [1.0]) * 2 + [2.0, 1.0])

R_MAX = 1.5
PHI_HALF = 0.7
TAU_K = 50.0

PARAMS = [
    ("N_a", 360.0, 60.0, 3000.0, True),
    ("N_b", 350.0, 40.0, 3000.0, True),
    ("s_a", 0.003, 1e-4, 1.0, True),
    ("s_b", 0.0008, 1e-5, 1.0, True),
    ("k_Ia", 0.002, 1e-5, 1.0, True),
    ("k_Ib", 0.005, 1e-5, 1.0, True),
    ("q", 0.015, 1e-3, 1.5, True),
    ("o", 0.001, 1e-5, 0.1, True),
    ("tau_on", 7.0, 1.0, 60.0, True),
    ("c_L", 0.012, 1e-4, 0.5, True),
    ("k_X", 0.12, 0.005, 1.5, True),
    ("m0", 0.33, 0.0, 0.9, False),
    ("k_conv", 0.3, 0.0, 0.7, False),
    ("k_A", 0.1, 0.0, 3.0, False),
    ("k_B", 0.045, 0.0, 0.7, False),
    ("tau_E", 28.0, 1.0, 500.0, True),
    ("tau_D", 13.0, 1.0, 1000.0, True),
    ("x_q", 0.005, 1e-5, 1.0, True),
    ("tau_R", 50.0, 2.0, 2000.0, True),
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


def _community(M, s, N, drive, th, inc, phi, chM, mech):
    L, Mm, X, D, W1, W2, W3, V1, V2, V3, K = s
    A = L + Mm + X
    W = W1 + W2 + W3 + V1 + V2 + V3
    pool = _pos(M, N - A - W - D, 1.0)
    raw = drive + th["q"] * A / N + th["o"]
    if "A" in mech:
        raw = raw * K
    r = raw / (1.0 + raw / R_MAX)
    new = r * pool
    k = 3.0 / th["tau_on"]
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
    fa = (sa[0] + sa[1] + sa[2]) / th["N_a"]
    fb = (sb[0] + sb[1] + sb[2]) / th["N_b"]
    loc = seed * _pos(M, 1.0 - br, 0.01)
    intro = seed * br
    cross_a = th["k_Ia"] * intro * fb
    cross_b = th["k_Ib"] * intro * fa
    dR = 0.0 * R
    if "C" in mech:
        cross_a = cross_a + th["x_q"] * R * fb
        cross_b = cross_b + th["x_q"] * R * fa
        dR = (intro / 10.0 - R) / th["tau_R"]
    da = _community(M, sa, th["N_a"], th["s_a"] * loc + cross_a, th, inc, phi, chM, mech)
    db = _community(M, sb, th["N_b"], th["s_b"] * loc + cross_b, th, inc, phi, chM, mech)
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
