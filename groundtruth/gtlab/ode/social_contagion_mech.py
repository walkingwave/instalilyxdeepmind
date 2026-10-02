"""Grey-box ODE: social_contagion, v2 base + three switchable mechanisms (A, B, C).
NUMPY + math only (ships verbatim).

Base, per community i in {a, b}: loyal L_i, incentive-led M_i, onboarding queue W_i, initial
leavers X_i; pool P_i = N_i - L_i - M_i - W_i - X_i (smooth positive part); observed
A_i = L_i + M_i + X_i. Recruitment per pool member rho_i = seed_i + q A_i / N_i (+ cross word
of mouth under C), times credibility K_i under A, saturating at R_MAX. Recruits leave W at
1/tau_on and split phi = c/(c + 0.7) into M, the rest into L; a paid incentive converts L into
M at k_conv c. L churns at c_L, X at k_X. Without B, M churns like L (c_L): the incentive then
has no effect on anyone's churn.

A credibility: K_i in [0, 1], dK_i = (1 - K_i)/tau_K - k_A K_i W_i / N_i; recruitment x K_i.
B incentive expectations: memory E, dE = (c - E)/tau_E; M churn = c_L + k_B (E - c)_+.
C cross-community ties: T in [0, 1], dT = (beta - T)/TAU_T; rho_a += k_C T A_b/N_b, and
   symmetrically for b.
"""
import math

import numpy as np

FAMILY = "social_contagion_mech"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility falls with long onboarding queues, scales recruitment",
         "B": "incentive memory; churn of incentive-led members when incentive is removed",
         "C": "cross-community ties grow with bridge outreach, add cross word of mouth"}
N_SUB = 2

STATE = ["La", "Ma", "Wa", "Xa", "Lb", "Mb", "Wb", "Xb", "Ka", "Kb", "E", "T"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4] * 8 + [1.0, 1.0, 2.0, 1.0])

R_MAX = 1.5
PHI_HALF = 0.7
TAU_T = 30.0

PARAMS = [
    ("N_a", 250.0, 60.0, 3000.0, True),
    ("N_b", 230.0, 40.0, 3000.0, True),
    ("s_a", 0.0045, 1e-4, 1.0, True),
    ("s_b", 0.001, 1e-4, 1.0, True),
    ("q", 0.0124, 1e-3, 1.5, True),
    ("tau_on", 8.4, 1.0, 100.0, True),
    ("c_L", 0.0045, 1e-4, 0.5, True),
    ("k_X", 0.086, 0.005, 1.5, True),
    ("k_conv", 0.05, 0.0, 0.7, False),
    ("m0", 0.165, 0.0, 0.9, False),
    ("k_A", 0.5, 0.0, 1.5, False),
    ("tau_K", 30.0, 2.0, 300.0, True),
    ("k_B", 0.045, 0.0, 0.7, False),
    ("tau_E", 50.0, 1.0, 500.0, True),
    ("k_C", 0.02, 0.0, 1.0, False),
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


def _community(M, L, Mm, W, X, K, N, seed_term, cross, th, inc, chM, mech):
    A = L + Mm + X
    pool = _pos(M, N - A - W, 1.0)
    raw = seed_term + th["q"] * A / N + cross
    if "A" in mech:
        raw = raw * K
    r = raw / (1.0 + raw / R_MAX)
    on = W / th["tau_on"]
    phi = inc / (inc + PHI_HALF)
    conv = th["k_conv"] * inc * L
    dW = r * pool - on
    dL = (1.0 - phi) * on - th["c_L"] * L - conv
    dM = phi * on + conv - chM * Mm
    dX = -th["k_X"] * X
    if "A" in mech:
        dK = (1.0 - K) / th["tau_K"] - th["k_A"] * K * W / N
    else:
        dK = 0.0 * K
    return dL, dM, dW, dX, dK


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    La, Ma, Wa, Xa, Lb, Mb, Wb, Xb, Ka, Kb, E, T = s
    Na, Nb = th["N_a"], th["N_b"]
    chM = th["c_L"] + 0.0 * E
    dE = 0.0 * E
    if "B" in mech:
        chM = th["c_L"] + th["k_B"] * _pos(M, E - inc, 0.05)
        dE = (inc - E) / th["tau_E"]
    ca = 0.0 * T
    cb = 0.0 * T
    dT = 0.0 * T
    if "C" in mech:
        ca = th["k_C"] * T * (Lb + Mb + Xb) / Nb
        cb = th["k_C"] * T * (La + Ma + Xa) / Na
        dT = (br - T) / TAU_T
    dLa, dMa, dWa, dXa, dKa = _community(M, La, Ma, Wa, Xa, Ka, Na, th["s_a"] * seed, ca, th, inc, chM, mech)
    dLb, dMb, dWb, dXb, dKb = _community(M, Lb, Mb, Wb, Xb, Kb, Nb, th["s_b"] * seed, cb, th, inc, chM, mech)
    return _pack(M, [dLa, dMa, dWa, dXa, dLb, dMb, dWb, dXb, dKa, dKb, dE, dT])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1] + s[3], s[4] + s[5] + s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    z = 0.0 * a + 0.0 * m0
    one = z + 1.0
    return _pack(M, [(1.0 - m0) * a, z, z, m0 * a, (1.0 - m0) * b, z, z, m0 * b, one, one, z, z])
