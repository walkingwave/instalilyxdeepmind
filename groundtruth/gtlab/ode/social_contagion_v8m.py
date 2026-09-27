"""Grey-box ODE: social_contagion v8. NUMPY + math only (ships verbatim).

Per community i in {a, b}: loyal L_i, incentive-led M_i, onboarding chain W1_i -> W2_i -> W3_i
(Erlang-3 delay, mean tau_on), initial leavers X_i, disappointed former members D_i.
Pool P_i = (N_i - L_i - M_i - W_i - X_i - D_i)_+ ; observed A_i = L_i + M_i + X_i.
Recruitment per pool member rho_i = s_i s (1 - k_br beta)_+ + q A_i / N_i, times credibility K_i
under A, saturating at R_MAX. Onboarded recruits split phi = c/(c + 0.7) into M, the rest into L;
a paid incentive converts L into M at k_conv c. L churns at c_L, X at k_X, M at c_L (+ the B term).
Every leaver enters D and returns to the pool at 1/tau_D.

A credibility: dK_i = (1 - K_i)/tau_K - k_A K_i W_i / N_i; recruitment x K_i.
B incentive expectations: dE = (c - E)/tau_E; M churn = c_L + k_B (E - c)_+.
C cross-community ties: not modelled (bridge reallocates seeding effort through k_br only).
"""
import math

import numpy as np

FAMILY = "social_contagion_v8m"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility falls with long onboarding queues, scales recruitment",
         "B": "incentive memory; churn of incentive-led members when incentive is removed",
         "C": "cross-community ties (not modelled here)"}
N_SUB = 2
N_STAGE = 3

STATE = ["La", "Ma", "W1a", "W2a", "W3a", "Xa", "Da",
         "Lb", "Mb", "W1b", "W2b", "W3b", "Xb", "Db", "Ka", "Kb", "E"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4] * 14 + [1.0, 1.0, 2.0])

R_MAX = 1.5
PHI_HALF = 0.7

PARAMS = [
    ("N_a", 280.0, 60.0, 3000.0, True),
    ("N_b", 220.0, 40.0, 3000.0, True),
    ("s_a", 0.008, 1e-4, 1.0, True),
    ("s_b", 0.0016, 1e-4, 1.0, True),
    ("q", 0.03, 1e-3, 1.5, True),
    ("tau_on", 8.0, 1.0, 100.0, True),
    ("c_L", 0.01, 1e-4, 0.5, True),
    ("k_X", 0.15, 0.005, 1.5, True),
    ("k_conv", 0.03, 0.0, 0.7, False),
    ("m0", 0.2, 0.0, 0.9, False),
    ("k_A", 0.3, 0.0, 1.5, False),
    ("tau_K", 60.0, 2.0, 300.0, True),
    ("k_B", 0.05, 0.0, 0.7, False),
    ("tau_E", 30.0, 1.0, 500.0, True),
    ("k_br", 0.3, 0.0, 1.0, False),
    ("tau_D", 30.0, 1.0, 1000.0, True),
    ("o", 0.0015, 1e-5, 0.1, True),
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


def _community(M, L, Mm, W1, W2, W3, X, D, K, N, seed_term, th, inc, chM, mech):
    A = L + Mm + X
    W = W1 + W2 + W3
    pool = _pos(M, N - A - W - D, 1.0)
    raw = seed_term + th["q"] * A / N + th["o"]
    if "A" in mech:
        raw = raw * K
    r = raw / (1.0 + raw / R_MAX)
    k = N_STAGE / th["tau_on"]
    on = k * W3
    phi = inc / (inc + PHI_HALF)
    conv = th["k_conv"] * inc * L
    outL = th["c_L"] * L
    outM = chM * Mm
    outX = th["k_X"] * X
    ret = D / th["tau_D"]
    dW1 = r * pool - k * W1
    dW2 = k * W1 - k * W2
    dW3 = k * W2 - on
    dL = (1.0 - phi) * on - outL - conv
    dM = phi * on + conv - outM
    dX = -outX
    dD = outL + outM + outX - ret
    if "A" in mech:
        dK = (1.0 - K) / th["tau_K"] - th["k_A"] * K * W / N
    else:
        dK = 0.0 * K
    return dL, dM, dW1, dW2, dW3, dX, dD, dK


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    La, Ma, W1a, W2a, W3a, Xa, Da, Lb, Mb, W1b, W2b, W3b, Xb, Db, Ka, Kb, E = s
    chM = th["c_L"] + 0.0 * E
    dE = 0.0 * E
    if "B" in mech:
        chM = th["c_L"] + th["k_B"] * _pos(M, E - inc, 0.05)
        dE = (inc - E) / th["tau_E"]
    eff = seed * _pos(M, 1.0 - th["k_br"] * br, 0.01)
    da = _community(M, La, Ma, W1a, W2a, W3a, Xa, Da, Ka, th["N_a"], th["s_a"] * eff, th, inc, chM, mech)
    db = _community(M, Lb, Mb, W1b, W2b, W3b, Xb, Db, Kb, th["N_b"], th["s_b"] * eff, th, inc, chM, mech)
    return _pack(M, list(da[:7]) + list(db[:7]) + [da[7], db[7], dE])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1] + s[5], s[7] + s[8] + s[12]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    z = 0.0 * a + 0.0 * m0
    one = z + 1.0
    return _pack(M, [(1.0 - m0) * a, z, z, z, z, m0 * a, z,
                     (1.0 - m0) * b, z, z, z, z, m0 * b, z, one, one, z])
