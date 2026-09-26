"""Grey-box ODE: social_contagion, alternate v3 (capped conversion share rho, no bridge). See social_contagion_min.py.
NUMPY + math only (ships verbatim).

Per community i in {a, b}: L_i loyal members, M_i incentive-expecting members, W_i onboarding
queue, X_i initial members who leave regardless (the dip every run starts with, incentive or not).
pool_i = N_i - L_i - M_i - W_i - X_i. Everyone joins through W (lag tau_on). Recruitment per
pool member = seeding share + word of mouth q * A_i / N_i, saturating at R_MAX. Seeding goes to
both communities at fixed rates s_a s and s_b s (b receives recruits even at bridge 0; a bridge
split fitted to exactly zero on the compose run, so bridge is not used). Recruits leaving W are
split by the incentive at that moment: phi = inc / (inc + PHI_HALF) into M, the rest into L.
While the incentive is paid it also converts loyal members into incentive-expecting ones at
k_conv * inc * (1 - M / (rho N))_+ : at most a share rho of the community is incentive-led.
M stays while paid (churn k_M / (1 + k_ret inc)) and churns at k_M once it stops. Loyal members
churn at c_L. X churns at k_M. Churned members return to the pool.
Initial members: fraction m0 in X, the rest in L; M and W start empty (brief).

Mechanism letters are accepted for interface compatibility but every term is always active.
"""
import math

import numpy as np

FAMILY = "social_contagion_min2"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "onboarding queue (always on)", "B": "incentive expectation + churn (always on)", "C": "unused"}
N_SUB = 2

STATE = ["La", "Ma", "Wa", "Xa", "Lb", "Mb", "Wb", "Xb"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e4)

R_MAX = 1.5
PHI_HALF = 0.7

PARAMS = [
    ("N_a", 215.0, 60.0, 3000.0, True),
    ("N_b", 160.0, 40.0, 3000.0, True),
    ("s_a", 0.006, 1e-4, 1.0, True),
    ("s_b", 0.0025, 1e-4, 1.0, True),
    ("q", 0.03, 1e-3, 1.5, True),
    ("tau_on", 6.0, 1.0, 100.0, True),
    ("c_L", 0.012, 1e-4, 0.5, True),
    ("k_M", 0.09, 0.005, 1.5, True),
    ("k_ret", 15.0, 0.0, 50.0, False),
    ("k_conv", 0.01, 0.0, 0.2, False),
    ("m0", 0.25, 0.0, 0.9, False),
    ("rho", 0.8, 0.3, 1.0, False),
]


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


def _community(M, L, Mm, W, X, N, seed_term, th, inc):
    A = L + Mm + X
    pool = _pos(M, N - A - W, 1.0)
    raw = seed_term + th["q"] * A / N
    r = raw / (1.0 + raw / R_MAX)
    inflow = r * pool
    on = W / th["tau_on"]
    phi = inc / (inc + PHI_HALF)
    chM = th["k_M"] / (1.0 + th["k_ret"] * inc)
    conv = th["k_conv"] * inc * L * _pos(M, 1.0 - Mm / (th["rho"] * N), 0.02)
    dW = inflow - on
    dL = (1.0 - phi) * on - th["c_L"] * L - conv
    dM = phi * on + conv - chM * Mm
    dX = -th["k_M"] * X
    return dL, dM, dW, dX


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    La, Ma, Wa, Xa, Lb, Mb, Wb, Xb = s
    sa = th["s_a"] * seed
    sb = th["s_b"] * seed
    dLa, dMa, dWa, dXa = _community(M, La, Ma, Wa, Xa, th["N_a"], sa, th, inc)
    dLb, dMb, dWb, dXb = _community(M, Lb, Mb, Wb, Xb, th["N_b"], sb, th, inc)
    return _pack(M, [dLa, dMa, dWa, dXa, dLb, dMb, dWb, dXb])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1] + s[3], s[4] + s[5] + s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    z = 0.0 * a
    return _pack(M, [(1.0 - m0) * a, z, z, m0 * a, (1.0 - m0) * b, z, z, m0 * b])
