"""Grey-box ODE: social_contagion, minimal two-class adoption with an onboarding queue.
NUMPY + math only (ships verbatim).

Per community i in {a, b}: L_i loyal members, M_i incentive-led members, W_i onboarding queue,
pool_i = N_i - L_i - M_i - W_i. Everyone joins through W (onboarding lag tau_on). Recruitment
per pool member r_i = seeding share (a gets 1 - bridge, b gets bridge) scaled by the incentive,
plus word of mouth q * A_i / N_i. Recruits split into M with fraction
phi = phi_max * inc / (inc + PHI_HALF); the half-point is fixed (only inc in {0, 2} was observed;
a free half-point runs to zero and makes a cliff at tiny incentives). Loyal members churn at
c_L; incentive-led members churn at k_M / (1 + k_ret * inc), so they stay while the incentive is
paid and leave fast when it stops. Churned members return to the pool.
Initial members: fraction m0 in M, the rest in L; queues start empty (brief).

Mechanism letters are accepted for interface compatibility but every term is always active.
"""
import math

import numpy as np

FAMILY = "social_contagion_min"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "onboarding queue (always on)", "B": "incentive-led churn (always on)", "C": "unused"}
N_SUB = 2

STATE = ["La", "Ma", "Wa", "Lb", "Mb", "Wb"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e4)

R_MAX = 1.5
PHI_HALF = 0.7

PARAMS = [
    ("N_a", 220.0, 60.0, 3000.0, True),
    ("N_b", 150.0, 40.0, 3000.0, True),
    ("s_a", 0.02, 1e-4, 1.0, True),
    ("s_b", 0.005, 1e-4, 1.0, True),
    ("q", 0.05, 1e-3, 1.5, True),
    ("k_inc", 1.0, 0.0, 5.0, False),
    ("tau_on", 15.0, 1.0, 100.0, True),
    ("phi_max", 0.8, 0.0, 1.0, False),
    ("c_L", 0.015, 1e-4, 0.5, True),
    ("k_M", 0.1, 0.005, 1.5, True),
    ("k_ret", 20.0, 0.0, 50.0, False),
    ("m0", 0.3, 0.0, 0.9, False),
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


def _community(M, L, W, Mm, N, seed_term, th, inc):
    A = L + Mm
    pool = _pos(M, N - A - W, 1.0)
    raw = seed_term + th["q"] * A / N
    r = raw / (1.0 + raw / R_MAX)
    inflow = r * pool
    on = W / th["tau_on"]
    phi = th["phi_max"] * inc / (inc + PHI_HALF)
    chM = th["k_M"] / (1.0 + th["k_ret"] * inc)
    dW = inflow - on
    dL = (1.0 - phi) * on - th["c_L"] * L
    dM = phi * on - chM * Mm
    return dL, dM, dW


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    La, Ma, Wa, Lb, Mb, Wb = s
    boost = seed * (1.0 + th["k_inc"] * inc)
    sa = th["s_a"] * boost * (1.0 - br)
    sb = th["s_b"] * boost * br
    dLa, dMa, dWa = _community(M, La, Wa, Ma, th["N_a"], sa, th, inc)
    dLb, dMb, dWb = _community(M, Lb, Wb, Mb, th["N_b"], sb, th, inc)
    return _pack(M, [dLa, dMa, dWa, dLb, dMb, dWb])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1], s[3] + s[4]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    return _pack(M, [(1.0 - m0) * a, m0 * a, 0.0 * a, (1.0 - m0) * b, m0 * b, 0.0 * b])
