"""Grey-box ODE: market v9c = v9b + burst inventory through a settlement stage, H clears slower under tax. NUMPY + math only.

Price / volume exactly as market_v8t2 (moving anchor, speed-limited follower, reset imbalance W).
Depth   depth = D (1 - R) + H
        D  = quoted capacity, one pool: dD/dt = (D* - D) / tau_D  (tau_D ~ 7.5: every mid-run step,
             up or down, is a single exponential with that time constant)
        D* = d0 (1 - m1 tax / (tax + X_H)) (1 - m2 (1 - g)): any interior tax removes about half the
             capacity (plateau ~44-47 from tax 0.025 to 0.0425), a trading freeze removes about half again
        H  = inventory the reset order burst puts on dealer books (reset only, never from a mid-run step),
             through one settlement stage G: dG/dt = c_H B - G / TAU_G, dH/dt = G / TAU_G - H k_H,
             k_H = g exp(-a_H tax) / tau_H1 + (1 - g) / tau_H0 (clears through trading; a tax slows it,
             a freeze stalls it)
        R  = lost risk capacity, dR/dt = (k_R max(-dP/dt, 0) - R) / TAU_R.
Reset: B0 = volume0 - V0, P0 = A0 = Q = price0, v0 = 0, D0 = depth0, R0 = H0 = 0, W0 = 1.
Mechanism letters accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "market_v9c"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up (burst inventory H)", "B": "risk capacity loss R", "C": "momentum (price follower)"}
N_SUB = 2

STATE = ["B", "P", "v", "A", "D", "R", "H", "Q", "W", "G"]
_NS = len(STATE)
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1e4, 1e4, 0.9, 200.0, 1e4, 1.0, 200.0])

TAU_B = 2.7
TAU_R = 20.0
W_X = 0.0005
N_RC = 7.6
X_C = 0.0452
C_G = 0.0184
V0 = 1.77
X_H = 0.005
C_W = 1.139
TAU_W = 70.71
R_C = 0.0795
TAU_G = 3.0

PARAMS = [
    ("c_p", 1.076, 0.01, 10.0, True),
    ("d0", 90.5, 60.0, 130.0, False),
    ("m1", 0.55, 0.0, 0.95, False),
    ("m2", 0.5, 0.0, 0.95, False),
    ("tau_D", 7.5, 1.0, 40.0, True),
    ("c_H", 0.08, 0.0, 1.0, False),
    ("tau_H1", 12.0, 1.0, 200.0, True),
    ("tau_H0", 40.0, 1.0, 400.0, True),
    ("a_H", 10.0, 0.0, 100.0, False),
    ("k_R", 0.4545, 0.0, 2.0, False),
    ("p_lo", 69.85, 55.0, 90.0, False),
    ("k_x", 140.9, 0.0, 400.0, False),
    ("tau_A", 9.551, 2.0, 100.0, True),
    ("tau_Au", 58.13, 5.0, 300.0, True),
    ("kap", 0.0882, 0.005, 2.0, True),
    ("s_v", 0.6322, 0.05, 5.0, True),
    ("tau_v", 14.44, 0.5, 60.0, True),
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

    @staticmethod
    def pw(a, b):
        return (a if a > 1e-12 else 1e-12) ** b


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))

    @staticmethod
    def pw(a, b):
        return np.power(np.maximum(a, 1e-12), b)


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


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    B, P, v, A, D, R, H, Q, W, G = s
    dB = -B / TAU_B
    g = 1.0 / (1.0 + M.exp((tax - X_C) / W_X))
    hr = M.pw(rate / R_C, N_RC)
    Aeq = Q + (th["p_lo"] - th["k_x"] * tax * g - Q) * hr / (1.0 + hr)
    ea = Aeq - A
    su = 0.5 * (1.0 + ea / M.sqrt(ea * ea + 1.0))
    dA = ea * ((1.0 - su) / th["tau_A"] + su / th["tau_Au"])
    dP = g * v
    zv = th["kap"] * (A - P) / th["s_v"]
    zv = M.min(M.max(zv, -20.0), 20.0)
    ez = M.exp(2.0 * zv)
    dv = g * (th["s_v"] * (ez - 1.0) / (ez + 1.0) - v) / th["tau_v"]
    fall = 0.5 * (M.sqrt(dP * dP + 0.01) - dP)
    dR = (th["k_R"] * fall - R) / TAU_R
    dG = th["c_H"] * B - G / TAU_G
    dH = G / TAU_G - H * (g * M.exp(-th["a_H"] * tax) / th["tau_H1"] + (1.0 - g) / th["tau_H0"])
    Dstar = th["d0"] * (1.0 - th["m1"] * tax / (tax + X_H)) * (1.0 - th["m2"] * (1.0 - g))
    dD = (Dstar - D) / th["tau_D"]
    return _pack(M, [dB, dP, dv, dA, dD, dR, dH, 0.0 * Q, -W / TAU_W, dG])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, A, D, R, H, Q, W, G = s
    rate, tax = uu
    g = 1.0 / (1.0 + M.exp((tax - X_C) / W_X))
    gap = A - P
    vol = V0 + B + C_G * M.sqrt(gap * gap + 0.25) + th["c_p"] * g * M.sqrt(v * v + 0.0001) + C_W * W * g * (rate * 10.0 + tax * 20.0)
    dep = D * (1.0 - R) + H
    return _pack(M, [P, vol, dep])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    P = M.max(y[0], 0.0)
    B = M.max(y[1] - V0 - 0.5 * C_G, 0.0)
    D = M.max(y[2], 0.0)
    z = 0.0 * P
    return _pack(M, [B, P, z, P, D, z, z, P, z + 1.0, z])
