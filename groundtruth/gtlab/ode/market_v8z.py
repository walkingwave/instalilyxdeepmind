"""Grey-box ODE: market v8z (moving price anchor, speed-limited price, reset imbalance). NUMPY + math only.

Price   The traded price P follows an anchor A (what the hidden stocks support) through the order
        pipeline, with a speed limit:
            dP/dt = g v,   dv/dt = g (s_v tanh(kap (A - P) / s_v) - v) / tau_v
        g = 1 / (1 + exp((tax - X_C) / W_X)) freezes trading above the tax threshold X_C.
        Anchor: Qe = w Q + (1 - w) p_a, Q = price at reset (constant state), p_a = common level;
            phi(r) = (r/R_C)^N / (1 + (r/R_C)^N)    (interest above ~R_C drains working cash)
            A_eq  = Qe + (p_lo - k_x tax g - Qe) phi(r)
            dA/dt = (A_eq - A) / tau_A  when falling,  / tau_Au when recovering (smooth switch).
Volume  v = V0 + B + C_G |A - P| + c_p g |v| + c_W W g (10 rate + 20 tax)
        B = reset order backlog (dB/dt = -B / 2.7); W = reset warehouse imbalance, dW/dt = -W / tau_W,
        W0 = 1: the adjustment flow is larger while controls are on and stops when trading freezes.
Depth   (FRAC D_f + (1 - FRAC) D_s) (1 - R) + H
        D* = d0 exp(-a_t tax); dD_f/dt = (D* - D_f) / TAU_F; dD_s/dt = k(e) e, e = D* - D_s,
        k = k_sd when shrinking, K_SU when refilling (smooth switch);
        R = lost risk capacity, dR/dt = (k_R max(-dP/dt, 0) - R) / TAU_R (adverse moves);
        H = inventory the reset burst brings to dealer books, dH/dt = C_B B - H / TAU_HC.
Reset: B0 = volume0 - V0, P0 = A0 = Q = price0, v0 = 0, D_f0 = D_s0 = depth0, R0 = H0 = 0, W0 = 1.
Mechanism letters accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "market_v8z"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up (burst inventory H)", "B": "risk capacity loss R", "C": "momentum (price follower)"}
N_SUB = 2

STATE = ["B", "P", "v", "A", "Df", "Ds", "R", "H", "Q", "W"]
_NS = len(STATE)
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1e4, 1e4, 1e4, 0.9, 200.0, 1e4, 1.0])

TAU_B = 2.7
TAU_R = 20.0
W_X = 0.0005
N_RC = 7.6
R_C = 0.0795
FRAC = 0.448
TAU_HC = 11.1
X_C = 0.0452
K_SU = 0.133
C_G = 0.0184
V0 = 1.91
TAU_F = 4.0
C_B = 0.05

PARAMS = [
    ("c_p", 0.8396, 0.01, 10.0, True),
    ("c_W", 1.056, 0.0, 5.0, False),
    ("tau_W", 62.15, 5.0, 300.0, True),
    ("d0", 92.12, 60.0, 130.0, False),
    ("a_t", 27.49, 2.0, 100.0, True),
    ("k_sd", 0.02923, 0.005, 0.3, True),
    ("k_R", 0.4666, 0.0, 2.0, False),
    ("p_lo", 69.87, 55.0, 90.0, False),
    ("w", 0.8425, 0.0, 1.0, False),
    ("p_a", 100.1, 80.0, 110.0, False),
    ("k_x", 140.6, 0.0, 400.0, False),
    ("tau_A", 7.647, 2.0, 100.0, True),
    ("tau_Au", 70.8, 5.0, 300.0, True),
    ("kap", 0.09208, 0.005, 2.0, True),
    ("s_v", 0.6021, 0.05, 5.0, True),
    ("tau_v", 15.05, 0.5, 60.0, True),
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
    B, P, v, A, Df, Ds, R, H, Q, W = s
    dB = -B / TAU_B
    g = 1.0 / (1.0 + M.exp((tax - X_C) / W_X))
    hr = M.pw(rate / R_C, N_RC)
    Qe = th["w"] * Q + (1.0 - th["w"]) * th["p_a"]
    Aeq = Qe + (th["p_lo"] - th["k_x"] * tax * g - Qe) * hr / (1.0 + hr)
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
    dH = C_B * B - H / TAU_HC
    Dstar = th["d0"] * M.exp(-th["a_t"] * tax)
    dDf = (Dstar - Df) / TAU_F
    e = Dstar - Ds
    sw = 0.5 * (1.0 + e / M.sqrt(e * e + 4.0))
    dDs = (th["k_sd"] + (K_SU - th["k_sd"]) * sw) * e
    return _pack(M, [dB, dP, dv, dA, dDf, dDs, dR, dH, 0.0 * Q, -W / th["tau_W"]])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, A, Df, Ds, R, H, Q, W = s
    rate, tax = uu
    g = 1.0 / (1.0 + M.exp((tax - X_C) / W_X))
    gap = A - P
    vol = V0 + B + C_G * M.sqrt(gap * gap + 0.25) + th["c_p"] * g * M.sqrt(v * v + 0.0001) + th["c_W"] * W * g * (rate * 10.0 + tax * 20.0)
    dep = (FRAC * Df + (1.0 - FRAC) * Ds) * (1.0 - R) + H
    return _pack(M, [P, vol, dep])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    P = M.max(y[0], 0.0)
    B = M.max(y[1] - V0 - 0.5 * C_G, 0.0)
    D = M.max(y[2], 0.0)
    z = 0.0 * P
    return _pack(M, [B, P, z, P, D, D, z, z, P, z + 1.0])
