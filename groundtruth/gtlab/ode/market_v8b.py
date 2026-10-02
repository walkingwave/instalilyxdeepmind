"""Grey-box ODE: market v8 (moving price anchor, risk-capacity depth loss). NUMPY + math only.

Price   = a damped follower of a slow anchor A. At reset A = price0 (the price sits at the level
          the hidden stocks support). A relaxes to A_eq(rate) = p_a - dl * (rate / 0.1)^gam with
          time constant tau_A (interest drains working cash, which lowers what buyers pay; with the
          rate off the anchor returns to p_a). The price follows through the order pipeline:
          dP/dt = g * v, dv/dt = g * (kap * (A - P) - v) / tau_v, a second-order (S-shaped) follower.
          g = transaction-tax gate, 1 / (1 + exp((tax - x_c) / 0.0005)): above x_c trading freezes.
Volume  = v0 + B + c_g * |A - P|: base flow, the reset order backlog B (tau 2.7), and trading
          driven by the gap between the anchor and the traded price.
Depth   = (frac * D_f + (1 - frac) * D_s) * (1 - R) + H:
          two dealer pools relaxing to d0 * exp(-a_t * tax) (fast pool tau_f; slow pool k_sd when
          shrinking, k_su when refilling); R = lost risk capacity, dR/dt = (k_R * fall - R) / TAU_R,
          fall = smooth max(-dP/dt, 0) (adverse moves cut dealer capacity); H = inventory the
          reset burst brings to dealer books, dH/dt = c_B * B - H / TAU_H.

Reset: B0 = volume0 - v0, P0 = A0 = price0, velocity 0, D_f0 = D_s0 = depth0, R0 = H0 = 0.
Mechanism letters accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "market_v8b"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up (burst inventory H)", "B": "risk capacity loss R", "C": "momentum (price follower)"}
N_SUB = 2

STATE = ["B", "P", "v", "A", "Df", "Ds", "R", "H"]
_NS = len(STATE)
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1e4, 1e4, 1e4, 0.9, 200.0])

TAU_B = 2.7
TAU_R = 20.0
TAU_H = 8.0
W_X = 0.0005

PARAMS = [
    ("v0", 1.9, 0.5, 6.0, True),
    ("c_g", 0.1, 0.001, 1.0, True),
    ("d0", 90.0, 60.0, 130.0, False),
    ("a_t", 25.0, 2.0, 100.0, True),
    ("tau_f", 5.0, 1.5, 20.0, True),
    ("k_sd", 0.03, 0.005, 0.3, True),
    ("k_su", 0.13, 0.005, 0.5, True),
    ("frac", 0.35, 0.05, 0.95, False),
    ("c_B", 0.02, 0.0005, 0.3, True),
    ("k_R", 0.3, 0.0, 2.0, False),
    ("p_a", 94.0, 80.0, 110.0, False),
    ("dl", 22.0, 0.0, 60.0, False),
    ("gam", 1.5, 0.5, 4.0, False),
    ("tau_A", 45.0, 15.0, 300.0, True),
    ("kap", 0.1, 0.005, 2.0, True),
    ("tau_v", 5.0, 0.5, 60.0, True),
    ("x_c", 0.0442, 0.042, 0.047, False),
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
    B, P, v, A, Df, Ds, R, H = s
    dB = -B / TAU_B
    g = 1.0 / (1.0 + M.exp((tax - th["x_c"]) / W_X))
    Aeq = th["p_a"] - th["dl"] * M.pw(rate * 10.0, th["gam"])
    dA = (Aeq - A) / th["tau_A"]
    dP = g * v
    dv = g * (th["kap"] * (A - P) - v) / th["tau_v"]
    fall = 0.5 * (M.sqrt(dP * dP + 0.01) - dP)
    dR = (th["k_R"] * fall - R) / TAU_R
    dH = th["c_B"] * B - H / TAU_H
    Dstar = th["d0"] * M.exp(-th["a_t"] * tax)
    dDf = (Dstar - Df) / th["tau_f"]
    e = Dstar - Ds
    sw = 0.5 * (1.0 + e / M.sqrt(e * e + 4.0))
    dDs = (th["k_sd"] + (th["k_su"] - th["k_sd"]) * sw) * e
    return _pack(M, [dB, dP, dv, dA, dDf, dDs, dR, dH])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, A, Df, Ds, R, H = s
    gap = A - P
    vol = th["v0"] + B + th["c_g"] * M.sqrt(gap * gap + 0.25)
    dep = (th["frac"] * Df + (1.0 - th["frac"]) * Ds) * (1.0 - R) + H
    return _pack(M, [P, vol, dep])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    P = M.max(y[0], 0.0)
    B = M.max(y[1] - th["v0"] - 0.5 * th["c_g"], 0.0)
    D = M.max(y[2], 0.0)
    z = 0.0 * P
    return _pack(M, [B, P, z, P, D, D, z, z])
