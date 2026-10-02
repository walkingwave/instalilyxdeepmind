"""Grey-box ODE: market_x2 = agent-flow market: producers' working cash with self-reinforcing financing
drain, a universal fundamental, committed orders that outlive a freeze. NUMPY + math only.

  Q  = full-cash clearing level. Starts at the reset price and relaxes to the common fundamental p_full
       (tau_Q): the reset price premium is not permanent (every run returns to ~92 at zero controls).
  C  = producers' working cash (1 = full at reset). Revenue from final consumption regrows it while trading,
       logistically (revenue scales with funded production): rho g (C + c0)(1 - C).
       Financing drains it: interest on borrowed cash (1 - C + b0) at a rate-driven cost a_r (r/0.1)^n_r,
       and the tax on sales a_x g x/0.05. Borrowing grows as cash falls, so the drain accelerates.
  A  = p_lo + (Q - p_lo) C: producers' reservation value p_lo when cash is gone.
  P, v = committed orders: price velocity v builds toward g s_v tanh(kap (A - P)/s_v) with the
       preparation/execution time tau_v; the price keeps moving while committed orders execute, also
       under a freeze (a new policy does not cancel commitments), and new orders stop when frozen.
  g  = trading gate at the tax x_c (fitted).
  B, W volume terms as before. Depth: D toward D* = d0 (1 - m1 s((x - x_d)/w_d)) (1 - m2 (1-g) r/0.1)
       (dealer quoting thins sharply past a tax level; a freeze strands inventory whose funding costs capacity
       in proportion to the rate), reset-burst inventory G/H settling slower under tax, risk capacity R.
Reset: C = 1, P = A = Q = price0, v = 0, B = volume0 - V0, D = depth0, R = G = H = 0, W = 1.
"""
import math

import numpy as np

FAMILY = "market_x2"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up (burst inventory H)", "B": "risk capacity loss R", "C": "momentum (price follower)"}
N_SUB = 2

STATE = ["B", "P", "v", "C", "D", "R", "H", "Q", "W", "G"]
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1.0, 1e4, 0.9, 200.0, 1e4, 1.0, 200.0])

TAU_B = 2.7
TAU_R = 20.0
W_X = 0.0005
C_G = 0.0184
V0 = 1.77
C_W = 1.139
TAU_W = 70.71
TAU_G = 3.0
W_R = 0.004
TAU_D = 8.0
C_H = 0.0759
TAU_H1 = 9.57
A_H = 38.2

PARAMS = [
    ("c_p", 0.9, 0.01, 10.0, True),
    ("d0", 91.0, 60.0, 130.0, False),
    ("m1", 0.52, 0.0, 0.95, False),
    ("x_d", 0.018, 0.002, 0.035, False),
    ("w_d", 0.003, 0.0005, 0.02, True),
    ("m2", 0.5, 0.0, 0.95, False),
    ("k_R", 0.5, 0.0, 2.0, False),
    ("x_R", 0.034, 0.0, 0.05, False),
    ("w_r", 0.75, 0.0, 1.0, False),
    ("x_c", 0.0445, 0.0426, 0.0458, False),
    ("p_full", 93.0, 80.0, 105.0, False),
    ("tau_Q", 150.0, 10.0, 2000.0, True),
    ("p_lo", 73.5, 55.0, 90.0, False),
    ("rho", 0.03, 0.001, 0.5, True),
    ("c0", 0.1, 0.001, 2.0, True),
    ("a_r", 0.05, 0.001, 2.0, True),
    ("n_r", 2.0, 0.5, 10.0, False),
    ("a_x", 0.01, 0.0, 0.5, False),
    ("b0", 0.2, 0.01, 3.0, True),
    ("kap", 0.088, 0.005, 2.0, True),
    ("s_v", 0.62, 0.05, 5.0, True),
    ("tau_v", 14.0, 0.5, 60.0, True),
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


def _gate(M, tax, th):
    return 1.0 / (1.0 + M.exp((tax - th["x_c"]) / W_X))


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    B, P, v, C, D, R, H, Q, W, G = s
    dB = -B / TAU_B
    g = _gate(M, tax, th)
    rr = M.pw(rate / 0.1, th["n_r"])
    dC = th["rho"] * g * (C + th["c0"]) * (1.0 - C) - (th["a_r"] * rr + th["a_x"] * g * tax / 0.05) * (1.0 - C + th["b0"])
    A = th["p_lo"] + (Q - th["p_lo"]) * C
    dQ = (th["p_full"] - Q) / th["tau_Q"]
    zv = th["kap"] * (A - P) / th["s_v"]
    zv = M.min(M.max(zv, -20.0), 20.0)
    ez = M.exp(2.0 * zv)
    dv = (g * th["s_v"] * (ez - 1.0) / (ez + 1.0) - v) / th["tau_v"]
    dP = v
    fall = 0.5 * (M.sqrt(dP * dP + 0.01) - dP)
    sR = 1.0 / (1.0 + M.exp((tax - th["x_R"]) / W_R))
    dR = (th["k_R"] * fall * (1.0 - th["w_r"] + th["w_r"] * rr / (1.0 + rr)) * sR - R) / TAU_R
    dG = C_H * B - G / TAU_G
    dH = G / TAU_G - H * M.exp(-A_H * tax) / TAU_H1
    sd = 1.0 / (1.0 + M.exp(-(tax - th["x_d"]) / th["w_d"]))
    Dstar = th["d0"] * (1.0 - th["m1"] * sd) * (1.0 - th["m2"] * (1.0 - g) * rate * 10.0)
    dD = (Dstar - D) / TAU_D
    return _pack(M, [dB, dP, dv, dC, dD, dR, dH, dQ, -W / TAU_W, dG])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, C, D, R, H, Q, W, G = s
    rate, tax = uu
    g = _gate(M, tax, th)
    A = th["p_lo"] + (Q - th["p_lo"]) * C
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
    return _pack(M, [B, P, z, z + 1.0, D, z, z, P, z + 1.0, z])
