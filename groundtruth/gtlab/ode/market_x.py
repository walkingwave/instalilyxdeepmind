"""Grey-box ODE: market_x = agent-flow market with a producers' working-cash stock. NUMPY + math only.

Physical reading of the brief:
  C  = producers' working cash (fraction of full; 1 at reset). Production consumes it, final consumption
       returns revenue, so it regrows logistically while trading (revenue scales with the cash that funds
       production): rho g (C + c0)(1 - C). Interest on borrowed working cash drains it above the producers'
       margin (Hill threshold phi(r)): - delta phi(r) C. A transcritical balance: small rates lower C a
       little, rates past the margin drive C to its floor.
  A  = clearing price the cash state supports: producers' reservation value p_f when cash is gone,
       the reset price Q when cash is full: A = p_f + (Q - p_f) C, p_f = p_lo - k_x x g.
  P, v = committed orders move the price: preparation/execution inertia tau_v and a top speed s_v
       (a new policy does not cancel committed orders); the price moves only while trading (g).
  B  = reset order burst (execution, tau 2.7); W = warehouses' half-full reset imbalance, adjusting while
       controls are on and trading is open.
  D  = quoted dealer capacity (one pool, tau 8) toward D* = d0 (1 - m1 x/(x+X_H)) (1 - m2 (1-g) r/0.1):
       a tax thins dealer quoting; a freeze strands inventory on books and the funding it ties up costs
       capacity in proportion to the rate.
  G, H = reset-burst inventory on dealer books through one settlement stage; settlement slows under tax.
  R  = risk capacity lost after adverse moves (rate-weighted, not when a high tax has thinned inventory).
Reset: C = 1, P = A = Q = price0, v = 0, B = volume0 - V0, D = depth0, R = G = H = 0, W = 1.
Mechanism letters accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "market_x"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up (burst inventory H)", "B": "risk capacity loss R", "C": "momentum (price follower)"}
N_SUB = 2

STATE = ["B", "P", "v", "C", "D", "R", "H", "Q", "W", "G"]
_NS = len(STATE)
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1.0, 1e4, 0.9, 200.0, 1e4, 1.0, 200.0])

TAU_B = 2.7
TAU_R = 20.0
W_X = 0.0005
N_RC = 7.6
R_C = 0.0795
X_C = 0.0452
C_G = 0.0184
V0 = 1.77
X_H = 0.005
C_W = 1.139
TAU_W = 70.71
TAU_G = 3.0
W_R = 0.004
TAU_D = 8.0

PARAMS = [
    ("c_p", 0.96, 0.01, 10.0, True),
    ("d0", 91.5, 60.0, 130.0, False),
    ("m1", 0.58, 0.0, 0.95, False),
    ("m2", 0.5, 0.0, 0.95, False),
    ("c_H", 0.076, 0.0, 1.0, False),
    ("tau_H1", 9.6, 1.0, 200.0, True),
    ("a_H", 38.0, 0.0, 100.0, False),
    ("k_R", 0.49, 0.0, 2.0, False),
    ("x_R", 0.0337, 0.0, 0.05, False),
    ("w_r", 0.77, 0.0, 1.0, False),
    ("p_lo", 73.0, 55.0, 90.0, False),
    ("k_x", 20.0, 0.0, 400.0, False),
    ("rho", 0.03, 0.001, 0.5, True),
    ("c0", 0.2, 0.001, 2.0, True),
    ("delta", 0.1, 0.005, 2.0, True),
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


def _anchor(M, C, Q, tax, g, th):
    pf = th["p_lo"] - th["k_x"] * tax * g
    return pf + (Q - pf) * C


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    B, P, v, C, D, R, H, Q, W, G = s
    dB = -B / TAU_B
    g = 1.0 / (1.0 + M.exp((tax - X_C) / W_X))
    hr = M.pw(rate / R_C, N_RC)
    phi = hr / (1.0 + hr)
    dC = th["rho"] * g * (C + th["c0"]) * (1.0 - C) - th["delta"] * phi * C
    A = _anchor(M, C, Q, tax, g, th)
    dP = g * v
    zv = th["kap"] * (A - P) / th["s_v"]
    zv = M.min(M.max(zv, -20.0), 20.0)
    ez = M.exp(2.0 * zv)
    dv = g * (th["s_v"] * (ez - 1.0) / (ez + 1.0) - v) / th["tau_v"]
    fall = 0.5 * (M.sqrt(dP * dP + 0.01) - dP)
    sR = 1.0 / (1.0 + M.exp((tax - th["x_R"]) / W_R))
    dR = (th["k_R"] * fall * (1.0 - th["w_r"] + th["w_r"] * phi) * sR - R) / TAU_R
    dG = th["c_H"] * B - G / TAU_G
    dH = G / TAU_G - H * M.exp(-th["a_H"] * tax) / th["tau_H1"]
    Dstar = th["d0"] * (1.0 - th["m1"] * tax / (tax + X_H)) * (1.0 - th["m2"] * (1.0 - g) * rate * 10.0)
    dD = (Dstar - D) / TAU_D
    return _pack(M, [dB, dP, dv, dC, dD, dR, dH, 0.0 * Q, -W / TAU_W, dG])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, C, D, R, H, Q, W, G = s
    rate, tax = uu
    g = 1.0 / (1.0 + M.exp((tax - X_C) / W_X))
    A = _anchor(M, C, Q, tax, g, th)
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
