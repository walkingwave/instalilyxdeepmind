"""Grey-box ODE: supply_chain, single-class variant of supply_chain_min v7. NUMPY + math only.

Same production, dispatch, conveyor, terminal and wear structure as supply_chain_min (v7), with
one retail stock R selling at smin(dem + kd R, KS R). 7 states, 11 parameters.
"""
import math

import numpy as np

FAMILY = "supply_chain_min2"
OBS = ["shipments", "inventory_supplier", "inventory_retail"]
CTRL = ["order_quantity", "lead_time_buy", "product_mix", "production_effort",
        "receiving_effort", "maintenance"]
MECHS = {"A": "adaptive production commitment (always on)",
         "B": "machine heat/wear on the terminal, reset by maintenance (always on)",
         "C": "unused"}
N_SUB = 2
SCAP = 362.0
KS = 1.5
TAU_HEAT = 100.0
WLOSS = 0.5
RCAP = 1e5

STATE = ["S", "C1", "C2", "Q1", "Q2", "R", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([SCAP, 1e4, 1e4, 1e6, 1e6, RCAP, 1.5])

PARAMS = [
    ("p0", 11.3, 1.0, 100.0, True),
    ("kc", 0.45, 0.0, 5.0, False),
    ("tau_c", 10.0, 1.0, 100.0, True),
    ("d0", 0.0, 0.0, 100.0, False),
    ("d1", 18.0, 0.0, 100.0, False),
    ("kq0", 1.0, 0.1, 1.5, True),
    ("kl", 8.0, 0.0, 60.0, False),
    ("a0", 57.0, 5.0, 200.0, True),
    ("dem", 27.0, 5.0, 100.0, True),
    ("kd", 0.01, 0.001, 0.2, True),
    ("kcool", 0.07, 0.01, 1.0, True),
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


def _smin(M, a, b, e):
    return 0.5 * (a + b - M.sqrt((a - b) * (a - b) + e * e))


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


def _flows(M, s, uu, th):
    S, C1, C2, Q1, Q2, R, W = s
    order, lead, mix, prod, recv, maint = uu
    P = prod * (th["p0"] + C2)
    dcap = th["d0"] + th["d1"] * recv
    D = _smin(M, _smin(M, order, dcap, 1.0), P + KS * S, 1.0)
    kq = th["kq0"] / (1.0 + th["kl"] * lead)
    g = 1.0 - WLOSS / (1.0 + M.exp(-(W - 1.0) / 0.05))
    A = _smin(M, kq * Q2, g * th["a0"] * recv, 1.0)
    sale = _smin(M, th["dem"] + th["kd"] * R, KS * R, 1.0)
    return P, D, kq, A, sale


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    S, C1, C2, Q1, Q2, R, W = s
    order, maint = uu[0], uu[5]
    P, D, kq, A, sale = _flows(M, s, uu, th)
    dS = P - D
    dC1 = (th["kc"] * order - C1) / th["tau_c"]
    dC2 = (C1 - C2) / th["tau_c"]
    dQ1 = D - kq * Q1
    dQ2 = kq * Q1 - A
    dR = A - sale
    dW = (1.0 - maint) / TAU_HEAT - th["kcool"] * maint * W
    return _pack(M, [dS, dC1, dC2, dQ1, dQ2, dR, dW])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    P, D, kq, A, sale = _flows(M, s, uu, th)
    return _pack(M, [A, s[0], s[5]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    ship = M.max(y[0], 0.0)
    S = M.min(M.max(y[1], 0.0), SCAP)
    R = M.max(y[2], 0.0)
    z = 0.0 * ship
    return _pack(M, [S, z, z, ship, z, R, z])
