"""Grey-box ODE: supply_chain, minimal stock-flow. NUMPY + math only (ships verbatim).

Supplier stock S is filled by production and hard-capped at SCAP (361.6 +- 1.2 in both real
runs). Production = effort * (p0 + C2), where C1 -> C2 is a
two-stage commitment cascade that follows kc * order_quantity with time constant tau_c
(adaptive production commitments: the pulse run shows production jumping from ~19 to ~70 per
tick about 30 ticks after orders start). Orders withdraw only available stock, through a
dispatch cap: D = smin(smin(order, S), dcap). Dispatched goods travel a two-stage conveyor
Q1 -> Q2 at rate kq = kq0 / (1 + kl * lead_time_buy) and arrive at retail at
A = smin(kq * Q2, g * a0). Machine heat W rises at
(1 - maintenance) / tau_heat and cools at kcool * maintenance * W; once W passes 1 the
terminal throughput gate g = 1 - wloss * sigmoid((W - 1) / 0.05) drops (the pulse run halves
shipments at tick 101 after 100 ticks without maintenance and recovers within ~10 ticks of
maintenance). Retail stock R sells at dem + kd * R (smooth-min with R so it stops at zero;
the real drain is ~17/tick at low stock and 24-30/tick at 130-210 units) and is hard-capped at
RCAP = 5000.0   # interior hold (p3.hold_mid) showed retail stock past 1,100 and still rising: no 362 cap on retail

x0: S = inventory_supplier, R = inventory_retail, Q1 = shipments (initial in-transit content),
C1 = C2 = Q2 = W = 0.

Mechanism letters are accepted for interface compatibility but every term is always active;
fit with --mech AB.
"""
import math

import numpy as np

FAMILY = "supply_chain_min"
OBS = ["shipments", "inventory_supplier", "inventory_retail"]
CTRL = ["order_quantity", "lead_time_buy", "product_mix", "production_effort",
        "receiving_effort", "maintenance"]
MECHS = {"A": "adaptive production commitment (always on)",
         "B": "machine heat/wear on the terminal, reset by maintenance (always on)",
         "C": "unused"}
N_SUB = 2
SCAP = 362.0

STATE = ["S", "C1", "C2", "Q1", "Q2", "R", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
RCAP = 5000.0   # interior hold (p3.hold_mid) showed retail stock past 1,100 and still rising: no 362 cap on retail
STATE_HI = np.array([SCAP, 1e4, 1e4, 1e6, 1e6, RCAP, 1.5])

PARAMS = [
    ("p0", 11.3, 1.0, 100.0, True),
    ("kc", 0.45, 0.0, 5.0, False),
    ("tau_c", 12.0, 2.0, 100.0, True),
    ("dcap", 27.0, 5.0, 300.0, True),
    ("kq0", 1.0, 0.1, 3.0, True),
    ("kl", 8.0, 0.0, 60.0, False),
    ("a0", 20.0, 1.0, 100.0, True),
    ("dem", 17.0, 1.0, 100.0, True),
    ("kd", 0.05, 0.02, 0.5, True),
    ("tau_heat", 100.0, 20.0, 1000.0, True),
    ("wloss", 0.5, 0.0, 0.9, False),
    ("kcool", 0.06, 0.01, 1.0, True),
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
    D = _smin(M, _smin(M, order, S, 1.0), th["dcap"], 1.0)
    kq = th["kq0"] / (1.0 + th["kl"] * lead)
    g = 1.0 - th["wloss"] / (1.0 + M.exp(-(W - 1.0) / 0.05))
    A = _smin(M, kq * Q2, g * th["a0"], 1.0)
    sale = _smin(M, th["dem"] + th["kd"] * R, R, 1.0)
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
    dW = (1.0 - maint) / th["tau_heat"] - th["kcool"] * maint * W
    return _pack(M, [dS, dC1, dC2, dQ1, dQ2, dR, dW])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    P, D, kq, A, sale = _flows(M, s, uu, th)
    return _pack(M, [A, s[0], s[5]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    ship = M.max(y[0], 0.0)
    S = M.min(M.max(y[1], 0.0), SCAP)
    R = M.min(M.max(y[2], 0.0), RCAP)
    z = 0.0 * ship
    return _pack(M, [S, z, z, ship, z, R, z])
