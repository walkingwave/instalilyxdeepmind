"""Grey-box ODE: supply_chain canon2 = canon1 with the brief's transport wording taken literally.
NUMPY + math only (ships verbatim).
- Conveyors start empty at reset (the y0 shipments reading seeds nothing).
- Travel commitment is fixed at departure: dispatch splits into a rush lane (share 1 - lead,
  rate kf) and a normal lane (share lead, rate kn) per goods class; goods already on a lane keep
  that lane's rate when lead_time_buy changes. Both lanes feed the class's terminal buffer Q2c,
  which the terminal passes at min(kq2 * Q2, g a0 r, cT (1 - kmT m)).
Rest as canon1 / v8b (maintenance treatment cap, heat/wear, congested transport on total Q).
"""
import math

import numpy as np

FAMILY = "supply_chain_canon2"
OBS = ["shipments", "inventory_supplier", "inventory_retail"]
CTRL = ["order_quantity", "lead_time_buy", "product_mix", "production_effort",
        "receiving_effort", "maintenance"]
MECHS = {"A": "adaptive production commitment (always on)",
         "B": "machine heat/wear on the terminal, reset by maintenance (always on)",
         "C": "unused"}
N_SUB = 2
SCAP = 362.0
KS = 1.5          # max fraction of a stock withdrawn per tick (dispatch from S, sales from R)
TAU_HEAT = 100.0  # ticks to the wear threshold without maintenance (pulse run: tick 101)
WLOSS = 0.5       # throughput lost past the threshold (18.4 -> 9.2)
RCAP = 1e5

STATE = ["S", "C1", "C2", "QAf", "QAn", "QBf", "QBn", "Q2A", "Q2B", "RA", "RB", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([SCAP, 1e4, 1e4, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, RCAP, RCAP, 1.5])

PARAMS = [
    ("p0", 11.3, 1.0, 100.0, True),
    ("kc", 0.45, 0.0, 5.0, False),
    ("tau_c", 10.0, 1.0, 100.0, True),
    ("qm", 1200.0, 50.0, 5000.0, True),
    ("dq", 0.5, 0.01, 5.0, True),
    ("kf", 0.5, 0.02, 3.0, True),
    ("kn", 0.15, 0.005, 3.0, True),
    ("kq2", 0.5, 0.02, 5.0, True),
    ("a0", 57.0, 5.0, 200.0, True),
    ("dem", 27.0, 5.0, 100.0, True),
    ("fb", 0.5, 0.2, 0.8, False),
    ("kd", 0.01, 0.001, 0.2, True),
    ("kcool", 0.0102, 1e-4, 1.0, True),
    ("ws", 0.5, 0.0, 1.0, False),
    ("cT", 80.0, 5.0, 400.0, True),
    ("kmT", 0.7, 0.0, 0.99, False),
    ("wl", 0.5, 0.05, 0.95, False),
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
    S, C1, C2, QAf, QAn, QBf, QBn, Q2A, Q2B, RA, RB, W = s
    order, lead, mix, prod, recv, maint = uu
    P = prod * (th["p0"] + C2)
    room = M.max(th["qm"] - (QAf + QAn + QBf + QBn + Q2A + Q2B), 0.0)
    D = _smin(M, _smin(M, order, th["dq"] * room, 1.0), P + KS * S, 1.0)
    shB = mix * (1.0 - th["ws"] * S / SCAP)
    g = 1.0 - th["wl"] / (1.0 + M.exp(-(W - 1.0) / 0.05))
    Q2 = Q2A + Q2B
    A = _smin(M, _smin(M, th["kq2"] * Q2, g * th["a0"] * recv, 1.0), th["cT"] * (1.0 - th["kmT"] * maint), 1.0)
    fA = Q2A / (Q2 + 1e-6)
    fb = th["fb"]
    sA = _smin(M, th["dem"] * (1.0 - fb) + th["kd"] * RA, KS * RA, 1.0)
    sB = _smin(M, th["dem"] * fb + th["kd"] * RB, KS * RB, 1.0)
    return P, D, shB, A, fA, sA, sB


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    S, C1, C2, QAf, QAn, QBf, QBn, Q2A, Q2B, RA, RB, W = s
    order, lead, maint = uu[0], uu[1], uu[5]
    P, D, shB, A, fA, sA, sB = _flows(M, s, uu, th)
    AA = fA * A
    AB = A - AA
    kf, kn = th["kf"], th["kn"]
    DA = (1.0 - shB) * D
    DB = shB * D
    return _pack(M, [P - D, (th["kc"] * order - C1) / th["tau_c"], (C1 - C2) / th["tau_c"],
                     (1.0 - lead) * DA - kf * QAf, lead * DA - kn * QAn,
                     (1.0 - lead) * DB - kf * QBf, lead * DB - kn * QBn,
                     kf * QAf + kn * QAn - AA, kf * QBf + kn * QBn - AB,
                     AA - sA, AB - sB, (1.0 - maint) / TAU_HEAT - th["kcool"] * maint * W])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    P, D, shB, A, fA, sA, sB = _flows(M, s, uu, th)
    return _pack(M, [A, s[0], s[9] + s[10]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    S = M.min(M.max(y[1], 0.0), SCAP)
    R = 0.5 * M.max(y[2], 0.0)
    z = 0.0 * S
    return _pack(M, [S, z, z, z, z, z, z, z, z, R, R, z])
