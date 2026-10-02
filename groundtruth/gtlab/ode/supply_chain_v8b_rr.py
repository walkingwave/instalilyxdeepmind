"""Reset rule copy (organizer init doc): internal buffers and conveyors start empty (Q1A = Q1B = 0).
Grey-box ODE: supply_chain v8b = v8 with congested transport instead of a receiving-scaled
dispatch cap: D = smin(smin(order, dq * (qm - Q)+), P + KS S), Q = all goods on the conveyors.
Pulse run: production ~65/tick all dispatched while the conveyor fills behind the terminal
(19/tick at receiving 0.35); once Q nears qm (tick ~31) dispatch throttles to the terminal
rate and the supplier refills at ~45/tick. The ~1,000-unit backlog is what arrives after the
pulse. Rest as v8.

v8 notes: NUMPY + math only (ships verbatim).

v7b (supply_chain_min) plus class-resolved transport. Dispatch D is split into the two goods
classes: class B (selected by product_mix) takes the share mix * (1 - ws * S / SCAP), class A
the rest. With the supplier empty all goods flow straight from the line at the requested mix;
once the supplier holds stock, dispatch draws old batches and the class-B share falls. Each
class travels its own two-stage conveyor (Q1c -> Q2c, rate kq); the terminal passes
A = smin(kq (Q2A + Q2B), g a0 r) split pro rata to the class contents at the terminal, and
class arrivals feed their own retail stock RA, RB, each sold at dem_c + kd R_c.
Pulse run: before the supplier refills (tick 31) arrivals are 80% class B and the retail
stock builds (class A sells out); ~12 ticks later class A share rises, class-B stock drains.
inventory_retail = RA + RB. Every y0 entry used: S = supplier, RA = RB = retail/2,
Q1A = Q1B = shipments/2.
"""
import math

import numpy as np

FAMILY = "supply_chain_v8b_rr"
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

STATE = ["S", "C1", "C2", "Q1A", "Q1B", "Q2A", "Q2B", "RA", "RB", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([SCAP, 1e4, 1e4, 1e6, 1e6, 1e6, 1e6, RCAP, RCAP, 1.5])

PARAMS = [
    ("p0", 11.3, 1.0, 100.0, True),
    ("kc", 0.45, 0.0, 5.0, False),
    ("tau_c", 10.0, 1.0, 100.0, True),
    ("qm", 1200.0, 50.0, 5000.0, True),
    ("dq", 0.5, 0.01, 5.0, True),
    ("kq0", 1.5, 0.1, 3.0, True),
    ("kl", 8.0, 0.0, 60.0, False),
    ("a0", 57.0, 5.0, 200.0, True),
    ("dem", 27.0, 5.0, 100.0, True),
    ("fb", 0.5, 0.2, 0.8, False),
    ("kd", 0.01, 0.001, 0.2, True),
    ("kcool", 0.0102, 0.005, 1.0, True),
    ("ws", 0.5, 0.0, 1.0, False),
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
    S, C1, C2, Q1A, Q1B, Q2A, Q2B, RA, RB, W = s
    order, lead, mix, prod, recv, maint = uu
    P = prod * (th["p0"] + C2)
    room = M.max(th["qm"] - (Q1A + Q1B + Q2A + Q2B), 0.0)
    D = _smin(M, _smin(M, order, th["dq"] * room, 1.0), P + KS * S, 1.0)
    shB = mix * (1.0 - th["ws"] * S / SCAP)
    kq = th["kq0"] / (1.0 + th["kl"] * lead)
    g = 1.0 - WLOSS / (1.0 + M.exp(-(W - 1.0) / 0.05))
    Q2 = Q2A + Q2B
    A = _smin(M, kq * Q2, g * th["a0"] * recv, 1.0)
    fA = Q2A / (Q2 + 1e-6)
    fb = th["fb"]
    sA = _smin(M, th["dem"] * (1.0 - fb) + th["kd"] * RA, KS * RA, 1.0)
    sB = _smin(M, th["dem"] * fb + th["kd"] * RB, KS * RB, 1.0)
    return P, D, shB, kq, A, fA, sA, sB


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    S, C1, C2, Q1A, Q1B, Q2A, Q2B, RA, RB, W = s
    order, maint = uu[0], uu[5]
    P, D, shB, kq, A, fA, sA, sB = _flows(M, s, uu, th)
    AA = fA * A
    AB = A - AA
    return _pack(M, [P - D, (th["kc"] * order - C1) / th["tau_c"], (C1 - C2) / th["tau_c"],
                     (1.0 - shB) * D - kq * Q1A, shB * D - kq * Q1B, kq * Q1A - AA, kq * Q1B - AB,
                     AA - sA, AB - sB, (1.0 - maint) / TAU_HEAT - th["kcool"] * maint * W])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    P, D, shB, kq, A, fA, sA, sB = _flows(M, s, uu, th)
    return _pack(M, [A, s[0], s[7] + s[8]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    q = 0.5 * M.max(y[0], 0.0)
    S = M.min(M.max(y[1], 0.0), SCAP)
    R = 0.5 * M.max(y[2], 0.0)
    z = 0.0 * q
    return _pack(M, [S, z, z, z, z, z, z, R, R, z])
