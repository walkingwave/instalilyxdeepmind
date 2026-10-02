"""Grey-box ODE: supply_chain, minimal stock-flow (v7). NUMPY + math only (ships verbatim).

Supplier stock S is filled by production and hard-capped at SCAP (361.6 +- 1.2 in both real
runs). Production = effort * (p0 + C2), where C1 -> C2 is a two-stage commitment cascade that
follows kc * order_quantity with time constant tau_c (adaptive production commitments).
Dispatch takes the current production flow plus at most KS * S of the stock per tick, limited
by the order and by a forward-transport cap d0 + d1 * receiving_effort:
D = smin(smin(order, d0 + d1 r), P + KS S). With the stock drained through a rate (not a
level) the supplier reads 0 whenever the order exceeds production, which is what all three runs
show (the v6 form smin(order, S) let S hover at the production level).
Dispatched goods travel a two-stage conveyor Q1 -> Q2 at rate kq = kq0 / (1 + kl * lead_time)
and arrive at retail at A = smin(kq Q2, g a0 r): the terminal is staffed by receiving effort
(19.5/tick at r = 0.35 in the pulse run, 46 in the interior hold at 0.925). Machine heat W
rises at (1 - maintenance) / TAU_HEAT and cools at kcool * maintenance * W; past W = 1 the
terminal gate g = 1 - WLOSS * sigmoid((W - 1) / 0.05) halves throughput (the pulse run halves
shipments at tick 101 after 100 ticks without maintenance: TAU_HEAT = 100, WLOSS = 0.5).
Retail holds two classes RA, RB (the brief: two goods classes, product mix selects new
production and dispatch, initial stocks split into fixed class shares). Arrivals split
(1 - mix) : mix; each class sells at its own demand dem_A = (1 - fb) dem, dem_B = fb dem,
plus kd * stock, never more than KS * stock per tick. The three runs all drain the initial
retail stock at 27.5/tick (both classes selling), the pulse (mix 0.8, arrivals 20) sells 17/tick
and banks class B at 3/tick, the interior hold (mix 0.65, arrivals 35) sells 27 and banks 8/tick,
and sales grow ~0.009 per unit stock up to 1,100 units. inventory_retail = RA + RB, uncapped.

x0: S = inventory_supplier, RA = RB = inventory_retail / 2, Q1 = shipments (initial
in-transit content), C1 = C2 = Q2 = W = 0.

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
KS = 1.5          # max fraction of a stock withdrawn per tick (dispatch from S, sales from R)
TAU_HEAT = 100.0  # ticks to the wear threshold without maintenance (pulse run: tick 101)
WLOSS = 0.5       # throughput lost past the threshold (18.4 -> 9.2)
RCAP = 1e5

STATE = ["S", "C1", "C2", "Q1", "Q2", "RA", "RB", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([SCAP, 1e4, 1e4, 1e6, 1e6, RCAP, RCAP, 1.5])

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
    ("fb", 0.5, 0.2, 0.8, False),
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
    S, C1, C2, Q1, Q2, RA, RB, W = s
    order, lead, mix, prod, recv, maint = uu
    P = prod * (th["p0"] + C2)
    dcap = th["d0"] + th["d1"] * recv
    D = _smin(M, _smin(M, order, dcap, 1.0), P + KS * S, 1.0)
    kq = th["kq0"] / (1.0 + th["kl"] * lead)
    g = 1.0 - WLOSS / (1.0 + M.exp(-(W - 1.0) / 0.05))
    A = _smin(M, kq * Q2, g * th["a0"] * recv, 1.0)
    fb = th["fb"]
    sA = _smin(M, th["dem"] * (1.0 - fb) + th["kd"] * RA, KS * RA, 1.0)
    sB = _smin(M, th["dem"] * fb + th["kd"] * RB, KS * RB, 1.0)
    return P, D, kq, A, sA, sB


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    S, C1, C2, Q1, Q2, RA, RB, W = s
    order, mix, maint = uu[0], uu[2], uu[5]
    P, D, kq, A, sA, sB = _flows(M, s, uu, th)
    dS = P - D
    dC1 = (th["kc"] * order - C1) / th["tau_c"]
    dC2 = (C1 - C2) / th["tau_c"]
    dQ1 = D - kq * Q1
    dQ2 = kq * Q1 - A
    dRA = (1.0 - mix) * A - sA
    dRB = mix * A - sB
    dW = (1.0 - maint) / TAU_HEAT - th["kcool"] * maint * W
    return _pack(M, [dS, dC1, dC2, dQ1, dQ2, dRA, dRB, dW])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    P, D, kq, A, sA, sB = _flows(M, s, uu, th)
    return _pack(M, [A, s[0], s[5] + s[6]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    ship = M.max(y[0], 0.0)
    S = M.min(M.max(y[1], 0.0), SCAP)
    R = 0.5 * M.max(y[2], 0.0)
    z = 0.0 * ship
    return _pack(M, [S, z, z, ship, z, R, R, z])
