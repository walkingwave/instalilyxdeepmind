"""Grey-box ODE: supply_chain v8f = v8e with (1) the class switch acting only as the supplier
nears its cap: class-B share mix (1 - ws (S / SCAP)^8) (0.13 of the full shift at S = 280,
0.7 at 347; the pulse run's sales jump comes at tick 43, when the supplier first reads ~350,
not at tick 31 when it starts refilling); (2) initial retail stock split in the demand
shares, RA = (1 - fb) R0, RB = fb R0, so both classes sell out together (all three runs
empty the initial retail stock in ~3-4 ticks at 27.5/tick, even with arrivals running).
Same 13 parameters.

v8e notes: supply_chain v8e = v8d without the slow commitment stage (C3, ks, tau_s),
kcool free again. Keeps: congested transport (v8b), affine terminal cap g (a0 + a1 r) (v8c),
class split at the terminal by supplier stock (v8d), tau_c = 1. 13 parameters. The slow stage
is identified only by the interior hold; fitted without it, it filled the supplier on that
hold (leave-one-out supplier 0.75), so we drop it.

v8d notes: supply_chain v8d = v8c with the class split made at the terminal instead of
at dispatch: one conveyor Q1 -> Q2 (no class-resolved transport), arrivals split
class B : class A = mix (1 - ws S / SCAP) : rest, with S the current supplier stock. The pulse
run's retail sales jump (16.7 -> 24/tick at tick 43, arrivals unchanged at 19.2) comes right
after the supplier reaches its cap (ticks 38-42), not after the ~80-tick dilution a mixed
1,600-unit conveyor would impose; the switch therefore acts on what the terminal releases.

v8c notes: supply_chain v8c = v8b with (1) an affine terminal cap g (a0 + a1 r): 19.2 at
receiving 0.35 (pulse), 37.4 at 0.925 (interior hold, from tick 236), ~55 at 1.5 (post-pulse
burst); (2) a slow commitment stage C3 -> ks * order with time constant tau_s: the interior
hold's production creeps from 34.6 to ~40 (arrivals pinned at the terminal cap 37.4 from tick
236, the surplus refills the supplier from tick 385 at 2.3/tick). tau_c fixed at 1 (the fits
put it on its lower bound), kcool fixed at the v8b value. 14 parameters.

v8b notes: supply_chain v8b = v8 with congested transport instead of a receiving-scaled
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

FAMILY = "supply_chain_v8f"
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
TAU_C = 1.0
WLOSS = 0.5       # throughput lost past the threshold (18.4 -> 9.2)
RCAP = 1e5

STATE = ["S", "C1", "C2", "Q1", "Q2", "RA", "RB", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([SCAP, 1e4, 1e4, 1e6, 1e6, RCAP, RCAP, 1.5])

PARAMS = [
    ("p0", 11.3, 1.0, 100.0, True),
    ("kc", 0.45, 0.0, 5.0, False),
    ("qm", 1200.0, 50.0, 5000.0, True),
    ("dq", 0.5, 0.01, 5.0, True),
    ("kq0", 1.5, 0.1, 3.0, True),
    ("kl", 8.0, 0.0, 60.0, False),
    ("a0", 8.0, 0.0, 100.0, False),
    ("a1", 32.0, 1.0, 200.0, True),
    ("dem", 27.0, 5.0, 100.0, True),
    ("fb", 0.5, 0.2, 0.8, False),
    ("kd", 0.01, 0.001, 0.2, True),
    ("ws", 0.5, 0.0, 1.0, False),
    ("kcool", 0.0157, 0.005, 1.0, True),
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
    room = M.max(th["qm"] - (Q1 + Q2), 0.0)
    D = _smin(M, _smin(M, order, th["dq"] * room, 1.0), P + KS * S, 1.0)
    sr = S / SCAP
    sr = sr * sr
    sr = sr * sr
    shB = mix * (1.0 - th["ws"] * sr * sr)
    kq = th["kq0"] / (1.0 + th["kl"] * lead)
    g = 1.0 - WLOSS / (1.0 + M.exp(-(W - 1.0) / 0.05))
    A = _smin(M, kq * Q2, g * (th["a0"] + th["a1"] * recv), 1.0)
    fA = 1.0 - shB
    fb = th["fb"]
    sA = _smin(M, th["dem"] * (1.0 - fb) + th["kd"] * RA, KS * RA, 1.0)
    sB = _smin(M, th["dem"] * fb + th["kd"] * RB, KS * RB, 1.0)
    return P, D, shB, kq, A, fA, sA, sB


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    S, C1, C2, Q1, Q2, RA, RB, W = s
    order, maint = uu[0], uu[5]
    P, D, shB, kq, A, fA, sA, sB = _flows(M, s, uu, th)
    AA = fA * A
    AB = A - AA
    return _pack(M, [P - D, (th["kc"] * order - C1) / TAU_C, (C1 - C2) / TAU_C,
                     D - kq * Q1, kq * Q1 - A,
                     AA - sA, AB - sB, (1.0 - maint) / TAU_HEAT - th["kcool"] * maint * W])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    P, D, shB, kq, A, fA, sA, sB = _flows(M, s, uu, th)
    return _pack(M, [A, s[0], s[5] + s[6]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    q = M.max(y[0], 0.0)
    S = M.min(M.max(y[1], 0.0), SCAP)
    R = M.max(y[2], 0.0)
    fb = th["fb"]
    z = 0.0 * q
    return _pack(M, [S, z, z, q, z, (1.0 - fb) * R, fb * R, z])
