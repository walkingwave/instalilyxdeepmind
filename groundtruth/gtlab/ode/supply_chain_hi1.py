"""Discrete-time (dt = 1) literal-code family: supply_chain hi1.

Reading of the tick data: dispatch reaches the terminal after exactly 3 ticks (integer delay
line), the supplier gives up the order quantity in one tick (orders withdraw available stock
only), the terminal and the transport pass at most a fixed number of units per tick (hard min),
retail sells min(demand, stock). Structure otherwise as v8c: production commitment (fast stage
+ slow stage) driven by the order quantity, congestion limit on goods in transit, two goods
classes (product mix, old batches dilute the class-B share), heat/wear on the terminal.

One tick, in order:
  P  = prod * (p0 + C2 + C3)
  D  = min(order, dq * (qm - goods in transit), S + P)          orders withdraw available stock
  S' = min(S + P - D, SCAP)
  line: D enters slot 0; the slot-2 content (dispatched 3 ticks ago) joins the terminal queue Q
  A  = min(kq * Q, v0 / (1 + kl * lead), g * (a0 + a1 * recv) * (1 - km * maint))
  retail class c: sold_c = min(dem_c + kd R_c, R_c + A_c)
  C1 += ac (kc order - C1); C2 += ac (C1 - C2); C3 += (ks order - C3) / tau_s
  W  += (1 - maint) / 100 - kcool maint W;   g = 1 - wl * [W > 1] (steep logistic)

Runtime: the shared RK4 rollout is used with f = 0 (state unchanged by the integrator) and the
map is applied inside h, which writes the new state into x in place and returns the observation
after the tick. N_SUB = 1. Numpy + math only (ships verbatim).
"""
import math

import numpy as np

FAMILY = "supply_chain_hi1"
OBS = ["shipments", "inventory_supplier", "inventory_retail"]
CTRL = ["order_quantity", "lead_time_buy", "product_mix", "production_effort",
        "receiving_effort", "maintenance"]
MECHS = {"A": "adaptive production commitment",
         "B": "congested transport",
         "C": "machine heat/wear (kept as a terminal derate)"}
N_SUB = 1
SCAP = 362.0
DELAY = 3
TAU_HEAT = 100.0

STATE = ["S", "C1", "C2", "C3", "LA0", "LA1", "LA2", "LB0", "LB1", "LB2", "QA", "QB", "RA", "RB", "W"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([SCAP] + [1e4] * 3 + [1e6] * 8 + [1e5, 1e5, 2.0])

PARAMS = [
    ("p0", 11.5, 1.0, 100.0, True),
    ("kc", 0.35, 0.0, 5.0, False),
    ("ac", 0.5, 0.02, 1.0, True),
    ("ks", 0.3, 0.0, 1.0, False),
    ("tau_s", 300.0, 20.0, 2000.0, True),
    ("qm", 1294.0, 50.0, 5000.0, True),
    ("dq", 1.0, 0.02, 1.0, True),
    ("kq", 0.5, 0.02, 1.0, True),
    ("v0", 60.0, 5.0, 500.0, True),
    ("kl", 2.0, 0.0, 60.0, False),
    ("a0", 8.0, 0.0, 100.0, False),
    ("a1", 34.6, 1.0, 200.0, True),
    ("km", 0.05, 0.0, 0.9, False),
    ("dem", 21.8, 5.0, 100.0, True),
    ("fb", 0.646, 0.2, 0.8, False),
    ("kd", 0.0167, 0.0, 0.2, False),
    ("ws", 0.4, 0.0, 1.0, False),
    ("kcool", 0.0157, 0.001, 1.0, True),
    ("wl", 0.5, 0.0, 0.9, False),
]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def step(x, u, th):
    """x [..., NS], u [..., 6] or [6] -> (x_next, y). Works for 1-d and batched arrays."""
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    g_ = lambda i: x[..., i]
    S, C1, C2, C3 = g_(0), g_(1), g_(2), g_(3)
    LA = [g_(4), g_(5), g_(6)]
    LB = [g_(7), g_(8), g_(9)]
    QA, QB, RA, RB, W = g_(10), g_(11), g_(12), g_(13), g_(14)
    order, lead, mix, prod, recv, maint = [u[..., j] for j in range(6)]
    P = prod * (th["p0"] + C2 + C3)
    transit = LA[0] + LA[1] + LA[2] + LB[0] + LB[1] + LB[2] + QA + QB
    room = np.maximum(th["qm"] - transit, 0.0)
    D = np.minimum(np.minimum(order, th["dq"] * room), S + P)
    D = np.maximum(D, 0.0)
    S2 = np.minimum(np.maximum(S + P - D, 0.0), SCAP)
    shB = mix * (1.0 - th["ws"] * S / SCAP)
    QA = QA + LA[2]
    QB = QB + LB[2]
    Q = QA + QB
    g = 1.0 - th["wl"] * _sig((W - 1.0) / 0.01)
    capT = g * (th["a0"] + th["a1"] * recv) * (1.0 - th["km"] * maint)
    vq = th["v0"] / (1.0 + th["kl"] * lead)
    A = np.maximum(np.minimum(np.minimum(th["kq"] * Q, vq), capT), 0.0)
    fA = QA / (Q + 1e-9)
    AA = fA * A
    AB = A - AA
    QA2 = np.maximum(QA - AA, 0.0)
    QB2 = np.maximum(QB - AB, 0.0)
    fb = th["fb"]
    RA1 = RA + AA
    RB1 = RB + AB
    sA = np.minimum(th["dem"] * (1.0 - fb) + th["kd"] * RA, RA1)
    sB = np.minimum(th["dem"] * fb + th["kd"] * RB, RB1)
    RA2 = RA1 - sA
    RB2 = RB1 - sB
    a = th["ac"]
    C1n = C1 + a * (th["kc"] * order - C1)
    C2n = C2 + a * (C1 - C2)
    C3n = C3 + (th["ks"] * order - C3) / th["tau_s"]
    Wn = W + (1.0 - maint) / TAU_HEAT - th["kcool"] * maint * W
    vals = [S2, C1n, C2n, C3n, (1.0 - shB) * D, LA[0], LA[1], shB * D, LB[0], LB[1],
            QA2, QB2, RA2, RB2, Wn]
    xn = np.stack(np.broadcast_arrays(*vals), axis=-1)
    y = np.stack(np.broadcast_arrays(A, S2, RA2 + RB2), axis=-1)
    return xn, y


def f(x, u, th, mech):
    return np.zeros_like(np.asarray(x, float))


def h(x, u, th, mech):
    xn, y = step(x, u, th)
    x[...] = np.minimum(np.maximum(xn, STATE_LO), STATE_HI)
    return y


def x0(y0, th, mech):
    y = np.asarray(y0, float)
    S = np.minimum(np.maximum(y[..., 1], 0.0), SCAP)
    R = 0.5 * np.maximum(y[..., 2], 0.0)
    z = 0.0 * S
    return np.stack([S, z, z, z, z, z, z, z, z, z, z, z, R, R, z], axis=-1)


def simulate(y0, U, th):
    x = np.asarray(x0(y0, th, None), float)
    Y = np.empty((len(U), 3))
    for t in range(len(U)):
        x, Y[t] = step(x, U[t], th)
        x = np.minimum(np.maximum(x, STATE_LO), STATE_HI)
    return Y
