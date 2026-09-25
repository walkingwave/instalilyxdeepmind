"""Grey-box ODE: wildlife, minimal two-region prey / resource / predator. NUMPY + math only.

Per region (north scale 1, south scale ks on the density scale):
    P' = r P / (1 + P/c) - dl L P - hq * hunting * P
    L' = rho (P - L)
    Q' = s (q0 + q1 P - Q) Q / (Q + qh)
Prey birth is crowded by nursery competition 1/(1+P/c); death dl*L follows a lagged density L
(exhausted food / depleted cover renewing at rate rho), which gives the single overshoot seen
after every release (a delayed logistic). Hunting is a proportional take hq*hunting per head
(a saturating take hq*hunting*P/(P+p0) fitted p0 to its upper bound twice). Predators relax toward a capacity that rises weakly with prey, at a rate
that is linear for large Q and logistic below qh; they do not feed back on prey (initial prey
growth was identical at Q=8.7 and Q=2.0). Habitat protection and corridor access carry no term:
a food-renewal effect of habitat fitted to exactly zero on the two runs we have.

Reset: prey and predators from y0, lagged density L = 0 in both regions (a fitted reset
fraction l0 lowered leave-one-run-out, so it is fixed at zero).
Mechanism letters are accepted for interface compatibility but every term is always active.
"""
import math

import numpy as np

FAMILY = "wildlife_min"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {"A": "finite food stock (always on)", "B": "saturating harvest (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["PN", "LN", "QN", "PS", "LS", "QS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e4, 1e3, 1e4, 1e4, 1e3])

PARAMS = [
    ("r", 0.3, 0.05, 1.5, True),
    ("c", 40.0, 3.0, 500.0, True),
    ("dl", 6.4e-4, 1e-5, 0.1, True),
    ("rho", 0.033, 0.005, 0.5, True),
    ("hq", 0.032, 0.001, 1.0, True),
    ("qh", 10.0, 0.05, 50.0, True),
    ("s", 0.19, 0.003, 0.5, True),
    ("q0", 1.55, 0.1, 20.0, True),
    ("q1", 0.0075, 1e-4, 0.5, True),
    ("ks", 0.83, 0.3, 1.5, False),
]


class _PY:
    max = max
    min = min


class _NP:
    max = np.maximum
    min = np.minimum


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


def _region(th, P, L, Q, scale, take):
    c = th["c"] * scale
    dl = th["dl"] / scale
    grow = th["r"] * P / (1.0 + P / c)
    dP = grow - dl * L * P - take * P
    dL = th["rho"] * (P - L)
    Kq = th["q0"] + th["q1"] * P
    dQ = th["s"] * (Kq - Q) * Q / (Q + th["qh"])
    return dP, dL, dQ


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, LN, QN, PS, LS, QS = s
    take = th["hq"] * hunt
    dPN, dLN, dQN = _region(th, PN, LN, QN, 1.0, take)
    dPS, dLS, dQS = _region(th, PS, LS, QS, th["ks"], take)
    return _pack(M, [dPN, dLN, dQN, dPS, dLS, dQS])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0], s[2], s[3], s[5]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    PN = M.max(y[0], 0.0)
    QN = M.max(y[1], 0.01)
    PS = M.max(y[2], 0.0)
    QS = M.max(y[3], 0.01)
    zero = 0.0 * PN
    return _pack(M, [PN, zero, QN, PS, zero, QS])
