"""Grey-box ODE (variant 2: linear predator relaxation, south corridor emigration): wildlife, minimal two-region prey / lagged density / predator. NUMPY + math only.

Per region (north scale 1, south scale ks on the density scale):
    P' = r P / (1 + P/c) - dl L P - hq * hunting * (1 - e_i * habitat) * P
    L' = rho (P / g(habitat) - L),      g(h) = kh + (1 - kh) h
    Q' = s (q0 - qc * corridor - Q)          (south prey also lose mc * corridor * P: emigration)
Prey birth is crowded by nursery competition 1/(1+P/c); death dl*L follows a lagged density L
(exhausted food / depleted cover renewing at rate rho), a delayed logistic that gives the single
overshoot seen after every release. Habitat protection sets the carrying capacity through the
lag target P/g(h): losing habitat raises the density the food stock "feels", so the decline is
slow at first and then sharp, and the settle is at half the protected level. Habitat also
shelters prey from hunting, in the north only (e_n = en, e_s = 0): at habitat 1 the north loses
a quarter under quota 6 while the south loses two thirds; at habitat 0.2 both regions fall at
the same -0.10 per head. Predators relax toward a capacity that the corridor lowers (they follow
the animals out); the relaxation rate falls with Q (slow decay from 8-12, fast settle near 2).
Prey do not depend on predators and predators no longer depend on prey (q1 fitted below 0.0015
once the corridor term carried the pulse-run level).

Reset: prey and predators from y0, lagged density L = 0 in both regions.
Mechanism letters are accepted for interface compatibility but every term is always active.
"""
import math

import numpy as np

FAMILY = "wildlife_min2"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {"A": "finite food stock (always on)", "B": "sheltered hunting exposure (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["PN", "LN", "QN", "PS", "LS", "QS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e4, 1e3, 1e4, 1e4, 1e3])

PARAMS = [
    ("r", 0.19, 0.05, 1.5, True),
    ("c", 100.0, 3.0, 500.0, True),
    ("dl", 7.2e-4, 1e-5, 0.1, True),
    ("rho", 0.045, 0.005, 0.5, True),
    ("hq", 0.018, 0.001, 1.0, True),
    ("ks", 0.82, 0.3, 1.5, False),
    ("kh", 0.4, 0.05, 1.0, True),
    ("en", 0.4, 0.0, 0.95, False),
    ("s", 0.2, 0.003, 0.5, True),
    ("mc", 0.02, 0.0, 0.3, False),
    ("q0", 2.3, 0.1, 20.0, True),
    ("qc", 0.65, 0.0, 3.0, False),
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


def _region(th, P, L, Q, scale, take, g, Kq, leave=0.0):
    c = th["c"] * scale
    dl = th["dl"] / scale
    grow = th["r"] * P / (1.0 + P / c)
    dP = grow - dl * L * P - take * P - leave * P
    dL = th["rho"] * (P / g - L)
    dQ = th["s"] * (Kq - Q)
    return dP, dL, dQ


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, LN, QN, PS, LS, QS = s
    g = th["kh"] + (1.0 - th["kh"]) * hab
    Kq = th["q0"] - th["qc"] * cor
    take = th["hq"] * hunt
    dPN, dLN, dQN = _region(th, PN, LN, QN, 1.0, take * (1.0 - th["en"] * hab), g, Kq)
    dPS, dLS, dQS = _region(th, PS, LS, QS, th["ks"], take, g, Kq, th["mc"] * cor)
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
