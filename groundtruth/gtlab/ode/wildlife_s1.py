"""Grey-box ODE: wildlife with switchable named mechanisms. NUMPY + math only.

Per region i (north scale 1, south scale ks), observed prey = adults P + juveniles J:
    births   b  = r * F * P,                F = R if B else 1
    A:  J' = b - mJ J - take J,             maturation mJ J / (1 + J/(cJ s_i)), the rest of the
        juveniles leaving the nursery die (poor condition under food competition)
        P' = mJ J / (1 + J/(cJ s_i)) - d0 P - take P - emig + settle
    not A: births recruit directly, J = 0
    B:  R' = w (1 - bh (1 - hab)) (1 - R) - kR (P / s_i) R      (food, renewal set by habitat)
    not B: R = 1
    take = hq * hunting * (1 - e_i * hab),  e_N = en, e_S = 0 (sheltered hunting, north)
    emig = em * corridor * P;  transit pools T_NS, T_SN fed by emig, arriving at rate 1/tau
    (travellers in transit still arrive after the corridor closes)
    C:  settle = arrivals * max(0, 1 - P_dest / (ksc s_dest)), unsettled arrivals are lost
    not C: settle = arrivals
    Q' = s (q0 - qc * corridor - Q)          (predators as in wildlife_min v7)
Revision r2 (extra letter "J" in the mechanism set, e.g. "ABJ"): observed prey = adults P only,
juveniles hidden, so the nursery stage delays recruitment instead of adding to the count at once.
Reset: P, Q from y0; J = 0, R = 1 (full food), transit pools empty.
"""
import math

import numpy as np

FAMILY = "wildlife_s1"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
# "J" is not a mechanism: it switches the observation to adults only (revision r2).
MECHS = {
    "A": "juvenile condition: births pass a nursery stage with food competition",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition: arrivals settle only where space is available",
}
N_SUB = 2

STATE = ["PN", "JN", "RN", "QN", "PS", "JS", "RS", "QS", "TNS", "TSN"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e4, 1.0, 1e3, 1e4, 1e4, 1.0, 1e3, 1e4, 1e4])

PARAMS = [
    ("r", 0.40, 0.05, 1.5, True),
    ("d0", 0.10, 0.001, 0.5, True),
    ("ks", 0.83, 0.3, 1.5, False),
    ("hq", 0.026, 0.001, 1.0, True),
    ("en", 0.5, 0.0, 0.95, False),
    ("s", 0.064, 0.003, 0.5, True),
    ("q0", 2.45, 0.1, 20.0, True),
    ("qc", 0.76, 0.0, 3.0, False),
    ("em", 0.02, 0.0, 0.3, False),
    ("tau", 8.0, 1.0, 100.0, True),
    ("mJ", 0.2, 0.01, 1.5, True),        # A
    ("cJ", 50.0, 1.0, 1000.0, True),     # A
    ("w", 0.05, 0.002, 1.5, True),       # B
    ("kR", 0.00125, 1e-5, 0.02, True),   # B
    ("bh", 0.5, 0.0, 1.0, False),        # B
    ("ksc", 150.0, 20.0, 2000.0, True),  # C
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


def _region(M, th, mech, P, J, R, Q, scale, take, hab, Kq, emig, arr):
    A = "A" in mech
    B = "B" in mech
    F = R if B else 1.0
    b = th["r"] * F * P
    if A:
        lv = th["mJ"] * J
        rec = lv / (1.0 + J / (th["cJ"] * scale))
        dJ = b - lv - take * J
    else:
        rec = b
        dJ = -J
    if "C" in mech:
        arr = arr * M.max(1.0 - P / (th["ksc"] * scale), 0.0)
    dP = rec - th["d0"] * P - take * P - emig + arr
    if B:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (1.0 - R) - th["kR"] * (P / scale) * R
    else:
        dR = 1.0 - R
    dQ = th["s"] * (Kq - Q)
    return dP, dJ, dR, dQ


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, JN, RN, QN, PS, JS, RS, QS, TNS, TSN = s
    Kq = th["q0"] - th["qc"] * cor
    take = th["hq"] * hunt
    ktr = 1.0 / th["tau"]
    eN = th["em"] * cor * PN
    eS = th["em"] * cor * PS
    dPN, dJN, dRN, dQN = _region(M, th, mech, PN, JN, RN, QN, 1.0, take * (1.0 - th["en"] * hab),
                                 hab, Kq, eN, ktr * TSN)
    dPS, dJS, dRS, dQS = _region(M, th, mech, PS, JS, RS, QS, th["ks"], take, hab, Kq, eS, ktr * TNS)
    dTNS = eN - ktr * TNS
    dTSN = eS - ktr * TSN
    return _pack(M, [dPN, dJN, dRN, dQN, dPS, dJS, dRS, dQS, dTNS, dTSN])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    if "J" in mech:
        return _pack(M, [s[0], s[3], s[4], s[7]])
    return _pack(M, [s[0] + s[1], s[3], s[4] + s[5], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    PN = M.max(y[0], 0.0)
    QN = M.max(y[1], 0.01)
    PS = M.max(y[2], 0.0)
    QS = M.max(y[3], 0.01)
    z = 0.0 * PN
    return _pack(M, [PN, z, 1.0 + z, QN, PS, z, 1.0 + z, QS, z, z])
