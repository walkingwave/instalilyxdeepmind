"""Grey-box ODE: wildlife v8. NUMPY + math only.

Per region i (north scale 1, south scale ks); observed prey = adults P, observed predators = Q.
    births   b = r F P,  F = R if B else 1
    A:  recruitment b / (1 + P/(cJ s_i))       (nursery food competition, instantaneous)
    B:  R' = w (1 - bh (1 - hab)) (1 - R) - kR (P/s_i) R      (finite food renewal)
    harvest  H hunt (1 - e_i hab) P / (P + Ph s_i),  e_N = en, e_S = 0   (quota, saturating)
    P' = rec - d0 P - harvest - em cor P + sv T_in / tau
    Q' = cq Q (q0 + q1 Z_i - Q) - emq cor Q + sv V_in / tau
         Z_i = (1 + xq (1 - hab)) P / (P + 100 s_i)     (predator food: prey, more exposed without cover)
    transit pools per direction and species: T' = em cor P_from - T/tau, V' = emq cor Q_from - V/tau;
    a fraction sv of travellers arrives (the rest are lost); travellers still arrive after closing.
Reset: P, Q from y0; R = 1; pools empty.
"""
import math

import numpy as np

FAMILY = "wildlife_v8"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: nursery food competition caps recruitment",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2

STATE = ["PN", "RN", "QN", "PS", "RS", "QS", "TNS", "TSN", "VNS", "VSN"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1.0, 1e3, 1e4, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3])

PARAMS = [
    ("r", 0.75, 0.05, 3.0, True),
    ("d0", 0.5, 0.001, 2.0, True),
    ("ks", 0.80, 0.3, 1.5, False),
    ("H", 1.0, 0.01, 20.0, True),
    ("en", 0.4, 0.0, 0.99, False),
    ("Ph", 30.0, 0.5, 2000.0, True),
    ("em", 0.03, 0.0005, 0.5, True),
    ("tau", 10.0, 0.5, 150.0, True),
    ("sv", 0.8, 0.0, 1.0, False),
    ("cJ", 100.0, 5.0, 5000.0, True),     # A
    ("w", 0.033, 0.002, 1.5, True),       # B
    ("kR", 8e-5, 1e-6, 0.02, True),       # B
    ("bh", 0.5, 0.0, 1.0, False),         # B
    ("q0", 1.6, 0.0, 10.0, False),
    ("q1", 1.4, 0.0, 10.0, False),
    ("cq", 0.01, 0.0005, 0.5, True),
    ("xq", 0.5, 0.0, 5.0, False),
    ("emq", 0.05, 0.0005, 0.5, True),
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


def _region(th, mech, P, R, Q, sc, hunt, hab, cor, e, arrP, arrQ):
    B = "B" in mech
    F = R if B else 1.0
    b = th["r"] * F * P
    rec = b / (1.0 + P / (th["cJ"] * sc)) if "A" in mech else b
    harv = th["H"] * hunt * (1.0 - e * hab) * P / (P + th["Ph"] * sc)
    dP = rec - th["d0"] * P - harv - th["em"] * cor * P + th["sv"] * arrP
    if B:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    Z = (1.0 + th["xq"] * (1.0 - hab)) * P / (P + 100.0 * sc)
    dQ = th["cq"] * Q * (th["q0"] + th["q1"] * Z - Q) - th["emq"] * cor * Q + th["sv"] * arrQ
    return dP, dR, dQ


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN = s
    k = 1.0 / th["tau"]
    dPN, dRN, dQN = _region(th, mech, PN, RN, QN, 1.0, hunt, hab, cor, th["en"], k * TSN, k * VSN)
    dPS, dRS, dQS = _region(th, mech, PS, RS, QS, th["ks"], hunt, hab, cor, 0.0, k * TNS, k * VNS)
    dTNS = th["em"] * cor * PN - k * TNS
    dTSN = th["em"] * cor * PS - k * TSN
    dVNS = th["emq"] * cor * QN - k * VNS
    dVSN = th["emq"] * cor * QS - k * VSN
    return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0], s[2], s[3], s[5]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    PN = M.max(y[0], 0.0)
    QN = M.max(y[1], 0.01)
    PS = M.max(y[2], 0.0)
    QS = M.max(y[3], 0.01)
    z = 0.0 * PN
    return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z])
