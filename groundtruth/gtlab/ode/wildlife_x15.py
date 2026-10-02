"""Grey-box ODE: wildlife x15 (x14 with instantaneous nursery and self-seeding food regrowth). NUMPY + math only.

Per region i (north scale 1, south scale ks). Adults P spread over open pasture / mixed cover
(exposed) and sheltered browse. Animals relocate fast, so occupancy is at its equilibrium:
sheltered browse holds S = P K_i/(K_i + P) with capacity K_i = s_i ch_i hab (habitat protection adds
shelter), the rest E = P - S = P^2/(K_i + P) is exposed. Observed prey = P, predators = Q.
    A:  recruitment rec = r P / (1 + P/(cJ s_i F))   (young compete for nursery food; the nursery food
        is the renewing stock F = R if B else 1; the nursery stage is fast, so it is at equilibrium)
    P' = rec - d P - dh (1 - hab) E - a Q E - H hunt E^2/(E^2 + Ph^2) - em_i cor P + sv T_in/tau
         (only exposed animals are hunted, taken by predators or die of exposure)
    B:  R' = w (1 - bh (1 - hab)) (rl + (1 - rl) R) (1 - R) - kR (P/s_i) R   (rl < 1: regrowth partly from what is left)
    predator food Z = E/(E + 100 s_i) (predators hunt in the open),  G' = (Z - G)/tz
    Q' = cq (q0 + q1 G - Q) Q/(Q + 4) - em cor Q + sv V_in/tau
    transit pools per direction: T' = em_from cor P_from - T/tau, V' = em cor Q_from - V/tau;
    a fraction sv of travellers arrives; travellers still arrive after the corridor closes.
Reset: P, Q from y0; R = 1; pools empty; G = Z at full habitat.
"""
import math

import numpy as np

FAMILY = "wildlife_x15"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: young animals compete for nursery food before recruiting",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2

STATE = ["PN", "RN", "QN", "PS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1.0, 1e3, 1e4, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2])

PARAMS = [
    ("r", 0.34, 0.05, 5.0, True),
    ("d", 0.11, 0.01, 2.0, True),
    ("ks", 0.80, 0.3, 1.5, False),
    ("H", 1.55, 0.01, 50.0, True),
    ("Ph", 3.3, 0.5, 500.0, True),
    ("em", 0.030, 0.0005, 0.5, True),
    ("emS", 0.044, 0.0005, 0.5, True),
    ("tau", 26.6, 0.5, 150.0, True),
    ("sv", 0.79, 0.0, 1.0, False),
    ("cJ", 33.0, 2.0, 1e5, True),         # A
    ("w", 0.0152, 0.002, 1.5, True),      # B
    ("kR", 2.3e-4, 1e-6, 0.02, True),     # B
    ("bh", 0.41, 0.0, 1.0, False),        # B
    ("rl", 0.5, 0.0, 1.0, False),         # B
    ("dh", 0.055, 0.0, 1.0, False),
    ("a", 0.005, 0.0, 0.2, False),
    ("chN", 300.0, 1.0, 1000.0, True),
    ("chS", 300.0, 1.0, 1000.0, True),
    ("q0", 1.73, 0.0, 10.0, False),
    ("q1", 2.6, 0.0, 10.0, False),
    ("cq", 0.094, 0.002, 2.0, True),
    ("tz", 36.5, 0.5, 300.0, True),
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


def _region(th, mech, P, R, Q, G, sc, ch, hunt, hab, cor, em, arrP, arrQ):
    F = R if "B" in mech else 1.0
    if "A" in mech:
        rec = th["r"] * P / (1.0 + P / (th["cJ"] * sc * F + 1e-9))
    else:
        rec = th["r"] * F * P
    K = sc * ch * hab
    E = P * P / (K + P + 1e-9)
    E2 = E * E
    harv = th["H"] * hunt * E2 / (E2 + th["Ph"] ** 2)
    dP = rec - th["d"] * P - th["dh"] * (1.0 - hab) * E - th["a"] * Q * E - harv - em * cor * P + th["sv"] * arrP
    if "B" in mech:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (th["rl"] + (1.0 - th["rl"]) * R) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    dG = (E / (E + 100.0 * sc) - G) / th["tz"]
    dQ = th["cq"] * (th["q0"] + th["q1"] * G - Q) * Q / (Q + 4.0) - th["em"] * cor * Q + th["sv"] * arrQ
    return dP, dR, dQ, dG


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
    k = 1.0 / th["tau"]
    dPN, dRN, dQN, dGN = _region(th, mech, PN, RN, QN, GN, 1.0, th["chN"], hunt, hab, cor, th["em"], k * TSN, k * VSN)
    dPS, dRS, dQS, dGS = _region(th, mech, PS, RS, QS, GS, th["ks"], th["chS"], hunt, hab, cor, th["emS"], k * TNS, k * VNS)
    dTNS = th["em"] * cor * PN - k * TNS
    dTSN = th["emS"] * cor * PS - k * TSN
    dVNS = th["em"] * cor * QN - k * VNS
    dVSN = th["em"] * cor * QS - k * VSN
    return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS])


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
    KN = th["chN"]
    KS = th["ks"] * th["chS"]
    EN = PN * PN / (KN + PN + 1e-9)
    ES = PS * PS / (KS + PS + 1e-9)
    GN = EN / (EN + 100.0)
    GS = ES / (ES + 100.0 * th["ks"])
    return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS])
