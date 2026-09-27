"""Grey-box ODE: wildlife x7 (patch refuge + explicit nursery stage). NUMPY + math only.

Per region i (north scale 1, south scale ks). Adults split into an exposed pool E (open pasture +
mixed cover) and a sheltered pool S (sheltered browse); juveniles J sit in the nursery and are not
counted. Observed prey = E + S, predators = Q.
    P = E + S; births r P into the nursery
    A:  J' = r P - mJ J;  of the juveniles leaving the nursery a fraction 1/(1 + J/(cJ s_i F)) matures
        (young compete for nursery food; the nursery food is the renewing stock F = R if B else 1)
    recruits  rec = mJ J / (1 + J/(cJ s_i F)), shared between E and S in proportion to their sizes
    shelter capacity  K_i = s_i ch hab;  sheltered target S* = P K_i/(K_i + P);  relocation m (S* - S)
    E' = rec E/P - (d + dh (1 - hab)) E - H hunt E^2/(E^2 + Ph^2) - em_i cor E + sv T_in/tau - flux
    S' = rec S/P - d S - em_i cor S + flux                (sheltered animals are not hunted)
    B:  R' = w (1 - bh (1 - hab)) (1 - R) - kR (P/s_i) R
    predator food Z = (E + phi S)/(E + phi S + 100 s_i),  G' = (Z - G)/tz
    Q' = cq (q0 + q1 G - Q) Q/(Q + 4) - em cor Q + sv V_in/tau
    transit pools per direction: T' = em_from cor P_from - T/tau, V' = em cor Q_from - V/tau.
Reset: P, Q from y0; S = S*(hab = 1); J = r P/mJ; R = 1; pools empty; G = Z.
"""
import math

import numpy as np

FAMILY = "wildlife_x7"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: young animals compete for nursery food before recruiting",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2

STATE = ["EN", "SN", "JN", "RN", "QN", "ES", "SS", "JS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e4, 1e5, 1.0, 1e3, 1e4, 1e4, 1e5, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2])

PARAMS = [
    ("r", 0.45, 0.05, 5.0, True),
    ("d", 0.28, 0.01, 2.0, True),
    ("ks", 0.80, 0.3, 1.5, False),
    ("H", 1.43, 0.01, 50.0, True),
    ("Ph", 6.0, 0.5, 500.0, True),
    ("em", 0.0315, 0.0005, 0.5, True),
    ("emS", 0.048, 0.0005, 0.5, True),
    ("tau", 24.6, 0.5, 150.0, True),
    ("sv", 0.79, 0.0, 1.0, False),
    ("cJ", 560.0, 5.0, 1e5, True),        # A
    ("mJ", 0.3, 0.02, 5.0, True),         # A
    ("w", 0.0155, 0.002, 1.5, True),      # B
    ("kR", 2.1e-4, 1e-6, 0.02, True),     # B
    ("bh", 0.75, 0.0, 1.0, False),        # B
    ("phi", 0.05, 0.0, 1.0, False),
    ("dh", 0.03, 0.0, 1.0, False),
    ("ch", 250.0, 1.0, 1000.0, True),
    ("m", 4.6, 0.005, 20.0, True),
    ("q0", 1.74, 0.0, 10.0, False),
    ("q1", 2.23, 0.0, 10.0, False),
    ("cq", 0.094, 0.002, 2.0, True),
    ("tz", 33.5, 0.5, 300.0, True),
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


def _region(th, mech, E, S, J, R, Q, G, sc, hunt, hab, cor, em, arrP, arrQ):
    P = E + S
    F = R if "B" in mech else 1.0
    if "A" in mech:
        dJ = th["r"] * P - th["mJ"] * J
        rec = th["mJ"] * J / (1.0 + J / (th["cJ"] * sc * F + 1e-9))
    else:
        dJ = 0.0 * J
        rec = th["r"] * F * P
    ip = 1.0 / (P + 1e-9)
    K = sc * th["ch"] * hab
    Sst = P * K / (K + P + 1e-9)
    flux = th["m"] * (Sst - S)
    E2 = E * E
    harv = th["H"] * hunt * E2 / (E2 + th["Ph"] ** 2)
    dE = rec * E * ip - (th["d"] + th["dh"] * (1.0 - hab)) * E - harv - em * cor * E + th["sv"] * arrP - flux
    dS = rec * S * ip - th["d"] * S - em * cor * S + flux
    if "B" in mech:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    Pz = E + th["phi"] * S
    dG = (Pz / (Pz + 100.0 * sc) - G) / th["tz"]
    dQ = th["cq"] * (th["q0"] + th["q1"] * G - Q) * Q / (Q + 4.0) - th["em"] * cor * Q + th["sv"] * arrQ
    return dE, dS, dJ, dR, dQ, dG


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    EN, SN, JN, RN, QN, ES, SS, JS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
    k = 1.0 / th["tau"]
    dEN, dSN, dJN, dRN, dQN, dGN = _region(th, mech, EN, SN, JN, RN, QN, GN, 1.0, hunt, hab, cor, th["em"], k * TSN, k * VSN)
    dES, dSS, dJS, dRS, dQS, dGS = _region(th, mech, ES, SS, JS, RS, QS, GS, th["ks"], hunt, hab, cor, th["emS"], k * TNS, k * VNS)
    dTNS = th["em"] * cor * (EN + SN) - k * TNS
    dTSN = th["emS"] * cor * (ES + SS) - k * TSN
    dVNS = th["em"] * cor * QN - k * VNS
    dVSN = th["em"] * cor * QS - k * VSN
    return _pack(M, [dEN, dSN, dJN, dRN, dQN, dES, dSS, dJS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1], s[4], s[5] + s[6], s[9]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    PN = M.max(y[0], 0.0)
    QN = M.max(y[1], 0.01)
    PS = M.max(y[2], 0.0)
    QS = M.max(y[3], 0.01)
    z = 0.0 * PN
    KN = th["ch"]
    KS = th["ks"] * th["ch"]
    SN = PN * KN / (KN + PN + 1e-9)
    SS = PS * KS / (KS + PS + 1e-9)
    JN = th["r"] * PN / th["mJ"]
    JS = th["r"] * PS / th["mJ"]
    ZN = PN - SN + th["phi"] * SN
    ZS = PS - SS + th["phi"] * SS
    GN = ZN / (ZN + 100.0)
    GS = ZS / (ZS + 100.0 * th["ks"])
    return _pack(M, [PN - SN, SN, JN, 1.0 + z, QN, PS - SS, SS, JS, 1.0 + z, QS, z, z, z, z, GN, GS])
