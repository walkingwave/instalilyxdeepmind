"""Grey-box ODE: wildlife x5 (x4 + predators feed on exposed prey). NUMPY + math only.

Per region i (north scale 1, south scale ks). Prey split into an exposed pool E (open pasture +
mixed cover) and a sheltered pool S (sheltered browse); observed prey = E + S, predators = Q.
    P = E + S; per-capita recruitment g = r / (1 + P/(cJ s_i F)),  F = R if B else 1
        (A: young animals compete for nursery food; the nursery food is the renewing stock R, so the
         crowding scale shrinks when food is depleted)
    shelter capacity  K_i = s_i (c0 + ch_i hab);  sheltered target  S* = P K_i / (K_i + P)
    relocation flux   m (S* - S)            (animals move between patches)
    E' = g E - d E - H hunt E^2/(E^2 + Ph^2) - em_i cor E + sv T_in/tau - flux
    S' = g S - d S - em_i cor S + flux                (sheltered animals are not hunted)
    B:  R' = w (1 - bh (1 - hab)) (1 - R) - kR (P/s_i) R
    Q' = cq (q0 + q1 G - Q) Q/(Q + 4) - em cor Q + sv V_in/tau,  G' = (Z - G)/tz,
        Z = (E + phi S)/(E + phi S + 100 s_i)   (predators hunt mostly in the open)
    transit pools per direction: T' = em_from cor P_from - T/tau, V' = em cor Q_from - V/tau.
Reset: P, Q from y0; S = S*(hab = 1); R = 1; pools empty; G = Z.
"""
import math

import numpy as np

FAMILY = "wildlife_x5"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: nursery food competition caps recruitment",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2

STATE = ["EN", "SN", "RN", "QN", "ES", "SS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e4, 1.0, 1e3, 1e4, 1e4, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2])

PARAMS = [
    ("r", 0.9, 0.05, 3.0, True),
    ("d", 0.7, 0.01, 2.0, True),
    ("ks", 0.80, 0.3, 1.5, False),
    ("H", 1.4, 0.01, 50.0, True),
    ("Ph", 27.0, 0.5, 500.0, True),
    ("em", 0.028, 0.0005, 0.5, True),
    ("emS", 0.04, 0.0005, 0.5, True),
    ("tau", 33.0, 0.5, 150.0, True),
    ("sv", 0.75, 0.0, 1.0, False),
    ("cJ", 900.0, 5.0, 5000.0, True),     # A
    ("w", 0.026, 0.002, 1.5, True),       # B
    ("kR", 2.4e-5, 1e-6, 0.02, True),     # B
    ("bh", 0.34, 0.0, 1.0, False),        # B
    ("phi", 0.3, 0.0, 1.0, False),
    ("c0", 3.0, 0.01, 200.0, True),
    ("chN", 20.0, 0.0, 300.0, False),
    ("chS", 5.0, 0.0, 300.0, False),
    ("m", 3.0, 0.005, 20.0, True),
    ("q0", 1.62, 0.0, 10.0, False),
    ("q1", 1.51, 0.0, 10.0, False),
    ("cq", 0.097, 0.002, 2.0, True),
    ("tz", 63.0, 0.5, 300.0, True),
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


def _region(th, mech, E, S, R, Q, G, sc, ch, hunt, hab, cor, em, arrP, arrQ):
    P = E + S
    F = R if "B" in mech else 1.0
    g = th["r"] / (1.0 + P / (th["cJ"] * sc * F + 1e-9)) if "A" in mech else th["r"] * F
    K = sc * (th["c0"] + ch * hab)
    Sst = P * K / (K + P + 1e-9)
    flux = th["m"] * (Sst - S)
    E2 = E * E
    harv = th["H"] * hunt * E2 / (E2 + th["Ph"] ** 2)
    dE = g * E - th["d"] * E - harv - em * cor * E + th["sv"] * arrP - flux
    dS = g * S - th["d"] * S - em * cor * S + flux
    if "B" in mech:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    Pz = E + th["phi"] * S
    dG = (Pz / (Pz + 100.0 * sc) - G) / th["tz"]
    dQ = th["cq"] * (th["q0"] + th["q1"] * G - Q) * Q / (Q + 4.0) - th["em"] * cor * Q + th["sv"] * arrQ
    return dE, dS, dR, dQ, dG


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    EN, SN, RN, QN, ES, SS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
    k = 1.0 / th["tau"]
    dEN, dSN, dRN, dQN, dGN = _region(th, mech, EN, SN, RN, QN, GN, 1.0, th["chN"], hunt, hab, cor, th["em"], k * TSN, k * VSN)
    dES, dSS, dRS, dQS, dGS = _region(th, mech, ES, SS, RS, QS, GS, th["ks"], th["chS"], hunt, hab, cor, th["emS"], k * TNS, k * VNS)
    dTNS = th["em"] * cor * (EN + SN) - k * TNS
    dTSN = th["emS"] * cor * (ES + SS) - k * TSN
    dVNS = th["em"] * cor * QN - k * VNS
    dVSN = th["em"] * cor * QS - k * VSN
    return _pack(M, [dEN, dSN, dRN, dQN, dES, dSS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1], s[3], s[4] + s[5], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    PN = M.max(y[0], 0.0)
    QN = M.max(y[1], 0.01)
    PS = M.max(y[2], 0.0)
    QS = M.max(y[3], 0.01)
    z = 0.0 * PN
    KN = th["c0"] + th["chN"]
    KS = th["ks"] * (th["c0"] + th["chS"])
    SN = PN * KN / (KN + PN + 1e-9)
    SS = PS * KS / (KS + PS + 1e-9)
    ZN = PN - SN + th["phi"] * SN
    ZS = PS - SS + th["phi"] * SS
    GN = ZN / (ZN + 100.0)
    GS = ZS / (ZS + 100.0 * th["ks"])
    return _pack(M, [PN - SN, SN, 1.0 + z, QN, PS - SS, SS, 1.0 + z, QS, z, z, z, z, GN, GS])
