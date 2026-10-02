"""Grey-box ODE: wildlife_tm = wild9p (v8h with predators on their own transit) + mechanism C. NUMPY + math only.

Per region i (north scale 1, south scale ks); observed prey = P, observed predators = Q.
    births   b = r F P,  F = R if B else 1
    A:  recruitment b / (1 + P/(cJ s_i))                         (nursery food competition)
    B:  R' = w (1 - bh (1 - hab)) (1 - R) - kR (P/s_i) R         (finite food renewal)
    C:  arrivals settle with probability 1/(1 + P_dest/(Kc s_dest)); the rest are lost
        (settlement competition; flag "cq": predator arrivals too, 1/(1 + Q_dest/(Kq s_dest)))
    harvest  H hunt P^2 / (P^2 + Ph^2)
    P' = rec - (D0 + dh_i (1 - hab)) P - harvest - em_i cor P + sv set_P T_in / tau [- predation]
    Q' = cq (q0 + q1 G_i - Q) Q/(Q + 4) - emq cor Q + svq set_Q V_in / tauq
    Z_i = P/(P + 100 s_i), G_i' = (Z_i - G_i)/tz
    transit pools per direction: T' = em_from cor P_from - T/tau, V' = emq cor Q_from - V/tauq.
Flags:
    cq:  settlement competition also applies to predator arrivals (Kq)
    pd:  predation on prey: - ap Q P/(P + 100 s_i)
    sh:  shelter occupancy lags protection: S_i' = (hab - S_i)/ts, harvest x (1 - xh S_i), S_i(0) = s0
Reset: P, Q from y0; R = 1; pools empty; G = Z.
"""
import math

import numpy as np

FAMILY = "wildlife_tm"
FLAGS = []
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: nursery food competition caps recruitment",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition: arrivals settle less where the destination is crowded",
}
N_SUB = 2
D0 = 0.7

STATE = ["PN", "RN", "QN", "PS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS", "SN", "SS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1.0, 1e3, 1e4, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2, 1.0, 1.0])

PARAMS = [
    ('r', 0.8808542347385102, 0.05, 3.0, True),
    ('ks', 0.7975559776432426, 0.3, 1.5, False),
    ('H', 1.3679210929831758, 0.01, 50.0, True),
    ('Ph', 25.10751258551428, 0.5, 500.0, True),
    ('em', 0.022415886831882822, 0.0005, 0.5, True),
    ('emS', 0.03480665493839356, 0.0005, 0.5, True),
    ('tau', 88.06126139556702, 0.5, 150.0, True),
    ('sv', 0.9009507418792193, 0.0, 1.0, False),
    ('cJ', 898.4621785824578, 5.0, 5000.0, True),
    ('w', 0.025934196891793958, 0.002, 1.5, True),
    ('kR', 2.3528179848824946e-05, 1e-06, 0.02, True),
    ('bh', 0.40625352293732203, 0.0, 1.0, False),
    ('dhN', 0.047178310123975935, 0.0, 1.0, False),
    ('dhS', 0.030439676088799085, 0.0, 1.0, False),
    ('q0', 1.7679310288109504, 0.0, 10.0, False),
    ('q1', 1.128539203225617, 0.0, 10.0, False),
    ('cq', 0.10132402353824638, 0.002, 2.0, True),
    ('tz', 19.622238239382696, 0.5, 300.0, True),
    ('emq', 0.029743134949109507, 0.0005, 0.5, True),
    ('tauq', 18.1073226714383, 0.5, 150.0, True),
    ('svq', 0.8056096865981848, 0.0, 1.0, False),
    ('Kc', 200.0, 2.0, 1e5, True),
    ('Kq', 20.0, 0.2, 1e4, True),
    ('ap', 0.5, 1e-3, 20.0, True),
    ('ts', 30.0, 1.0, 500.0, True),
    ('xh', 0.5, 0.0, 1.0, False),
    ('s0', 0.5, 0.0, 1.0, False),
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


def _region(M, th, mech, P, R, Q, G, Sh, sc, hunt, hab, cor, dh, em, arrP, arrQ):
    B = "B" in mech
    F = R if B else 1.0
    b = th["r"] * F * P
    rec = b / (1.0 + P / (th["cJ"] * sc)) if "A" in mech else b
    P2 = P * P
    Ph = th["Ph"]
    harv = th["H"] * hunt * P2 / (P2 + Ph * Ph)
    if "sh" in FLAGS:
        harv = harv * (1.0 - th["xh"] * Sh)
    dS = (hab - Sh) / th["ts"]
    Z = P / (P + 100.0 * sc)
    setP = 1.0
    setQ = 1.0
    if "C" in mech:
        setP = 1.0 / (1.0 + P / (th["Kc"] * sc))
        if "cq" in FLAGS:
            setQ = 1.0 / (1.0 + Q / (th["Kq"] * sc))
    dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th["sv"] * setP * arrP
    if "pd" in FLAGS:
        dP = dP - th["ap"] * Q * Z
    if B:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    dG = (Z - G) / th["tz"]
    outQ = th["emq"] * cor * Q
    dQ = th["cq"] * (th["q0"] + th["q1"] * G - Q) * Q / (Q + 4.0) - outQ + th["svq"] * setQ * arrQ
    return dP, dR, dQ, dG, outQ, dS


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS, SN, SS = s
    k = 1.0 / th["tau"]
    kq = 1.0 / th["tauq"]
    dPN, dRN, dQN, dGN, oQN, dSN = _region(M, th, mech, PN, RN, QN, GN, SN, 1.0, hunt, hab, cor, th["dhN"], th["em"], k * TSN, kq * VSN)
    dPS, dRS, dQS, dGS, oQS, dSS = _region(M, th, mech, PS, RS, QS, GS, SS, th["ks"], hunt, hab, cor, th["dhS"], th["emS"], k * TNS, kq * VNS)
    dTNS = th["em"] * cor * PN - k * TNS
    dTSN = th["emS"] * cor * PS - k * TSN
    dVNS = oQN - kq * VNS
    dVSN = oQS - kq * VSN
    return _pack(M, [dPN, dRN, dQN, dPS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS, dSN, dSS])


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
    GN = PN / (PN + 100.0)
    GS = PS / (PS + 100.0 * th["ks"])
    S0 = th["s0"] + z
    return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS, S0, S0])
