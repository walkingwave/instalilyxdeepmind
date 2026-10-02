"""Grey-box ODE: wildlife_wild9n (v8h with switches ['phr', 'gx', 'pt']). NUMPY + math only.

Base = v8h (see wildlife_v8h.py). Switches:
    phr: harvest refuge grows with cover, per region: Ph_i = Ph (1 + hP_i hab)
    phs: harvest refuge scales with region size: Ph_i = Ph s_i
    pt:  predators start journeys in proportion to prey availability, em P/(P + Pt s_i)
    ptq: predators start journeys less when crowded out: em Q/(Q + Qt) (per predator)
    qx:  predator relaxation target lowered by the corridor inside the region: q0 + q1 G - xq cor
    ref: refuge Pr s_i of prey never exposed to harvest: harvest on max(P - Pr s_i, 0)
    hr:  harvest also removes food competitors: renewal pool recovers (none)
    hsx: south harvest factor xS on the requested quota
    gx:  predators feed on exposed prey: food Z = P e/(P e + 100 s_i), e = 1 + xg_i (1 - hab)
    bhr: own habitat weight on food renewal in the south, bhS
"""
import math

import numpy as np

FAMILY = "wildlife_wild9n"
FLAGS = ['phr', 'gx', 'pt']
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: nursery food competition caps recruitment",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2
D0 = 0.7

STATE = ["PN", "RN", "QN", "PS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1.0, 1e3, 1e4, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2])

PARAMS = [
    ('r', 0.879476323218783, 0.05, 3.0, True),
    ('ks', 0.7991713725708542, 0.3, 1.5, False),
    ('H', 1.3567344001368091, 0.01, 50.0, True),
    ('Ph', 24.944829387699713, 0.5, 500.0, True),
    ('em', 0.02593781706810343, 0.0005, 0.5, True),
    ('emS', 0.04033527068702279, 0.0005, 0.5, True),
    ('tau', 26.39036409931354, 0.5, 150.0, True),
    ('sv', 0.7800573507586477, 0.0, 1.0, False),
    ('cJ', 900.1504294955678, 5.0, 5000.0, True),
    ('w', 0.026202108245787555, 0.002, 1.5, True),
    ('kR', 2.3675571495756507e-05, 1e-06, 0.02, True),
    ('bh', 0.3462105711884271, 0.0, 1.0, False),
    ('dhN', 0.043905550461308965, 0.0, 1.0, False),
    ('dhS', 0.025175861577830756, 0.0, 1.0, False),
    ('q0', 1.753304368495746, 0.0, 10.0, False),
    ('q1', 1.1579882592866027, 0.0, 10.0, False),
    ('cq', 0.10086003859133522, 0.002, 2.0, True),
    ('tz', 27.25352098237159, 0.5, 300.0, True),
    ('hPN', 0.3, 0.0, 20.0, False),
    ('hPS', 0.3, 0.0, 20.0, False),
    ('xgN', 0.1, 0.0, 10.0, False),
    ('xgS', 0.1, 0.0, 10.0, False),
    ('Pt', 2.0, 0.1, 500.0, True),
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


def _region(M, th, mech, P, R, Q, G, sc, south, hunt, hab, cor, dh, em, arrP, arrQ):
    B = "B" in mech
    F = R if B else 1.0
    b = th["r"] * F * P
    rec = b / (1.0 + P / (th["cJ"] * sc)) if "A" in mech else b
    Ph = th["Ph"]
    if "phr" in FLAGS:
        Ph = Ph * (1.0 + (th["hPS"] if south else th["hPN"]) * hab)
    if "phs" in FLAGS:
        Ph = Ph * sc
    Pe = M.max(P - th["Pr"] * sc, 0.0) if "ref" in FLAGS else P
    Hq = th["H"] * (th["xS"] if ("hsx" in FLAGS and south) else 1.0)
    P2 = Pe * Pe
    harv = Hq * hunt * P2 / (P2 + Ph * Ph)
    dP = rec - (D0 + dh * (1.0 - hab)) * P - harv - em * cor * P + th["sv"] * arrP
    bh = th["bhS"] if ("bhr" in FLAGS and south) else th["bh"]
    if B:
        dR = th["w"] * (1.0 - bh * (1.0 - hab)) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    Pz = P * (1.0 + (th["xgS"] if south else th["xgN"]) * (1.0 - hab)) if "gx" in FLAGS else P
    Z = Pz / (Pz + 100.0 * sc)
    dG = (Z - G) / th["tz"]
    qt = th["q0"] + th["q1"] * G - (th["xq"] * cor if "qx" in FLAGS else 0.0)
    eq = th["em"]
    if "pt" in FLAGS:
        eq = eq * P / (P + th["Pt"] * sc)
    if "ptq" in FLAGS:
        eq = eq * Q / (Q + th["Qt"])
    outQ = eq * cor * Q
    dQ = th["cq"] * (qt - Q) * Q / (Q + 4.0) - outQ + th["sv"] * arrQ
    return dP, dR, dQ, dG, outQ


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, RN, QN, PS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
    k = 1.0 / th["tau"]
    dPN, dRN, dQN, dGN, oQN = _region(M, th, mech, PN, RN, QN, GN, 1.0, False, hunt, hab, cor, th["dhN"], th["em"], k * TSN, k * VSN)
    dPS, dRS, dQS, dGS, oQS = _region(M, th, mech, PS, RS, QS, GS, th["ks"], True, hunt, hab, cor, th["dhS"], th["emS"], k * TNS, k * VNS)
    dTNS = th["em"] * cor * PN - k * TNS
    dTSN = th["emS"] * cor * PS - k * TSN
    dVNS = oQN - k * VNS
    dVSN = oQS - k * VSN
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
    GN = PN / (PN + 100.0)
    GS = PS / (PS + 100.0 * th["ks"])
    return _pack(M, [PN, 1.0 + z, QN, PS, 1.0 + z, QS, z, z, z, z, GN, GS])
