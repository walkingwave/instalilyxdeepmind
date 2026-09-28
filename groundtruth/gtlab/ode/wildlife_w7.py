"""Grey-box ODE: wildlife_w7 (wildlife z9/x13 family with shelter and predator-travel variants). NUMPY + math only.

Per region i (north scale 1, south scale ks). Adults P spread over exposed ground and sheltered browse;
occupancy sits at its equilibrium. Sheltered capacity K_i = s_i (c0 + ch_i hab^kp)
(c0 = 0 unless switch c0; ch_S = ch unless switch chs gives the south its own slope chS; kp = 1 unless
switch kp). Exposed E = P^2/(K + P).
    A:  J' = r P - mJ J;  rec = mJ J / (1 + J/(cJ s_i F))            (nursery competition)
    P' = rec - d P - dh_i (1 - hab) E - H hunt phi E^2/(E^2 + Ph^2) - em_i cor P + sv T_in/tau
         phi = P/(P + Pe) with switch hs (hunting effort falls with scarcity), else 1;
         switch xh: the south's take is xh times the north's at the same exposure
    B:  R' = w (1 - bh (1 - hab)) (rl + (1 - rl) R) (1 - R) - kR (P/s_i) R  (finite food renewal)
    G' = (E/(E + 100 s_i) - G)/tz;  Q' = cq (q0 + q1 G - Q) Q/(Q + 4) - eq cor Q + sv V_in/tau
         eq = em, or em P/(P + Pt) with switch pt (predators travel only where prey are worth following)
    transit pools T, V per direction relax with time tau; fraction sv arrives.
Reset: P, Q from y0; J = r P/mJ; R = 1; pools empty; G at full habitat.
"""
import math

import numpy as np

FAMILY = "wildlife_w7"
FLAGS = ['c0', 'pt']
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: young animals compete for nursery food before recruiting",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2

STATE = ["PN", "JN", "RN", "QN", "PS", "JS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e5, 1.0, 1e3, 1e4, 1e5, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2])

PARAMS = [
    ('r', 0.3320885136602903, 0.05, 5.0, True),
    ('d', 0.13859799841071582, 0.01, 2.0, True),
    ('ks', 0.8009250355178825, 0.3, 1.5, False),
    ('H', 1.2844940657344488, 0.01, 50.0, True),
    ('Ph', 0.5000000000000001, 0.5, 500.0, True),
    ('em', 0.028137718705122265, 0.0005, 0.5, True),
    ('emS', 0.03938141991686534, 0.0005, 0.5, True),
    ('tau', 25.299153804596468, 0.5, 150.0, True),
    ('sv', 0.7329379258650502, 0.0, 1.0, False),
    ('cJ', 15.006652109910599, 2.0, 100000.0, True),
    ('mJ', 4.999976797083815, 0.02, 5.0, True),
    ('w', 0.027901143998126605, 0.002, 1.5, True),
    ('kR', 0.00022547921877772275, 1e-06, 0.02, True),
    ('bh', 0.4792271345328071, 0.0, 1.0, False),
    ('rl', 0.3534526642293274, 0.0, 1.0, False),
    ('dhN', 0.21763190272500857, 0.0, 1.0, False),
    ('dhS', 0.1185763821644351, 0.0, 1.0, False),
    ('ch', 583.8652637953509, 1.0, 1000.0, True),
    ('q0', 1.8688364715067762, 0.0, 10.0, False),
    ('q1', 4.118109211480462, 0.0, 10.0, False),
    ('cq', 0.09932318785774699, 0.002, 2.0, True),
    ('tz', 19.554814141340614, 0.5, 300.0, True),
    ('c0', 321.30740812803293, 0.0, 500.0, False),
    ('Pt', 5.100018015853493, 0.5, 500.0, True),
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


def _cap(M, th, sc, hab, south):
    slope = th["chS"] if ("chs" in FLAGS and south) else th["ch"]
    hv = M.max(hab, 0.0) ** th["kp"] if "kp" in FLAGS else hab
    return sc * ((th["c0"] if "c0" in FLAGS else 0.0) + slope * hv)


def _qtravel(th, P):
    if "pt" in FLAGS:
        return th["em"] * P / (P + th["Pt"] + 1e-9)
    return th["em"] + 0.0 * P


def _region(M, th, mech, P, J, R, Q, G, sc, south, dh, hunt, hab, cor, em, arrP, arrQ):
    F = R if "B" in mech else 1.0
    if "A" in mech:
        dJ = th["r"] * P - th["mJ"] * J
        rec = th["mJ"] * J / (1.0 + J / (th["cJ"] * sc * F + 1e-9))
    else:
        dJ = 0.0 * J
        rec = th["r"] * F * P
    K = _cap(M, th, sc, hab, south)
    E = P * P / (K + P + 1e-9)
    E2 = E * E
    phi = P / (P + th["Pe"]) if "hs" in FLAGS else 1.0
    harv = th["H"] * hunt * phi * E2 / (E2 + th["Ph"] ** 2)
    if "xh" in FLAGS and south:
        harv = th["xh"] * harv
    dP = rec - th["d"] * P - dh * (1.0 - hab) * E - harv - em * cor * P + th["sv"] * arrP
    if "B" in mech:
        dR = th["w"] * (1.0 - th["bh"] * (1.0 - hab)) * (th["rl"] + (1.0 - th["rl"]) * R) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    dG = (E / (E + 100.0 * sc) - G) / th["tz"]
    dQ = th["cq"] * (th["q0"] + th["q1"] * G - Q) * Q / (Q + 4.0) - _qtravel(th, P) * cor * Q + th["sv"] * arrQ
    return dP, dJ, dR, dQ, dG


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    PN, JN, RN, QN, PS, JS, RS, QS, TNS, TSN, VNS, VSN, GN, GS = s
    k = 1.0 / th["tau"]
    dPN, dJN, dRN, dQN, dGN = _region(M, th, mech, PN, JN, RN, QN, GN, 1.0, False, th["dhN"], hunt, hab, cor, th["em"], k * TSN, k * VSN)
    dPS, dJS, dRS, dQS, dGS = _region(M, th, mech, PS, JS, RS, QS, GS, th["ks"], True, th["dhS"], hunt, hab, cor, th["emS"], k * TNS, k * VNS)
    dTNS = th["em"] * cor * PN - k * TNS
    dTSN = th["emS"] * cor * PS - k * TSN
    dVNS = _qtravel(th, PN) * cor * QN - k * VNS
    dVSN = _qtravel(th, PS) * cor * QS - k * VSN
    return _pack(M, [dPN, dJN, dRN, dQN, dPS, dJS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0], s[3], s[4], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    PN = M.max(y[0], 0.0)
    QN = M.max(y[1], 0.01)
    PS = M.max(y[2], 0.0)
    QS = M.max(y[3], 0.01)
    z = 0.0 * PN
    KN = _cap(M, th, 1.0, 1.0, False)
    KS = _cap(M, th, th["ks"], 1.0, True)
    EN = PN * PN / (KN + PN + 1e-9)
    ES = PS * PS / (KS + PS + 1e-9)
    JN = th["r"] * PN / th["mJ"]
    JS = th["r"] * PS / th["mJ"]
    GN = EN / (EN + 100.0)
    GS = ES / (ES + 100.0 * th["ks"])
    return _pack(M, [PN, JN, 1.0 + z, QN, PS, JS, 1.0 + z, QS, z, z, z, z, GN, GS])
