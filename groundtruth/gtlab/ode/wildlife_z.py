"""Grey-box ODE: wildlife_z (wildlife x13 with long-hold variants). NUMPY + math only.

Base (x13): per region i (north scale 1, south scale ks). Adults P spread over exposed ground (open
pasture, mixed cover) and sheltered browse; occupancy sits at its equilibrium. Sheltered capacity
K_i = s_i (c0 + ch hab) (c0 = 0 unless the base-refuge switch is on). Exposed
E = P (P/K)^n / (1 + (P/K)^n) (n = 1 unless the refuge-sharpness switch is on).
    A:  J' = r P - mJ J;  rec = mJ J / (1 + J/(cJ s_i F))            (nursery competition)
    P' = rec - d P - dh_i (1 - hab) E - H q phi E^2/(E^2 + Ph^2) - em_i cor P + sv T_in/tau
         q = hunting quota (or 7 (quota/7)^gq with the quota-power switch)
         phi = hunting effort (1 unless the effort switch is on: phi' = (P/(P + Pe) - phi)/te,
               hunters give up slowly when prey are scarce; with the static switch hs the effort
               sits at phi = P/(P + Pe): hunters' effort falls with the catch per trip;
               hsE: the same with the exposed prey, phi = E/(E + Pe))
    B:  R' = w (1 - bh (1 - hab)) (rl + (1 - rl) R) (1 - R) - kR (P/s_i) R  (finite food renewal)
    G' = (E/(E + 100 s_i) - G)/tz;  Q' = cq (q0 + q1 G - Q) Q/(Q + 4) - eq cor Q + sv V_in/tau
         eq = em (or em P/(P + Pt) with the predators-follow-prey switch: few predator journeys
         leave a region whose prey are scarce)
    transit pools T, V per direction relax with time tau; fraction sv arrives.
Switch bh2: the south's renewal loss without protection is xb times the north's (bh_S = xb bh).
Switches cg / hg: the corridor and habitat settings act through cor^cg and hab^hg (journeys or
protection that are not proportional to the setting); both 1 in x13.
Reset: P, Q from y0; J = r P/mJ; R = 1; pools empty; G at full habitat; phi = P/(P + Pe).
"""
import math

import numpy as np

FAMILY = "wildlife_z"
FLAGS = []
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: young animals compete for nursery food before recruiting",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition (not modelled in this family)",
}
N_SUB = 2

STATE = ["PN", "JN", "RN", "QN", "PS", "JS", "RS", "QS", "TNS", "TSN", "VNS", "VSN", "GN", "GS", "FN", "FS"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e4, 1e5, 1.0, 1e3, 1e4, 1e5, 1.0, 1e3, 1e4, 1e4, 1e3, 1e3, 1e2, 1e2, 1.0, 1.0])

PARAMS = [
    ("r", 0.34207679970469973, 0.05, 5.0, True),
    ("d", 0.1194228887535048, 0.01, 2.0, True),
    ("ks", 0.7973974425455057, 0.3, 1.5, False),
    ("H", 1.4837942564157593, 0.01, 50.0, True),
    ("Ph", 5.625205605740534, 0.5, 500.0, True),
    ("em", 0.029831319484129967, 0.0005, 0.5, True),
    ("emS", 0.04357353540292663, 0.0005, 0.5, True),
    ("tau", 26.015797658807568, 0.5, 150.0, True),
    ("sv", 0.7837118570099956, 0.0, 1.0, False),
    ("cJ", 56.79263697154018, 2.0, 100000.0, True),
    ("mJ", 1.0666509014380816, 0.02, 5.0, True),
    ("w", 0.026272303773532145, 0.002, 1.5, True),
    ("kR", 0.00023886252908292288, 1e-06, 0.02, True),
    ("bh", 0.44591734084087487, 0.0, 1.0, False),
    ("rl", 0.4142525668797434, 0.0, 1.0, False),
    ("dhN", 0.057184040388278026, 0.0, 1.0, False),
    ("dhS", 0.03486202534795581, 0.0, 1.0, False),
    ("ch", 253.38328194689427, 1.0, 1000.0, True),
    ("q0", 1.7274609037374886, 0.0, 10.0, False),
    ("q1", 2.27072445633324, 0.0, 10.0, False),
    ("cq", 0.09380610959307359, 0.002, 2.0, True),
    ("tz", 35.928181829793374, 0.5, 300.0, True),
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


def _exposed(th, P, K):
    if "n" in FLAGS:
        z = (P / (K + 1e-9)) ** th["nR"]
        return P * z / (1.0 + z)
    return P * P / (K + P + 1e-9)


def _cap(th, sc, hab):
    return sc * ((th["c0"] if "c0" in FLAGS else 0.0) + th["ch"] * hab)


def _qtravel(th, P):
    if "pt" in FLAGS:
        return th["em"] * P / (P + th["Pt"] + 1e-9)
    return th["em"] + 0.0 * P


def _region(th, mech, P, J, R, Q, G, Fh, sc, dh, hunt, hab, cor, em, arrP, arrQ, bh=None):
    F = R if "B" in mech else 1.0
    if "A" in mech:
        dJ = th["r"] * P - th["mJ"] * J
        rec = th["mJ"] * J / (1.0 + J / (th["cJ"] * sc * F + 1e-9))
    else:
        dJ = 0.0 * J
        rec = th["r"] * F * P
    E = _exposed(th, P, _cap(th, sc, hab))
    E2 = E * E
    q = 7.0 * (hunt / 7.0) ** th["gq"] if "g" in FLAGS else hunt
    if "eff" in FLAGS:
        phi = Fh
    elif "hs" in FLAGS:
        phi = P / (P + th["Pe"])
    elif "hsE" in FLAGS:
        phi = E / (E + th["Pe"])
    else:
        phi = 1.0
    harv = th["H"] * q * phi * E2 / (E2 + th["Ph"] ** 2)
    dP = rec - th["d"] * P - dh * (1.0 - hab) * E - harv - em * cor * P + th["sv"] * arrP
    if "B" in mech:
        bh = th["bh"] if bh is None else bh
        dR = th["w"] * (1.0 - bh * (1.0 - hab)) * (th["rl"] + (1.0 - th["rl"]) * R) * (1.0 - R) - th["kR"] * (P / sc) * R
    else:
        dR = 1.0 - R
    dG = (E / (E + 100.0 * sc) - G) / th["tz"]
    dQ = th["cq"] * (th["q0"] + th["q1"] * G - Q) * Q / (Q + 4.0) - _qtravel(th, P) * cor * Q + th["sv"] * arrQ
    if "eff" in FLAGS:
        dF = (P / (P + th["Pe"]) - Fh) / th["te"]
    else:
        dF = 0.0 * Fh
    return dP, dJ, dR, dQ, dG, dF


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    hunt, hab, cor = uu
    if "cg" in FLAGS:
        cor = M.max(cor, 0.0) ** th["cg"]
    if "hg" in FLAGS:
        hab = M.max(hab, 0.0) ** th["hg"]
    PN, JN, RN, QN, PS, JS, RS, QS, TNS, TSN, VNS, VSN, GN, GS, FN, FS = s
    k = 1.0 / th["tau"]
    dPN, dJN, dRN, dQN, dGN, dFN = _region(th, mech, PN, JN, RN, QN, GN, FN, 1.0, th["dhN"], hunt, hab, cor, th["em"], k * TSN, k * VSN)
    dPS, dJS, dRS, dQS, dGS, dFS = _region(th, mech, PS, JS, RS, QS, GS, FS, th["ks"], th["dhS"], hunt, hab, cor, th["emS"], k * TNS, k * VNS,
                                                  bh=th["bh"] * th["xb"] if "bh2" in FLAGS else None)
    dTNS = th["em"] * cor * PN - k * TNS
    dTSN = th["emS"] * cor * PS - k * TSN
    dVNS = _qtravel(th, PN) * cor * QN - k * VNS
    dVSN = _qtravel(th, PS) * cor * QS - k * VSN
    return _pack(M, [dPN, dJN, dRN, dQN, dPS, dJS, dRS, dQS, dTNS, dTSN, dVNS, dVSN, dGN, dGS, dFN, dFS])


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
    EN = _exposed(th, PN, _cap(th, 1.0, 1.0))
    ES = _exposed(th, PS, _cap(th, th["ks"], 1.0))
    JN = th["r"] * PN / th["mJ"]
    JS = th["r"] * PS / th["mJ"]
    GN = EN / (EN + 100.0)
    GS = ES / (ES + 100.0 * th["ks"])
    if "eff" in FLAGS:
        FN = PN / (PN + th["Pe"])
        FS = PS / (PS + th["Pe"])
    else:
        FN = 1.0 + z
        FS = 1.0 + z
    return _pack(M, [PN, JN, 1.0 + z, QN, PS, JS, 1.0 + z, QS, z, z, z, z, GN, GS, FN, FS])
