"""Grey-box ODE: wildlife. NUMPY + math only (ships verbatim in submissions).

Structure (regions n = north, s = south)
- Prey split into exposed (open pasture) and sheltered (cover/browse) patches; habitat
  protection raises the sheltered target fraction and the resource-limited capacity
  (weight 1 in the north, hab_s in the south). Predators hunt exposed prey only
  (Holling type II). Logistic, resource-limited prey births.
- Hunting quota requests harvest from exposed prey (smooth min with availability).
- Corridor access controls NEW emigration; animals already in the 2-stage transit
  pipeline still arrive.
Mechanisms: A juvenile stage (births pass a nursery with density-dependent competition),
B slow resource depletion (grazing depletes a resource that renews with tau_res),
C settlement competition (arrivals only settle when the destination has space).
"""
import math

import numpy as np

FAMILY = "wildlife"
OBS = ["prey_north", "predator_north", "prey_south", "predator_south"]
CTRL = ["hunting_quota", "habitat_protection", "corridor_access"]
MECHS = {
    "A": "juvenile condition: births pass a nursery stage with food competition",
    "B": "finite food renewal: grazing depletes a slowly renewing resource",
    "C": "settlement competition: arrivals settle only where space is available",
}
N_SUB = 2
SH_REF = 0.6    # sheltered fraction at reset (fixed convention) relative to 'shelter'

STATE = ["PeN", "PsN", "ZN", "JN", "RN", "PeS", "PsS", "ZS", "JS", "RS", "T1", "T2", "U1", "U2"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e6)
STATE_HI[4] = 1.0
STATE_HI[9] = 1.0

PARAMS = [
    ("r_n", 0.05, 1e-3, 1.0, True),
    ("r_s", 0.05, 1e-3, 1.0, True),
    ("K_n", 1200.0, 20.0, 1e6, True),
    ("K_s", 1000.0, 20.0, 1e6, True),
    ("hab_k", 0.5, 0.0, 3.0, False),
    ("hab_s", 0.3, 0.0, 1.0, False),
    ("a_pred", 0.001, 1e-6, 0.1, True),
    ("h_t", 0.5, 0.0, 20.0, False),
    ("shelter", 0.5, 0.0, 0.95, False),
    ("k_ex", 0.1, 0.005, 2.0, True),
    ("e_conv", 0.2, 1e-3, 1.0, True),
    ("m_pred", 0.03, 1e-3, 0.5, True),
    ("m_prey", 0.005, 0.0, 0.2, False),
    ("q_k", 0.5, 0.02, 20.0, True),
    ("h50", 50.0, 1.0, 1e5, True),         # harvest half-saturation (prey scarcity)
    ("em", 0.01, 0.0, 0.2, False),
    ("tau_tr", 10.0, 1.0, 200.0, True),
    ("tau_juv", 10.0, 1.0, 200.0, True),   # A
    ("juv_k", 0.001, 0.0, 0.1, False),     # A
    ("tau_res", 100.0, 5.0, 5000.0, True),  # B
    ("dep_k", 0.5, 0.0, 10.0, False),      # B
    ("set_k", 1.0, 0.0, 20.0, False),      # C
]

# ---- numeric helpers (duplicated in every family file: files ship standalone) ----
class _PY:
    max = max
    min = min
    tanh = math.tanh

    @staticmethod
    def exp(z):
        return math.exp(z if z < 50.0 else 50.0)

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0

    @staticmethod
    def log(z):
        return math.log(z) if z > 1e-300 else -690.0


class _NP:
    max = np.maximum
    min = np.minimum
    tanh = np.tanh

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))

    @staticmethod
    def log(z):
        return np.log(np.maximum(z, 1e-300))


def _smin(M, a, b, e):
    return 0.5 * (a + b - M.sqrt((a - b) * (a - b) + e * e))


def _pos(M, a, e):
    return 0.5 * (a + M.sqrt(a * a + e * e))


def _unpack(x, u):
    if np.ndim(x) == 1:
        return _PY, x.tolist(), np.asarray(u, dtype=float).tolist()
    return _NP, [x[..., i] for i in range(x.shape[-1])], [u[..., j] for j in range(u.shape[-1])]


def _pack(M, vals):
    if M is _PY:
        return np.array(vals, dtype=float)
    return np.stack(np.broadcast_arrays(*vals), axis=-1).astype(float)


def _sig(M, z):
    return 1.0 / (1.0 + M.exp(-z))


def _yunpack(y0):
    if np.ndim(y0) == 1:
        return _PY, np.asarray(y0, dtype=float).tolist()
    return _NP, [y0[..., j] for j in range(y0.shape[-1])]


# ---- model ----
def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    quota, hp, cor = uu
    PeN, PsN, ZN, JN, RN, PeS, PsS, ZS, JS, RS, T1, T2, U1, U2 = s
    hk = th["hab_k"]
    wts = (1.0, th["hab_s"])
    Ks = (th["K_n"], th["K_s"])
    rs = (th["r_n"], th["r_s"])
    reg = ((PeN, PsN, ZN, JN, RN), (PeS, PsS, ZS, JS, RS))
    pe_tot = PeN + PeS + 1e-6
    harv = quota * th["q_k"] * pe_tot / (pe_tot + th["h50"])
    ktr = 2.0 / th["tau_tr"]
    arrive = (ktr * U2, ktr * T2)    # into north from south (U), into south from north (T)
    out = []
    d = [0.0] * _NS
    sf = th["shelter"] * (0.2 + 0.8 * hp)
    for k in range(2):
        Pe, Ps, Z, J, Rr = reg[k]
        P = Pe + Ps
        Kk = Ks[k] * (1.0 + hk * hp * wts[k])
        food = Rr if "B" in mech else 1.0
        births = rs[k] * P * food * _pos(M, 1.0 - (P + J) / Kk, 1e-3)
        pred = th["a_pred"] * Z * Pe / (1.0 + th["a_pred"] * th["h_t"] * Pe)
        h_k = harv * Pe / pe_tot
        emig = cor * th["em"] * Pe
        arr = arrive[k]
        if "C" in mech:
            arr = arr / (1.0 + th["set_k"] * P / Kk)
        ex = th["k_ex"] * (sf * P - Ps)
        if "A" in mech:
            mat = J / th["tau_juv"]
            dJ = births - mat - th["juv_k"] * J * J
            recruit = mat
        else:
            dJ = -J
            recruit = births
        i = 5 * k
        d[i] = recruit - pred - h_k - emig + arr - ex - th["m_prey"] * Pe
        d[i + 1] = ex - th["m_prey"] * Ps
        d[i + 2] = th["e_conv"] * pred - th["m_pred"] * Z
        d[i + 3] = dJ
        if "B" in mech:
            d[i + 4] = (1.0 - Rr) * (1.0 + hk * hp * wts[k]) / th["tau_res"] - th["dep_k"] * (P / Kk) * Rr / th["tau_res"]
        out.append(emig)
    d[10] = out[0] - ktr * T1
    d[11] = ktr * (T1 - T2)
    d[12] = out[1] - ktr * U1
    d[13] = ktr * (U1 - U2)
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[0] + s[1], s[2], s[5] + s[6], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    sf = th["shelter"] * SH_REF
    pn = M.max(y[0], 0.0)
    ps = M.max(y[2], 0.0)
    z = 0.0 * pn
    return _pack(M, [pn * (1.0 - sf), pn * sf, M.max(y[1], 0.0), z, 1.0 + z,
                     ps * (1.0 - sf), ps * sf, M.max(y[3], 0.0), z, 1.0 + z, z, z, z, z])
