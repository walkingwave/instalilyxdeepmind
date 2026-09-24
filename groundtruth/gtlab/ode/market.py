"""Grey-box ODE: market. NUMPY + math only (ships verbatim in submissions).

Structure
- Producer warehouse Gp and working cash Cp (production consumes cash, final consumption
  returns revenue); consumer warehouse Gc drained by consumption. Stocks are fractions of
  storage capacity.
- Desired buying/selling depend on price (reservation value p_ref, elasticities), tax
  (wedge between buyer and seller price) and interest rate (carrying cost: sellers sell
  faster, buyers stock less). Orders pass a 2-stage preparation/execution pipeline, so a
  policy change cannot cancel committed orders.
- Matched volume = smooth min(buy, sell); the imbalance is absorbed by finite dealer
  books up to depth; dealer inventory is hedged out slowly.
- log price moves with imbalance / depth and with dealer inventory pressure.
- depth = dealer capacity * (1 - |inventory|/limit) * funding * risk factors.
Mechanisms: A settlement tie-up (funding tied by dealer volume, recovers tau_f),
B risk capacity loss after adverse moves (tau_rk), C momentum investors (EMA of returns
shifts buying).
"""
import math

import numpy as np

FAMILY = "market"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {
    "A": "settlement tie-up: dealer inventory ties funding until settlement",
    "B": "risk capacity loss after adverse price moves",
    "C": "momentum: investors shift exposure toward recently successful strategies",
}
N_SUB = 2

STATE = ["Gp", "Cp", "Gc", "B1", "B2", "S1", "S2", "logp", "dealer", "tied", "risk", "mom"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e4)
STATE_HI[0] = 1.0
STATE_HI[1] = 1.0
STATE_HI[2] = 1.0
STATE_LO[7] = -12.0
STATE_HI[7] = 12.0
STATE_LO[8] = -1e3
STATE_HI[8] = 1e3
STATE_HI[9] = 1.0
STATE_HI[10] = 1e3
STATE_LO[11] = -1.0
STATE_HI[11] = 1.0

PARAMS = [
    ("p_ref", 1.0, 0.01, 1000.0, True),
    ("a_b", 0.1, 1e-3, 10.0, True),
    ("a_s", 0.1, 1e-3, 10.0, True),
    ("e_b", 1.5, 0.1, 8.0, False),
    ("e_s", 1.5, 0.1, 8.0, False),
    ("k_rate", 3.0, 0.0, 40.0, False),
    ("k_tax", 2.0, 0.0, 20.0, False),
    ("prod", 0.05, 1e-3, 2.0, True),
    ("cons", 0.05, 1e-3, 2.0, True),
    ("cash_use", 1.0, 0.01, 20.0, True),
    ("tau_o", 3.0, 0.5, 50.0, True),
    ("kd", 10.0, 0.05, 1e4, True),
    ("dmax", 0.5, 0.01, 100.0, True),
    ("kappa", 0.2, 1e-4, 5.0, True),
    ("kinv", 0.02, 0.0, 1.0, False),
    ("tau_hedge", 20.0, 1.0, 2000.0, True),
    ("vol_k", 10.0, 0.01, 1e5, True),
    ("tau_f", 30.0, 2.0, 1000.0, True),   # A
    ("f_k", 1.0, 0.0, 50.0, False),       # A
    ("tau_rk", 150.0, 5.0, 5000.0, True),  # B
    ("rk_k", 50.0, 0.0, 2000.0, False),   # B
    ("tau_m", 10.0, 1.0, 500.0, True),    # C
    ("m_k", 5.0, 0.0, 200.0, False),      # C
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
def _alg(M, s, uu, th, mech):
    rate, tax = uu
    Gp, Cp, Gc, B1, B2, S1, S2, lp, dn, tied, risk, mom = s
    p = M.exp(lp)
    pc = p * (1.0 + th["k_tax"] * tax) / th["p_ref"]
    ps = p * _pos(M, 1.0 - th["k_tax"] * tax, 1e-3) / th["p_ref"]
    fac_b = 1.0 / (1.0 + th["k_rate"] * rate)
    if "C" in mech:
        fac_b = fac_b * (1.0 + 0.9 * M.tanh(th["m_k"] * mom))
    b_des = th["a_b"] * (1.0 - Gc) * M.exp(-th["e_b"] * M.log(pc)) * fac_b
    s_des = th["a_s"] * Gp * M.exp(th["e_s"] * M.log(ps)) * (1.0 + th["k_rate"] * rate)
    b_des = M.min(b_des, 5.0)
    s_des = M.min(s_des, 5.0)
    ab = B2 / th["tau_o"]
    as_ = S2 / th["tau_o"]
    depth = th["kd"] * _pos(M, 1.0 - M.sqrt(dn * dn + 1e-8) / th["dmax"], 1e-3)
    if "A" in mech:
        depth = depth * (1.0 - tied)
    if "B" in mech:
        depth = depth / (1.0 + risk)
    m = _pos(M, _smin(M, ab, as_, 1e-3), 1e-5)
    imb = ab - as_
    aimb = M.sqrt(imb * imb + 1e-8)
    dshare = depth / (depth + aimb * th["kd"] * 0.1 + 1e-9)
    dt = imb * dshare
    adt = aimb * dshare
    return b_des, s_des, ab, as_, depth, m, imb, dt, adt, p


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    Gp, Cp, Gc, B1, B2, S1, S2, lp, dn, tied, risk, mom = s
    b_des, s_des, ab, as_, depth, m, imb, dt, adt, p = _alg(M, s, uu, th, mech)
    to = th["tau_o"]
    prod = th["prod"] * _smin(M, 4.0 * Cp, 1.0, 0.02) * (1.0 - Gp)
    cons = th["cons"] * Gc
    dlp = th["kappa"] * imb / (depth + 0.01 * th["kd"]) - th["kinv"] * dn / th["dmax"]
    d = [0.0] * _NS
    d[0] = prod - m - _pos(M, -dt, 1e-5)
    d[1] = th["cash_use"] * (cons - prod) - 0.1 * rate * Cp
    d[2] = m + _pos(M, dt, 1e-5) - cons
    d[3] = b_des - B1 / to
    d[4] = (B1 - B2) / to
    d[5] = s_des - S1 / to
    d[6] = (S1 - S2) / to
    d[7] = dlp
    d[8] = -dt - dn / th["tau_hedge"]
    if "A" in mech:
        d[9] = th["f_k"] * adt / th["kd"] * (1.0 - tied) - tied / th["tau_f"]
    if "B" in mech:
        d[10] = th["rk_k"] * _pos(M, -dn * dlp, 1e-6) / th["dmax"] - risk / th["tau_rk"]
    if "C" in mech:
        d[11] = (dlp - mom) / th["tau_m"]
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    b_des, s_des, ab, as_, depth, m, imb, dt, adt, p = _alg(M, s, uu, th, mech)
    return _pack(M, [p, th["vol_k"] * (m + adt), depth])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    lp = M.log(M.max(y[0], 1e-3))
    z = 0.0 * lp
    return _pack(M, [0.5 + z, 1.0 + z, 0.5 + z, z, z, z, z, lp, z, z, z, z])
