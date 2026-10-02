"""Grey-box ODE: ad_auction v8c (v8b with a price floor: p = ((bid + pb) / (1.5 + pb))^gam, tau1 fixed at T1 = 5.4), pool + rival capital + preparation + fulfilment with idle capacity. NUMPY + math only.

Auction: opportunities per tick A = N * breadth * (1 - X), X = fraction of the targeted pool temporarily removed by
exposure. Rival capital in our audience R = 1 - kr * X (rivals move their capital away from a depleted audience,
fast compared with a tick); the rival bid level is b0 * R. We win w = bid^n / (bid^n + (b0 R)^n) / (1 + kw (breadth - 0.1))
of the opportunities at a cost per impression p = (bid / 1.5)^gam; impressions = min(w A, cap / p); spend = p * imp;
win_rate = imp / A. Exposure removes kd people per impression; they return with tau_e.
Preparation: P = prepared fraction of the audience (0 at reset, "available and unprepared"), grows with the exposure
per person e = imp / (N breadth) as dP = kp * e / (1 + e / E0) * (1 - P); purchases start at c * imp * (g0 + (1 - g0) P).
Fulfilment: two stages (tau1, tau2); completions are capped at F + K / TK where K is idle fulfilment capacity (full at
Kmax on reset, drained when the cap binds, refilled at the spare rate F - conv otherwise). This gives the burst above F
at the start of a heavy pulse and the flat backlog plateau at F after it.
Fixed shape constants: Hill exponent NW = 1.3, preparation saturation E0 = 0.2, idle-capacity release time TK = 13.
Mechanism letters are accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "ad_auction_v8c"
OBS = ["win_rate", "spend", "conversions"]
CTRL = ["bid", "budget_cap", "targeting_breadth"]
MECHS = {"A": "exposure removes reachable people (always on)",
         "B": "fulfilment queue with limited capacity (always on)",
         "C": "rival capital leaves a depleted audience (always on)"}
N_SUB = 2

STATE = ["X", "Q1", "Q2", "P", "K"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1.0, 1e6, 1e6, 1.0, 1e6])
NW = 1.3
E0 = 0.2
TK = 13.0
T1 = 5.4

PARAMS = [
    ("N", 266.57444890842277, 20.0, 5000.0, True),
    ("b0", 3.7570624317891013, 0.2, 30.0, True),
    ("gam", 0.43572325596942457, 0.0, 1.0, False),
    ("kd", 0.1377930226653349, 0.005, 2.0, True),
    ("tau_e", 64.09332789718184, 3.0, 400.0, True),
    ("c", 0.27830590697833635, 0.01, 2.0, True),
    ("pb", 0.3, 0.0, 5.0, False),
    ("tau2", 5.873626459349504, 0.7, 60.0, True),
    ("F", 5.352445688085872, 0.5, 100.0, True),
    ("kw", 0.3795776870425216, 0.0, 3.0, False),
    ("kr", 0.34533303133202176, 0.0, 0.95, False),
    ("g0", 0.4561093124899725, 0.0, 1.0, False),
    ("kp", 0.31651315365893223, 0.01, 50.0, True),
    ("Kmax", 30.112438093173882, 0.01, 500.0, True),
]


class _PY:
    max = max
    min = min

    @staticmethod
    def exp(z):
        return math.exp(z if z < 50.0 else 50.0)

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0

    @staticmethod
    def pow(a, b):
        return math.pow(a if a > 1e-12 else 1e-12, b)


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))

    @staticmethod
    def pow(a, b):
        return np.power(np.maximum(a, 1e-12), b)


def _pos(M, a, e):
    return 0.5 * (a + M.sqrt(a * a + e * e))


def _smin(M, a, b, e):
    return 0.5 * (a + b - M.sqrt((a - b) * (a - b) + e * e))


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


def _alg(M, s, uu, th):
    bid, cap, br = uu
    X, Q1, Q2, P, K = s
    R = _pos(M, 1.0 - th["kr"] * X, 1e-3)
    bn = M.pow(bid, NW)
    w = bn / (bn + M.pow(th["b0"] * R, NW)) / (1.0 + th["kw"] * (br - 0.1))
    p = M.pow((bid + th["pb"]) / (1.5 + th["pb"]), th["gam"])
    A = th["N"] * br * _pos(M, 1.0 - X, 1e-3)
    want = w * A
    afford = cap / (p + 1e-6)
    imp = M.max(_smin(M, want, afford, 0.5), 0.0)
    imp = M.min(imp, afford)
    spend = M.min(p * imp, cap)
    win = imp / (A + 1e-6)
    conv = M.max(_smin(M, Q2 / th["tau2"], th["F"] + K / TK, 0.1), 0.0)
    return w, p, A, imp, spend, win, conv


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    bid, cap, br = uu
    X, Q1, Q2, P, K = s
    w, p, A, imp, spend, win, conv = _alg(M, s, uu, th)
    e = imp / (th["N"] * br)
    r1 = Q1 / T1
    dX = th["kd"] * e - X / th["tau_e"]
    dQ1 = th["c"] * (th["g0"] + (1.0 - th["g0"]) * P) * imp - r1
    dQ2 = r1 - conv
    dP = th["kp"] * e / (1.0 + e / E0) * (1.0 - P)
    sp = th["F"] - conv
    dK = _pos(M, sp, 1e-3) * _pos(M, 1.0 - K / th["Kmax"], 1e-3) - _pos(M, -sp, 1e-3)
    return _pack(M, [dX, dQ1, dQ2, dP, dK])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    w, p, A, imp, spend, win, conv = _alg(M, s, uu, th)
    return _pack(M, [win, spend, conv])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    return _pack(M, [z, z, z, z, z + th["Kmax"]])
