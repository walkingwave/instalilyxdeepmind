"""Grey-box ODE: ad_auction v8 (exploratory superset of ad_auction_min). NUMPY + math only.

Adds to ad_auction_min (every addition is off at its default):
  R  rival capital in our audience (1 at reset), relaxing to 1 - kr * X with time tau_r: rivals leave a depleted
     audience. The won fraction uses the rival bid level b0 * R; the cost per impression scales with R^gr.
  cb conversions per impression grow with the bid: c * (bid / 1.5)^cb.
  P  prepared fraction of the audience (0 at reset): exposure prepares people at rate kp * e * (1 - P), preparation
     fades with tau_p; purchases start at c * imp * (g0 + (1 - g0) * P).
  e0 preparation saturation: dP uses e / (1 + e / e0) (e0 large: proportional to exposure; small: time-driven).
  K  idle fulfilment capacity (full at reset, Kmax): completions are capped at F + K / tau_k; K drains when the cap
     binds and refills at the spare rate F - conv otherwise (Kmax -> 0: plain capacity F).
  ab audience size exponent: the targeted pool is N * breadth^ab (1: proportional to breadth).
  md exposure removal exponent: people removed per tick = kd * e0 * (e / e0)^md, e = imp / (N breadth), e0 = 0.2.
"""
import math

import numpy as np

FAMILY = "ad_auction_v8"
OBS = ["win_rate", "spend", "conversions"]
CTRL = ["bid", "budget_cap", "targeting_breadth"]
MECHS = {"A": "exposure removes reachable people (always on)",
         "B": "fulfilment queue with limited capacity (always on)", "C": "unused"}
N_SUB = 2

STATE = ["X", "Q1", "Q2", "R", "P", "K"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1.0, 1e6, 1e6, 5.0, 1.0, 1e6])

PARAMS = [
    ("N", 255.0, 20.0, 5000.0, True),
    ("b0", 3.0, 0.2, 30.0, True),
    ("nw", 1.5, 0.3, 4.0, False),
    ("gam", 0.35, 0.0, 1.0, False),
    ("kd", 0.1, 0.005, 2.0, True),
    ("tau_e", 40.0, 3.0, 400.0, True),
    ("c", 0.25, 0.01, 2.0, True),
    ("tau1", 12.0, 0.7, 60.0, True),
    ("tau2", 8.0, 0.7, 60.0, True),
    ("F", 6.0, 0.5, 100.0, True),
    ("kb", 1.0, 0.0, 10.0, False),
    ("kw", 0.3, 0.0, 3.0, False),
    ("tau_r", 60.0, 3.0, 600.0, True),
    ("kr", 0.0, 0.0, 3.0, False),
    ("gr", 0.0, 0.0, 2.0, False),
    ("cb", 0.0, -1.0, 1.0, False),
    ("g0", 1.0, 0.0, 1.0, False),
    ("kp", 1.0, 0.01, 50.0, True),
    ("tau_p", 60.0, 3.0, 600.0, True),
    ("md", 1.0, 0.3, 3.0, False),
    ("e0", 1e3, 1e-3, 1e3, True),
    ("Kmax", 1e-3, 1e-3, 500.0, True),
    ("tau_k", 10.0, 1.0, 100.0, True),
    ("ab", 1.0, 0.3, 1.5, False),
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
    X, Q1, Q2, R, P, K = s
    Rp = _pos(M, R, 1e-3)
    bn = M.pow(bid, th["nw"])
    w = bn / (bn + M.pow(th["b0"] * Rp, th["nw"])) / (1.0 + th["kw"] * (br - 0.1))
    p = M.pow(bid / 1.5, th["gam"]) * M.pow(Rp, th["gr"])
    A = th["N"] * M.pow(br, th["ab"]) * _pos(M, 1.0 - X, 1e-3)
    want = w * A
    afford = cap / (p + 1e-6)
    imp = M.max(_smin(M, want, afford, 0.5), 0.0)
    imp = M.min(imp, afford)
    spend = M.min(p * imp, cap)
    win = imp / (A + 1e-6)
    conv = M.max(_smin(M, Q2 / th["tau2"], th["F"] + K / th["tau_k"], 0.1), 0.0)
    return w, p, A, imp, spend, win, conv


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    bid, cap, br = uu
    X, Q1, Q2, R, P, K = s
    w, p, A, imp, spend, win, conv = _alg(M, s, uu, th)
    e = imp / (th["N"] * M.pow(br, th["ab"]))
    work = 1.0 + th["kb"] * (br - 0.1)
    r1 = Q1 / (th["tau1"] * work)
    dX = th["kd"] * 0.2 * M.pow(e / 0.2, th["md"]) - X / th["tau_e"]
    cq = th["c"] * M.pow(bid / 1.5, th["cb"]) * (th["g0"] + (1.0 - th["g0"]) * P)
    dQ1 = cq * imp - r1
    dQ2 = r1 - conv
    dR = (1.0 - th["kr"] * X - R) / th["tau_r"]
    dP = th["kp"] * e / (1.0 + e / th["e0"]) * (1.0 - P) - P / th["tau_p"]
    sp = th["F"] - conv
    dK = _pos(M, sp, 1e-3) * _pos(M, 1.0 - K / th["Kmax"], 1e-3) - _pos(M, -sp, 1e-3)
    return _pack(M, [dX, dQ1, dQ2, dR, dP, dK])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    w, p, A, imp, spend, win, conv = _alg(M, s, uu, th)
    return _pack(M, [win, spend, conv])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    return _pack(M, [z, z, z, z + 1.0, z, z + th["Kmax"]])
