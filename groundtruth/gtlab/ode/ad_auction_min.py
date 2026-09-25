"""Grey-box ODE: ad_auction, minimal pool + fulfilment queue. NUMPY + math only (ships verbatim).

Auction: opportunities per tick A = N * breadth * (1 - X), X = fraction of the targeted pool
temporarily removed by exposure. We win a fraction w(bid) = bid^n / (bid^n + b0^n) of them at a
cost per impression p(bid) = (bid / 1.5)^gam (second-price flavour: sublinear in the bid).
Impressions won = min(w * A, cap / p); spend = p * impressions <= budget_cap; win_rate =
impressions / A, so it equals w(bid) when the cap is slack and cap / (p * A) when it binds.
Exposure removes kd people per impression, they return with tau_e. Purchases start at c per
impression and pass a two-stage fulfilment chain whose first stage is slower for broader
audiences (work per purchase 1 + kb * (breadth - 0.1)); completions are capped at F per tick.

Broader audiences carry more rival pressure: the won fraction is divided by 1 + kw * (breadth - 0.1).
Reset convention (brief and data: conversions are 0 on the first two ticks of every run whatever
the initial report says): X = 0, empty queues. The initial report is a random reading, not a state.
Mechanism letters are accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "ad_auction_min"
OBS = ["win_rate", "spend", "conversions"]
CTRL = ["bid", "budget_cap", "targeting_breadth"]
MECHS = {"A": "exposure removes reachable people (always on)",
         "B": "fulfilment queue with limited capacity (always on)", "C": "unused"}
N_SUB = 2

STATE = ["X", "Q1", "Q2"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1.0, 1e6, 1e6])

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
    X, Q1, Q2 = s
    bn = M.pow(bid, th["nw"])
    w = bn / (bn + M.pow(th["b0"], th["nw"])) / (1.0 + th["kw"] * (br - 0.1))
    p = M.pow(bid / 1.5, th["gam"])
    A = th["N"] * br * _pos(M, 1.0 - X, 1e-3)
    want = w * A
    afford = cap / (p + 1e-6)
    imp = M.max(_smin(M, want, afford, 0.5), 0.0)
    imp = M.min(imp, afford)
    spend = M.min(p * imp, cap)
    win = imp / (A + 1e-6)
    conv = M.max(_smin(M, Q2 / th["tau2"], th["F"], 0.1), 0.0)
    return w, p, A, imp, spend, win, conv


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    bid, cap, br = uu
    X, Q1, Q2 = s
    w, p, A, imp, spend, win, conv = _alg(M, s, uu, th)
    work = 1.0 + th["kb"] * (br - 0.1)
    r1 = Q1 / (th["tau1"] * work)
    dX = th["kd"] * imp / (th["N"] * br) - X / th["tau_e"]
    dQ1 = th["c"] * imp - r1
    dQ2 = r1 - conv
    return _pack(M, [dX, dQ1, dQ2])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    w, p, A, imp, spend, win, conv = _alg(M, s, uu, th)
    return _pack(M, [win, spend, conv])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    return _pack(M, [z, z, z])
