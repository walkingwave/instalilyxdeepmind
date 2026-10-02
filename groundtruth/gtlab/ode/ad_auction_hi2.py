"""Grey-box ODE: ad_auction hi2 = hi1 + second-price payment: the price per impression follows the rival
bid level, p = (bid / 1.5)^gam * R^kpr (R = rival capital left in the audience). NUMPY + math only.
hi1 notes: ad_auction hi1 = v8b with the audience split into breadth bins.

v8b keeps exposure removal X and preparation P as single fractions of "the targeted audience", so
when targeting is widened after a narrow phase the depleted narrow core is spread over the new, fresh
people and the fresh audience never shows up (spend and conversions too low after broadening).
Here the pool is nested along breadth: bin k covers breadth (0.1 k, 0.1 (k+1)], coverage
c_k = clip((breadth - 0.1 k) / 0.1, 0, 1). Each bin keeps its own removed fraction X_k and prepared
fraction P_k. Available people A = N sum_k 0.1 c_k (1 - X_k); impressions go to bins in proportion to
their available people. Per bin exposure (bin average) eb_k = imp c_k (1 - X_k) / A,
dX_k = kd eb_k - X_k / tau_e, dP_k = c_k kp e_k / (1 + e_k / E0) (1 - P_k) with e_k = imp (1 - X_k) / A.
Rival capital uses the removed fraction of the targeted audience Xe = 1 - A / (N breadth).
Purchases c sum_k imp_k (g0 + (1 - g0) P_k); fulfilment, auction and price as v8b.
With every X_k, P_k equal this is exactly v8b.
"""
import math

import numpy as np

FAMILY = "ad_auction_hi2"
OBS = ["win_rate", "spend", "conversions"]
CTRL = ["bid", "budget_cap", "targeting_breadth"]
MECHS = {"A": "exposure removes reachable people (always on)",
         "B": "fulfilment queue with limited capacity (always on)",
         "C": "rival capital leaves a depleted audience (always on)"}
N_SUB = 2
NB = 10
W = 1.0 / NB
_LO = np.arange(NB) * W

STATE = [f"X{k}" for k in range(NB)] + [f"P{k}" for k in range(NB)] + ["Q1", "Q2", "K"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1.0] * (2 * NB) + [1e6, 1e6, 1e6])
NW = 1.3
E0 = 0.2
TK = 13.0

PARAMS = [
    ("N", 266.51705246677216, 20.0, 5000.0, True),
    ("b0", 3.8319568601909597, 0.2, 30.0, True),
    ("gam", 0.44095712682781824, 0.0, 1.0, False),
    ("kd", 0.13824034735164845, 0.005, 2.0, True),
    ("tau_e", 63.95965934504543, 3.0, 400.0, True),
    ("c", 0.26793555409085906, 0.01, 2.0, True),
    ("tau1", 5.43316932471045, 0.7, 60.0, True),
    ("tau2", 5.793772654366348, 0.7, 60.0, True),
    ("F", 5.39384526673784, 0.5, 100.0, True),
    ("kw", 0.37140060555771154, 0.0, 3.0, False),
    ("kr", 0.3762393353980513, 0.0, 0.95, False),
    ("g0", 0.5058757683206756, 0.0, 1.0, False),
    ("kp", 0.33318076592460205, 0.01, 50.0, True),
    ("Kmax", 28.603860972230635, 0.01, 500.0, True),
    ("kpr", 0.0, 0.0, 3.0, False),
]


def _pos(a, e):
    return 0.5 * (a + np.sqrt(a * a + e * e))


def _smin(a, b, e):
    return 0.5 * (a + b - np.sqrt((a - b) * (a - b) + e * e))


def _pw(a, b):
    return np.power(np.maximum(a, 1e-12), b)


def _alg(x, u, th):
    bid, cap, br = u[..., 0], u[..., 1], u[..., 2]
    Xb = x[..., 0:NB]
    Q2, K = x[..., 2 * NB + 1], x[..., 2 * NB + 2]
    cov = np.clip((br[..., None] - _LO) / W, 0.0, 1.0)
    av = cov * _pos(1.0 - Xb, 1e-3)
    A = th["N"] * W * av.sum(-1)
    Xe = 1.0 - A / (th["N"] * br)
    R = _pos(1.0 - th["kr"] * Xe, 1e-3)
    bn = _pw(bid, NW)
    w = bn / (bn + _pw(th["b0"] * R, NW)) / (1.0 + th["kw"] * (br - 0.1))
    p = _pw(bid / 1.5, th["gam"]) * _pw(R, th["kpr"])
    want = w * A
    afford = cap / (p + 1e-6)
    imp = np.maximum(_smin(want, afford, 0.5), 0.0)
    imp = np.minimum(imp, afford)
    spend = np.minimum(p * imp, cap)
    win = imp / (A + 1e-6)
    conv = np.maximum(_smin(Q2 / th["tau2"], th["F"] + K / TK, 0.1), 0.0)
    return cov, av, A, imp, spend, win, conv


def _psc(a, e):
    return 0.5 * (a + math.sqrt(a * a + e * e))


def _alg1(x, u, th):
    bid, cap, br = float(u[0]), float(u[1]), float(u[2])
    N = th["N"]
    cov = [0.0] * NB
    av = [0.0] * NB
    sa = 0.0
    for k in range(NB):
        c = (br - k * W) / W
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        cov[k] = c
        a = c * _psc(1.0 - x[k], 1e-3)
        av[k] = a
        sa += a
    A = N * W * sa
    R = _psc(1.0 - th["kr"] * (1.0 - A / (N * br)), 1e-3)
    bn = math.pow(max(bid, 1e-12), NW)
    w = bn / (bn + math.pow(max(th["b0"] * R, 1e-12), NW)) / (1.0 + th["kw"] * (br - 0.1))
    p = math.pow(max(bid / 1.5, 1e-12), th["gam"]) * math.pow(max(R, 1e-12), th["kpr"])
    want = w * A
    afford = cap / (p + 1e-6)
    d = want - afford
    imp = 0.5 * (want + afford - math.sqrt(d * d + 0.25))
    imp = min(max(imp, 0.0), afford)
    spend = min(p * imp, cap)
    win = imp / (A + 1e-6)
    a2 = x[2 * NB + 1] / th["tau2"]
    b2 = th["F"] + x[2 * NB + 2] / TK
    d = a2 - b2
    conv = max(0.5 * (a2 + b2 - math.sqrt(d * d + 0.01)), 0.0)
    return cov, av, sa, A, imp, spend, win, conv


def _f1(x, u, th):
    x = x.tolist()
    cov, av, sa, A, imp, spend, win, conv = _alg1(x, u, th)
    out = [0.0] * _NS
    kd, te, kp, g0 = th["kd"], th["tau_e"], th["kp"], th["g0"]
    ia = imp / (A + 1e-6)
    buy = 0.0
    for k in range(NB):
        Xk, Pk = x[k], x[NB + k]
        ek = ia * _psc(1.0 - Xk, 1e-3)
        out[k] = kd * cov[k] * ek - Xk / te
        out[NB + k] = cov[k] * kp * ek / (1.0 + ek / E0) * (1.0 - Pk)
        buy += av[k] * (g0 + (1.0 - g0) * Pk)
    buy = th["c"] * imp * buy / (sa + 1e-12)
    Q1, K = x[2 * NB], x[2 * NB + 2]
    r1 = Q1 / th["tau1"]
    out[2 * NB] = buy - r1
    out[2 * NB + 1] = r1 - conv
    sp = th["F"] - conv
    out[2 * NB + 2] = _psc(sp, 1e-3) * _psc(1.0 - K / th["Kmax"], 1e-3) - _psc(-sp, 1e-3)
    return np.array(out)


def f(x, u, th, mech):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return _f1(x, u, th)
    u = np.broadcast_to(np.asarray(u, float), x.shape[:-1] + (3,))
    br = u[..., 2]
    Xb, Pb = x[..., 0:NB], x[..., NB:2 * NB]
    Q1 = x[..., 2 * NB]
    K = x[..., 2 * NB + 2]
    cov, av, A, imp, spend, win, conv = _alg(x, u, th)
    share = av / (av.sum(-1)[..., None] + 1e-12)                        # imp_k / imp
    ek = (imp / (A + 1e-6))[..., None] * _pos(1.0 - Xb, 1e-3)            # per covered person
    eb = cov * ek
    kd, te, kp, g0 = (np.asarray(th[n], float)[..., None] for n in ("kd", "tau_e", "kp", "g0"))
    dX = kd * eb - Xb / te
    dP = cov * kp * ek / (1.0 + ek / E0) * (1.0 - Pb)
    g = g0 + (1.0 - g0) * Pb
    buy = th["c"] * imp * (share * g).sum(-1)
    r1 = Q1 / th["tau1"]
    dQ1 = buy - r1
    dQ2 = r1 - conv
    sp = th["F"] - conv
    dK = _pos(sp, 1e-3) * _pos(1.0 - K / th["Kmax"], 1e-3) - _pos(-sp, 1e-3)
    return np.concatenate([dX, dP, dQ1[..., None], dQ2[..., None], dK[..., None]], axis=-1)


def h(x, u, th, mech):
    x = np.asarray(x, float)
    if x.ndim == 1:
        r = _alg1(x.tolist(), u, th)
        return np.array([r[6], r[5], r[7]])
    u = np.broadcast_to(np.asarray(u, float), x.shape[:-1] + (3,))
    cov, av, A, imp, spend, win, conv = _alg(x, u, th)
    return np.stack(np.broadcast_arrays(win, spend, conv), axis=-1)


def x0(y0, th, mech):
    y0 = np.asarray(y0, float)
    z = np.zeros(y0.shape[:-1] + (_NS,))
    z[..., -1] = th["Kmax"]
    return z
