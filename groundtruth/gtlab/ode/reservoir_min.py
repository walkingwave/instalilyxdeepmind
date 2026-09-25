"""Grey-box ODE: reservoir, minimal. NUMPY + math only (ships verbatim).

States: level L (volume), quality Q, return flow R, clock c (ticks since reset).
    dL/dt = inflow(c) + R - delivered(L, u) - seep * L,   L clipped at L_FULL (full pool, 941.3)
    dQ/dt = (q_base + q_aer * aeration - Q) / tau_q
    dR/dt = (k_ret * irrigation - R) / tau_ret                 delayed return from irrigated land
    dc/dt = 1
    inflow(c)    = q_m + q_a * sin(2 pi c / per + phi)          season, phase locked to reset
    delivered    = softmin(release + irrigation, c_out * (L / 500)^p_out)   head-limited discharge
    spill        = gate(L) * softplus(inflow + R - delivered - seep * L)   net inflow spills when full
Observation after the tick: [L, inflow(c) + R, delivered + spill, Q].

Reset convention: L0, Q0 from the observed initial readings; R0 = 0, c0 = 0. The initial inflow and
outflow readings carry no state (both jump to their algebraic values at the first tick).

Mechanism letters are accepted for interface compatibility but every term is always active;
fit with pairs=("AB",).
"""
import math

import numpy as np

FAMILY = "reservoir_min"
OBS = ["level", "inflow", "outflow", "quality"]
CTRL = ["release_rate", "irrigation_allocation", "withdrawal_depth", "aeration"]
MECHS = {"A": "head-limited discharge (always on)", "B": "seepage + soft spill (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["L", "Q", "R", "c"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([941.3, 1.0, 5.0, 1e9])

_LREF = 500.0
_L_FULL = 941.3
_TWO_PI = 2.0 * math.pi

PARAMS = [
    ("q_m", 11.43, 8.0, 15.0, False),
    ("q_a", 2.17, 0.5, 5.0, False),
    ("per", 67.8, 40.0, 110.0, True),
    ("phi", 0.04, -1.6, 1.6, False),
    ("seep", 0.0014, 1e-4, 0.02, True),
    ("c_out", 13.3, 5.0, 30.0, True),
    ("p_out", 0.33, 0.05, 1.0, False),
    ("q_base", 0.935, 0.5, 1.0, False),
    ("q_aer", 0.01, 0.0, 0.05, False),
    ("tau_q", 3.0, 0.67, 50.0, True),
    ("k_ret", 0.05, 0.0, 0.3, False),
    ("tau_ret", 40.0, 5.0, 50.0, True),
]


class _PY:
    max = max
    min = min
    sin = math.sin

    @staticmethod
    def exp(z):
        return math.exp(z if z < 50.0 else 50.0)

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0

    @staticmethod
    def pow(a, b):
        return math.pow(a, b) if a > 0.0 else 0.0


class _NP:
    max = np.maximum
    min = np.minimum
    sin = np.sin

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))

    @staticmethod
    def pow(a, b):
        return np.power(np.maximum(a, 0.0), b)


def _pos(M, a, e):
    return 0.5 * (a + M.sqrt(a * a + e * e))


def _smin(M, a, b, e):
    d = a - b
    return 0.5 * (a + b - M.sqrt(d * d + e * e))


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


def _inflow(M, c, th):
    return th["q_m"] + th["q_a"] * M.sin(_TWO_PI * c / th["per"] + th["phi"])


def _delivered(M, L, uu, th):
    rel, irr, dep, aer = uu
    cap = th["c_out"] * M.pow(L / _LREF, th["p_out"])
    return _smin(M, rel + irr, cap, 0.5)


def _spill(M, L, net):
    gate = 1.0 / (1.0 + M.exp(-(L - _L_FULL + 2.0) / 0.5))
    return gate * _pos(M, net, 0.5)


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    L, Q, R, c = s
    irr, aer = uu[1], uu[3]
    dL = _inflow(M, c, th) + R - _delivered(M, L, uu, th) - th["seep"] * L
    dQ = (th["q_base"] + th["q_aer"] * aer - Q) / th["tau_q"]
    dR = (th["k_ret"] * irr - R) / th["tau_ret"]
    dc = 1.0 + 0.0 * c
    return _pack(M, [dL, dQ, dR, dc])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    L, Q, R, c = s
    delivered = _delivered(M, L, uu, th)
    net = _inflow(M, c, th) + R - delivered - th["seep"] * L
    return _pack(M, [L, _inflow(M, c, th) + R, delivered + _spill(M, L, net), Q])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    L = M.max(y[0], 0.0)
    Q = M.min(M.max(y[3], 0.0), 1.0)
    c = 0.0 * (y[1] + y[2])
    return _pack(M, [L, Q, c, c])
