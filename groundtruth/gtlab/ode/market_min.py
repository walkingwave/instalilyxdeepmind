"""Grey-box ODE: market, minimal. NUMPY + math only (ships verbatim).

Volume  = base flow v0 + reset order backlog B draining at 1/TAU_B (fixed 2.7: the burst decays
          by a factor 0.69 per tick in every run) (the burst seen after every
          reset: 60-80 at t=0, ~2 after 20 ticks, whatever the controls).
Depth   = frac * D_f + (1 - frac) * D_s: two dealer-capacity pools relaxing to the same target
          d0 * exp(-a_t * tax - a_r * rate); the fast pool with tau_f, the slow pool with an
          asymmetric rate (k_sd when shrinking, k_su when refilling: depth falls slowly under
          a tax and refills fast after it is lifted).
Price   = dP/dt = (m - k_r * rate) * g(tax) * s(P), g = exp(-a_x * tax) (a transaction tax
          freezes trading, so it freezes the price), s = clip((P - p_lo)/10, 0, 1) a soft floor
          at the producers' reservation value; m = momentum, dm/dt = (k_m * dP/dt - m)/tau_m
          (investors chase recent moves: a fall that starts under a rate step keeps accelerating
          after the step is lifted, as long as trading is not taxed).

Reset: B0 = volume0 - v0, P0 = price0, m0 = 0, D_f0 = D_s0 = depth0.
Mechanism letters are accepted for interface compatibility; every term is always active.
"""
import math

import numpy as np

FAMILY = "market_min"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "tax freezes trading (always on)", "B": "price momentum (always on)", "C": "unused"}
N_SUB = 2

STATE = ["B", "P", "m", "Df", "Ds"]
_NS = len(STATE)
STATE_LO = np.array([0.0, 0.0, -50.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 50.0, 1e4, 1e4])

TAU_B = 2.7
TAU_M = 10.0
W_LO = 10.0

PARAMS = [
    ("v0", 2.2, 0.5, 6.0, True),
    ("d0", 91.0, 60.0, 130.0, False),
    ("a_t", 28.0, 2.0, 100.0, True),
    ("a_r", 2.0, 0.0, 30.0, False),
    ("tau_f", 5.0, 1.5, 20.0, True),
    ("k_sd", 0.025, 0.005, 0.3, True),
    ("k_su", 0.08, 0.005, 0.5, True),
    ("frac", 0.42, 0.05, 0.95, False),
    ("k_r", 7.0, 0.2, 60.0, True),
    ("a_x", 47.0, 0.0, 120.0, False),
    ("k_m", 1.5, 0.0, 4.0, False),
    ("p_lo", 75.0, 20.0, 90.0, False),
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


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))


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


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    B, P, m, Df, Ds = s
    dB = -B / TAU_B
    # price
    g = M.exp(-th["a_x"] * tax)
    sf = M.min(M.max((P - th["p_lo"]) / W_LO, 0.0), 1.0)
    dP = (m - th["k_r"] * rate) * g * sf
    dm = (th["k_m"] * dP - m) / TAU_M
    # depth
    Dstar = th["d0"] * M.exp(-th["a_t"] * tax - th["a_r"] * rate)
    dDf = (Dstar - Df) / th["tau_f"]
    e = Dstar - Ds
    sw = 0.5 * (1.0 + e / M.sqrt(e * e + 4.0))
    k = th["k_sd"] + (th["k_su"] - th["k_sd"]) * sw
    dDs = k * e
    return _pack(M, [dB, dP, dm, dDf, dDs])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, m, Df, Ds = s
    vol = th["v0"] + B
    dep = th["frac"] * Df + (1.0 - th["frac"]) * Ds
    return _pack(M, [P, vol, dep])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    P = M.max(y[0], 0.0)
    B = M.max(y[1] - th["v0"], 0.0)
    D = M.max(y[2], 0.0)
    m = 0.0 * P
    return _pack(M, [B, P, m, D, D])
