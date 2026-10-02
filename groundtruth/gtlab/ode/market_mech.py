"""Grey-box ODE: market, minimal base + switchable brief mechanisms. NUMPY + math only (ships verbatim).

Base (always on)
  Volume  = v0 + B, reset order backlog B draining with TAU_B = 2.7 (ratio 0.69 per tick in every run).
  Depth   D relaxes first-order to D* = d0 * exp(-a_t * tax - a_r * rate), rate k_dn when shrinking,
          k_up when refilling (smooth switch).
  Price   dP = g * (k_p * (p_a - P) - k_r * rate): a restoring pull toward the anchor p_a and a
          rate-driven sell pressure (equilibrium p_a - k_r * rate / k_p). No floor.
          g = 1 / (1 + exp((tax - x_c) / W_X)): trading, and so the price, freezes once the tax
          passes the threshold x_c (width W_X = 0.002 fixed); below it the tax does not slow the price.
Mechanisms (exactly two active, letters as in the brief)
  A settlement tie-up: T in [0, 1], dT = g_A * (volume + C_A * |dP|) * (1 - T) - T / tau_A.
      Tied funding scales the price move and the depth refill rate by (1 - T).
  B risk capacity: E = EMA of price (TAU_E), adverse move = max(E - P, 0);
      dR = g_R * max(E - P, 0) - R / tau_R. Lost capacity weakens the pull (/(1 + R)),
      raises the rate impact (*(1 + R)) and lowers the depth target (/(1 + R)).
  C momentum: m = EMA (TAU_M) of k_m * dP_core; dP = dP_core + m.
Reset: B0 = volume0 - v0, P0 = E0 = price0, D0 = depth0, T0 = R0 = m0 = 0.
"""
import math

import numpy as np

FAMILY = "market_mech"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {
    "A": "settlement tie-up: trading ties dealer funding until settlement",
    "B": "risk capacity loss after adverse price moves",
    "C": "momentum: exposure shifts toward recently successful strategies",
}
N_SUB = 2

STATE = ["B", "P", "D", "T", "E", "R", "m"]
_NS = len(STATE)
STATE_LO = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -5.0])
STATE_HI = np.array([1e4, 400.0, 1e3, 1.0, 400.0, 5.0, 5.0])

TAU_B = 2.7
TAU_E = 10.0
TAU_M = 10.0
C_A = 20.0
W_X = 0.002

PARAMS = [
    ("v0", 2.3, 0.5, 6.0, True),
    ("d0", 90.0, 60.0, 130.0, False),
    ("a_t", 28.0, 2.0, 100.0, True),
    ("a_r", 2.0, 0.0, 30.0, False),
    ("k_dn", 0.1, 0.01, 1.0, True),
    ("k_up", 0.15, 0.01, 1.0, True),
    ("p_a", 93.0, 70.0, 120.0, False),
    ("k_p", 0.02, 0.001, 0.3, True),
    ("k_r", 3.0, 0.05, 40.0, True),
    ("x_c", 0.046, 0.02, 0.07, False),
    ("g_A", 0.002, 1e-4, 0.02, True),      # A
    ("tau_A", 20.0, 2.0, 200.0, True),     # A
    ("g_R", 0.03, 1e-3, 0.5, True),        # B
    ("tau_R", 30.0, 3.0, 300.0, True),     # B
    ("k_m", 1.0, 0.0, 2.5, False),         # C
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
    B, P, D, T, E, R, m = s
    z = 0.0 * P
    dB = -B / TAU_B
    fa = (1.0 - T) if "A" in mech else 1.0
    rr = (1.0 + R) if "B" in mech else 1.0
    g = 1.0 / (1.0 + M.exp((tax - th["x_c"]) / W_X))
    core = g * fa * (th["k_p"] * (th["p_a"] - P) / rr - th["k_r"] * rate * rr)
    if "C" in mech:
        dP = core + m
        dm = (th["k_m"] * core - m) / TAU_M
    else:
        dP = core
        dm = z
    Dstar = th["d0"] * M.exp(-th["a_t"] * tax - th["a_r"] * rate) / rr
    e = Dstar - D
    sw = 0.5 * (1.0 + e / M.sqrt(e * e + 4.0))
    k = th["k_dn"] + (th["k_up"] * fa - th["k_dn"]) * sw
    dD = k * e
    if "A" in mech:
        act = th["v0"] + B + C_A * M.sqrt(dP * dP + 1e-6)
        dT = th["g_A"] * act * (1.0 - T) - T / th["tau_A"]
    else:
        dT = z
    dE = (P - E) / TAU_E
    if "B" in mech:
        dR = th["g_R"] * M.max(E - P, 0.0) - R / th["tau_R"]
    else:
        dR = z
    return _pack(M, [dB, dP, dD, dT, dE, dR, dm])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, D, T, E, R, m = s
    return _pack(M, [P, th["v0"] + B, D])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    P = M.max(y[0], 1.0)
    B = M.max(y[1] - th["v0"], 0.0)
    D = M.max(y[2], 0.0)
    z = 0.0 * P
    return _pack(M, [B, P, D, z, P, z, z])
