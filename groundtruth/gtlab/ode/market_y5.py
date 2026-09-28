"""Grey-box ODE: market_y5 = market_y3 with market_y4's two-stage order pipeline (orders prepared, O, then executed,
v; each stage tau_v); p_full keeps market_y3 bounds.
market_y3 = market_y plus consumers' warehouses as an explicit inventory (storage demand).
  Iw = consumers' warehouse fill (0.5 at reset: "warehouses start half full"). Desired fill I* = i0 - i_r r/0.1
       (the rate is the carrying cost of storage). Restocking/destocking orders k_w (I* - Iw) execute through the
       gate g_eff; excess storage demand lifts the supported price: A1 = p_lo + (Q - p_lo) C + k_I (I* - Iw).
       At reset with i0 > 0.5 the warehouses fill (price hump at zero controls); a rate lowers I* and the
       destocking pushes the price down until the warehouses reach the new fill.
  Pinned vs market_y: c_p, w_d, x_R, tau_Q; B acts on depth only (no k_BP).
market_y = agent-flow market (producers' working cash, committed orders, dealer books) with the
three dealer mechanisms as real switches. NUMPY + math only.

Base (always on):
  Q  = full-cash clearing level: starts at the reset price, relaxes to the common fundamental p_full (tau_Q).
  C  = producers' working cash (1 = full at reset). Revenue from consumption regrows it while trading,
       rho g (C + c0)(1 - C); interest on borrowed cash drains it, a_r (r/0.1)^n_r (1 - C + b0);
       while frozen, production keeps consuming cash with no revenue: c_f (1 - g) C.
  A  = cash-supported level p_lo + (Q - p_lo) C, lagged (falls with tau_A, rises with tau_Au: dumped goods
       must be absorbed before the price recovers).
  P, v = committed orders: v builds toward g_eff s_v tanh(kap (A - P)/s_v) with the preparation/execution time
       tau_v; committed orders keep executing under a freeze (a new policy does not cancel them);
       g_eff = g_l + (1 - g_l) g, g = trading gate at the tax x_c.
  Volume: V0 + reset burst B + |A - P| + c_p g |v| + reset adjustment W.
  Depth: D -> D* (tau 8), reset-burst inventory G -> H on dealer books settling (slower under tax).
Mechanisms (exactly two active):
  A settlement tie-up of funding: inventory stranded on dealer books by a freeze, S -> (1 - g) r/0.1, ties funding
       at the rate: D* = d0 (1 - m1 s(x)) (1 - m2 S). Off: a freeze thins quotes by m2 regardless of the rate.
  B risk-capacity loss after adverse moves: R builds with price falls (partly rate-gated, cut off by a tax above
       x_R) and recovers with tau_R; depth = D (1 - R) + H; lost capacity also amplifies the price impact of
       a fall by (1 + k_BP R).
  C investor momentum: exposure shifts toward recent winners, M = EMA of price velocity (TAU_M); the order
       flow target gets + k_M M (a started move keeps going and overshoots).
Reset: C = 1, P = A = Q = price0, v = 0, B = volume0 - V0, D = depth0, R = G = H = S = M = 0, W = 1.
"""
import math

import numpy as np

FAMILY = "market_y5"
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up of funding (stranded inventory costs capacity at the rate)",
         "B": "risk-capacity loss after adverse moves",
         "C": "investor momentum toward recent winners"}
N_SUB = 2

STATE = ["B", "P", "v", "C", "D", "R", "H", "Q", "W", "G", "A", "S", "M", "Iw", "O"]
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, -20.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1.0, 1e4, 0.9, 200.0, 1e4, 1.0, 200.0, 1e4, 1.0, 20.0, 1.0, 20.0])

TAU_B = 2.7
W_X = 0.0005
C_G = 0.0184
V0 = 1.77
C_W = 1.139
TAU_W = 70.71
TAU_G = 3.0
W_R = 0.004
TAU_D = 8.0
C_H = 0.0759
TAU_H1 = 9.57
A_H = 38.2
W_RR = 0.7
TAU_S = 1.0
TAU_M = 12.0
C_P = 1.145
W_D = 0.0014
X_R = 0.037
TAU_Q = 204.0

PARAMS = [
    ("d0", 92.5, 60.0, 130.0, False),
    ("m1", 0.52, 0.0, 0.95, False),
    ("x_d", 0.0187, 0.002, 0.035, False),
    ("m2", 0.5, 0.0, 0.95, False),
    ("k_R", 0.84, 0.0, 2.0, False),
    ("tau_R", 20.0, 3.0, 200.0, True),
    ("k_M", 0.3, 0.0, 0.95, False),
    ("g_l", 0.031, 0.0, 0.3, False),
    ("x_c", 0.0444, 0.0426, 0.0458, False),
    ("p_full", 91.0, 86.0, 99.0, False),
    ("p_lo", 74.5, 55.0, 90.0, False),
    ("rho", 0.2, 0.001, 0.5, True),
    ("c0", 0.72, 0.001, 2.0, True),
    ("a_r", 1.0, 0.001, 2.0, True),
    ("n_r", 2.0, 0.5, 10.0, False),
    ("b0", 0.3, 0.01, 3.0, True),
    ("c_f", 0.005, 0.0, 0.1, False),
    ("i0", 0.6, 0.3, 1.0, False),
    ("i_r", 0.1, 0.0, 0.5, False),
    ("k_w", 0.03, 0.001, 0.3, True),
    ("k_I", 10.0, 0.0, 60.0, False),
    ("tau_A", 15.5, 1.0, 100.0, True),
    ("tau_Au", 22.7, 2.0, 300.0, True),
    ("kap", 0.092, 0.005, 2.0, True),
    ("s_v", 0.45, 0.05, 5.0, True),
    ("tau_v", 4.0, 0.5, 60.0, True),
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
    def pw(a, b):
        return (a if a > 1e-12 else 1e-12) ** b


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
    def pw(a, b):
        return np.power(np.maximum(a, 1e-12), b)


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


def _gate(M, tax, th):
    return 1.0 / (1.0 + M.exp((tax - th["x_c"]) / W_X))


def _cash_drift(M, C, g, rate, th):
    rr = M.pw(rate / 0.1, th["n_r"])
    dC = th["rho"] * g * (C + th["c0"]) * (1.0 - C) - th["a_r"] * rr * (1.0 - C + th["b0"]) - th["c_f"] * (1.0 - g) * C
    return dC, rr


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    B, P, v, C, D, R, H, Q, W, G, A, S, Mo, Iw, O = s
    onA, onB, onC = ("A" in mech), ("B" in mech), ("C" in mech)
    dB = -B / TAU_B
    g = _gate(M, tax, th)
    dC, rr = _cash_drift(M, C, g, rate, th)
    geff = th["g_l"] + (1.0 - th["g_l"]) * g
    Istar = th["i0"] - th["i_r"] * rate * 10.0
    dI = geff * th["k_w"] * (Istar - Iw)
    ea = th["p_lo"] + (Q - th["p_lo"]) * C + th["k_I"] * (Istar - Iw) - A
    su = 0.5 * (1.0 + ea / M.sqrt(ea * ea + 1.0))
    dA = ea * ((1.0 - su) / th["tau_A"] + su / th["tau_Au"])
    dS = ((1.0 - g) * rate * 10.0 - S) / TAU_S if onA else 0.0 * S
    dQ = (th["p_full"] - Q) / TAU_Q
    zv = th["kap"] * (A - P) / th["s_v"]
    zv = M.min(M.max(zv, -20.0), 20.0)
    ez = M.exp(2.0 * zv)
    push = th["s_v"] * (ez - 1.0) / (ez + 1.0)
    if onC:
        push = push + th["k_M"] * Mo
    dO = (geff * push - O) / th["tau_v"]
    dv = (O - v) / th["tau_v"]
    dP = v
    dM = (v - Mo) / TAU_M if onC else 0.0 * Mo
    if onB:
        fall = 0.5 * (M.sqrt(dP * dP + 0.01) - dP)
        sR = 1.0 / (1.0 + M.exp((tax - X_R) / W_R))
        dR = th["k_R"] * fall * (1.0 - W_RR + W_RR * rr / (1.0 + rr)) * sR / 20.0 - R / th["tau_R"]
    else:
        dR = 0.0 * R
    dG = C_H * B - G / TAU_G
    dH = G / TAU_G - H * M.exp(-A_H * tax) / TAU_H1
    sd = 1.0 / (1.0 + M.exp(-(tax - th["x_d"]) / W_D))
    if onA:
        Dstar = th["d0"] * (1.0 - th["m1"] * sd) * (1.0 - th["m2"] * S)
    else:
        Dstar = th["d0"] * (1.0 - th["m1"] * sd) * (1.0 - th["m2"] * (1.0 - g))
    dD = (Dstar - D) / TAU_D
    return _pack(M, [dB, dP, dv, dC, dD, dR, dH, dQ, -W / TAU_W, dG, dA, dS, dM, dI, dO])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, C, D, R, H, Q, W, G, A, S, Mo, Iw, O = s
    rate, tax = uu
    g = _gate(M, tax, th)
    gap = A - P
    vol = V0 + B + C_G * M.sqrt(gap * gap + 0.25) + C_P * g * M.sqrt(v * v + 0.0001) + C_W * W * g * (rate * 10.0 + tax * 20.0)
    dep = D * (1.0 - R) + H
    return _pack(M, [P, vol, dep])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    P = M.max(y[0], 0.0)
    B = M.max(y[1] - V0 - 0.5 * C_G, 0.0)
    D = M.max(y[2], 0.0)
    z = 0.0 * P
    return _pack(M, [B, P, z, z + 1.0, D, z, z, P, z + 1.0, z, P, z, z, z + 0.5, z])
