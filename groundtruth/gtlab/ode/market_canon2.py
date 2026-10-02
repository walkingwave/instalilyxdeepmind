"""Grey-box ODE: market_canon2 = market_y3 with a literal dealer-quote switch (hysteresis), a slow funding tie-up
and a frozen price leak proportional to the gap.
  Gate state g (1 = dealers quote, 0 = frozen). Reset: g = 0 (orders and commitments start empty, the book opens
       only if the tax is below the opening threshold). A quoting market keeps quoting until the tax exceeds x_hi;
       a frozen market reopens only when the tax falls below x_lo < x_hi:
       dg = (s((x_lo + (x_hi - x_lo) g - tax)/W_X) - g)/TAU_G.
  F (mechanism A, slow) = dealer funding tied up by inventory stranded in a freeze, priced at the rate:
       dF = (1 - g)(r/0.1)^n_F (1 - F)/tau_Fi - F exp(-b_F r/0.1 - a_F tax/0.05)/tau_Fo;  D* x (1 - m3 F).
  Frozen leak: v target = g_eff push + (1 - g) lam_f (A - P) (slide proportional to the gap).
  Everything else is market_y3 (see that module); with x_lo = x_hi, g0 = 1, m3 = 0, lam_f = 0 it is market_y3.
"""
import math

import numpy as np

FAMILY = "market_canon2"
SLOW_F = True
LIN_LEAK = True
TAU_G = 1.0
OBS = ["price", "volume", "depth"]
CTRL = ["interest_rate", "transaction_tax"]
MECHS = {"A": "settlement tie-up of funding (stranded inventory costs capacity at the rate)",
         "B": "risk-capacity loss after adverse moves",
         "C": "investor momentum toward recent winners"}
N_SUB = 2

STATE = ["B", "P", "v", "C", "D", "R", "H", "Q", "W", "G", "A", "S", "M", "Iw", "g", "F"]
STATE_LO = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0])
STATE_HI = np.array([1e4, 1e4, 20.0, 1.0, 1e4, 0.9, 200.0, 1e4, 1.0, 200.0, 1e4, 1.0, 20.0, 1.0, 1.0, 1.0])

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
    ("x_lo", 0.0386, 0.0343, 0.0458, False),
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
    ("tau_v", 7.9, 0.5, 60.0, True),
    ("m3", 0.3, 0.0, 0.95, False),
    ("n_F", 1.0, 0.0, 6.0, False),
    ("tau_Fi", 60.0, 5.0, 400.0, True),
    ("tau_Fo", 5.0, 0.5, 200.0, True),
    ("b_F", 3.0, 0.0, 10.0, False),
    ("a_F", 1.0, 0.0, 6.0, False),
    ("lam_f", 0.003, 0.0, 0.05, False),
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


def _gate_target(M, tax, g, th):
    xl = M.min(th["x_lo"], th["x_c"])
    return 1.0 / (1.0 + M.exp((tax - xl - (th["x_c"] - xl) * g) / W_X))


def _cash_drift(M, C, g, rate, th):
    rr = M.pw(rate / 0.1, th["n_r"])
    dC = th["rho"] * g * (C + th["c0"]) * (1.0 - C) - th["a_r"] * rr * (1.0 - C + th["b0"]) - th["c_f"] * (1.0 - g) * C
    return dC, rr


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rate, tax = uu
    B, P, v, C, D, R, H, Q, W, G, A, S, Mo, Iw, g, F = s
    onA, onB, onC = ("A" in mech), ("B" in mech), ("C" in mech)
    dB = -B / TAU_B
    dg = (_gate_target(M, tax, g, th) - g) / TAU_G
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
    tgt = geff * push
    if LIN_LEAK:
        tgt = tgt + (1.0 - g) * th["lam_f"] * (A - P)
    dv = (tgt - v) / th["tau_v"]
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
        if SLOW_F:
            Dstar = Dstar * (1.0 - th["m3"] * F)
    else:
        Dstar = th["d0"] * (1.0 - th["m1"] * sd) * (1.0 - th["m2"] * (1.0 - g))
    dD = (Dstar - D) / TAU_D
    if onA and SLOW_F:
        dF = ((1.0 - g) * M.pw(rate * 10.0, th["n_F"]) * (1.0 - F) / th["tau_Fi"]
              - F * M.exp(-th["b_F"] * rate * 10.0 - th["a_F"] * tax * 20.0) / th["tau_Fo"])
    else:
        dF = 0.0 * F
    return _pack(M, [dB, dP, dv, dC, dD, dR, dH, dQ, -W / TAU_W, dG, dA, dS, dM, dI, dg, dF])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    B, P, v, C, D, R, H, Q, W, G, A, S, Mo, Iw, g, F = s
    rate, tax = uu
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
    return _pack(M, [B, P, z, z + 1.0, D, z, z, P, z + 1.0, z, P, z, z, z + 0.5, z, z])
