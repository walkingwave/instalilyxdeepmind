"""Grey-box ODE: power_grid w4 = power_grid_w2 with one capacity for the whole reserve fleet and the
interconnector scaling kept on the request (as v9c).

Reff = smin((c_r (1-beta) Rs + draw) icf, p_l), icf = i0 + (1-i0) ic. The fleet delivery cannot
exceed p_l power units whatever the request; the interconnector scales what reaches the grid as in
v9c. w2 split the capacity into p_l + p_r ic, which only the pulse run (ic 0.2) can separate: with
that run held out the fit put all of it on the interconnector (p_l -> 0).
Other terms as power_grid_w2: remote delivery g_i (ic - 1), frequency-sensitive load kd fdev,
fitted share floor sh_i0.
"""
import math

import numpy as np

FAMILY = "power_grid_w4"
OBS = ["load", "frequency", "renewable_share"]
CTRL = ["price_signal", "reserve_dispatch", "charging_allowance", "interconnector"]
MECHS = {"A": "thermostat synchronisation (always on)", "B": "governor + reserve stock (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["Ds", "x1", "x2", "Rs", "fdev", "sh", "G", "E", "Th"]
_NS = len(STATE)
STATE_LO = np.array([0.0, -32.0, -60.0, 0.0, -2.5, 0.0, -80.0, 0.0, 0.0])
STATE_HI = np.array([260.0, 32.0, 60.0, 150.0, 2.5, 1.0, 80.0, 1.0, 260.0])

A_S = 1.0
A_R = 0.6
A_SH = 1.5
A_TH = 0.12
A_F = 1.0     # frequency lag, per tick
G_LIM = 40.0  # governor output limit, power units
K_C = 80.0    # reserve request that halves the renewable share
DSAT = 3.0    # saturation scale of the synchronisation kick
IC0 = 0.15
SH_IC0 = 0.5
SMIN_W = 3.0  # width of the smooth minimum, power units

PARAMS = [
    ("d0", 126.35363, 60.0, 250.0, False),
    ("d1", 20.89758, 0.0, 80.0, False),
    ("w", 0.09957, 0.03, 0.4, True),
    ("zeta", 0.29103, 0.02, 1.0, True),
    ("kick", 0.53744, 0.0, 1.0, False),
    ("kf", 0.04227, 0.005, 0.3, True),
    ("gb", -5.49506, -40.0, 40.0, False),
    ("a_g", 0.01826, 0.005, 1.0, True),
    ("kg", 10.63558, 0.0, 200.0, False),
    ("c_r", 1.35666, 0.0, 2.0, False),
    ("cap", 321.88525, 20.0, 20000.0, True),
    ("p_ch", 76.08126, 0.0, 150.0, False),
    ("s0", 0.35706, 0.05, 0.9, False),
    ("k_th", 0.3632, -2.0, 2.0, False),
    ("f_hi", 1.73317, 0.5, 3.5, False),
    ("beta", 0.5514, 0.0, 1.0, False),
    ("kd", 2.0, 0.0, 20.0, False),
    ("g_i", 10.0, 0.0, 80.0, False),
    ("i0", 0.15, 0.0, 1.0, False),
    ("sh_i0", 0.5, 0.0, 1.0, False),
    ("p_l", 60.0, 0.0, 250.0, False),
]


class _PY:
    max = max
    min = min

    @staticmethod
    def tanh(z):
        return math.tanh(z)


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def tanh(z):
        return np.tanh(z)


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


def _demand(th, price):
    return th["d0"] - th["d1"] * price


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    price, res, chg, ic = uu
    Ds, x1, x2, Rs, fdev, sh, G, E, Th = s
    D = _demand(th, price)
    gap = D - Ds
    load = Ds + x1 + th["kd"] * fdev
    w = th["w"]
    dDs = A_S * gap
    dx1 = x2
    dx2 = -2.0 * th["zeta"] * w * x2 - w * w * x1 + th["kick"] * A_S * DSAT * M.tanh(gap / DSAT)
    dRs = A_R * (res - Rs)
    Dref = _demand(th, 0.8)
    draw = th["c_r"] * th["beta"] * Rs * E
    icf = th["i0"] + (1.0 - th["i0"]) * ic
    Rreq = (th["c_r"] * (1.0 - th["beta"]) * Rs + draw) * icf
    Pcap = th["p_l"]
    dd = Rreq - Pcap
    Reff = 0.5 * (Rreq + Pcap - (dd * dd + SMIN_W * SMIN_W) ** 0.5)
    Pch = th["p_ch"] * chg * (1.0 - E)
    dE = (Pch - draw) / th["cap"]
    bal = Dref + th["gb"] + th["g_i"] * (ic - 1.0) + G + Reff - Pch * icf - load
    ft = th["kf"] * bal
    fh = th["f_hi"]
    ft = M.min(ft, fh * M.tanh(ft / fh))
    dfdev = A_F * (ft - fdev)
    gl = G_LIM
    gt = M.max(M.min(-th["kg"] * fdev, gl), -gl)
    dG = th["a_g"] * (gt - G)
    dTh = A_TH * (load - Th)
    q = res / K_C
    sh_t = th["s0"] * (th["sh_i0"] + (1.0 - th["sh_i0"]) * ic) / (1.0 + q * q) * (1.0 - th["k_th"] * (Th - Dref) / 100.0)
    dsh = A_SH * (sh_t - sh)
    return _pack(M, [dDs, dx1, dx2, dRs, dfdev, dsh, dG, dE, dTh])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    Ds, x1, x2, Rs, fdev, sh, G, E, Th = s
    return _pack(M, [Ds + x1 + th["kd"] * fdev, 50.0 + fdev, sh])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    Ds = M.max(y[0], 0.0)
    fdev = M.max(M.min(y[1] - 50.0, 2.5), -2.5)
    sh = M.max(M.min(y[2], 1.0), 0.0)
    z = 0.0 * Ds
    return _pack(M, [Ds, z, z, z, fdev, sh, z, z + 1.0, Ds])
