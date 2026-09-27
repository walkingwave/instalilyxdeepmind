"""Grey-box ODE: power_grid v8. Thermostat rebound + inertia/governor frequency + reserve stock
+ interconnector temperature on the renewable connection. NUMPY + math only.

Load (as power_grid_min): D(p) = d0 - d1 p; the fast component Ds relaxes to D at A_S; the gap
kicks a damped oscillator (x1, x2) through a saturating tanh; load = Ds + x1.

Supply / frequency: reserve request Rs follows reserve_dispatch at A_R. The reserve fleet holds a
stock E in [0, 1] (share of energy left): delivered reserve Reff = c_r Rs (IC0 + (1-IC0) ic) E,
dE = -Reff / cap + k_ch chg (1 - E) (charging_allowance refills through the shared connection).
Governor output G (deviation from the base G0 = D(0.8) + gb) follows -kg fdev at rate a_g,
clipped to [-g_lim, g_lim] via the target. The frequency deviation relaxes at a_f toward
kf (G0 + G + Reff - load).

Renewable share: target s0 (0.5 + 0.5 ic) / (1 + (Rs/k_c)^2) (1 - k_th (Th - D(0.8)) / 100),
Th = interconnector temperature, a filtered load (rate A_TH): heavy recent flows curtail the
renewable connection. sh relaxes at A_SH.

x0 uses every y0 entry: Ds = D(0.8) (every reset is the same population at price 0.8), the
offset o = load0 - D(0.8) decays at a_o (it adds to load), Th = load0, fdev = f0 - 50, sh = share0; oscillator, Rs, G at rest,
E full. Every term is always active (fit with --mech AB).
"""
import math

import numpy as np

FAMILY = "power_grid_v8b"
OBS = ["load", "frequency", "renewable_share"]
CTRL = ["price_signal", "reserve_dispatch", "charging_allowance", "interconnector"]
MECHS = {"A": "thermostat synchronisation (always on)", "B": "governor + reserve stock (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["Ds", "x1", "x2", "Rs", "fdev", "sh", "G", "E", "Th", "o"]
_NS = len(STATE)
STATE_LO = np.array([0.0, -32.0, -60.0, 0.0, -2.5, 0.0, -80.0, 0.0, 0.0, -80.0])
STATE_HI = np.array([260.0, 32.0, 60.0, 150.0, 2.5, 1.0, 80.0, 1.0, 260.0, 80.0])

A_S = 1.0
A_R = 0.6
A_SH = 1.5
A_TH = 0.12
IC0 = 0.15
SH_IC0 = 0.5

PARAMS = [
    ("d0", 126.0, 60.0, 250.0, False),
    ("d1", 21.2, 0.0, 80.0, False),
    ("w", 0.10, 0.03, 0.4, True),
    ("zeta", 0.28, 0.02, 1.0, True),
    ("kick", 0.46, 0.0, 1.0, False),
    ("Dsat", 4.4, 0.5, 100.0, True),
    ("kf", 0.04, 0.005, 0.3, True),
    ("gb", -8.0, -40.0, 40.0, False),
    ("a_f", 1.0, 0.2, 4.0, True),
    ("a_g", 0.05, 0.005, 1.0, True),
    ("kg", 20.0, 0.0, 200.0, False),
    ("g_lim", 30.0, 2.0, 80.0, True),
    ("c_r", 0.6, 0.0, 2.0, False),
    ("cap", 4000.0, 200.0, 50000.0, True),
    ("k_ch", 0.05, 0.001, 1.0, True),
    ("s0", 0.374, 0.05, 0.9, False),
    ("k_c", 72.0, 10.0, 500.0, True),
    ("k_th", 0.3, -2.0, 2.0, False),
    ("a_o", 0.25, 0.02, 3.0, True),
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
    Ds, x1, x2, Rs, fdev, sh, G, E, Th, o = s
    D = _demand(th, price)
    gap = D - Ds
    load = Ds + x1 + o
    w = th["w"]
    dDs = A_S * gap
    dx1 = x2
    dx2 = -2.0 * th["zeta"] * w * x2 - w * w * x1 + th["kick"] * A_S * th["Dsat"] * M.tanh(gap / th["Dsat"])
    dRs = A_R * (res - Rs)
    Dref = _demand(th, 0.8)
    Reff = th["c_r"] * Rs * (IC0 + (1.0 - IC0) * ic) * E
    dE = -Reff / th["cap"] + th["k_ch"] * chg * (1.0 - E)
    bal = Dref + th["gb"] + G + Reff - load
    dfdev = th["a_f"] * (th["kf"] * bal - fdev)
    gl = th["g_lim"]
    gt = M.max(M.min(-th["kg"] * fdev, gl), -gl)
    dG = th["a_g"] * (gt - G)
    dTh = A_TH * (load - Th)
    q = Rs / th["k_c"]
    sh_t = th["s0"] * (SH_IC0 + (1.0 - SH_IC0) * ic) / (1.0 + q * q) * (1.0 - th["k_th"] * (Th - Dref) / 100.0)
    dsh = A_SH * (sh_t - sh)
    return _pack(M, [dDs, dx1, dx2, dRs, dfdev, dsh, dG, dE, dTh, -th["a_o"] * o])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    Ds, x1, x2, Rs, fdev, sh, G, E, Th, o = s
    return _pack(M, [Ds + x1 + o, 50.0 + fdev, sh])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    L0 = M.max(y[0], 0.0)
    Ds = L0 * 0.0 + _demand(th, 0.8)
    o = M.max(M.min(L0 - Ds, 80.0), -80.0)
    fdev = M.max(M.min(y[1] - 50.0, 2.5), -2.5)
    sh = M.max(M.min(y[2], 1.0), 0.0)
    z = 0.0 * Ds
    return _pack(M, [Ds, z, z, z, fdev, sh, z, z + 1.0, L0, o])
