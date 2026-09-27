"""Grey-box ODE: power_grid v8. Thermostat rebound + inertia/governor frequency + reserve stock
+ interconnector temperature on the renewable connection. NUMPY + math only.

Load (as power_grid_min): D(p) = d0 - d1 p; the fast component Ds relaxes to D at A_S; the gap
kicks a damped oscillator (x1, x2) through a saturating tanh; load = Ds + x1.

Supply / frequency: reserve request Rs follows reserve_dispatch at A_R. The reserve fleet holds a
storage part (share beta of the request) with stock E in [0, 1]; it runs at draw = beta c_r Rs E,
the sustained part at (1 - beta) c_r Rs; the interconnector passes
Reff = ((1 - beta) c_r Rs + draw) (IC0 + (1-IC0) ic). Refilling takes grid power Pch = p_ch chg (1 - E) (charging
allowance on the shared connection), dE = (Pch - draw) / cap. Governor output G (deviation from
the base G0 = D(0.8) + gb) follows -kg fdev at rate a_g, target clipped to +-G_LIM. The frequency
deviation relaxes at A_F toward kf (G0 + G + Reff - Pch - load), soft-capped above at f_hi
(surplus beyond the cap is curtailed).

Renewable share: target s0 (0.5 + 0.5 ic) / (1 + (r/K_C)^2) (D(0.8) / Th)^k_th with r the reserve_dispatch request
itself (the share moves within the first tick of a reserve step) Th = interconnector temperature, a filtered load (rate A_TH): heavy recent flows curtail the
renewable connection. sh relaxes at A_SH.

x0 uses every y0 entry: Ds = Th = load0, fdev = f0 - 50, sh = share0; oscillator, Rs, G at rest,
E full. Every term is always active (fit with --mech AB).
"""
import math

import numpy as np

FAMILY = "power_grid_v8h"
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
A_TH = 0.2
A_F = 1.0     # frequency lag, per tick
G_LIM = 40.0  # governor output limit, power units
K_C = 80.0    # reserve request that halves the renewable share
DSAT = 3.0    # saturation scale of the synchronisation kick
IC0 = 0.15
SH_IC0 = 0.5

PARAMS = [
    ("d0", 126.0, 60.0, 250.0, False),
    ("d1", 21.2, 0.0, 80.0, False),
    ("w", 0.10, 0.03, 0.4, True),
    ("zeta", 0.28, 0.02, 1.0, True),
    ("kick", 0.46, 0.0, 1.0, False),
    ("kf", 0.04, 0.005, 0.3, True),
    ("gb", -8.0, -40.0, 40.0, False),
    ("a_g", 0.05, 0.005, 1.0, True),
    ("kg", 20.0, 0.0, 200.0, False),
    ("c_r", 0.6, 0.0, 2.0, False),
    ("cap", 300.0, 20.0, 20000.0, True),
    ("p_ch", 20.0, 0.0, 150.0, False),
    ("s0", 0.374, 0.05, 0.9, False),
    ("k_th", 0.3, -2.0, 2.0, False),
    ("f_hi", 1.9, 0.5, 2.5, False),
    ("beta", 0.3, 0.0, 1.0, False),
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
    load = Ds + x1
    w = th["w"]
    dDs = A_S * gap
    dx1 = x2
    dx2 = -2.0 * th["zeta"] * w * x2 - w * w * x1 + th["kick"] * A_S * DSAT * M.tanh(gap / DSAT)
    dRs = A_R * (res - Rs)
    Dref = _demand(th, 0.8)
    draw = th["c_r"] * th["beta"] * Rs * E
    Reff = (th["c_r"] * (1.0 - th["beta"]) * Rs + draw) * (IC0 + (1.0 - IC0) * ic)
    Pch = th["p_ch"] * chg * (1.0 - E)
    dE = (Pch - draw) / th["cap"]
    bal = Dref + th["gb"] + G + Reff - Pch - load
    ft = th["kf"] * bal
    fh = th["f_hi"]
    ft = M.min(ft, fh * M.tanh(ft / fh))
    dfdev = A_F * (ft - fdev)
    gl = G_LIM
    gt = M.max(M.min(-th["kg"] * fdev, gl), -gl)
    dG = th["a_g"] * (gt - G)
    dTh = A_TH * (load - Th)
    q = res / K_C
    sh_t = th["s0"] * (SH_IC0 + (1.0 - SH_IC0) * ic) / (1.0 + q * q) * (Dref / M.max(Th, 20.0)) ** th["k_th"]
    dsh = A_SH * (sh_t - sh)
    return _pack(M, [dDs, dx1, dx2, dRs, dfdev, dsh, dG, dE, dTh])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    Ds, x1, x2, Rs, fdev, sh, G, E, Th = s
    return _pack(M, [Ds + x1, 50.0 + fdev, sh])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    Ds = M.max(y[0], 0.0)
    fdev = M.max(M.min(y[1] - 50.0, 2.5), -2.5)
    sh = M.max(M.min(y[2], 1.0), 0.0)
    z = 0.0 * Ds
    return _pack(M, [Ds, z, z, z, fdev, sh, z, z + 1.0, Ds])
