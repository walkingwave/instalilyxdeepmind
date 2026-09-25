"""Grey-box ODE: power_grid, minimal thermostat-rebound + droop grid. NUMPY + math only.

Load: desired demand D(p) = d0 - d1 * price. The fast component Ds relaxes to D at a fixed
rate A_S (1/tick). A price step synchronises a group of thermostats: while Ds is still moving,
the gap D - Ds kicks a damped oscillator (x1, x2) with natural period 2 pi / w; the kick
saturates (tanh, scale Dsat) so that the rebound amplitude is nearly independent of the step
size, as observed. load = Ds + x1. x1 is the synchronised group's excess over the dispersed
demand; it is clipped to +-32 power units (largest observed rebound 29.4): only a bounded share
of the thermostat population can be in step, so load stays within [D(2) - 32, D(0) + 32].

Supply / frequency: reserve request Rs follows reserve_dispatch with a fixed thermal lag A_R
(0.6/tick) and is delivered through the interconnector, Rs_eff = c_r * Rs * (0.15 + 0.85 * ic).
Conventional generation is a fixed droop base G0 = D(0.8) (reference-price load). Balance
bal = G0 + Rs_eff - load; the frequency deviation fdev relaxes at A_F (0.5/tick) toward
kf * (bal - k_asym * max(-bal, 0)): governors have output limits, so a deficit moves the
frequency more than an equal surplus. fdev is clipped to +-2.5 Hz (observed range 49.2-51.9)
so unseen sustained deficits cannot run the frequency away.

Renewable share: sh relaxes at A_SH (1.5/tick) toward s0 * (0.5 + 0.5 * ic) / (1 + (Rs/k_c)^2)
(reserve dispatch curtails the renewable connection). charging_allowance has no visible effect
in the data and is ignored.

x0 uses every y0 entry: Ds = load0, fdev = f0 - 50, sh = share0; oscillator at rest, Rs = 0.
Mechanism letters are accepted for interface compatibility; every term is always active
(fit with pairs=("AB",)).
"""
import math

import numpy as np

FAMILY = "power_grid_min"
OBS = ["load", "frequency", "renewable_share"]
CTRL = ["price_signal", "reserve_dispatch", "charging_allowance", "interconnector"]
MECHS = {"A": "thermostat synchronisation (always on)", "B": "droop governor (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["Ds", "x1", "x2", "Rs", "fdev", "sh"]
_NS = len(STATE)
STATE_LO = np.array([0.0, -32.0, -60.0, 0.0, -2.5, 0.0])
STATE_HI = np.array([260.0, 32.0, 60.0, 150.0, 2.5, 1.0])

A_S = 1.0     # fast load relaxation, per tick
A_R = 0.6     # reserve thermal lag, per tick
A_F = 0.5     # frequency measurement / inertia lag, per tick
A_SH = 1.5    # renewable share filter, per tick
IC0 = 0.15    # interconnector floor for reserve delivery
SH_IC0 = 0.5  # interconnector floor for the renewable connection

PARAMS = [
    ("d0", 126.0, 60.0, 250.0, False),
    ("d1", 21.5, 0.0, 80.0, False),
    ("w", 0.10, 0.03, 0.4, True),
    ("zeta", 0.28, 0.02, 1.0, True),
    ("kick", 0.49, 0.0, 1.0, False),
    ("Dsat", 3.7, 0.5, 100.0, True),
    ("kf", 0.0225, 0.005, 0.3, True),
    ("k_asym", 2.7, 0.0, 4.0, False),
    ("c_r", 0.59, 0.0, 2.0, False),
    ("s0", 0.374, 0.05, 0.9, False),
    ("k_c", 72.0, 10.0, 500.0, True),
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
    Ds, x1, x2, Rs, fdev, sh = s
    D = _demand(th, price)
    gap = D - Ds
    load = Ds + x1
    w = th["w"]
    dDs = A_S * gap
    dx1 = x2
    dx2 = -2.0 * th["zeta"] * w * x2 - w * w * x1 + th["kick"] * A_S * th["Dsat"] * M.tanh(gap / th["Dsat"])
    dRs = A_R * (res - Rs)
    G0 = _demand(th, 0.8)
    Rs_eff = th["c_r"] * Rs * (IC0 + (1.0 - IC0) * ic)
    bal = G0 + Rs_eff - load
    dfdev = A_F * (th["kf"] * (bal - th["k_asym"] * M.max(-bal, 0.0)) - fdev)
    q = Rs / th["k_c"]
    sh_t = th["s0"] * (SH_IC0 + (1.0 - SH_IC0) * ic) / (1.0 + q * q)
    dsh = A_SH * (sh_t - sh)
    return _pack(M, [dDs, dx1, dx2, dRs, dfdev, dsh])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    Ds, x1, x2, Rs, fdev, sh = s
    return _pack(M, [Ds + x1, 50.0 + fdev, sh])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    Ds = M.max(y[0], 0.0)
    fdev = M.max(M.min(y[1] - 50.0, 2.5), -2.5)
    sh = M.max(M.min(y[2], 1.0), 0.0)
    z = 0.0 * Ds
    return _pack(M, [Ds, z, z, z, fdev, sh])
