"""Grey-box ODE: power_grid pg9 = power_grid_w5 with the load replaced by a thermostatic load population.

Load. Two classes c of cooling loads (weights phi, 1 - phi; thermal time constants tau_c), each held as
off/on densities on a normalised temperature grid x in [X_LO, X_HI] (M cells). Off loads warm toward 1,
on loads cool toward 0:  dx/dt = (1 - x)/tau (off),  -x/tau (on). The price sets the band centre
s(p) = s08 + s1 (p - 0.8) + s2 (p - 0.8)^2, width w; loads switch on above s + w/2 and off below
s - w/2 at a smooth rate r (logistic in x, width dx/2). Upwind transport (its numerical diffusion plus
the second class disperses synchronisation). Load = base + P * (fraction on) + z, z a decaying reset
offset (z0 = zk (y0 - load_ref), dz/dt = -az z). Every reset starts from the stationary population at
price 0.8 (brief: same asynchronous population, no random phases), relaxed in x0.
A price step up empties the on state (floor = base), the population warms together and switches back
on nearly at once (rebound); a step down fills it (ceiling = base + P).

Frequency, governor, reserve, storage, interconnector and share: exactly power_grid_w5, with the load
from the population and Dref = base + P (1 - s08) (the stationary load at 0.8 for a thin band).
"""
import math

import numpy as np

FAMILY = "power_grid_pg9"
OBS = ["load", "frequency", "renewable_share"]
CTRL = ["price_signal", "reserve_dispatch", "charging_allowance", "interconnector"]
MECHS = {"A": "thermostat synchronisation (always on)", "B": "governor + reserve stock (always on)",
         "C": "unused"}
N_SUB = 1

M = 32
K = 2
X_LO, X_HI = 0.3, 0.95
DX = (X_HI - X_LO) / M
XC = X_LO + DX * (np.arange(M) + 0.5)
XE = X_LO + DX * np.arange(1, M)
NP_ = K * M
REST = ["z", "Rs", "fdev", "sh", "G", "E", "Th"]
STATE = [f"off{c}_{i}" for c in range(K) for i in range(M)] + [f"on{c}_{i}" for c in range(K) for i in range(M)] + REST
_NS = len(STATE)
STATE_LO = np.concatenate([np.zeros(2 * NP_), np.array([-80.0, 0.0, -2.5, 0.0, -80.0, 0.0, 0.0])])
STATE_HI = np.concatenate([np.ones(2 * NP_), np.array([80.0, 150.0, 2.5, 1.0, 80.0, 1.0, 260.0])])
IZ = 2 * NP_

A_R = 0.6
A_SH = 1.5
A_TH = 0.12
A_F = 1.0
G_LIM = 40.0
K_C = 80.0
SMIN_W = 3.0
RELAX = 300       # ticks of relaxation to the stationary population in x0

PARAMS = [
    ("base", 50.9, 20.0, 100.0, False),
    ("P", 137.6, 40.0, 250.0, False),
    ("s08", 0.58, 0.35, 0.75, False),
    ("s1", 0.1275, 0.0, 0.5, False),
    ("s2", 0.008, -0.3, 0.3, False),
    ("w", 0.236, 0.08, 0.3, False),
    ("tau1", 53.2, 25.0, 400.0, True),
    ("tau2", 31.2, 25.0, 300.0, True),
    ("phi", 0.34, 0.0, 1.0, False),
    ("r", 0.20, 0.05, 1.2, True),
    ("az", 0.42, 0.02, 3.0, True),
    ("zk", 0.106, 0.0, 1.5, False),
    ("kf", 0.0384, 0.005, 0.3, True),
    ("gb", -9.21, -40.0, 40.0, False),
    ("a_g", 0.0554, 0.005, 1.0, True),
    ("kg", 4.59, 0.0, 200.0, False),
    ("c_r", 1.09, 0.0, 2.0, False),
    ("cap", 304.0, 20.0, 20000.0, True),
    ("p_ch", 86.1, 0.0, 150.0, False),
    ("s0", 0.365, 0.05, 0.9, False),
    ("k_th", 0.25, -2.0, 2.0, False),
    ("f_hi", 1.89, 0.5, 3.5, False),
    ("beta", 0.503, 0.0, 1.0, False),
    ("g_i", 13.7, 0.0, 80.0, False),
    ("i0", 0.01, 0.0, 1.0, False),
    ("sh_i0", 0.432, 0.0, 1.0, False),
    ("p_l", 21.9, 0.0, 250.0, False),
    ("p_r", 82.2, 0.0, 250.0, False),
]


def _p(th, n):
    return np.asarray(th[n], dtype=float)


def _col(v):
    return np.asarray(v, dtype=float)[..., None, None]


def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _band(price, th):
    dp = price - 0.8
    return _p(th, "s08") + _p(th, "s1") * dp + _p(th, "s2") * dp * dp


def _pop_deriv(off, on, S, th):
    """off, on [..., K, M]; S [...]. Returns d_off, d_on."""
    tau = np.stack(np.broadcast_arrays(_p(th, "tau1"), _p(th, "tau2")), axis=-1)[..., None]   # [..., K, 1]
    w = _col(_p(th, "w"))
    Sc = _col(S)
    r = _col(_p(th, "r"))
    eps = 0.5 * DX
    up = _sig((XC - (Sc + 0.5 * w)) / eps)
    lo = _sig(((Sc - 0.5 * w) - XC) / eps)
    Fo = (1.0 - XE) / tau * off[..., :-1] / DX
    Fn = XE / tau * on[..., 1:] / DX
    zo = np.zeros(Fo.shape[:-1] + (1,))
    d_off = np.concatenate([zo, Fo], axis=-1) - np.concatenate([Fo, zo], axis=-1)
    d_on = np.concatenate([Fn, zo], axis=-1) - np.concatenate([zo, Fn], axis=-1)
    sw = r * (up * off - lo * on)
    return d_off - sw, d_on + sw


def _split(x):
    off = x[..., :NP_].reshape(x.shape[:-1] + (K, M))
    on = x[..., NP_:2 * NP_].reshape(x.shape[:-1] + (K, M))
    return off, on


def _load(x, th):
    on = x[..., NP_:2 * NP_]
    return _p(th, "base") + _p(th, "P") * on.sum(axis=-1) + x[..., IZ]


def _dref(th):
    return _p(th, "base") + _p(th, "P") * (1.0 - _p(th, "s08"))


_POP_KEYS = ("tau1", "tau2", "w", "r", "s08", "s1", "s2")
_REST_KEYS = ("base", "P", "s08", "az", "c_r", "beta", "i0", "p_l", "p_r", "p_ch", "cap", "gb", "g_i", "kf",
              "f_hi", "kg", "a_g", "s0", "sh_i0", "k_th")
_A_CACHE = {}
_TH_CACHE = [None, None, None]


def _pop_matrix(price, pkey, th):
    """[A(price, theta); ones on the on-cells] as one (2NP + 1, 2NP) matrix: A @ pop and the on fraction."""
    key = (price,) + pkey
    A = _A_CACHE.get(key)
    if A is None:
        if len(_A_CACHE) > 4096:
            _A_CACHE.clear()
        E = np.eye(2 * NP_)
        off, on = _split(E)
        d_off, d_on = _pop_deriv(off, on, _band(np.full(2 * NP_, price), th), th)
        A = np.concatenate([d_off.reshape(2 * NP_, NP_), d_on.reshape(2 * NP_, NP_)], axis=-1).T
        ones = np.concatenate([np.zeros(NP_), np.ones(NP_)])[None]
        A = np.ascontiguousarray(np.concatenate([A, ones], axis=0))
        _A_CACHE[key] = A
    return A


def _th_consts(th):
    if _TH_CACHE[0] is not th:
        _TH_CACHE[0] = th
        _TH_CACHE[1] = tuple(float(th[n]) for n in _POP_KEYS)
        _TH_CACHE[2] = tuple(float(th[n]) for n in _REST_KEYS)
    return _TH_CACHE[1], _TH_CACHE[2]


def _f_fast(x, u, th):
    price, res, chg, ic = u.tolist() if hasattr(u, "tolist") else [float(v) for v in u]
    pkey, c = _th_consts(th)
    (base, P, s08, az, c_r, beta, i0, p_l, p_r, p_ch, cap, gb, g_i, kf, f_hi, kg, a_g, s0, sh_i0, k_th) = c
    A = _pop_matrix(price, pkey, th)
    dpop = A @ x[:2 * NP_]
    z, Rs, fdev, sh, G, E, Th = x[IZ:].tolist()
    load = base + P * float(dpop[-1]) + z
    dz = -az * z
    dRs = A_R * (res - Rs)
    Dref = base + P * (1.0 - s08)
    draw = c_r * beta * Rs * E
    icf = i0 + (1.0 - i0) * ic
    Rreq = c_r * (1.0 - beta) * Rs + draw
    Pcap = p_l + p_r * ic
    dd = Rreq - Pcap
    Reff = 0.5 * (Rreq + Pcap - math.sqrt(dd * dd + SMIN_W * SMIN_W))
    Pch = p_ch * chg * (1.0 - E)
    dE = (Pch - draw) / cap
    bal = Dref + gb + g_i * (ic - 1.0) + G + Reff - Pch * icf - load
    ft = kf * bal
    ft = min(ft, f_hi * math.tanh(ft / f_hi))
    dfdev = A_F * (ft - fdev)
    gt = max(min(-kg * fdev, G_LIM), -G_LIM)
    dG = a_g * (gt - G)
    dTh = A_TH * (load - Th)
    q = res / K_C
    sh_t = s0 * (sh_i0 + (1.0 - sh_i0) * ic) / (1.0 + q * q) * (1.0 - k_th * (Th - Dref) / 100.0)
    dsh = A_SH * (sh_t - sh)
    out = np.empty(_NS)
    out[:IZ] = dpop[:IZ]
    out[IZ:] = (dz, dRs, dfdev, dsh, dG, dE, dTh)
    return out


def f(x, u, th, mech):
    if np.ndim(x) == 1 and np.ndim(th["tau1"]) == 0:
        return _f_fast(x, u, th)
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    price, res, chg, ic = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    off, on = _split(x)
    S = _band(price, th)
    d_off, d_on = _pop_deriv(off, on, S, th)
    z, Rs, fdev, sh, G, E, Th = (x[..., IZ + i] for i in range(7))
    g = lambda n: _p(th, n)
    load = _load(x, th)
    dz = -g("az") * z
    dRs = A_R * (res - Rs)
    Dref = _dref(th)
    draw = g("c_r") * g("beta") * Rs * E
    icf = g("i0") + (1.0 - g("i0")) * ic
    Rreq = g("c_r") * (1.0 - g("beta")) * Rs + draw
    Pcap = g("p_l") + g("p_r") * ic
    dd = Rreq - Pcap
    Reff = 0.5 * (Rreq + Pcap - np.sqrt(dd * dd + SMIN_W * SMIN_W))
    Pch = g("p_ch") * chg * (1.0 - E)
    dE = (Pch - draw) / g("cap")
    bal = Dref + g("gb") + g("g_i") * (ic - 1.0) + G + Reff - Pch * icf - load
    ft = g("kf") * bal
    fh = g("f_hi")
    ft = np.minimum(ft, fh * np.tanh(ft / fh))
    dfdev = A_F * (ft - fdev)
    gt = np.clip(-g("kg") * fdev, -G_LIM, G_LIM)
    dG = g("a_g") * (gt - G)
    dTh = A_TH * (load - Th)
    q = res / K_C
    sh_t = g("s0") * (g("sh_i0") + (1.0 - g("sh_i0")) * ic) / (1.0 + q * q) * (1.0 - g("k_th") * (Th - Dref) / 100.0)
    dsh = A_SH * (sh_t - sh)
    rest = np.stack(np.broadcast_arrays(dz, dRs, dfdev, dsh, dG, dE, dTh), axis=-1)
    lead = rest.shape[:-1]
    return np.concatenate([d_off.reshape(lead + (NP_,)), d_on.reshape(lead + (NP_,)), rest], axis=-1)


def h(x, u, th, mech):
    x = np.asarray(x, dtype=float)
    load = _load(x, th)
    return np.stack(np.broadcast_arrays(load, 50.0 + x[..., IZ + 2], x[..., IZ + 3]), axis=-1)


def _stationary(th, lead):
    S = np.broadcast_to(_band(np.asarray(0.8), th), lead)
    w = _col(_p(th, "w"))
    Sc = _col(S)
    tau = np.stack(np.broadcast_arrays(_p(th, "tau1"), _p(th, "tau2")), axis=-1)[..., None]
    band = ((XC > Sc - 0.5 * w) & (XC < Sc + 0.5 * w)).astype(float)
    o = band * tau / (1.0 - XC)
    n = band * tau / XC
    tot = o.sum(-1, keepdims=True) + n.sum(-1, keepdims=True) + 1e-12
    phi = _p(th, "phi")
    wts = np.stack(np.broadcast_arrays(phi, 1.0 - phi), axis=-1)[..., None]
    off = np.broadcast_to(o / tot * wts, lead + (K, M)).copy()
    on = np.broadcast_to(n / tot * wts, lead + (K, M)).copy()
    dt = 0.5
    for _ in range(2 * RELAX):
        a, b = _pop_deriv(off, on, S, th)
        off = np.maximum(off + dt * a, 0.0)
        on = np.maximum(on + dt * b, 0.0)
    return off, on


def x0(y0, th, mech):
    y0 = np.asarray(y0, dtype=float)
    lead = y0.shape[:-1]
    off, on = _stationary(th, lead)
    lref = _p(th, "base") + _p(th, "P") * on.sum(axis=(-2, -1))
    z = _p(th, "zk") * (y0[..., 0] - lref)
    fdev = np.clip(y0[..., 1] - 50.0, -2.5, 2.5)
    sh = np.clip(y0[..., 2], 0.0, 1.0)
    zero = 0.0 * fdev
    Th = lref + z
    rest = np.stack(np.broadcast_arrays(z, zero, fdev, sh, zero, zero + 1.0, Th), axis=-1)
    return np.concatenate([off.reshape(lead + (NP_,)), on.reshape(lead + (NP_,)), rest], axis=-1)
