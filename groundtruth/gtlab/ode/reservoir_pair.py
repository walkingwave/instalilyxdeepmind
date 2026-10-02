"""Grey-box ODE: reservoir pair = str9 with the three delayed-return paths of the brief behind
mechanism letters, plus an optional screen-fouling store.

Brief: "Groundwater, irrigated land and deposited material may return water or contaminants after a
delay" (exactly two of three apply). Letters:
  A groundwater return:  G = g_gw softplus_20(L_gw - L), observed in inflow and added to the balance
  B irrigated-land return: dR/dt = (k_ret irrigation - R) / tau_ret, observed in inflow
  C deposited material:  dN/dt = (1 - w_L + w_L L / L_full - N) / tau_n, quality target -= q_n N
Base (always on): head-limited discharge, bank storage, inflow-proportional loss, stratification
deficit D, seasonal inflow, surcharge store P.
Screen fouling (base per the brief, off when g_f = 0): attached material F in [0, 1),
    dF/dt = g_f (1 - aeration)(1 - F) - F (r_fl release / 12 + r_a aeration),  F(0) = 0,
    outlet capacity c_out (L / 500)^p_out (1 - F).
With letters AC, g_f = 0 and d0 = 0 this is the str9 'c' model.

Original str9 notes follow.
reservoir str9 = v8 + level-driven groundwater inflow G = g_gw softplus(L_gw - L)
(observed in inflow, enters the balance) + slow quality stock N (reference profile at reset, N0 = n0):
    dN/dt = (1 - w_L + w_L L / L_full - N) / tau_n,  Q target -= q_n N + q_aer aeration.
Reset: N0 = n0, D0 = d0 (the reference profile). With n0 = q_aer = d0 = 0 (defaults when the
theta vector is shorter than PARAMS) this is the str9 'b' model.

Original v8 notes follow. NUMPY + math only (ships verbatim).

reservoir_min plus three structural terms, each forced by a residual of the calibrated-scale fit:
  * bank storage B: water that goes into the banks while the stage rises and comes back while it
    falls. Loss = seep * L + f_in * inflow + k_b * (L - B), dB/dt = (L - B) / tau_b. It explains why
    the fill slows at high stage (loss ~2.4/tick at L ~ 880 while filling) although the loss at full
    pool is only ~1.2/tick and decays after the pool fills (1.9 -> 1.0 over ~40 ticks in both runs).
  * inflow-proportional loss f_in: at full pool the outflow tracks 0.92 x inflow - 0.28 in both runs
    (gain < 1 at the season frequency; not explained by a spill lag, best lag 0.5 tick).
  * slow quality state D: under the pulse (no aeration, deep withdrawal) quality falls from 0.955 to
    0.933 over ~150 ticks and recovers only partly. The two controls moved together in our data, so
    the drive is split equally between them (the min-norm choice).
The spill passes through a short surcharge store P (tau_s = 0.5 tick, fixed).

States: level L, quality Q, return flow R, clock c, surcharge P, deficit D, bank B.
    net   = inflow(c) + R - delivered(L, u) - seep * L - f_in * inflow(c) - k_b * (L - B)
    dL/dt = net / A(L),  A(L) = (L / 500)^beta (beta fixed 0 unless listed in PARAMS);  L <= L_FULL
    dP/dt = gate(L) * softplus(net) - P / tau_s
    dB/dt = (L - B) / tau_b
    dQ/dt = (q_base - q_d * D - Q) / tau_q
    dD/dt = (0.5 * (1 - aeration) + 0.5 * withdrawal_depth - D) / tau_d
    dR/dt = (k_ret * irrigation - R) / tau_ret
    inflow(c) = q_m + q_a * sin(2 pi c / 67.75 + phi)   (period read off both runs; phi fixed 0)
    delivered = softmin(release + irrigation, c_out * (L / 500)^p_out)
Observation after the tick: [L, inflow(c) + R, delivered + P / tau_s, Q].
Reset: L0, Q0 from the readings, B0 = L0 (banks in equilibrium), R0 = D0 = 0, c0 = 0,
P0 = tau_s * gate(L0) * softplus(net at the recovery release 2).
"""
import math

import numpy as np

FAMILY = "reservoir_pair"
OBS = ["level", "inflow", "outflow", "quality"]
CTRL = ["release_rate", "irrigation_allocation", "withdrawal_depth", "aeration"]
MECHS = {"A": "groundwater return G(L)", "B": "irrigated-land return R", "C": "deposited-material quality stock N"}
N_SUB = 2

STATE = ["L", "Q", "R", "c", "P", "D", "B", "N", "F"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([941.3, 1.0, 5.0, 1e9, 500.0, 1.0, 1000.0, 1.0, 0.95])

_LREF = 500.0
_L_FULL = 941.3
_TWO_PI = 2.0 * math.pi
_PER = 67.75
_K = {"phi": 0.0, "tau_s": 0.5, "beta": 0.0, "f_in": 0.0, "g_gw": 0.0, "L_gw": 500.0, "q_n": 0.0, "tau_n": 500.0, "w_L": 0.0, "n0": 0.0, "q_aer": 0.0, "d0": 0.0, "g_f": 0.0, "r_fl": 0.05, "r_a": 0.05}

PARAMS = [
    ("q_m", 11.28, 8.0, 15.0, False),
    ("q_a", 2.22, 0.5, 5.0, False),
    ("seep", 0.0013, 1e-4, 0.02, True),
    ("c_out", 13.8, 5.0, 30.0, True),
    ("p_out", 0.38, 0.05, 1.0, False),
    ("q_base", 0.952, 0.5, 1.0, False),
    ("tau_q", 3.5, 0.67, 50.0, True),
    ("k_ret", 0.035, 0.0, 0.3, False),
    ("tau_ret", 20.0, 5.0, 50.0, True),
    ("q_d", 0.02, 0.0, 0.1, False),
    ("tau_d", 115.0, 5.0, 400.0, True),
    ("k_b", 0.011, 0.0, 0.2, False),
    ("tau_b", 12.0, 3.0, 400.0, True),
    ("f_in", 0.05, 0.0, 0.3, False),
    ("g_gw", 0.002, 0.0, 0.02, False),
    ("L_gw", 500.0, 150.0, 941.0, False),
    ("q_n", 0.02, 0.0, 0.15, False),
    ("tau_n", 400.0, 30.0, 3000.0, True),
    ("w_L", 0.5, 0.0, 1.0, False),
    ("n0", 0.3, 0.0, 1.0, False),
    ("q_aer", 0.0, -0.03, 0.03, False),
    ("d0", 0.0, 0.0, 1.0, False),
    ("g_f", 0.0, 0.0, 0.05, False),
    ("r_fl", 0.05, 0.0, 0.5, False),
    ("r_a", 0.05, 0.0, 0.5, False),
]


def _p(th, n):
    return th[n] if n in th else _K[n]


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
    return th["q_m"] + th["q_a"] * M.sin(_TWO_PI * c / _PER + _p(th, "phi"))


def _delivered(M, L, uu, th, F):
    rel, irr, dep, aer = uu
    cap = th["c_out"] * M.pow(L / _LREF, th["p_out"]) * (1.0 - F)
    return _smin(M, rel + irr, cap, 0.5)


def _gate(M, L):
    return 1.0 / (1.0 + M.exp(-(L - _L_FULL + 2.0) / 0.5))


def _gw(M, L, th, mech):
    if "A" not in mech:
        return 0.0 * L
    return _p(th, "g_gw") * _pos(M, _p(th, "L_gw") - L, 20.0)


def _net(M, L, R, c, B, delivered, th, mech):
    q = _inflow(M, c, th)
    return q + R + _gw(M, L, th, mech) - delivered - th["seep"] * L - _p(th, "f_in") * q - th["k_b"] * (L - B)


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    L, Q, R, c, P, Dd, B, N, F = s
    rel, irr, dep, aer = uu
    net = _net(M, L, R, c, B, _delivered(M, L, uu, th, F), th, mech)
    dL = net / M.pow(M.max(L, 20.0) / _LREF, _p(th, "beta"))
    dP = _gate(M, L) * _pos(M, net, 0.5) - P / _p(th, "tau_s")
    dB = (L - B) / th["tau_b"]
    qn = _p(th, "q_n") if "C" in mech else 0.0
    dQ = (th["q_base"] - th["q_d"] * Dd - qn * N - _p(th, "q_aer") * aer - Q) / th["tau_q"]
    wl = _p(th, "w_L")
    dN = (1.0 - wl + wl * L / _L_FULL - N) / _p(th, "tau_n")
    dD = (0.5 * (1.0 - aer) + 0.5 * dep - Dd) / th["tau_d"]
    kr = th["k_ret"] if "B" in mech else 0.0
    dR = (kr * irr - R) / th["tau_ret"]
    dF = _p(th, "g_f") * (1.0 - aer) * (1.0 - F) - F * (_p(th, "r_fl") * rel / 12.0 + _p(th, "r_a") * aer)
    dc = 1.0 + 0.0 * c
    return _pack(M, [dL, dQ, dR, dc, dP, dD, dB, dN, dF])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    L, Q, R, c, P, Dd, B, N, F = s
    delivered = _delivered(M, L, uu, th, F)
    return _pack(M, [L, _inflow(M, c, th) + R + _gw(M, L, th, mech), delivered + P / _p(th, "tau_s"), Q])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    L = M.min(M.max(y[0], 0.0), _L_FULL)
    Q = M.min(M.max(y[3], 0.0), 1.0)
    c = 0.0 * (y[1] + y[2])
    net = _net(M, L, c, c, L, 2.0 + c, th, mech)
    P = _p(th, "tau_s") * _gate(M, L) * _pos(M, net, 0.5)
    return _pack(M, [L, Q, c, c, P, c + _p(th, "d0"), L, c + _p(th, "n0"), c])
