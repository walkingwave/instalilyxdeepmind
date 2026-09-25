"""Grey-box ODE: traffic, minimal two-route pipeline. NUMPY + math only (ships verbatim).

Admitted demand a = dem0 * ramp_metering, split equally between the two routes (toll changes the
vehicle mix, not the total: a toll term fitted to zero), and rejected by a finite approach buffer:
a_i * max(0, 1 - n_i / n_max). Each route is a chain of three first-order stages with rate r_pipe
(mean transit 3 / r_pipe ticks: the ~12-tick lag between admission and exit seen in every run).
Stage 1 -> 2 is the shared junction: route i gets service junc * share_i with
share_a = 0.5 + w_sig * (signal_timing - 0.5), so one route starves while the other flows.
Exit flow = capacity-limited drain of stage 3, capacity = cap * (1 - lc_i * lane_closure)
* (1 + clr * clearance_effort) with a different lane sensitivity per route. Every limit is the
smooth minimum smin(x, c) = x / (1 + (x/c)^4)^(1/4). Reported speed relaxes (rate rho) toward
v_free / (1 + n_i / n_ref) where n_i is the vehicle count on route i.

Reset convention: roads start empty (flows are 0 at t = 0 in every run), so the pipeline is
empty regardless of the initial flow values; speeds start at the observed initial speeds.

Mechanism letters are accepted for interface compatibility but every term is always active;
fit with pairs=("AB",).
"""
import math

import numpy as np

FAMILY = "traffic_min"
OBS = ["flow_a", "flow_b", "speed_a", "speed_b"]
CTRL = ["signal_timing", "lane_closure", "toll", "ramp_metering", "freight_priority",
        "clearance_effort"]
MECHS = {"A": "capacity-limited junction and exits (always on)",
         "B": "occupancy-dependent speed, finite approach buffer (always on)", "C": "unused"}
N_SUB = 2

STATE = ["p1a", "p2a", "p3a", "p1b", "p2b", "p3b", "sa", "sb"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 200.0, 200.0])

PARAMS = [
    ("dem0", 40.0, 5.0, 300.0, True),
    ("w_sig", 1.0, 0.0, 1.5, False),
    ("r_pipe", 0.25, 0.05, 1.5, True),
    ("junc", 40.0, 3.0, 600.0, True),
    ("cap", 15.0, 3.0, 300.0, True),
    ("lc_a", 0.3, 0.0, 1.0, False),
    ("lc_b", 0.7, 0.0, 1.0, False),
    ("clr", 0.8, 0.0, 3.0, False),
    ("v_free", 48.9, 30.0, 70.0, False),
    ("n_ref", 200.0, 10.0, 5000.0, True),
    ("rho", 0.25, 0.02, 1.5, True),
    ("n_max", 800.0, 50.0, 20000.0, True),
]


class _PY:
    max = max
    min = min

    @staticmethod
    def exp(z):
        return math.exp(z if z < 50.0 else 50.0)

    @staticmethod
    def pow(a, b):
        return math.pow(a, b) if a > 0.0 else 0.0


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def pow(a, b):
        return np.power(np.maximum(a, 0.0), b)


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


def _smin(M, x, c):
    q = x / c
    q2 = q * q
    return x / M.pow(1.0 + q2 * q2, 0.25)


def _exits(M, s, uu, th):
    sig, lane, toll, ramp, frt, clr = uu
    p1a, p2a, p3a, p1b, p2b, p3b, sa, sb = s
    r = th["r_pipe"]
    boost = 1.0 + th["clr"] * clr
    cap_a = th["cap"] * (1.0 - th["lc_a"] * lane) * boost + 1e-6
    cap_b = th["cap"] * (1.0 - th["lc_b"] * lane) * boost + 1e-6
    return _smin(M, r * p3a, cap_a), _smin(M, r * p3b, cap_b)


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    sig, lane, toll, ramp, frt, clr = uu
    p1a, p2a, p3a, p1b, p2b, p3b, sa, sb = s
    r = th["r_pipe"]
    n_a = p1a + p2a + p3a
    n_b = p1b + p2b + p3b
    a_tot = th["dem0"] * ramp
    a_a = 0.5 * a_tot * M.max(1.0 - n_a / th["n_max"], 0.0)
    a_b = 0.5 * a_tot * M.max(1.0 - n_b / th["n_max"], 0.0)
    share = M.min(M.max(0.5 + th["w_sig"] * (sig - 0.5), 0.02), 0.98)
    j_a = _smin(M, r * p1a, th["junc"] * share + 1e-6)
    j_b = _smin(M, r * p1b, th["junc"] * (1.0 - share) + 1e-6)
    out_a, out_b = _exits(M, s, uu, th)
    dp1a = a_a - j_a
    dp2a = j_a - r * p2a
    dp3a = r * p2a - out_a
    dp1b = a_b - j_b
    dp2b = j_b - r * p2b
    dp3b = r * p2b - out_b
    tgt_a = th["v_free"] / (1.0 + n_a / th["n_ref"])
    tgt_b = th["v_free"] / (1.0 + n_b / th["n_ref"])
    dsa = th["rho"] * (tgt_a - sa)
    dsb = th["rho"] * (tgt_b - sb)
    return _pack(M, [dp1a, dp2a, dp3a, dp1b, dp2b, dp3b, dsa, dsb])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    out_a, out_b = _exits(M, s, uu, th)
    return _pack(M, [out_a, out_b, s[6], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    sa = M.min(M.max(y[2], 1.0), 80.0)
    sb = M.min(M.max(y[3], 1.0), 80.0)
    return _pack(M, [z, z, z, z, z, z, sa, sb])
