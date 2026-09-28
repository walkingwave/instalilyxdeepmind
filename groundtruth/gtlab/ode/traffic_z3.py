"""Grey-box ODE: traffic z3 = traffic z2 with a saturating (logistic) toll response of demand:
  a_tot = dem0 ramp / (1 + exp(k_toll (toll - t50)))  (dem0 = demand at a free road with no toll).

traffic z2 = traffic z plus three brief-faithful base terms (always on):
  toll-elastic demand  a_tot = dem0 ramp exp(-k_toll (toll - 2.5)),
  clearance crew taken from junction operation  J (1 - j_c clr_eff),
  approach drivers diverting  -d_div p1_i^2 / n_max  (waiting drivers leave a long approach queue).

traffic z = the v8d pipeline with the brief's three history mechanisms as switches.

Base (always on, identical to traffic_v8d): admitted demand dem0 * ramp split between the routes,
finite approach buffer n_max, three first-order stages per route (approach p1, junction p2, exit p3),
junction service shared by signal timing, capacity-limited exits (lane sensitivity per route,
clearance boost), speed from a lagged, delay-weighted occupancy m_i relaxed at rate rho.

Mechanisms (exactly two active; letters in `mech`):
  A route learning: the admitted split q (share choosing route a) learns toward the faster route,
    dq/dt = (q* - q) / tau_L,  q* = 1 / (1 + exp(-beta_L (s_a - s_b) / 10)),  q(0) = 0.5.
  B crew fatigue / switching cost: the clearance crew reaches its assignment with a lag
    dc/dt = (clr - c) / tau_sw (switching cost), and tires with sustained clearance work,
    dF/dt = (c - F) / tau_F; exit boost = 1 + clr_k c (1 - k_F F).
  C persistent spillback fronts: a finite exit buffer K3 throttles junction -> exit transfers,
    r p2 (1 - p3/K3)^+; a spillback front G_i follows exit-buffer fullness (p3_i/K3)^2, rising at
    r_up and dissolving only at r_dn (persistent); fronts block the shared junction for both routes,
    J / (1 + c_J (G_a + G_b)), and count as occupancy w_G n_ref G_i in the speed law.
"""
import math

import numpy as np

FAMILY = "traffic_z3"
OBS = ["flow_a", "flow_b", "speed_a", "speed_b"]
CTRL = ["signal_timing", "lane_closure", "toll", "ramp_metering", "freight_priority",
        "clearance_effort"]
MECHS = {"A": "route learning (admitted split learns toward the faster route)",
         "B": "crew fatigue / switching cost (lagged, tiring clearance crew)",
         "C": "persistent spillback fronts (finite exit buffer, slow-dissolving fronts block the junction)"}
N_SUB = 2

STATE = ["p1a", "p2a", "p3a", "p1b", "p2b", "p3b", "sa", "sb", "ma", "mb", "q", "ce", "fat", "ga", "gb"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 200.0, 200.0, 1e5, 1e5, 1.0, 1.0, 1.0, 50.0, 50.0])

BASE = ["dem0", "w_sig", "r_pipe", "junc", "cap", "lc_a", "lc_b", "clr", "v_free", "n_ref", "rho",
        "n_max", "gam", "w1", "r_m", "kap", "k_toll", "t50", "j_c", "d_div"]
MECH_PARAMS = {"A": ["beta_L", "tau_L"], "B": ["tau_sw", "k_F", "tau_F", "ce0"],
               "C": ["K3", "r_up", "r_dn", "c_J", "w_G"]}

PARAMS = [
    ("dem0", 70.0, 5.0, 300.0, True),
    ("w_sig", 1.0014, 0.0, 1.5, False),
    ("r_pipe", 0.28976, 0.05, 1.5, True),
    ("junc", 34.857, 3.0, 600.0, True),
    ("cap", 13.205, 3.0, 300.0, True),
    ("lc_a", 0.11712, 0.0, 1.0, False),
    ("lc_b", 0.98903, 0.0, 1.0, False),
    ("clr", 0.70799, 0.0, 3.0, False),
    ("v_free", 48.881, 30.0, 70.0, False),
    ("n_ref", 194.41, 10.0, 5000.0, True),
    ("rho", 0.29518, 0.02, 1.5, True),
    ("n_max", 715.24, 50.0, 20000.0, True),
    ("gam", 1.1661, 0.5, 4.0, False),
    ("w1", 0.53168, 0.0, 8.0, False),
    ("r_m", 0.40144, 0.03, 3.0, True),
    ("kap", 0.76829, 0.0, 2.0, False),
    ("k_toll", 0.5, 0.0, 4.0, False),
    ("t50", 4.0, -5.0, 15.0, False),
    ("j_c", 0.1, 0.0, 0.9, False),
    ("d_div", 0.1, 0.0, 20.0, False),
    # A route learning
    ("beta_L", 1.0, 0.0, 20.0, False),
    ("tau_L", 30.0, 1.0, 5000.0, True),
    # B crew fatigue / switching
    ("tau_sw", 3.0, 0.2, 300.0, True),
    ("k_F", 0.3, 0.0, 1.0, False),
    ("tau_F", 200.0, 5.0, 5000.0, True),
    ("ce0", 0.5, 0.0, 1.0, False),
    # C persistent spillback fronts
    ("K3", 150.0, 5.0, 5000.0, True),
    ("r_up", 0.2, 0.005, 3.0, True),
    ("r_dn", 0.01, 1e-4, 1.0, True),
    ("c_J", 0.5, 0.0, 10.0, False),
    ("w_G", 0.2, 0.0, 5.0, False),
]


def free_for(mech):
    out = list(BASE)
    for m in sorted(mech):
        out += MECH_PARAMS[m]
    return out


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


def _exits(M, s, uu, th, mech):
    sig, lane, toll, ramp, frt, clr = uu
    p3a, p3b = s[2], s[5]
    r = th["r_pipe"]
    if "B" in mech:
        ce, fat = s[11], s[12]
        boost = 1.0 + th["clr"] * ce * (1.0 - th["k_F"] * fat)
    else:
        boost = 1.0 + th["clr"] * clr
    cap_a = th["cap"] * (1.0 - th["lc_a"] * lane) * boost + 1e-6
    cap_b = th["cap"] * (1.0 - th["lc_b"] * lane) * boost + 1e-6
    return _smin(M, r * p3a, cap_a), _smin(M, r * p3b, cap_b)


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    sig, lane, toll, ramp, frt, clr = uu
    p1a, p2a, p3a, p1b, p2b, p3b, sa, sb, ma, mb, q, ce, fat, ga, gb = s
    r = th["r_pipe"]
    n_a = p1a + p2a + p3a
    n_b = p1b + p2b + p3b
    a_tot = th["dem0"] * ramp / (1.0 + M.exp(th["k_toll"] * (toll - th["t50"])))
    qa = q if "A" in mech else 0.5
    a_a = a_tot * qa * M.max(1.0 - n_a / th["n_max"], 0.0)
    a_b = a_tot * (1.0 - qa) * M.max(1.0 - n_b / th["n_max"], 0.0)
    share = M.min(M.max(0.5 + th["w_sig"] * (sig - 0.5), 0.02), 0.98)
    c_eff = ce if "B" in mech else clr
    jc = th["junc"] * (1.0 - th["j_c"] * c_eff)
    if "C" in mech:
        jc = jc / (1.0 + th["c_J"] * (ga + gb))
        k3 = th["K3"]
        tr_a = r * p2a * M.max(1.0 - p3a / k3, 0.0)
        tr_b = r * p2b * M.max(1.0 - p3b / k3, 0.0)
    else:
        tr_a = r * p2a
        tr_b = r * p2b
    j_a = _smin(M, r * p1a, jc * share + 1e-6)
    j_b = _smin(M, r * p1b, jc * (1.0 - share) + 1e-6)
    out_a, out_b = _exits(M, s, uu, th, mech)
    div_a = th["d_div"] * p1a * p1a / th["n_max"]
    div_b = th["d_div"] * p1b * p1b / th["n_max"]
    dp1a = a_a - j_a - div_a
    dp2a = j_a - tr_a
    dp3a = tr_a - out_a
    dp1b = a_b - j_b - div_b
    dp2b = j_b - tr_b
    dp3b = tr_b - out_b
    w1 = th["w1"]
    rm = th["r_m"]
    occ_a = w1 * p1a * M.pow(0.5 / share, th["kap"]) + p2a + p3a
    occ_b = w1 * p1b * M.pow(0.5 / (1.0 - share), th["kap"]) + p2b + p3b
    if "C" in mech:
        occ_a = occ_a + th["w_G"] * th["n_ref"] * ga
        occ_b = occ_b + th["w_G"] * th["n_ref"] * gb
    dma = rm * (occ_a - ma)
    dmb = rm * (occ_b - mb)
    tgt_a = th["v_free"] / (1.0 + M.pow(ma / th["n_ref"], th["gam"]))
    tgt_b = th["v_free"] / (1.0 + M.pow(mb / th["n_ref"], th["gam"]))
    dsa = th["rho"] * (tgt_a - sa)
    dsb = th["rho"] * (tgt_b - sb)
    z = 0.0 * p1a
    if "A" in mech:
        qs = 1.0 / (1.0 + M.exp(-th["beta_L"] * (sa - sb) / 10.0))
        dq = (qs - q) / th["tau_L"]
    else:
        dq = z
    if "B" in mech:
        dce = (clr - ce) / th["tau_sw"]
        dfat = (ce - fat) / th["tau_F"]
    else:
        dce = z
        dfat = z
    if "C" in mech:
        ta = (p3a / th["K3"]) ** 2
        tb = (p3b / th["K3"]) ** 2
        dga = th["r_up"] * M.max(ta - ga, 0.0) - th["r_dn"] * M.max(ga - ta, 0.0)
        dgb = th["r_up"] * M.max(tb - gb, 0.0) - th["r_dn"] * M.max(gb - tb, 0.0)
    else:
        dga = z
        dgb = z
    return _pack(M, [dp1a, dp2a, dp3a, dp1b, dp2b, dp3b, dsa, dsb, dma, dmb, dq, dce, dfat, dga, dgb])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    out_a, out_b = _exits(M, s, uu, th, mech)
    return _pack(M, [out_a, out_b, s[6], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    sa = M.min(M.max(y[2], 1.0), 80.0)
    sb = M.min(M.max(y[3], 1.0), 80.0)
    ce = z + th["ce0"]
    return _pack(M, [z, z, z, z, z, z, sa, sb, z, z, z + 0.5, ce, z, z, z])
