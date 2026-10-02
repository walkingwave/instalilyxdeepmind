"""Grey-box ODE: traffic hi = traffic z8 (toll-elastic demand, lane closure, A route learning +
B crew fatigue) with three switchable structure changes, set by the module constants below.

  ps (param): sharpness of the capacity minimum smin(x, c) = x / (1 + (x/c)^ps)^(1/ps).
      ps = 4 is z8; large ps approaches the hard point-queue law min(demand, capacity).
  SPEED: target speed law on the lagged occupancy m_i of each route.
      hill    v_free / (1 + (m/n_ref)^gam)                         (z8)
      green   v_free ((1 - m/n_ref)^+)^gam                          (Greenshields at gam = 1)
      under   v_free exp(-(m/n_ref)^gam)                            (Underwood at gam = 1)
      tri     smin(v_free, v_free gam (1 - m/n_ref)^+ n_ref / m)    (triangular diagram, congested branch)
      journey v_free min(1, T0 (out_i + e) / (m_i + T0 e))^gam      (completed-journey speed, Little law)
      mix     w_j journey + (1 - w_j) hill                          (journey times + stopped/moving mix)
  KTR: number of free-flow travel stages between admission and the approach queue (0 = z8); each stage
      has rate KTR / D_tr, so admitted vehicles reach the queue after a sharp delay of about D_tr ticks.
"""
import math

import numpy as np

FAMILY = "traffic_hi_d"
SPEED = "hill"
KTR = 4
OBS = ["flow_a", "flow_b", "speed_a", "speed_b"]
CTRL = ["signal_timing", "lane_closure", "toll", "ramp_metering", "freight_priority",
        "clearance_effort"]
MECHS = {"A": "route learning (admitted split learns toward the faster route)",
         "B": "crew fatigue / switching cost (lagged, tiring clearance crew)",
         "C": "persistent spillback fronts (not modelled in this family)"}
N_SUB = 2

STATE = (["p1a", "p2a", "p3a", "p1b", "p2b", "p3b", "sa", "sb", "ma", "mb", "q", "ce", "fat"]
         + ["ta%d" % k for k in range(KTR)] + ["tb%d" % k for k in range(KTR)])
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1e5] * 6 + [200.0, 200.0, 1e5, 1e5, 1.0, 1.0, 1.0] + [1e5] * (2 * KTR))

BASE = ["dem0", "w_sig", "r_pipe", "junc", "cap", "L_a", "L_b", "clr", "v_free", "n_ref", "rho",
        "n_max", "gam", "w1", "r_m", "kap", "k_toll", "n_l"]
MECH_PARAMS = {"A": ["beta_L", "tau_L"], "B": ["tau_sw", "k_F", "tau_F", "ce0"], "C": []}
EXTRA = ["ps", "T0", "w_j", "D_tr"]

PARAMS = [
    ("dem0", 46.927, 5.0, 300.0, True),
    ("w_sig", 0.72055, 0.0, 1.5, False),
    ("r_pipe", 0.24254, 0.05, 1.5, True),
    ("junc", 35.343, 3.0, 600.0, True),
    ("cap", 16.733, 3.0, 300.0, True),
    ("L_a", 1.6836, 0.3, 50.0, True),
    ("L_b", 1.2262, 0.3, 50.0, True),
    ("clr", 0.38409, 0.0, 3.0, False),
    ("v_free", 48.714, 30.0, 70.0, False),
    ("n_ref", 318.99, 10.0, 5000.0, True),
    ("rho", 0.29726, 0.02, 1.5, True),
    ("n_max", 1069.7, 50.0, 20000.0, True),
    ("gam", 1.0587, 0.2, 4.0, False),
    ("w1", 1.7512, 0.0, 8.0, False),
    ("r_m", 0.24796, 0.03, 3.0, True),
    ("kap", 0.66022, 0.0, 2.0, False),
    ("k_toll", 0.196, -0.5, 1.0, False),
    ("n_l", 1.3642, 1.0, 12.0, False),
    ("beta_L", 0.26168, 0.0, 20.0, False),
    ("tau_L", 6.5424, 1.0, 5000.0, True),
    ("tau_sw", 0.71574, 0.2, 300.0, True),
    ("k_F", 0.58697, 0.0, 1.0, False),
    ("tau_F", 76.067, 5.0, 5000.0, True),
    ("ce0", 0.072468, 0.0, 1.0, False),
    ("ps", 4.0, 1.0, 40.0, True),
    ("T0", 10.0, 1.0, 60.0, True),
    ("w_j", 0.5, 0.0, 1.0, False),
    ("D_tr", 4.0, 0.5, 20.0, True),
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


def _smin(M, x, c, ps):
    q = M.min(x / c, 30.0)
    return c * q / M.pow(1.0 + M.pow(q, ps), 1.0 / ps)


def _exits(M, s, uu, th, mech):
    sig, lane, toll, ramp, frt, clr = uu
    p3a, p3b = s[2], s[5]
    r = th["r_pipe"]
    if "B" in mech:
        ce, fat = s[11], s[12]
        boost = 1.0 + th["clr"] * ce * (1.0 - th["k_F"] * fat)
    else:
        boost = 1.0 + th["clr"] * clr
    cap_a = th["cap"] * M.max(1.0 - M.pow(lane / th["L_a"], th["n_l"]), 0.0) * boost + 1e-6
    cap_b = th["cap"] * M.max(1.0 - M.pow(lane / th["L_b"], th["n_l"]), 0.0) * boost + 1e-6
    return _smin(M, r * p3a, cap_a, th["ps"]), _smin(M, r * p3b, cap_b, th["ps"])


def _speed(M, m, out, th):
    v = th["v_free"]
    x = m / th["n_ref"]
    if SPEED == "green":
        return v * M.pow(1.0 - x, th["gam"]) + 0.0 * m
    if SPEED == "under":
        return v * M.exp(-M.pow(x, th["gam"]))
    if SPEED == "tri":
        return _smin(M, v * th["gam"] * M.max(1.0 - x, 0.0) / (x + 1e-6), v + 0.0 * m, 4.0)
    hill = v / (1.0 + M.pow(x, th["gam"]))
    if SPEED == "hill":
        return hill
    e = 0.5
    jr = v * M.pow(M.min(th["T0"] * (out + e) / (m + th["T0"] * e), 1.0), th["gam"])
    if SPEED == "journey":
        return jr
    return th["w_j"] * jr + (1.0 - th["w_j"]) * hill


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    sig, lane, toll, ramp, frt, clr = uu
    p1a, p2a, p3a, p1b, p2b, p3b, sa, sb, ma, mb, q, ce, fat = s[:13]
    ta = s[13:13 + KTR]
    tb = s[13 + KTR:13 + 2 * KTR]
    r = th["r_pipe"]
    n_a = p1a + p2a + p3a
    n_b = p1b + p2b + p3b
    for k in range(KTR):
        n_a = n_a + ta[k]
        n_b = n_b + tb[k]
    a_tot = th["dem0"] * ramp * M.exp(-th["k_toll"] * (toll - 2.5))
    qa = q if "A" in mech else 0.5
    a_a = a_tot * qa * M.max(1.0 - n_a / th["n_max"], 0.0)
    a_b = a_tot * (1.0 - qa) * M.max(1.0 - n_b / th["n_max"], 0.0)
    dta = []
    dtb = []
    if KTR:
        kr = KTR / th["D_tr"]
        for k in range(KTR):
            dta.append(a_a - kr * ta[k])
            dtb.append(a_b - kr * tb[k])
            a_a = kr * ta[k]
            a_b = kr * tb[k]
    share = M.min(M.max(0.5 + th["w_sig"] * (sig - 0.5), 0.02), 0.98)
    jc = th["junc"]
    j_a = _smin(M, r * p1a, jc * share + 1e-6, th["ps"])
    j_b = _smin(M, r * p1b, jc * (1.0 - share) + 1e-6, th["ps"])
    out_a, out_b = _exits(M, s, uu, th, mech)
    dp1a = a_a - j_a
    dp2a = j_a - r * p2a
    dp3a = r * p2a - out_a
    dp1b = a_b - j_b
    dp2b = j_b - r * p2b
    dp3b = r * p2b - out_b
    w1 = th["w1"]
    rm = th["r_m"]
    occ_a = w1 * p1a * M.pow(0.5 / share, th["kap"]) + p2a + p3a
    occ_b = w1 * p1b * M.pow(0.5 / (1.0 - share), th["kap"]) + p2b + p3b
    dma = rm * (occ_a - ma)
    dmb = rm * (occ_b - mb)
    tgt_a = _speed(M, ma, out_a, th)
    tgt_b = _speed(M, mb, out_b, th)
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
    return _pack(M, [dp1a, dp2a, dp3a, dp1b, dp2b, dp3b, dsa, dsb, dma, dmb, dq, dce, dfat] + dta + dtb)


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
    return _pack(M, [z, z, z, z, z, z, sa, sb, z, z, z + 0.5, ce, z] + [z] * (2 * KTR))
