"""Grey-box ODE: hospital_queue canon3 = canon2 with fatigue as an overtime-hours stock that recovers
slowly: dF = O - F / tau_f (tau_f free), eff /= 1 + k_fat F. A server whose capacity sits just
above the load after a pulse holds whatever queue it has (p4/voi plateau ~92, compose 23).
canon = hosp9 (h9w1) plus the brief's three named mechanisms written
literally, each with an off switch so every pair can be tested. NUMPY + math only (ships verbatim).
- fatigue (as hosp9): eff *= 1 - k_fat F, F follows overtime with 30 ticks memory; off: k_fat = 0.
- handover: effective staff Se rises toward a staffing increase with lag tau_o (orientation) and
  follows decreases at once; off: tau_o at its floor (0.3).
- returning case mix: a share rho of discharges returns to the waiting list through a two-stage
  delay (mean tau_r); follow-up prevents a share pf * followup of returns and diverts k_fs * followup
  of staff; off: rho = 0.
- canon2: the treatment stage empties at kt per tick (hosp9: fixed 3.0), so the uncongested
  occupancy (queue = lam0 x residence, 23 = 11.5 x 2 in the recovery hold) is free of the capacities.
- admission of waiting patients drains at kdr per tick (hosp9: fixed 3.0), so an uncongested
  waiting list can empty within the tick as a discrete admit-if-free rule would.
"""
import math

import numpy as np

FAMILY = "hospital_queue_canon3"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {"A": "fatigue", "B": "handover", "C": "returning case mix"}  # switched by parameters
N_SUB = 2

STATE = ["W", "A1", "A2", "T", "wt", "F", "G", "Se", "R1", "R2"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 1e4, 1e4, 1e4, 20.0, 1e4, 1e4])

Q_W = 6.0        # referral gate width (patients)
TAU_FAT = 30.0   # fatigue memory (ticks)
K_DRAIN = 3.0    # max drain / fill rate of a stock per tick

PARAMS = [
    ("lam0", 11.5, 5.0, 25.0, False),
    ("c_a", 1.5, 0.2, 6.0, True),
    ("c_t", 2.0, 0.3, 10.0, True),
    ("tau_a", 2.0, 0.7, 15.0, True),
    ("chairs", 30.0, 5.0, 300.0, True),
    ("beds", 40.0, 5.0, 300.0, True),
    ("ot_gain", 0.4, 0.0, 1.5, False),
    ("k_fat", 0.4, 0.0, 0.9, False),
    ("k_l", 0.02, 0.001, 0.5, True),
    ("tau_w", 8.0, 1.0, 60.0, True),
    ("q_cap", 325.0, 250.0, 340.0, False),
    ("d_h", 0.05, 0.01, 1.0, True),
    ("eps_w", 1.0, 0.02, 5.0, True),
    ("c_w", 1.0, 0.3, 3.0, True),
    ("tau_dn", 8.0, 1.0, 120.0, True),
    ("b_ot", 0.0, 0.0, 3000.0, False),
    ("tau_g", 200.0, 10.0, 3000.0, True),
    ("w0", 5.0, 0.5, 100.0, True),
    ("kdr", 3.0, 0.5, 20.0, True),
    ("tau_o", 0.3, 0.3, 100.0, True),
    ("se0", 20.0, 1.0, 20.0, False),
    ("rho", 0.0, 0.0, 0.5, False),
    ("tau_r", 30.0, 2.0, 300.0, True),
    ("pf", 0.5, 0.0, 1.0, False),
    ("k_fs", 0.0, 0.0, 0.5, False),
    ("kt", 3.0, 0.3, 10.0, True),
    ("tau_f", 30.0, 3.0, 5000.0, True),
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


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))


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


def _flows(M, s, uu, th):
    W, A1, A2, T, wt, F, G, Se, R1, R2 = s
    S, E, D, U, O, Fu = uu
    eff = Se * (1.0 - th["k_fs"] * Fu) * (1.0 + th["ot_gain"] * O) / (1.0 + th["k_fat"] * F)
    r_a = th["c_a"] * eff * D / (D + th["d_h"]) + 1e-6
    r_t = th["c_t"] * eff * (1.0 - D) + 1e-6
    free_c = M.max(th["chairs"] - A1 - A2, 0.0)
    free_b = M.max(th["beds"] - T, 0.0)
    kd = th["kdr"]
    f1 = M.min(M.min(r_a, kd * W), kd * free_c)
    ka = 2.0 / th["tau_a"]
    f2a = ka * A1
    f2b = M.min(ka * A2, K_DRAIN * free_b)
    f3 = M.min(r_t, th["kt"] * T)
    return f1, f2a, f2b, f3


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    W, A1, A2, T, wt, F, G, Se, R1, R2 = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3 = _flows(M, s, uu, th)
    q = W + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - th["q_cap"]) / Q_W))
    kr = 2.0 / th["tau_r"]
    ret = kr * R2
    arr = (th["lam0"] + E + ret) * gate
    leave = th["k_l"] * W
    dW = arr - f1 - leave
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    tgt = th["c_w"] * W / (f1 + th["eps_w"]) + th["b_ot"] * G / (W + th["w0"])
    gap = tgt - wt
    up = 0.5 + 0.5 * gap / (M.sqrt(gap * gap) + 1e-9)
    dwt = gap * (up / th["tau_w"] + (1.0 - up) / th["tau_dn"])
    dF = O - F / th["tau_f"]
    dG = S * O / 20.0 - G / th["tau_g"]
    ds = S - Se
    upo = 0.5 + 0.5 * ds / (M.sqrt(ds * ds) + 1e-9)
    dSe = ds * (upo / th["tau_o"] + (1.0 - upo) * 4.0)
    dR1 = th["rho"] * (1.0 - th["pf"] * Fu) * f3 - kr * R1
    dR2 = kr * R1 - ret
    return _pack(M, [dW, dA1, dA2, dT, dwt, dF, dG, dSe, dR1, dR2])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3 = _flows(M, s, uu, th)
    return _pack(M, [s[4], s[0] + s[1] + s[2] + s[3], f3])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, wt, z, z, z + th["se0"], z, z])
