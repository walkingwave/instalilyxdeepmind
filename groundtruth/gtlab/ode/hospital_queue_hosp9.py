"""Grey-box ODE: hospital_queue, p3 pipeline with a wider wait law (lane hosp9).
NUMPY + math only (ships verbatim).

Queue and discharges are p3 exactly (same stocks, flows and parameters). Only the reported wait
changes; it never feeds back into the other stocks:
    target = c_w * W / (f1 + eps_w) + b_ot * G / (W + w0)
    wt rises toward target with tau_w and falls with tau_dn
    G = overtime staff-hours stock: dG = staffing * overtime / 20 - G / tau_g, G(0) = 0
With eps_w = 1, c_w = 1, tau_dn = tau_w, b_ot = 0 the family is p3.
"""
import math

import numpy as np

FAMILY = "hospital_queue_hosp9"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {"A": "fatigue (always on)", "B": "finite chairs/beds (always on)", "C": "unused"}
N_SUB = 2

STATE = ["W", "A1", "A2", "T", "wt", "F", "G"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 1e4, 1.0, 1e4])

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
    W, A1, A2, T, wt, F, G = s
    S, E, D, U, O, Fu = uu
    eff = S * (1.0 + th["ot_gain"] * O) * (1.0 - th["k_fat"] * F)
    r_a = th["c_a"] * eff * D / (D + th["d_h"]) + 1e-6
    r_t = th["c_t"] * eff * (1.0 - D) + 1e-6
    free_c = M.max(th["chairs"] - A1 - A2, 0.0)
    free_b = M.max(th["beds"] - T, 0.0)
    f1 = M.min(M.min(r_a, K_DRAIN * W), K_DRAIN * free_c)
    ka = 2.0 / th["tau_a"]
    f2a = ka * A1
    f2b = M.min(ka * A2, K_DRAIN * free_b)
    f3 = M.min(r_t, K_DRAIN * T)
    return f1, f2a, f2b, f3


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    W, A1, A2, T, wt, F, G = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3 = _flows(M, s, uu, th)
    q = W + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - th["q_cap"]) / Q_W))
    arr = (th["lam0"] + E) * gate
    leave = th["k_l"] * W
    dW = arr - f1 - leave
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    tgt = th["c_w"] * W / (f1 + th["eps_w"]) + th["b_ot"] * G / (W + th["w0"])
    gap = tgt - wt
    up = 0.5 + 0.5 * gap / (M.sqrt(gap * gap) + 1e-9)
    dwt = gap * (up / th["tau_w"] + (1.0 - up) / th["tau_dn"])
    dF = (O - F) / TAU_FAT
    dG = S * O / 20.0 - G / th["tau_g"]
    return _pack(M, [dW, dA1, dA2, dT, dwt, dF, dG])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3 = _flows(M, s, uu, th)
    W, A1, A2, T, wt, F, G = s
    return _pack(M, [wt, W + A1 + A2 + T, f3])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, wt, z, z])
