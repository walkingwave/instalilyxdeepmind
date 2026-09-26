"""Grey-box ODE: hospital_queue, waiting -> assessment -> treatment pipeline with a stranded cohort.
NUMPY + math only (ships verbatim).

Stocks: W waiting, A1/A2 two assessment stages (finite chairs), T in treatment (finite beds),
queue = W + A1 + A2 + T; wt reported wait; C stranded (deteriorated) cohort, a its mean age.
Work: eff = staffing * (1 + ot_gain * overtime). Diagnostic allocation D feeds assessment with a
saturating share D / (D + d_h) and treatment with (1 - D):
    r_a = c_a * eff * D / (D + d_h),   r_t = c_t * eff * (1 - D).
Flows (every rate <= K per tick, K = 3):
    f1 = min(r_a, K W, K (chairs - A1 - A2))     admission to assessment
    A1 -> A2 at 2 / tau_a;  A2 -> T = min(2 A2 / tau_a, K (beds - T))   (blocked when beds full)
    f3 = min(r_t, K T)                          treatment completion = discharges
Arrivals: base lam0 gated to zero near q_cap (referred elsewhere); elective E gated (cancelled)
near the lower centre q_e. Waiting patients leave at k_l * U * W (urgent priority pushes routine
cases out of the pending list).
Stranded cohort: waiting patients deteriorate at k_s * W * wt / (wt + WT_H) into C (cleared slowly, TAU_Z); their mean
age a grows g_z per tick and is diluted by newcomers. The reported wait relaxes (tau_w) to the Little
estimate W / (f1 + 1) of the ordinary pool, plus the cohort age once the ordinary pool is nearly
empty (W below w_th) so the cohort dominates the estimate:
    target = W / (f1 + 1) + a * C^2 / (C^2 + C_H^2) * sigma((w_th - W) / W_WIDTH).

Reset: W = queue0, wt = wait0, services and cohort empty.
Mechanism letters accepted but every term is always active; fit with pairs=("AB",).
"""
import math

import numpy as np

FAMILY = "hospital_queue_min2"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {"A": "stranded cohort (always on)", "B": "finite chairs/beds (always on)", "C": "unused"}
N_SUB = 2

STATE = ["W", "A1", "A2", "T", "wt", "C", "a"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 1e4, 333.0, 3e3])

Q_W = 6.0        # referral gate width (patients)
Q_E_W = 20.0     # elective cancellation gate width (patients)
K_DRAIN = 3.0    # max drain / fill rate of a stock per tick
TAU_A = 1.77     # assessment duration (two stages of TAU_A / 2)
D_H = 0.04       # diagnostic share half-saturation
Q_CAP = 325.0    # referral gate centre
TAU_Z = 150.0    # stranded cohort clearing time (bounds the cohort age at g_z * TAU_Z)
C_H = 5.0        # cohort size at which it carries half weight in the estimate
W_WIDTH = 5.0    # width of the ordinary-pool-empty switch (patients)
WT_H = 30.0      # reported wait at which deterioration runs at half rate

PARAMS = [
    ("lam0", 11.5, 5.0, 25.0, False),
    ("c_a", 0.7, 0.2, 6.0, True),
    ("c_t", 1.9, 0.3, 10.0, True),
    ("chairs", 30.0, 5.0, 300.0, True),
    ("beds", 15.0, 5.0, 300.0, True),
    ("ot_gain", 0.45, 0.0, 1.5, False),
    ("k_l", 0.02, 0.001, 0.5, True),
    ("tau_w", 10.0, 1.0, 60.0, True),
    ("q_e", 250.0, 120.0, 330.0, False),
    ("k_s", 6e-4, 1e-5, 1e-2, True),
    ("g_z", 2.0, 0.3, 6.0, False),
    ("w_th", 20.0, 2.0, 40.0, True),
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
    W, A1, A2, T, wt, C, a = s
    S, E, D, U, O, Fu = uu
    eff = S * (1.0 + th["ot_gain"] * O)
    r_a = th["c_a"] * eff * D / (D + D_H) + 1e-6
    r_t = th["c_t"] * eff * (1.0 - D) + 1e-6
    free_c = M.max(th["chairs"] - A1 - A2, 0.0)
    free_b = M.max(th["beds"] - T, 0.0)
    f1 = M.min(M.min(r_a, K_DRAIN * W), K_DRAIN * free_c)
    ka = 2.0 / TAU_A
    f2a = ka * A1
    f2b = M.min(ka * A2, K_DRAIN * free_b)
    f3 = M.min(r_t, K_DRAIN * T)
    return f1, f2a, f2b, f3


def _target(M, s, f1, th):
    W, A1, A2, T, wt, C, a = s
    sw = 1.0 / (1.0 + M.exp((W - th["w_th"]) / W_WIDTH))
    return W / (f1 + 1.0) + a * (C * C / (C * C + C_H * C_H)) * sw


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    W, A1, A2, T, wt, C, a = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3 = _flows(M, s, uu, th)
    q = W + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - Q_CAP) / Q_W))
    gate_e = 1.0 / (1.0 + M.exp((q - th["q_e"]) / Q_E_W))
    arr = th["lam0"] * gate + E * gate_e
    leave = th["k_l"] * U * W
    strand = th["k_s"] * W * wt / (wt + WT_H)
    dW = arr - f1 - leave - strand
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    dwt = (_target(M, s, f1, th) - wt) / th["tau_w"]
    dC = strand - C / TAU_Z
    da = th["g_z"] - a * strand / (C + 1.0) - a / TAU_Z
    return _pack(M, [dW, dA1, dA2, dT, dwt, dC, da])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3 = _flows(M, s, uu, th)
    W, A1, A2, T, wt, C, a = s
    return _pack(M, [wt, W + A1 + A2 + T, f3])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, wt, z, z])
