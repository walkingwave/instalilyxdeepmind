"""Grey-box ODE: hospital_queue, p3 pipeline + switchable mechanisms A/B/C + stranded-cohort wait.
NUMPY + math only (ships verbatim).

Base (always on): W waiting, A1/A2 two assessment stages (finite chairs), T treatment (finite beds),
queue = W + A1 + A2 + T, discharges = treatment completions f3.
    eff = S_e * (1 + ot_gain * O) * (1 - k_fat * F)          (S_e = S without B, F = 0 without A)
    r_a = c_a * eff * D / (D + D_H),  r_t = c_t * eff * (1 - D)
    f1 = min(r_a, K W, K (chairs - A1 - A2));  A1 -> A2 at 2/TAU_A;  A2 -> T = min(2 A2/TAU_A, K (beds - T))
    f3 = min(r_t, K T)
    arrivals (lam0 + E + R/tau_ret) gated to zero near q_cap (overflow referred elsewhere)
    leaving k_l * U * W (urgent priority pushes routine cases out of the pending list)
Reported wait (cohort sub-model, as in hospital_queue_min2 v4): waiting patients strand at
    s = k_s W w / (w + WT_H) into C (cleared over TAU_Z), whose mean age a grows g_z per tick;
    w* = W / (f1 + 1) + a C^2/(C^2 + C_H^2) sigma((W_TH - W)/W_WIDTH),  dw = (w* - w)/tau_w.
Mechanisms:
    A fatigue: dF = O (1 - F)/TAU_UP - F/tau_fat (overtime builds fatigue, slow recovery); eff *= 1 - k_fat F
    B orientation: dS_e = max(S - S_e, 0)/tau_h + min(S - S_e, 0)/max(DN * tau_h, 0.5)
       (new staff are fully effective only after tau_h; cuts act fast). S_e = S_E0 at reset.
    C returns: dR = r_ret * f3 * (1 - Fu) - R/tau_ret; R/tau_ret re-enters arrivals.
    (Tried and dropped, tags AC2/BC2: a finite program absorbing min(r_ret f3, p_cap Fu), g_z frozen at 4.5;
     those thetas use that 16-entry layout and do not load into this module.)
Parameters of an inactive mechanism have no effect (fit them with --free, or they stay at init).
Reset: W = queue0, w = wait0, services / cohort / fatigue / return pool empty.
"""
import math

import numpy as np

FAMILY = "hospital_queue_s1"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {
    "A": "fatigue: overtime builds F (slow recovery), effectiveness * (1 - k_fat F)",
    "B": "orientation: effective staff ramps up over tau_h, down fast",
    "C": "returns: discharges * (1 - followup) come back after tau_ret",
}
N_SUB = 2

STATE = ["W", "A1", "A2", "T", "wt", "C", "a", "F", "Se", "R"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 1e4, 333.0, 3e3, 1.0, 20.0, 1e3])

Q_W = 6.0        # referral gate width (patients)
K_DRAIN = 3.0    # max drain / fill rate of a stock per tick
TAU_A = 0.80     # assessment duration (p3 refit value)
D_H = 0.087      # diagnostic share half-saturation (p3 refit value)
TAU_Z = 150.0    # stranded cohort clearing time
C_H = 5.0        # cohort size at half weight in the estimate
W_WIDTH = 5.0    # ordinary-pool-empty switch width
W_TH = 40.0      # ordinary pool size below which the cohort shows (cohort v4 fit: 40, at bound)
WT_H = 30.0      # reported wait at which stranding runs at half rate
TAU_UP = 20.0    # A: fatigue build-up time at full overtime
DN = 0.25        # B: staff cuts act DN times faster than orientation
S_E0 = 20.0      # B: oriented staff at reset

PARAMS = [
    ("lam0", 10.6, 5.0, 25.0, False),
    ("c_a", 1.0, 0.2, 6.0, True),
    ("c_t", 0.95, 0.3, 10.0, True),
    ("chairs", 20.0, 5.0, 300.0, True),
    ("beds", 10.5, 5.0, 300.0, True),
    ("ot_gain", 0.5, 0.0, 1.5, False),
    ("k_l", 0.012, 0.001, 0.5, True),
    ("tau_w", 10.0, 1.0, 60.0, True),
    ("q_cap", 317.0, 250.0, 340.0, False),
    ("k_s", 2.4e-3, 1e-5, 1e-2, True),
    ("g_z", 4.7, 0.3, 6.0, False),
    # A
    ("k_fat", 0.3, 0.0, 0.9, False),
    ("tau_fat", 60.0, 5.0, 600.0, True),
    # B
    ("tau_h", 10.0, 1.0, 200.0, True),
    # C
    ("r_ret", 0.1, 0.0, 0.8, False),
    ("tau_ret", 40.0, 3.0, 300.0, True),
]
BASE = [p[0] for p in PARAMS[:11]]
MECH_PARAMS = {"A": ["k_fat", "tau_fat"], "B": ["tau_h"], "C": ["r_ret", "tau_ret"]}


def free_for(mech):
    """Parameter names that matter for a mechanism set (for ode_lab --free)."""
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


class _NP:
    max = np.maximum
    min = np.minimum

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))


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


def _flows(M, s, uu, th, mech):
    W, A1, A2, T, wt, C, a, F, Se, R = s
    S, E, D, U, O, Fu = uu
    staff = Se if "B" in mech else S
    eff = staff * (1.0 + th["ot_gain"] * O)
    if "A" in mech:
        eff = eff * (1.0 - th["k_fat"] * F)
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


def _target(M, s, f1):
    W, A1, A2, T, wt, C, a, F, Se, R = s
    sw = 1.0 / (1.0 + M.exp((W - W_TH) / W_WIDTH))
    return W / (f1 + 1.0) + a * (C * C / (C * C + C_H * C_H)) * sw


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    W, A1, A2, T, wt, C, a, F, Se, R = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3 = _flows(M, s, uu, th, mech)
    q = W + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - th["q_cap"]) / Q_W))
    z = 0.0 * W
    if "C" in mech:
        ret = R / th["tau_ret"]
        dR = th["r_ret"] * f3 * (1.0 - Fu) - ret
    else:
        ret = z
        dR = z
    arr = (th["lam0"] + E + ret) * gate
    leave = th["k_l"] * U * W
    strand = th["k_s"] * W * wt / (wt + WT_H)
    dW = arr - f1 - leave - strand
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    dwt = (_target(M, s, f1) - wt) / th["tau_w"]
    dC = strand - C / TAU_Z
    da = th["g_z"] - a * strand / (C + 1.0) - a / TAU_Z
    if "A" in mech:
        dF = O * (1.0 - F) / TAU_UP - F / th["tau_fat"]
    else:
        dF = z
    if "B" in mech:
        gap = S - Se
        dSe = M.max(gap, 0.0) / th["tau_h"] + M.min(gap, 0.0) / M.max(DN * th["tau_h"], 0.5)
    else:
        dSe = z
    return _pack(M, [dW, dA1, dA2, dT, dwt, dC, da, dF, dSe, dR])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3 = _flows(M, s, uu, th, mech)
    W, A1, A2, T = s[0], s[1], s[2], s[3]
    return _pack(M, [s[4], W + A1 + A2 + T, f3])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, wt, z, z, z, z + S_E0, z])
