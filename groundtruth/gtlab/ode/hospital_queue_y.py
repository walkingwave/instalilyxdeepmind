"""Grey-box ODE: hospital_queue y = p3 pipeline on a brief-faithful base, three mechanisms as switches.
NUMPY + math only (ships verbatim).

Base (always on):
  case types: non-elective waiting Wn (routine + urgent, arrivals lam0) and elective waiting We
  (arrivals = elective_scheduling); both gated to zero as the queue nears q_cap (overflow referred).
  Admission to assessment f1 = min(r_a, K W, K free chairs), W = Wn + We, shared by weight:
  non-elective (1 + U) (urgent priority favours urgent/routine admissions), elective e_w.
  Assessment A1 -> A2 (2/tau_a each); A2 -> T blocked when beds are full (the finished assessment
  holds its chair); discharges f3 = min(r_t, K T).
  Work: staff on the floor S_h = S* (1 - phi_f Fu) (follow-up diverts shared staff);
  eff = S_h (1 + g O) [ (1 - k_fat F) if A ];  r_a = c_a eff D/(D + d_h),  r_t = c_t eff (1 - D).
  Waiting non-electives leave at k_l Wn (1 + b_U (U - 0.6)); electives are cancelled at k_e We.
  Reported wait rises toward W/(f1 + 1) over tau_up and falls over tau_w.
Mechanisms (exactly two active; letters in `mech`):
  A fatigue:     dF = (O - F)/tau_f, eff x (1 - k_fat F)
  B orientation: S* = Se + rho (S - Se)+,  dSe = (S - Se)+/tau_o + (S - Se)-/0.5,  Se(0) = 20
                 (without B, S* = S)
  C returns:     discharges not absorbed by the finite follow-up program (min(f3, p_cap Fu)) return
                 with probability r_ret after ~tau_ret: dR = r_ret (f3 - absorbed) - R/tau_ret,
                 arrivals to Wn + R/tau_ret
Reset: Wn = queue0, wt = wait0, everything else empty, Se = 20.
"""
import math

import numpy as np

FAMILY = "hospital_queue_y"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {"A": "fatigue after overtime", "B": "orientation of new staff", "C": "returns unless follow-up absorbs discharges"}
N_SUB = 2

STATE = ["Wn", "We", "A1", "A2", "T", "wt", "F", "Se", "R"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 333.0, 1e4, 1.0, 20.0, 1e4])

Q_W = 6.0
K_DRAIN = 3.0
S0 = 20.0

PARAMS = [
    ("lam0", 10.6, 5.0, 25.0, False),
    ("c_a", 1.0, 0.2, 6.0, True),
    ("c_t", 0.95, 0.3, 10.0, True),
    ("tau_a", 0.8, 0.7, 15.0, True),
    ("chairs", 18.0, 5.0, 300.0, True),
    ("beds", 10.5, 5.0, 300.0, True),
    ("ot_gain", 0.55, 0.0, 1.5, False),
    ("k_l", 0.008, 0.001, 0.5, True),
    ("tau_w", 14.0, 1.0, 60.0, True),
    ("q_cap", 318.0, 250.0, 340.0, False),
    ("d_h", 0.08, 0.01, 1.0, True),
    ("tau_up", 4.0, 1.0, 60.0, True),
    ("e_w", 1.0, 0.05, 5.0, True),
    ("k_e", 0.01, 0.001, 0.5, True),
    ("b_U", 0.0, -1.5, 1.6, False),
    ("phi_f", 0.05, 0.0, 0.6, False),
    # A fatigue
    ("k_fat", 0.35, 0.0, 0.9, False),
    ("tau_f", 30.0, 5.0, 400.0, True),
    # B orientation
    ("tau_o", 30.0, 2.0, 2000.0, True),
    ("rho", 0.5, 0.0, 1.0, False),
    # C returns
    ("r_ret", 0.1, 0.0, 1.0, False),
    ("tau_ret", 40.0, 5.0, 400.0, True),
    ("p_cap", 5.0, 0.1, 30.0, True),
]
BASE = [p[0] for p in PARAMS[:16]]
MECH_PARAMS = {"A": ["k_fat", "tau_f"], "B": ["tau_o", "rho"], "C": ["r_ret", "tau_ret", "p_cap"]}


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
    Wn, We, A1, A2, T, wt, F, Se, R = s
    S, E, D, U, O, Fu = uu
    if "B" in mech:
        Sx = Se + th["rho"] * M.max(S - Se, 0.0)
    else:
        Sx = S
    eff = Sx * (1.0 - th["phi_f"] * Fu) * (1.0 + th["ot_gain"] * O)
    if "A" in mech:
        eff = eff * (1.0 - th["k_fat"] * F)
    r_a = th["c_a"] * eff * D / (D + th["d_h"]) + 1e-6
    r_t = th["c_t"] * eff * (1.0 - D) + 1e-6
    W = Wn + We
    free_c = M.max(th["chairs"] - A1 - A2, 0.0)
    free_b = M.max(th["beds"] - T, 0.0)
    f1 = M.min(M.min(r_a, K_DRAIN * W), K_DRAIN * free_c)
    ka = 2.0 / th["tau_a"]
    f2a = ka * A1
    f2b = M.min(ka * A2, K_DRAIN * free_b)
    f3 = M.min(r_t, K_DRAIN * T)
    wn = Wn * (1.0 + U)
    we = We * th["e_w"]
    sn = wn / (wn + we + 1e-9)
    return f1, f2a, f2b, f3, sn


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    Wn, We, A1, A2, T, wt, F, Se, R = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3, sn = _flows(M, s, uu, th, mech)
    q = Wn + We + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - th["q_cap"]) / Q_W))
    ret = 0.0 * R
    dR = 0.0 * R
    if "C" in mech:
        ret = R / th["tau_ret"]
        absorbed = M.min(f3, th["p_cap"] * Fu)
        dR = th["r_ret"] * (f3 - absorbed) - ret
    lv = th["k_l"] * Wn * M.max(1.0 + th["b_U"] * (U - 0.6), 0.0)
    dWn = (th["lam0"] + ret) * gate - sn * f1 - lv
    dWe = E * gate - (1.0 - sn) * f1 - th["k_e"] * We
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    gw = (Wn + We) / (f1 + 1.0) - wt
    dwt = M.max(gw, 0.0) / th["tau_up"] + M.min(gw, 0.0) / th["tau_w"]
    dF = (O - F) / th["tau_f"] if "A" in mech else 0.0 * F
    if "B" in mech:
        gap = S - Se
        dSe = M.max(gap, 0.0) / th["tau_o"] + M.min(gap, 0.0) / 0.5
    else:
        dSe = 0.0 * Se
    return _pack(M, [dWn, dWe, dA1, dA2, dT, dwt, dF, dSe, dR])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3, sn = _flows(M, s, uu, th, mech)
    Wn, We, A1, A2, T, wt, F, Se, R = s
    return _pack(M, [wt, Wn + We + A1 + A2 + T, f3])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, z, wt, z, z + S0, z])
