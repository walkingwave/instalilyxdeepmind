"""Grey-box ODE: hospital_queue v9g = v9f without the wave discount (reported discharges = f3; r_b unused).
v9f = v9 with an overtime law that is neutral at the steady state:
F relaxes to O over tau_f in [10, 100] and k_fat = ot_gain / (1 + ot_gain), so a sustained overtime O=1
ends with (1 + g)(1 - g/(1 + g)) = 1 (no permanent gain or loss); overtime helps only for ~tau_f ticks.
Grey-box ODE: hospital_queue v9 = v8f with fatigue recovery bounded to tau_f in [30, 400] ticks
(v8f fitted tau_f at its 5,000 bound: fatigue never cleared, so a 4,000-tick hold after overtime stayed
under-served). Same structure otherwise.
Grey-box ODE: hospital_queue v8f = v8b with an asymmetric wait estimate: the reported wait rises
toward W/(f1+1) over tau_up and falls over tau_w (the estimate reacts fast to a growing backlog and
forgets slowly).
v8b: v8 + wave-completion discount on reported discharges.
With little treatment work per tick patients finish in waves (most ticks 0, some 5-17); the
reported discharges are f3 * r^4 / (r^4 + r_b^4), r = treatment work rate r_t (the typical tick).
NUMPY + math only (ships verbatim).

Stocks: W waiting, A1/A2 assessment (finite chairs), T treatment (finite beds), wt reported wait,
F fatigue (0..1), Se effective staff (orientation lag).
    eff = Se * (1 + ot_gain * O) * (1 - ot_gain/(1 + ot_gain) * F)
    dSe = (S - Se) / tau_o  when S > Se (new staff need orientation), (S - Se) / 0.5 when S < Se
    dF  = (O - F) / tau_f   (overtime-neutral at the steady state)
Rest as p3: r_a = c_a eff D/(D+d_h), r_t = c_t eff (1-D); f1 = min(r_a, K W, K free chairs),
A1 -> A2 at 2/tau_a, A2 -> T blocked by beds, f3 = min(r_t, K T) = discharges; arrivals gated at q_cap;
leave k_l W; wait relaxes to W/(f1+1) over tau_w.
Reset: W = queue0, wt = wait0, services empty, Se = 20 (full staff), F = 0.
"""
import math

import numpy as np

FAMILY = "hospital_queue_v9g"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {"A": "fatigue (slow recovery)", "B": "orientation lag of new staff", "C": "unused"}
N_SUB = 2

STATE = ["W", "A1", "A2", "T", "wt", "F", "Se"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 1e4, 1.0, 20.0])

Q_W = 6.0
TAU_B = 20.0
K_DRAIN = 3.0
S0 = 20.0

PARAMS = [
    ("lam0", 11.0, 5.0, 25.0, False),
    ("c_a", 1.5, 0.2, 6.0, True),
    ("c_t", 1.5, 0.3, 10.0, True),
    ("tau_a", 1.5, 0.7, 15.0, True),
    ("chairs", 30.0, 5.0, 300.0, True),
    ("beds", 30.0, 5.0, 300.0, True),
    ("ot_gain", 0.5, 0.0, 1.5, False),
    ("k_l", 0.01, 0.001, 0.5, True),
    ("tau_w", 10.0, 1.0, 60.0, True),
    ("q_cap", 320.0, 250.0, 340.0, False),
    ("d_h", 0.1, 0.01, 1.0, True),
    ("tau_o", 60.0, 2.0, 1000.0, True),
    ("tau_f", 40.0, 10.0, 100.0, True),
    ("r_b", 2.5, 0.3, 8.0, False),
    ("tau_up", 8.0, 1.0, 60.0, True),
]


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


def _flows(M, s, uu, th):
    W, A1, A2, T, wt, F, Se = s
    S, E, D, U, O, Fu = uu
    g = th["ot_gain"]
    eff = Se * (1.0 + g * O) * (1.0 - g / (1.0 + g) * F)
    r_a = th["c_a"] * eff * D / (D + th["d_h"]) + 1e-6
    r_t = th["c_t"] * eff * (1.0 - D) + 1e-6
    free_c = M.max(th["chairs"] - A1 - A2, 0.0)
    free_b = M.max(th["beds"] - T, 0.0)
    f1 = M.min(M.min(r_a, K_DRAIN * W), K_DRAIN * free_c)
    ka = 2.0 / th["tau_a"]
    f2a = ka * A1
    f2b = M.min(ka * A2, K_DRAIN * free_b)
    f3 = M.min(r_t, K_DRAIN * T)
    return f1, f2a, f2b, f3, r_t


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    W, A1, A2, T, wt, F, Se = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3, r_t = _flows(M, s, uu, th)
    q = W + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - th["q_cap"]) / Q_W))
    arr = (th["lam0"] + E) * gate
    dW = arr - f1 - th["k_l"] * W
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    gw = W / (f1 + 1.0) - wt
    dwt = M.max(gw, 0.0) / th["tau_up"] + M.min(gw, 0.0) / th["tau_w"]
    dF = (O - F) / th["tau_f"]
    gap = S - Se
    dSe = M.max(gap, 0.0) / th["tau_o"] + M.min(gap, 0.0) / 0.5
    return _pack(M, [dW, dA1, dA2, dT, dwt, dF, dSe])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3, r_t = _flows(M, s, uu, th)
    W, A1, A2, T, wt, F, Se = s
    r4 = r_t * r_t * r_t * r_t
    rb = th["r_b"] * th["r_b"]
    return _pack(M, [wt, W + A1 + A2 + T, f3 + 0.0 * r4 * rb])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, wt, z, z + S0])
