"""Grey-box ODE: hospital_queue v8d = v8 + deterioration while waiting (heavier case mix).
Waiting patients deteriorate at k_det; Dw = deteriorated part of W. Deteriorated patients need
(1 + k_d) times the work, so patient service rates are divided by m = 1 + k_d Dw/(W + 1).
Long waits -> heavier mix -> slower service -> long waits: a congested hospital can stay
congested under the same action that drains a fresh one (history dependence).
Reported discharges carry the wave-completion discount f3 r^4/(r^4 + r_b^4), r = r_t / m
(few completions per tick come in waves; the typical tick sees none). No orientation lag.
NUMPY + math only (ships verbatim).

Stocks: W waiting, A1/A2 assessment (finite chairs), T treatment (finite beds), wt reported wait,
F fatigue (0..1), Se effective staff (orientation lag).
    eff = Se * (1 + ot_gain * O) * (1 - k_fat * F)
    dSe = (S - Se) / tau_o  when S > Se (new staff need orientation), (S - Se) / 0.5 when S < Se
    dF  = O (1 - F) / 20 - F / tau_f   (fatigue builds under overtime, clears over tau_f)
Rest as p3: r_a = c_a eff D/(D+d_h), r_t = c_t eff (1-D); f1 = min(r_a, K W, K free chairs),
A1 -> A2 at 2/tau_a, A2 -> T blocked by beds, f3 = min(r_t, K T) = discharges; arrivals gated at q_cap;
leave k_l W; wait relaxes to W/(f1+1) over tau_w.
Reset: W = queue0, wt = wait0, services empty, Se = 20 (full staff), F = 0.
"""
import math

import numpy as np

FAMILY = "hospital_queue_v8d"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {"A": "fatigue (slow recovery)", "B": "orientation lag of new staff", "C": "unused"}
N_SUB = 2

STATE = ["W", "A1", "A2", "T", "wt", "F", "Se", "Dw"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([333.0, 333.0, 333.0, 333.0, 1e4, 1.0, 20.0, 333.0])

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
    ("k_fat", 0.4, 0.0, 0.95, False),
    ("k_l", 0.01, 0.001, 0.5, True),
    ("tau_w", 10.0, 1.0, 60.0, True),
    ("q_cap", 320.0, 250.0, 340.0, False),
    ("d_h", 0.1, 0.01, 1.0, True),
    ("tau_f", 200.0, 5.0, 5000.0, True),
    ("k_det", 0.05, 0.001, 1.0, True),
    ("k_d", 0.7, 0.0, 4.0, False),
    ("r_b", 2.5, 0.3, 8.0, True),
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
    W, A1, A2, T, wt, F, Se, Dw = s
    S, E, D, U, O, Fu = uu
    eff = Se * (1.0 + th["ot_gain"] * O) * (1.0 - th["k_fat"] * F) / (1.0 + th["k_d"] * Dw / (W + 1.0))
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
    W, A1, A2, T, wt, F, Se, Dw = s
    S, E, D, U, O, Fu = uu
    f1, f2a, f2b, f3, r_t = _flows(M, s, uu, th)
    q = W + A1 + A2 + T
    gate = 1.0 / (1.0 + M.exp((q - th["q_cap"]) / Q_W))
    arr = (th["lam0"] + E) * gate
    dW = arr - f1 - th["k_l"] * W
    out = (f1 + th["k_l"] * W) * Dw / (W + 1.0)
    dDw = th["k_det"] * (W - Dw) - out
    dA1 = f1 - f2a
    dA2 = f2a - f2b
    dT = f2b - f3
    dwt = (W / (f1 + 1.0) - wt) / th["tau_w"]
    dF = O * (1.0 - F) / TAU_B - F / th["tau_f"]
    dSe = (S - Se) / 0.5
    return _pack(M, [dW, dA1, dA2, dT, dwt, dF, dSe, dDw])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    f1, f2a, f2b, f3, r_t = _flows(M, s, uu, th)
    W, A1, A2, T, wt, F, Se, Dw = s
    r4 = r_t * r_t * r_t * r_t
    rb = th["r_b"] * th["r_b"]
    return _pack(M, [wt, W + A1 + A2 + T, f3 * r4 / (r4 + rb * rb)])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    wt = M.max(y[0], 0.0)
    W = M.max(y[1], 0.0)
    z = 0.0 * W
    return _pack(M, [W, z, z, z, wt, z, z + S0, z])
