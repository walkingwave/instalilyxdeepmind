"""Grey-box ODE: hospital_queue. NUMPY + math only (ships verbatim in submissions).

Structure
- Three waiting queues (routine, urgent, elective) -> assessment chairs (finite) ->
  treatment queue (finite, blocks assessment completions) -> beds (finite) -> discharge.
- Work delivered = effective staff * (1 + overtime gain) * effectiveness * (1 - follow-up
  diversion); diagnostic allocation splits it between assessment and treatment.
- Urgent priority re-weights admissions into assessment; urgent cases need more work.
- Arrivals are referred elsewhere when the waiting pool nears capacity (logistic gate);
  waiting routine/urgent patients leave or deteriorate out of the queue.
- wait_time = Little estimate: waiting patients / smoothed admission throughput.
Mechanisms: A fatigue (overtime EMA lowers effectiveness), B handover/orientation
(staff increases become effective with a lag), C returning case mix (discharges return
after a two-stage delay unless absorbed by the finite follow-up program).
"""
import math

import numpy as np

FAMILY = "hospital_queue"
OBS = ["wait_time", "queue", "discharges"]
CTRL = ["staffing", "elective_scheduling", "diagnostic_allocation", "urgent_priority",
        "overtime", "followup_capacity"]
MECHS = {
    "A": "fatigue: overtime EMA lowers later effectiveness",
    "B": "handover/orientation: staff increases effective only after a ramp",
    "C": "returning case mix: discharges return after a delay unless follow-up absorbs them",
}
N_SUB = 2

STATE = ["q_routine", "q_urgent", "q_elective", "assess", "treat_q", "beds",
         "staff_oriented", "fatigue", "program", "ret1", "ret2", "thru"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e5)
STATE_HI[7] = 1.0

PARAMS = [
    ("lam_r", 1.5, 0.01, 50.0, True),
    ("lam_u", 0.8, 0.01, 50.0, True),
    ("el_k", 0.15, 0.001, 3.0, True),
    ("sh_r", 0.5, 0.02, 0.95, False),      # initial routine share of reported count
    ("sh_u", 0.5, 0.02, 0.98, False),      # initial urgent share of the remainder
    ("q_cap", 150.0, 5.0, 5000.0, True),
    ("renege", 0.01, 0.0, 0.3, False),
    ("det_u", 0.02, 0.0, 0.5, False),
    ("chairs", 10.0, 1.0, 500.0, True),
    ("beds", 30.0, 1.0, 2000.0, True),
    ("w_cap", 20.0, 1.0, 2000.0, True),
    ("rho_a", 0.6, 0.005, 10.0, True),
    ("rho_t", 0.3, 0.005, 10.0, True),
    ("w_u", 1.5, 0.3, 5.0, True),
    ("prio_k", 3.0, 0.0, 20.0, False),
    ("tau_a", 2.0, 0.5, 100.0, True),
    ("tau_t", 4.0, 0.5, 300.0, True),
    ("ot_gain", 0.5, 0.0, 1.5, False),
    ("fu_div", 0.15, 0.0, 0.7, False),
    ("tau_w", 5.0, 1.0, 200.0, True),
    ("fat_k", 0.4, 0.0, 0.9, False),       # A
    ("tau_fat", 100.0, 5.0, 3000.0, True),  # A
    ("tau_hand", 20.0, 1.0, 500.0, True),  # B
    ("staff0", 20.0, 1.0, 20.0, False),    # B: oriented staff at reset
    ("ret_frac", 0.2, 0.0, 0.9, False),    # C
    ("tau_ret", 45.0, 4.0, 400.0, True),   # C
    ("prog_cap", 20.0, 0.5, 1000.0, True),  # C
    ("tau_prog", 30.0, 2.0, 500.0, True),  # C
]


# ---- numeric helpers (duplicated in every family file: files ship standalone) ----
class _PY:
    max = max
    min = min
    tanh = math.tanh

    @staticmethod
    def exp(z):
        return math.exp(z if z < 50.0 else 50.0)

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0

    @staticmethod
    def log(z):
        return math.log(z) if z > 1e-300 else -690.0


class _NP:
    max = np.maximum
    min = np.minimum
    tanh = np.tanh

    @staticmethod
    def exp(z):
        return np.exp(np.minimum(z, 50.0))

    @staticmethod
    def sqrt(z):
        return np.sqrt(np.maximum(z, 0.0))

    @staticmethod
    def log(z):
        return np.log(np.maximum(z, 1e-300))


def _smin(M, a, b, e):
    return 0.5 * (a + b - M.sqrt((a - b) * (a - b) + e * e))


def _pos(M, a, e):
    return 0.5 * (a + M.sqrt(a * a + e * e))


def _unpack(x, u):
    if np.ndim(x) == 1:
        return _PY, x.tolist(), np.asarray(u, dtype=float).tolist()
    return _NP, [x[..., i] for i in range(x.shape[-1])], [u[..., j] for j in range(u.shape[-1])]


def _pack(M, vals):
    if M is _PY:
        return np.array(vals, dtype=float)
    return np.stack(np.broadcast_arrays(*vals), axis=-1).astype(float)


def _sig(M, z):
    return 1.0 / (1.0 + M.exp(-z))


def _yunpack(y0):
    if np.ndim(y0) == 1:
        return _PY, np.asarray(y0, dtype=float).tolist()
    return _NP, [y0[..., j] for j in range(y0.shape[-1])]


# ---- model ----
def _flows(M, s, u, th, mech):
    staff, el, da, up, ot, fc = u
    qr, qu, qe, A, W, B, so, fat, prog, r1, r2, thru = s
    se = _smin(M, staff, so, 0.2) if "B" in mech else staff
    eff = (1.0 - th["fat_k"] * fat) if "A" in mech else 1.0
    work = se * (1.0 + th["ot_gain"] * ot) * eff * (1.0 - th["fu_div"] * fc)
    mu_a = da * work * th["rho_a"]
    mu_t = (1.0 - da) * work * th["rho_t"]
    # admissions into assessment chairs, priority weighted
    pu = 0.2 + th["prio_k"] * up
    wsum = qr + pu * qu + qe + 1e-6
    qsum = qr + qu + qe
    adm = _smin(M, qsum, _pos(M, th["chairs"] - A, 0.05), 0.05)
    adm_r = adm * qr / wsum
    adm_u = adm * pu * qu / wsum
    adm_e = adm * qe / wsum
    wmix = (adm_r + th["w_u"] * adm_u + adm_e + 1e-6) / (adm + 1e-6)
    c_a = _smin(M, A / th["tau_a"], mu_a / wmix, 0.02)
    dis = _smin(M, B / th["tau_t"], mu_t, 0.02)
    adm_b = _smin(M, W, _pos(M, th["beds"] - B, 0.05) + dis, 0.02)
    # blocking: a finished assessment holds its chair while the treatment queue is full
    c_a = _smin(M, c_a, _pos(M, th["w_cap"] - W, 0.05) + adm_b, 0.02)
    return qsum, adm, adm_r, adm_u, adm_e, c_a, adm_b, dis


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    staff, el, da, up, ot, fc = uu
    qr, qu, qe, A, W, B, so, fat, prog, r1, r2, thru = s
    qsum, adm, adm_r, adm_u, adm_e, c_a, adm_b, dis = _flows(M, s, uu, th, mech)
    qc = th["q_cap"]
    gate = 1.0 / (1.0 + M.exp((qsum - qc) / (0.08 * qc)))
    ret = 0.0
    d_so = 0.0
    d_fat = 0.0
    d_prog = 0.0
    d_r1 = 0.0
    d_r2 = 0.0
    if "C" in mech:
        enroll = dis * fc * _pos(M, 1.0 - prog / th["prog_cap"], 0.01)
        kr = 2.0 / th["tau_ret"]
        ret = kr * r2
        d_prog = enroll - prog / th["tau_prog"]
        d_r1 = th["ret_frac"] * _pos(M, dis - enroll, 0.001) - kr * r1
        d_r2 = kr * r1 - kr * r2
    if "A" in mech:
        d_fat = (ot - fat) / th["tau_fat"]
    if "B" in mech:
        d_so = (staff - so) / th["tau_hand"]
    d = [
        (th["lam_r"] + ret) * gate - adm_r - th["renege"] * qr,
        th["lam_u"] * gate - adm_u - th["det_u"] * qu,
        th["el_k"] * el * gate - adm_e,
        adm - c_a,
        c_a - adm_b,
        adm_b - dis,
        d_so, d_fat, d_prog, d_r1, d_r2,
        (adm - thru) / th["tau_w"],
    ]
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    qr, qu, qe, A, W, B = s[:6]
    thru = s[11]
    qsum, adm, adm_r, adm_u, adm_e, c_a, adm_b, dis = _flows(M, s, uu, th, mech)
    wait = (qsum + W) / (thru + 0.05)
    return _pack(M, [wait, qsum + A + W + B, dis])


def x0(y0, th, mech):
    M = _PY if np.ndim(y0) == 1 else _NP
    y = y0.tolist() if M is _PY else [y0[..., j] for j in range(y0.shape[-1])]
    q0 = M.max(y[1], 0.0)
    wait0 = M.max(y[0], 0.05)
    qr = th["sh_r"] * q0
    qu = th["sh_u"] * (1.0 - th["sh_r"]) * q0
    qe = q0 - qr - qu
    thru = M.min(M.max(q0 / wait0 - 0.05, 0.01), 1e3)
    z = 0.0 * q0
    return _pack(M, [qr, qu, qe, z, z, z, th["staff0"] + z, z, z, z, z, thru])
