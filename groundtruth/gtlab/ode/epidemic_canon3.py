"""Grey-box ODE: epidemic canon3 = canon1 code with the additive control reading: beta max(0, 1 - close_eff closure - mask_eff mask). Notes of canon1 follow.
(canon1 base description:) age-structured SEIR-H simulator coded the textbook way.

Contacts are a sum of settings, C = HOME + (1 - closure) SCHOOL + WORK + COMMUNITY, with fixed
textbook shapes and free setting weights (w_home, w_school, w_comm; WORK is the unit). Masks act
outside the home only: every non-home setting is scaled by (1 - mask_eff mask). School closure
removes the school setting and moves a share `displace` of the children's school contacts into the
community setting (postponed play dates, childcare). Force of infection per group
lam_i = beta sum_j C_ij I_j (I_j = infectious fraction of group j).

Disease: S -> E -> I -> R per age group, age-specific recovery; severe cases are referred to
hospital through a referral stage (mean tau_p), wait in a queue when beds are full
(admission = min(demand, free beds + discharges)), stay los. Vaccination doses come from a clinic
workforce shared with the hospital: throughput vac_eff vac / (1 + k_clinic H / hcap), elders first
(weight 1 + v_eld); vaccinated susceptibles become immune (mechanism B: after TAU_DEV ticks).
Reported daily cases = symptom onsets passed through a two-tick reporting delay.

Mechanisms: A behavioural fatigue (compliance decays with the time average of restrictions),
B developing immunity, C postponed gatherings (restriction builds a debt released on reopening).

Variant switches (module constants) let the same code test other literal readings:
CLOSE_MODE "layer" (this file) | "mult" (beta (1 - close_eff closure) on every contact)
MASK_MODE "nonhome" (this file) | "all"; COMBINE "mult" (this file) | "add"
(beta (1 - close_eff closure - mask_eff mask), floored at 0).
"""
import math

import numpy as np

FAMILY = "epidemic_canon3"
CLOSE_MODE = "mult"
MASK_MODE = "all"
COMBINE = "add"
OBS = ["daily_cases", "hospital_load"]
CTRL = ["school_closure", "mask_mandate", "vaccination_rate"]
MECHS = {
    "A": "behavioural fatigue: compliance decays with cumulative restriction",
    "B": "developing immunity: vaccinated turn immune after a delay",
    "C": "postponed gatherings: restriction builds a debt that raises contacts on release",
}
N_SUB = 2
NDLY = 6
DLY = 2.0
NAGE = np.array([0.2, 0.55, 0.25])
EMIX = [0.3, 0.55, 0.15]
# textbook setting shapes (children, adults, elders); rows = infectee, cols = infector
HOME = [[1.0, 1.0, 0.3], [1.0, 1.0, 0.3], [0.3, 0.3, 1.0]]
SCHOOL = [[1.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
WORK = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
COMM = [[0.5, 0.5, 0.3], [0.5, 1.0, 0.5], [0.3, 0.5, 0.5]]

STATE = ([f"S{i}" for i in range(3)] + [f"E{i}" for i in range(3)] + [f"I{i}" for i in range(3)]
         + [f"R{i}" for i in range(3)] + ["H", "WL", "fatigue", "debt", "P"] + [f"V{i}" for i in range(3)] + ["P2"])
STATE = STATE + [f"D{k}" for k in range(NDLY)]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1.0)
STATE_HI[12] = 1e7
STATE_HI[13] = 1e7
STATE_HI[15] = 1e4
STATE_HI[16] = 1e7
STATE_HI[20] = 1e7
STATE_HI[21:] = 1e7

PARAMS = [
    ("pop", 20000.0, 1e3, 1e8, True),
    ("beta", 0.10, 0.003, 2.0, True),
    ("sigma", 0.2, 0.03, 1.5, True),
    ("gam_c", 0.25, 0.02, 2.0, True),
    ("gam_a", 0.25, 0.02, 2.0, True),
    ("gam_e", 0.5, 0.02, 3.0, True),
    ("sev_a", 0.07, 1e-5, 0.8, True),
    ("sev_e", 0.06, 1e-5, 0.8, True),
    ("los", 13.0, 1.0, 200.0, True),
    ("hcap", 155.0, 5.0, 1e6, True),
    ("w_home", 1.0, 0.0, 10.0, False),
    ("w_school", 3.0, 0.0, 20.0, False),
    ("w_comm", 1.0, 0.0, 10.0, False),
    ("displace", 0.3, 0.0, 1.0, False),
    ("mask_eff", 0.5, 0.0, 0.95, False),
    ("close_eff", 0.3, 0.0, 0.95, False),
    ("k_clinic", 1.5, 0.0, 5.0, False),
    ("v_eld", 1.0, 0.0, 30.0, False),
    ("r_I", 0.5, 0.0, 2.0, False),
    ("tau_p", 3.0, 0.5, 30.0, True),
    ("tau_fat", 130.0, 10.0, 5000.0, True),   # A
    ("fat_k", 0.5, 0.0, 1.0, False),         # A
    ("tau_wane", 115.0, 20.0, 1e4, True),
    ("vac_eff", 1.5, 0.1, 3.0, True),
    ("debt_k", 0.5, 0.0, 5.0, False),         # C
    ("tau_debt", 30.0, 2.0, 1000.0, True),    # C
]
TAU_WL = 1.0
TAU_DEV = 10.0
WL_LEAVE = 0.05
CAP_EPS = 0.002


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


def _yunpack(y0):
    if np.ndim(y0) == 1:
        return _PY, np.asarray(y0, dtype=float).tolist()
    return _NP, [y0[..., j] for j in range(y0.shape[-1])]


def _contacts(M, clo, msk, th, boost):
    """Effective contact matrix (3x3 list) times beta."""
    b = th["beta"] * boost
    C = [[0.0] * 3 for _ in range(3)]
    if CLOSE_MODE == "layer":
        wh, ws, wc = th["w_home"], th["w_school"], th["w_comm"]
        if MASK_MODE == "nonhome":
            mf = 1.0 - th["mask_eff"] * msk
        else:
            mf = 1.0
        disp = th["displace"] * ws * clo
        for i in range(3):
            for j in range(3):
                out = ws * (1.0 - clo) * SCHOOL[i][j] + WORK[i][j] + wc * COMM[i][j]
                if i == 0 and j == 0:
                    out = out + disp
                C[i][j] = wh * HOME[i][j] + mf * out
        g = 1.0 if MASK_MODE == "nonhome" else 1.0 - th["mask_eff"] * msk
        return [[b * g * C[i][j] for j in range(3)] for i in range(3)]
    if COMBINE == "add":
        g = _pos(M, 1.0 - th["close_eff"] * clo - th["mask_eff"] * msk, 1e-3)
    else:
        g = (1.0 - th["close_eff"] * clo) * (1.0 - th["mask_eff"] * msk)
    wh, ws, wc = th["w_home"], th["w_school"], th["w_comm"]
    for i in range(3):
        for j in range(3):
            C[i][j] = b * g * (wh * HOME[i][j] + ws * SCHOOL[i][j] + WORK[i][j] + wc * COMM[i][j])
    return C


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    clo, msk, vac = uu
    S = s[0:3]
    E = s[3:6]
    I = s[6:9]
    R = s[9:12]
    H, WL, fat, debt, P = s[12], s[13], s[14], s[15], s[16]
    V = s[17:20]
    P2 = s[20]
    comp = (1.0 - th["fat_k"] * fat) if "A" in mech else 1.0
    restr = 0.5 * (clo + msk)
    boost = 1.0
    if "C" in mech:
        boost = 1.0 + th["debt_k"] * (1.0 - restr) * debt / th["tau_debt"]
    C = _contacts(M, clo * comp, msk * comp, th, boost)
    lam = [C[i][0] * I[0] + C[i][1] * I[1] + C[i][2] * I[2] for i in range(3)]
    gam = [th["gam_c"], th["gam_a"], th["gam_e"]]
    sev = [0.0, th["sev_a"], th["sev_e"]]
    kE = th["sigma"]
    pop = th["pop"]
    hcap = th["hcap"]
    clinic = 1.0 / (1.0 + th["k_clinic"] * H / hcap)
    vrate = th["vac_eff"] * vac * clinic
    vw = [1.0, 1.0, 1.0 + th["v_eld"]]
    wane = 1.0 / th["tau_wane"]
    dev = "B" in mech
    d = [0.0] * _NS
    ref = 0.0
    for i in range(3):
        inf = lam[i] * S[i]
        v = vrate * vw[i] * S[i]
        d[i] = -inf - v + wane * R[i]
        if dev:
            infv = lam[i] * V[i]
            conv = V[i] / TAU_DEV
            d[17 + i] = v - infv - conv
            d[3 + i] = inf + infv - kE * E[i]
            d[9 + i] = gam[i] * I[i] + conv - wane * R[i]
        else:
            d[3 + i] = inf - kE * E[i]
            d[9 + i] = gam[i] * I[i] + v - wane * R[i]
        d[6 + i] = kE * E[i] - gam[i] * I[i]
        ref = ref + NAGE[i] * sev[i] * gam[i] * I[i]
    ref = ref * pop
    kp = 2.0 / th["tau_p"]
    arr = kp * P2
    demand = arr + WL / TAU_WL
    room = _pos(M, hcap - H, CAP_EPS * hcap) + H / th["los"]
    adm = _smin(M, demand, room, CAP_EPS * hcap)
    d[12] = adm - H / th["los"]
    d[13] = arr - adm - WL_LEAVE * WL
    if "A" in mech:
        d[14] = (restr - fat) / th["tau_fat"]
    if "C" in mech:
        d[15] = restr - (1.0 - restr) * debt / th["tau_debt"] - 0.002 * debt
    d[16] = ref - kp * P
    d[20] = kp * P - arr
    kd = NDLY / DLY
    ons = pop * kE * (NAGE[0] * E[0] + NAGE[1] * E[1] + NAGE[2] * E[2])
    d[21] = kd * (ons - s[21])
    for k in range(1, NDLY):
        d[21 + k] = kd * (s[20 + k] - s[21 + k])
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[20 + NDLY], s[12]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    cases = M.max(y[0], 0.0)
    kE = th["sigma"]
    etot = cases / (th["pop"] * kE)
    gam = [th["gam_c"], th["gam_a"], th["gam_e"]]
    E = [M.min(etot * EMIX[i] / NAGE[i], 0.3) for i in range(3)]
    I = [M.min(th["r_I"] * E[i] * kE / gam[i], 0.3) for i in range(3)]
    R = [0.0 * cases] * 3
    S = [_pos(M, 1.0 - E[i] - I[i] - R[i], 1e-4) for i in range(3)]
    z = 0.0 * cases
    return _pack(M, S + E + I + R + [M.max(y[1], 0.0), z, z, z, z, z, z, z, z] + [cases] * NDLY)
