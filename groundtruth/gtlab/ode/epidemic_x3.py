"""Grey-box ODE: epidemic x3 = x where behaviour (A) keeps two traces of restriction history:
fatigue lowers compliance while restricted (fat_k), and learned caution lowers community and
school contacts at all times, so it persists after release (hab_k).

epidemic x: rebuild of v8g around where contacts happen). NUMPY + math only.

Contacts are split by setting: home, school, community. School closure removes a share of
school contacts and relocates a share rho of them into homes (children with children and with
adults). Masks act fully on school and community contacts and only a fraction mask_home at
home. Postponed gatherings (C) boost community contacts only. Everything else as v8g:
SEIR per age group with waning, per-capita vaccination through a clinic whose availability
falls with bed pressure, Erlang-2 referral pipeline, beds with a waiting list, fatigue (A),
developing immunity (B), gathering debt (C).
"""
import math

import numpy as np

FAMILY = "epidemic_x3"
OBS = ["daily_cases", "hospital_load"]
CTRL = ["school_closure", "mask_mandate", "vaccination_rate"]
MECHS = {
    "A": "behavioural fatigue: compliance decays with cumulative restriction",
    "B": "developing immunity: vaccinated turn immune after a delay (waning always on)",
    "C": "postponed gatherings: restriction builds a debt that raises community contacts on release",
}
N_SUB = 2
NAGE = np.array([0.2, 0.55, 0.25])
HOME = [[0.8, 0.9, 0.2], [0.9, 0.8, 0.3], [0.2, 0.3, 0.6]]
SCHOOL = [[2.0, 0.1, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 0.0]]
COMM = [[0.2, 0.5, 0.3], [0.5, 1.1, 0.3], [0.3, 0.3, 0.4]]
EMIX = [0.3, 0.55, 0.15]

STATE = ([f"S{i}" for i in range(3)] + [f"E{i}" for i in range(3)] + [f"I{i}" for i in range(3)]
         + [f"R{i}" for i in range(3)] + ["H", "WL", "fatigue", "debt", "P"] + [f"V{i}" for i in range(3)] + ["P2"])
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1.0)
STATE_HI[12] = 1e7
STATE_HI[13] = 1e7
STATE_HI[15] = 1e4
STATE_HI[16] = 1e7
STATE_HI[20] = 1e7

PARAMS = [
    ("pop", 18730.0, 1e3, 1e8, True),
    ("beta", 0.2726, 0.003, 2.0, True),
    ("sigma", 0.158, 0.03, 1.5, True),
    ("gam_c", 0.4638, 0.02, 2.0, True),
    ("gam_a", 0.5842, 0.02, 2.0, True),
    ("gam_e", 0.2628, 0.02, 3.0, True),
    ("sev_a", 0.1015, 1e-5, 0.8, True),
    ("sev_e", 0.1152, 1e-5, 0.8, True),
    ("los", 9.246, 1.0, 200.0, True),
    ("hcap", 154.62, 5.0, 1e6, True),
    ("school", 0.99, 0.0, 1.0, False),
    ("mask_eff", 0.3644, 0.0, 0.95, False),
    ("k_clinic", 15.0, 0.0, 40.0, False),
    ("v_eld", 5.267, 0.0, 30.0, False),
    ("r_I", 0.401, 0.0, 2.0, False),
    ("tau_p", 4.855, 0.5, 30.0, True),
    ("tau_fat", 250.8, 10.0, 5000.0, True),   # A
    ("fat_k", 0.8539, 0.0, 1.0, False),       # A
    ("tau_wane", 91.52, 20.0, 1e4, True),
    ("vac_eff", 4.239, 0.1, 300.0, True),
    ("debt_k", 0.11, 0.0, 5.0, False),        # C
    ("tau_debt", 119.5, 2.0, 1000.0, True),   # C
    ("home_k", 1.0, 0.1, 5.0, True),
    ("rho", 0.5, 0.0, 2.0, False),
    ("mask_home", 0.3, 0.0, 1.0, False),
    ("hab_k", 0.05, 0.0, 1.0, False),         # A
]
TAU_WL = 1.0
TAU_DEV = 10.0
WL_LEAVE = 0.05
CAP_EPS = 0.002

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
def _lam(M, I, clo, msk, comp, th, boost, hab):
    c = th["school"] * clo * comp
    mm = th["mask_eff"] * msk * comp
    mo = (1.0 - mm) * hab
    mh = 1.0 - th["mask_home"] * mm
    hk = th["home_k"]
    rr = th["rho"] * c
    b = th["beta"]
    lam = []
    for i in range(3):
        home = hk * (HOME[i][0] * I[0] + HOME[i][1] * I[1] + HOME[i][2] * I[2])
        if i == 0:
            home = home + rr * (0.8 * I[0] + 0.6 * I[1])
        elif i == 1:
            home = home + rr * 0.6 * I[0]
        sch = (1.0 - c) * (SCHOOL[i][0] * I[0] + SCHOOL[i][1] * I[1] + SCHOOL[i][2] * I[2])
        com = boost * (COMM[i][0] * I[0] + COMM[i][1] * I[1] + COMM[i][2] * I[2])
        lam.append(b * (mh * home + mo * (sch + com)))
    return lam


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
    hab = (1.0 - th["hab_k"] * fat) if "A" in mech else 1.0
    lam = _lam(M, I, clo, msk, comp, th, boost, hab)
    gam = [th["gam_c"], th["gam_a"], th["gam_e"]]
    sev = [0.0, th["sev_a"], th["sev_e"]]
    sg = th["sigma"]
    pop = th["pop"]
    hcap = th["hcap"]
    clinic = 1.0 / (1.0 + th["k_clinic"] * H / hcap)
    we = 1.0 + th["v_eld"]
    vrate = th["vac_eff"] * vac * clinic
    vw = [1.0, 1.0, we]
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
            d[3 + i] = inf + infv - sg * E[i]
            d[9 + i] = gam[i] * I[i] + conv - wane * R[i]
        else:
            d[3 + i] = inf - sg * E[i]
            d[9 + i] = gam[i] * I[i] + v - wane * R[i]
        d[6 + i] = sg * E[i] - gam[i] * I[i]
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
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    cases = th["pop"] * th["sigma"] * (NAGE[0] * s[3] + NAGE[1] * s[4] + NAGE[2] * s[5])
    return _pack(M, [cases, s[12]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    cases = M.max(y[0], 0.0)
    etot = cases / (th["pop"] * th["sigma"])
    gam = [th["gam_c"], th["gam_a"], th["gam_e"]]
    E = [M.min(etot * EMIX[i] / NAGE[i], 0.3) for i in range(3)]
    I = [M.min(th["r_I"] * E[i] * th["sigma"] / gam[i], 0.3) for i in range(3)]
    R = [0.0 * cases] * 3
    S = [_pos(M, 1.0 - E[i] - I[i] - R[i], 1e-4) for i in range(3)]
    z = 0.0 * cases
    return _pack(M, S + E + I + R + [M.max(y[1], 0.0), z, z, z, z, z, z, z, z])
