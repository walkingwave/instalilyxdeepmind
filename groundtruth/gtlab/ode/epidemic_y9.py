"""Grey-box ODE: epidemic y9 = y7 with the clinic-pressure bound of y6 (k_clinic <= 20).

epidemic y7 = y2 with a curved mask response: exposure x (1 - mask_eff m^mask_q).
Masks at 0.85 cut the infection rate to about x0.70 (p3, p4); mask 1 alone (p6) holds the first
wave lower than a straight line through that point predicts (x11 held out on p6 peaks 10 %
high). mask_q > 1 makes the last part of the mandate count more (compliance of the last
holdouts), mask_q < 1 less. m^q is computed as exp(q log m) with m floored at 1e-9.

epidemic y2 = x11 with a single latent stage and a two-tick reporting delay.

Every switch in every run shows the same shape: the next two observations stay exactly on the old
trend, then the slope changes at once and stays changed (p6 vaccination-only switch at t=100:
39.2, 39.0, then +3.1, +4.1, +3.5, +3.5; p1 start: 188.0, 187.8, then +30.6, +36.9, +34.1). An
Erlang-2 latent stage bends the slope over two to three ticks instead. Here onsets leave one
exponential latent stage (so the onset slope turns at the switch) and are reported through a
chain of NDLY stages with mean DLY = 2 ticks (a near-pure delay), all started at the observed
cases. Everything else is x11.
"""
import math

import numpy as np

FAMILY = "epidemic_y9"
OBS = ["daily_cases", "hospital_load"]
CTRL = ["school_closure", "mask_mandate", "vaccination_rate"]
MECHS = {
    "A": "behavioural fatigue: compliance decays with cumulative restriction",
    "B": "developing immunity: vaccinated turn immune after a delay (waning always on)",
    "C": "postponed gatherings: restriction builds a debt that raises contacts on release",
}
N_SUB = 3
NDLY = 6
DLY = 2.0
NAGE = np.array([0.2, 0.55, 0.25])
CBASE = [[3.0, 1.5, 0.5], [1.5, 2.0, 0.6], [0.5, 0.6, 1.0]]
EMIX = [0.3, 0.55, 0.15]

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
    ("pop", 22000.0, 1e3, 1e8, True),
    ("beta", 0.22, 0.003, 2.0, True),
    ("sigma", 0.13, 0.03, 1.5, True),
    ("gam_c", 0.26, 0.02, 2.0, True),
    ("gam_a", 0.30, 0.02, 2.0, True),
    ("gam_e", 1.0, 0.02, 3.0, True),
    ("sev_a", 0.07, 1e-5, 0.8, True),
    ("sev_e", 0.06, 1e-5, 0.8, True),
    ("los", 13.0, 1.0, 200.0, True),
    ("hcap", 155.0, 5.0, 1e6, True),
    ("school", 0.95, 0.0, 1.0, False),
    ("mask_eff", 0.45, 0.0, 0.95, False),
    ("k_clinic", 1.5, 0.0, 20.0, False),
    ("v_eld", 1.0, 0.0, 30.0, False),
    ("r_I", 0.5, 0.0, 2.0, False),
    ("tau_p", 3.0, 0.5, 30.0, True),
    ("tau_fat", 130.0, 10.0, 5000.0, True),   # A
    ("fat_k", 0.54, 0.0, 1.0, False),        # A
    ("tau_wane", 115.0, 20.0, 1e4, True),
    ("vac_eff", 1.5, 0.1, 3.0, True),
    ("debt_k", 0.5, 0.0, 5.0, False),         # C
    ("tau_debt", 30.0, 2.0, 1000.0, True),    # C
    ("mask_q", 1.0, 0.4, 3.0, True),
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
    cl = clo * comp
    red = th["school"] * cl
    mfac = 1.0 - th["mask_eff"] * M.exp(th["mask_q"] * M.log(M.max(msk, 1e-9))) * comp
    restr = 0.5 * (clo + msk)
    boost = 1.0
    if "C" in mech:
        boost = 1.0 + th["debt_k"] * (1.0 - restr) * debt / th["tau_debt"]
    b = th["beta"] * mfac * boost
    c00 = CBASE[0][0] * (1.0 - red)
    c01 = CBASE[0][1] + 0.3 * CBASE[0][0] * red
    lam0 = b * (c00 * I[0] + c01 * I[1] + CBASE[0][2] * I[2])
    lam1 = b * (c01 * I[0] + CBASE[1][1] * I[1] + CBASE[1][2] * I[2])
    lam2 = b * (CBASE[2][0] * I[0] + CBASE[2][1] * I[1] + CBASE[2][2] * I[2])
    lam = [lam0, lam1, lam2]
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
    kE = sg
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
