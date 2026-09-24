"""Grey-box ODE: epidemic. NUMPY + math only (ships verbatim in submissions).

Structure
- Three age groups (children, adults, elderly), SEIR fractions per group; a fixed contact
  matrix scaled by beta. School closure removes (compliance-weighted) school contacts
  from the child row and moves part of them into child-adult home contacts; masks scale
  all transmission.
- Hospital: severity-weighted referrals from I, finite beds (smooth min), overflow
  waits on a waiting list that drains as beds free (or leaves). hospital_load = beds used.
- Vaccination moves S -> R at rate v * clinic availability, clinic = 1/(1 + k*H/hcap)
  (hospital pressure reduces clinic workforce).
- daily_cases = onsets = pop * sum_i n_i * sigma * E_i.
Mechanisms: A behavioural fatigue (compliance decays with an EMA of restriction), B waning
immunity (R -> S), C postponed gatherings (restriction accumulates a debt released later).
"""
import math

import numpy as np

FAMILY = "epidemic"
OBS = ["daily_cases", "hospital_load"]
CTRL = ["school_closure", "mask_mandate", "vaccination_rate"]
MECHS = {
    "A": "behavioural fatigue: compliance decays with cumulative restriction",
    "B": "developing/waning immunity: R returns to S",
    "C": "postponed gatherings: restriction builds a debt that raises contacts on release",
}
N_SUB = 2
NAGE = np.array([0.2, 0.55, 0.25])          # population shares
CBASE = [[3.0, 1.5, 0.5], [1.5, 2.0, 0.6], [0.5, 0.6, 1.0]]
EMIX = [0.3, 0.55, 0.15]                      # fixed age mix of initial exposed onsets

STATE = ([f"S{i}" for i in range(3)] + [f"E{i}" for i in range(3)] + [f"I{i}" for i in range(3)]
         + [f"R{i}" for i in range(3)] + ["H", "WL", "fatigue", "debt"])
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1.0)
STATE_HI[12] = 1e7
STATE_HI[13] = 1e7
STATE_HI[15] = 1e4

PARAMS = [
    ("pop", 1e6, 1e4, 1e9, True),
    ("beta", 0.07, 0.003, 2.0, True),
    ("sigma", 0.25, 0.05, 1.5, True),
    ("gam_c", 0.2, 0.02, 1.5, True),
    ("gam_a", 0.15, 0.02, 1.5, True),
    ("gam_e", 0.1, 0.02, 1.5, True),
    ("sev_c", 0.002, 1e-5, 0.5, True),
    ("sev_a", 0.01, 1e-5, 0.5, True),
    ("sev_e", 0.06, 1e-5, 0.8, True),
    ("los", 8.0, 1.0, 200.0, True),
    ("hcap", 500.0, 5.0, 1e6, True),
    ("tau_wl", 2.0, 0.5, 50.0, True),
    ("wl_leave", 0.1, 0.0, 1.0, False),
    ("school", 0.6, 0.0, 0.95, False),
    ("close_eff", 0.8, 0.0, 1.0, False),
    ("mask_eff", 0.4, 0.0, 0.9, False),
    ("k_clinic", 1.0, 0.0, 20.0, False),
    ("imm0", 0.3, 0.0, 0.95, False),
    ("tau_fat", 200.0, 10.0, 5000.0, True),   # A
    ("fat_k", 0.5, 0.0, 0.95, False),         # A
    ("tau_wane", 400.0, 20.0, 1e4, True),     # B
    ("debt_k", 0.5, 0.0, 5.0, False),         # C
    ("tau_debt", 30.0, 2.0, 1000.0, True),    # C
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
def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    clo, msk, vac = uu
    S = s[0:3]
    E = s[3:6]
    I = s[6:9]
    R = s[9:12]
    H, WL, fat, debt = s[12], s[13], s[14], s[15]
    comp = (1.0 - th["fat_k"] * fat) if "A" in mech else 1.0
    cl = clo * comp
    red = th["close_eff"] * th["school"] * cl
    mfac = 1.0 - th["mask_eff"] * msk * comp
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
    sev = [th["sev_c"], th["sev_a"], th["sev_e"]]
    sg = th["sigma"]
    pop = th["pop"]
    hcap = th["hcap"]
    clinic = 1.0 / (1.0 + th["k_clinic"] * H / hcap)
    stot = NAGE[0] * S[0] + NAGE[1] * S[1] + NAGE[2] * S[2]
    vrate = vac * clinic / (stot + 0.02)
    wane = (1.0 / th["tau_wane"]) if "B" in mech else 0.0
    d = [0.0] * _NS
    ref = 0.0
    for i in range(3):
        inf = lam[i] * S[i]
        v = vrate * S[i]
        d[i] = -inf - v + wane * R[i]
        d[3 + i] = inf - sg * E[i]
        d[6 + i] = sg * E[i] - gam[i] * I[i]
        d[9 + i] = gam[i] * I[i] + v - wane * R[i]
        ref = ref + NAGE[i] * sev[i] * gam[i] * I[i]
    ref = ref * pop
    demand = ref + WL / th["tau_wl"]
    room = _pos(M, hcap - H, 0.01 * hcap) + H / th["los"]
    adm = _smin(M, demand, room, 0.01 * hcap)
    d[12] = adm - H / th["los"]
    d[13] = ref - adm - th["wl_leave"] * WL
    if "A" in mech:
        d[14] = (restr - fat) / th["tau_fat"]
    if "C" in mech:
        d[15] = restr - (1.0 - restr) * debt / th["tau_debt"] - 0.002 * debt
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
    I = [M.min(E[i] * th["sigma"] / gam[i], 0.3) for i in range(3)]
    R = [th["imm0"] + 0.0 * cases] * 3
    S = [_pos(M, 1.0 - E[i] - I[i] - R[i], 1e-4) for i in range(3)]
    z = 0.0 * cases
    return _pack(M, S + E + I + R + [M.max(y[1], 0.0), z, z, z])
