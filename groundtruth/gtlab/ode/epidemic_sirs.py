"""Grey-box ODE: epidemic, minimal SIRS + hospital. NUMPY + math only (ships verbatim).

One population, SEIRS fractions, hospital beds as a first-order stock fed by a fixed fraction of
recoveries. Interventions: school closure and masks each scale transmission multiplicatively;
vaccination moves S -> R at vac_eff * vaccination_rate, reduced by hospital pressure
(clinic = 1 / (1 + k_clinic * H / h_ref)). Waning immunity R -> S at 1 / tau_wane gives the
endemic plateau seen after every wave in the data.

Reset convention: I0, E0 from the observed initial cases (E0 = cases / (pop * sigma), I0 in
quasi-steady ratio sigma / gamma), R0 = imm0 (fixed), H0 = observed hospital_load.

Mechanism letters are accepted for interface compatibility but every term is always active;
fit with pairs=("AB",).
"""
import math

import numpy as np

FAMILY = "epidemic_sirs"
OBS = ["daily_cases", "hospital_load"]
CTRL = ["school_closure", "mask_mandate", "vaccination_rate"]
MECHS = {"A": "waning immunity (always on)", "B": "hospital pressure on clinics (always on)",
         "C": "unused"}
N_SUB = 2

STATE = ["S", "E", "I", "R", "H"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array([1.0, 1.0, 1.0, 1.0, 1e6])

PARAMS = [
    ("pop", 2.0e5, 1e3, 1e8, True),
    ("beta", 0.5, 0.02, 5.0, True),
    ("sigma", 0.3, 0.05, 2.0, True),
    ("gamma", 0.2, 0.02, 1.5, True),
    ("tau_wane", 150.0, 10.0, 5000.0, True),
    ("close_eff", 0.4, 0.0, 0.95, False),
    ("mask_eff", 0.4, 0.0, 0.95, False),
    ("vac_eff", 1.0, 0.0, 30.0, False),
    ("k_clinic", 1.0, 0.0, 20.0, False),
    ("h_ref", 100.0, 1.0, 1e5, True),
    ("hfrac", 0.01, 1e-5, 0.9, True),
    ("los", 20.0, 1.0, 300.0, True),
    ("imm0", 0.3, 0.0, 0.95, False),
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


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    clo, msk, vac = uu
    S, E, I, R, H = s
    b = th["beta"] * (1.0 - th["close_eff"] * clo) * (1.0 - th["mask_eff"] * msk)
    lam = b * I
    clinic = 1.0 / (1.0 + th["k_clinic"] * H / th["h_ref"])
    v = th["vac_eff"] * vac * clinic
    wane = 1.0 / th["tau_wane"]
    inf = lam * S
    dS = -inf - v * S + wane * R
    dE = inf - th["sigma"] * E
    dI = th["sigma"] * E - th["gamma"] * I
    dR = th["gamma"] * I + v * S - wane * R
    dH = th["hfrac"] * th["pop"] * th["gamma"] * I - H / th["los"]
    return _pack(M, [dS, dE, dI, dR, dH])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    cases = th["pop"] * th["sigma"] * s[1]
    return _pack(M, [cases, s[4]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    cases = M.max(y[0], 0.0)
    E = M.min(cases / (th["pop"] * th["sigma"]), 0.5)
    I = M.min(E * th["sigma"] / th["gamma"], 0.5)
    R = th["imm0"] + 0.0 * cases
    S = _pos(M, 1.0 - E - I - R, 1e-4)
    H = M.max(y[1], 0.0)
    return _pack(M, [S, E, I, R, H])
