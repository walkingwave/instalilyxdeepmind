"""Grey-box ODE: social_contagion. NUMPY + math only (ships verbatim in submissions).

Structure (communities a, b)
- Susceptible = N - interested - members - refractory.
- Interest (Bass-like): seeding outreach (local share 1 - bridge, cross introductions
  bridge) + local word of mouth q*M/N + cross-community influence; incentive multiplies.
- Interested people wait for onboarding by a workforce also needed by existing members
  (capacity w_cap - w_mem*members, smooth min); long waits drop out disappointed.
- Members churn into a refractory pool that returns to susceptible after tau_ref.
Mechanisms: A credibility (dropouts from unkept promises lower adoption), B incentive
expectations (churn rises when the incentive falls below its EMA), C cross-community ties
grow with bridge usage.
"""
import math

import numpy as np

FAMILY = "social_contagion"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {
    "A": "credibility: unkept onboarding promises lower later adoption",
    "B": "incentive expectations: removing an incentive raises churn",
    "C": "cross-community relationships grow with bridge outreach",
}
N_SUB = 1

STATE = ["Ia", "Ib", "Ma", "Mb", "Da", "Db", "cred", "expect", "ties"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e8)
STATE_HI[6] = 1.0
STATE_HI[7] = 2.0
STATE_HI[8] = 1.0

PARAMS = [
    ("N_a", 5000.0, 50.0, 1e8, True),
    ("N_b", 3000.0, 50.0, 1e8, True),
    ("p_seed", 5e-4, 1e-7, 0.05, True),
    ("q_a", 0.008, 1e-5, 2.0, True),
    ("q_b", 0.008, 1e-5, 2.0, True),
    ("cross", 0.05, 0.0, 1.0, False),
    ("inc_k", 0.5, 0.0, 10.0, False),
    ("w_cap", 50.0, 0.2, 1e5, True),
    ("w_mem", 0.005, 0.0, 0.2, False),
    ("tau_on", 3.0, 0.5, 100.0, True),
    ("drop", 0.02, 0.0, 0.5, False),
    ("churn", 0.005, 1e-5, 0.3, True),
    ("tau_ref", 60.0, 3.0, 2000.0, True),
    ("mix_b", 1.0, 0.0, 5.0, False),
    ("cred_k", 5.0, 0.0, 200.0, False),     # A
    ("tau_cred", 50.0, 3.0, 2000.0, True),  # A
    ("tau_ex", 30.0, 2.0, 1000.0, True),    # B
    ("ex_k", 2.0, 0.0, 30.0, False),        # B
    ("tie_k", 0.05, 0.0, 1.0, False),       # C
    ("tau_tie", 200.0, 5.0, 5000.0, True),  # C
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
    seed, inc, br = uu
    Ia, Ib, Ma, Mb, Da, Db, cr, ex, tb = s
    Na, Nb = th["N_a"], th["N_b"]
    Sa = _pos(M, Na - Ia - Ma - Da, 1e-3)
    Sb = _pos(M, Nb - Ib - Mb - Db, 1e-3)
    mtot = Ma + Mb + 1.0
    loc = 0.5 * seed * (1.0 - br)
    xin = th["mix_b"] * seed * br
    cx = th["cross"] + (tb if "C" in mech else 0.0)
    lam_a = th["p_seed"] * (loc + xin * Mb / mtot) + th["q_a"] * (Ma / Na + cx * (0.5 + br) * Mb / Nb)
    lam_b = th["p_seed"] * (loc + xin * Ma / mtot) + th["q_b"] * (Mb / Nb + cx * (0.5 + br) * Ma / Na)
    g = 1.0 + th["inc_k"] * inc
    if "A" in mech:
        g = g * cr
    cap = _pos(M, th["w_cap"] - th["w_mem"] * (Ma + Mb), 0.01 * th["w_cap"])
    itot = Ia + Ib + 1e-6
    onb = _smin(M, itot / th["tau_on"], cap, 0.01)
    onb = _pos(M, onb, 1e-5)
    oa = onb * Ia / itot
    ob = onb * Ib / itot
    ch = th["churn"]
    if "B" in mech:
        ch = ch * (1.0 + th["ex_k"] * _pos(M, ex - inc, 0.01))
    dr = th["drop"]
    tr = th["tau_ref"]
    d = [0.0] * _NS
    d[0] = lam_a * g * Sa - oa - dr * Ia
    d[1] = lam_b * g * Sb - ob - dr * Ib
    d[2] = oa - ch * Ma
    d[3] = ob - ch * Mb
    d[4] = ch * Ma + dr * Ia - Da / tr
    d[5] = ch * Mb + dr * Ib - Db / tr
    if "A" in mech:
        d[6] = (1.0 - cr) / th["tau_cred"] - th["cred_k"] * dr * itot / (th["w_cap"] * th["tau_cred"]) * cr
    if "B" in mech:
        d[7] = (inc - ex) / th["tau_ex"]
    if "C" in mech:
        d[8] = th["tie_k"] * br * (1.0 - tb) - tb / th["tau_tie"]
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    return _pack(M, [s[2], s[3]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    # members from the report; queues empty, full credibility, no incentive expectation
    return _pack(M, [z, z, M.max(y[0], 0.0), M.max(y[1], 0.0), z, z, 1.0 + z, z, z])
