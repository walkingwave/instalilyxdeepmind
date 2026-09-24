"""Grey-box ODE: ad_auction. NUMPY + math only (ships verbatim in submissions).

Structure
- Auction: win = sigmoid(k * (bid - rival price)) * bid/(bid + 0.05).
- Audiences: narrow core (always targeted) and a broad segment targeted in proportion to
  (breadth - 0.1)/0.9. Reachable fraction = 1 - fatigued - cooling down.
- spend = min(budget_cap, win * bid * c_imp * opportunities)  (hard cap, <= budget_cap).
- Won impressions start purchases (committed) that wait for shared fulfillment work;
  broad-audience purchases need more work. Completed purchases = conversions; converted
  customers cool down before becoming reachable again.
Mechanisms: A rival capital shift (rival price follows our spend), B exposure fatigue
(impressions temporarily remove reachable people), C broad-introduction priming (broad
exposure raises later narrow conversion).
"""
import math

import numpy as np

FAMILY = "ad_auction"
OBS = ["win_rate", "spend", "conversions"]
CTRL = ["bid", "budget_cap", "targeting_breadth"]
MECHS = {
    "A": "rival campaigns move a shared capital pool: rival price follows our spend",
    "B": "repeated exposure temporarily removes reachable people",
    "C": "broad introduction primes later follow-up conversion",
}
N_SUB = 1

STATE = ["Fn", "Fb", "Cn", "Cb", "prime", "Qn", "Qb", "rival"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1.0)
STATE_HI[5] = 1e6
STATE_HI[6] = 1e6
STATE_HI[7] = 100.0

PARAMS = [
    ("r0", 1.5, 0.05, 20.0, True),
    ("k_win", 2.0, 0.05, 30.0, True),
    ("reach", 100.0, 0.5, 1e6, True),
    ("c_imp", 0.1, 1e-4, 10.0, True),
    ("n_share", 0.5, 0.05, 0.95, False),
    ("p_n", 0.02, 1e-5, 0.5, True),
    ("p_b", 0.005, 1e-6, 0.5, True),
    ("fcap", 3.0, 0.02, 1000.0, True),
    ("work_b", 2.0, 0.2, 20.0, True),
    ("tau_f", 2.0, 0.5, 50.0, True),
    ("tau_cd", 30.0, 1.0, 1000.0, True),
    ("pop_n", 1000.0, 5.0, 1e7, True),
    ("pop_b", 3000.0, 5.0, 1e7, True),
    ("tau_A", 50.0, 2.0, 2000.0, True),     # A
    ("kA", 1.0, 0.0, 20.0, False),          # A
    ("fat_k", 0.5, 0.0, 20.0, False),       # B
    ("tau_fat", 20.0, 1.0, 1000.0, True),   # B
    ("prime_k", 2.0, 0.0, 30.0, False),     # C
    ("tau_pr", 50.0, 2.0, 2000.0, True),    # C
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
def _alg(M, s, uu, th, mech):
    bid, cap, br = uu
    Fn, Fb, Cn, Cb, P, Qn, Qb, rv = s
    rp = rv if "A" in mech else th["r0"]
    win = _sig(M, th["k_win"] * (bid - rp)) * bid / (bid + 0.05)
    wb = M.max((br - 0.1) / 0.9, 0.0)
    an = _pos(M, 1.0 - Fn - Cn, 1e-4)
    ab = _pos(M, 1.0 - Fb - Cb, 1e-4)
    ns = th["n_share"]
    opp_n = th["reach"] * ns * an
    opp_b = th["reach"] * (1.0 - ns) * wb * ab
    opp = opp_n + opp_b
    want = win * bid * th["c_imp"] * opp
    spend = M.min(_pos(M, _smin(M, want, cap, 0.01), 1e-4), cap)
    frac = spend / (want + 1e-9)
    imp_n = win * opp_n * frac
    imp_b = win * opp_b * frac
    wtot = Qn + th["work_b"] * Qb
    rate = _pos(M, _smin(M, wtot / th["tau_f"], th["fcap"], 0.01), 1e-5)
    conv_n = rate * Qn / (wtot + 1e-6)
    conv_b = rate * Qb / (wtot + 1e-6)
    return win, spend, imp_n, imp_b, conv_n, conv_b, an, ab


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    bid, cap, br = uu
    Fn, Fb, Cn, Cb, P, Qn, Qb, rv = s
    win, spend, imp_n, imp_b, conv_n, conv_b, an, ab = _alg(M, s, uu, th, mech)
    pn = th["p_n"] * (1.0 + th["prime_k"] * P) if "C" in mech else th["p_n"]
    d = [0.0] * _NS
    tcd = th["tau_cd"]
    d[2] = conv_n / th["pop_n"] - Cn / tcd
    d[3] = conv_b / th["pop_b"] - Cb / tcd
    d[5] = imp_n * pn - conv_n
    d[6] = imp_b * th["p_b"] - conv_b
    if "A" in mech:
        d[7] = (th["r0"] * (1.0 + th["kA"] * spend / 100.0) - rv) / th["tau_A"]
    if "B" in mech:
        d[0] = th["fat_k"] * imp_n / th["pop_n"] * an - Fn / th["tau_fat"]
        d[1] = th["fat_k"] * imp_b / th["pop_b"] * ab - Fb / th["tau_fat"]
    if "C" in mech:
        d[4] = imp_b / th["pop_b"] * (1.0 - P) - P / th["tau_pr"]
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    win, spend, imp_n, imp_b, conv_n, conv_b, an, ab = _alg(M, s, uu, th, mech)
    return _pack(M, [win, spend, conv_n + conv_b])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    z = 0.0 * y[0]
    # people available and unprepared, no pending purchases or exposure recovery
    return _pack(M, [z, z, z, z, z, z, z, th["r0"] + z])
