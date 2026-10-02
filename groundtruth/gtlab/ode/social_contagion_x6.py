"""Grey-box ODE: social_contagion x6 (x4 with a finite deliberative audience for organic adoption and
no workforce ceiling: the fits of x, x2, x4 put the ceiling far above every observed total). NUMPY + math only (ships verbatim).

Per community i in {a, b}:
  pool P_i = (N_i - A_i - W_i - D_i)_+, A_i = L_i + M_i + X_i observed, W_i = all waiting stages.
  interest rho_i = K_i [s_i s (1 - k_br beta)_+ + q A_i/N_i + o], r_i = rho_i/(1 + rho_i/1.5).
  A share phi(c) = c/(c + 0.7) of new interest joins with a paid promise (incentive-led cohort).
  Consideration: two Erlang stages C1 -> C2 at rate 2/tau_c (unpromised) and the same for promised
  cohorts (C1p, C2p); then the onboarding queue Q (Qp promised) served by a finite workforce.
  Onboarding queue Q drains at 1/tau_q (no binding workforce ceiling).
  Organic adoption comes from a finite deliberative audience: o (1 - A_i/G_i)_+, so with outreach
  stopped the community settles near G_i instead of filling the whole population.
  Waiting cohorts leave the queue at 1/tau_w into the disappointed pool D.
  Onboarded unpromised -> L (relationship-led), promised -> M (incentive-led).
  Paid offer converts L -> M at k_conv c. Churn: L at c_L, X (initial incentive-led members without
  a promise) at k_X, M at c_L + k_B (E - c)_+. Leavers -> D, back to the pool at 1/tau_D.
  x4: returning disappointed people reconsider: a share p_re re-enters the consideration chain
  directly (unpromised), the rest rejoins the pool. The paid offer draws an incentive-led audience:
  interest is multiplied by (1 + k_inc phi(c)).
A credibility: dK = (1 - K)/50 - k_A K (promised waiting)/N.
B incentive expectations: dE = (c - E)/tau_E.
C cross-community ties: not modelled.
"""
import math

import numpy as np

FAMILY = "social_contagion_x6"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility erodes with promised waiting cohorts",
         "B": "incentive expectations: incentive-led members leave when incentive falls below memory",
         "C": "cross-community ties (not modelled here)"}
N_SUB = 2

_PER = ["L", "M", "X", "D", "C1", "C2", "Q", "C1p", "C2p", "Qp", "K"]
STATE = [s + "a" for s in _PER] + [s + "b" for s in _PER] + ["E"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array(([1e4] * 10 + [1.0]) * 2 + [2.0])

R_MAX = 1.5
PHI_HALF = 0.7
TAU_K = 50.0

PARAMS = [
    ("N_a", 380.0, 60.0, 3000.0, True),
    ("N_b", 350.0, 40.0, 3000.0, True),
    ("s_a", 0.003, 1e-4, 1.0, True),
    ("s_b", 0.0008, 1e-5, 1.0, True),
    ("q", 0.015, 1e-3, 1.5, True),
    ("o", 0.001, 1e-5, 0.1, True),
    ("tau_c", 5.0, 0.5, 60.0, True),
    ("tau_q", 3.0, 0.2, 30.0, True),
    ("G_a", 90.0, 20.0, 3000.0, True),
    ("G_b", 80.0, 20.0, 3000.0, True),
    ("c_L", 0.012, 1e-4, 0.5, True),
    ("k_X", 0.12, 0.005, 1.5, True),
    ("m0", 0.3, 0.0, 0.9, False),
    ("k_conv", 0.3, 0.0, 0.7, False),
    ("k_A", 0.1, 0.0, 3.0, False),
    ("k_B", 0.045, 0.0, 0.7, False),
    ("tau_E", 30.0, 1.0, 500.0, True),
    ("k_br", 0.3, 0.0, 1.0, False),
    ("tau_D", 15.0, 1.0, 1000.0, True),
    ("p_re", 0.3, 0.0, 1.0, False),
    ("k_inc", 0.3, 0.0, 3.0, False),
]


class _PY:
    max = max
    min = min

    @staticmethod
    def sqrt(z):
        return math.sqrt(z) if z > 0.0 else 0.0


class _NP:
    max = np.maximum
    min = np.minimum

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


def _community(M, s, N, G, seed_term, th, inc, phi, chM, g, mech):
    L, Mm, X, D, C1, C2, Q, C1p, C2p, Qp, K = s
    A = L + Mm + X
    W = C1 + C2 + Q + C1p + C2p + Qp
    pool = _pos(M, N - A - W - D, 1.0)
    raw = seed_term + th["q"] * A / N + th["o"] * _pos(M, 1.0 - A / G, 0.02)
    if "A" in mech:
        raw = raw * K
    raw = raw * (1.0 + th["k_inc"] * phi)
    r = raw / (1.0 + raw / R_MAX)
    new = r * pool
    back = D / th["tau_D"]
    kc = 2.0 / th["tau_c"]
    ab = 0.0
    onb = g * Q
    onbp = g * Qp
    conv = th["k_conv"] * inc * L
    outL = th["c_L"] * L
    outM = chM * Mm
    outX = th["k_X"] * X
    dC1 = (1.0 - phi) * new + th["p_re"] * back - kc * C1
    dC2 = kc * (C1 - C2)
    dQ = kc * C2 - onb - ab * Q
    dC1p = phi * new - kc * C1p
    dC2p = kc * (C1p - C2p)
    dQp = kc * C2p - onbp - ab * Qp
    dL = onb - outL - conv
    dM = onbp + conv - outM
    dX = -outX
    dD = outL + outM + outX + ab * (Q + Qp) - back
    if "A" in mech:
        dK = (1.0 - K) / TAU_K - th["k_A"] * K * (C1p + C2p + Qp) / N
    else:
        dK = 0.0 * K
    return [dL, dM, dX, dD, dC1, dC2, dQ, dC1p, dC2p, dQp, dK]


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    n = len(_PER)
    sa, sb, E = s[:n], s[n:2 * n], s[2 * n]
    chM = th["c_L"] + 0.0 * E
    dE = 0.0 * E
    if "B" in mech:
        chM = th["c_L"] + th["k_B"] * _pos(M, E - inc, 0.05)
        dE = (inc - E) / th["tau_E"]
    phi = inc / (inc + PHI_HALF)
    Atot = sa[0] + sa[1] + sa[2] + sb[0] + sb[1] + sb[2]
    Qtot = sa[6] + sa[9] + sb[6] + sb[9]
    g = 1.0 / th["tau_q"] + 0.0 * Atot + 0.0 * Qtot
    eff = seed * _pos(M, 1.0 - th["k_br"] * br, 0.01)
    da = _community(M, sa, th["N_a"], th["G_a"], th["s_a"] * eff, th, inc, phi, chM, g, mech)
    db = _community(M, sb, th["N_b"], th["G_b"], th["s_b"] * eff, th, inc, phi, chM, g, mech)
    return _pack(M, da + db + [dE])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    n = len(_PER)
    return _pack(M, [s[0] + s[1] + s[2], s[n] + s[n + 1] + s[n + 2]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    z = 0.0 * a + 0.0 * m0
    one = z + 1.0
    return _pack(M, [(1.0 - m0) * a, z, m0 * a, z, z, z, z, z, z, z, one,
                     (1.0 - m0) * b, z, m0 * b, z, z, z, z, z, z, z, one, z])
