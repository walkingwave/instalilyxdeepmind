"""Grey-box ODE: social_contagion y5 (= y + bridge introductions: effort_i = s (s_i (1 - beta) + s_xi beta)) (brief rebuild, audience-segmented pools). NUMPY + math only.

Per community i in {a, b} (other community j):
  Two audience pools with fixed shares: incentive-led f_I N_i and the rest (relationship-led and
  deliberative) (1 - f_I) N_i. Each pool keeps its own members, waiting cohorts and disappointed.
  drive_i = s_i s (1 - beta) + q A_i/N_i  [+ C: q_x R A_j/N_j]
  relationship/deliberative interest rho_U = K (drive_i + o) -> unpromised cohorts W1..W3 -> L
  incentive-led interest rho_I = K k_inc phi(c) (drive_i + o) -> promised cohorts V1..V3 -> MI
  r = rho/(1 + rho/1.5); new = r x pool.
  Onboarding: Erlang-3, mean tau_on, times g (WF = 1: shared workforce net of members' needs,
  g = F_on/(F_on + h_Q Q_tot), F_on = (F_w - A_a - A_b)_+; WF = 0: g = 1).
  The paid offer converts L -> MU at k_conv c (relationship members that take a promise).
  Churn: L at c_L; MU and MI at c_L + c_off (1 - phi(c)/phi(2)) [+ B: k_B (E - c)_+];
  X (initial incentive-led members, no promise) at k_X. Leavers -> DU / DI, back after tau_D.
A credibility: dK = (1 - K)/50 - k_A K (promised waiting)/N.
B incentive expectations: dE = (c - E)/tau_E.
C cross-community relationships: dR = (s beta/10 - R)/tau_R, cross word of mouth q_x R A_j/N_j.
"""
import math

import numpy as np

FAMILY = "social_contagion_y5"
OBS = ["adopters_a", "adopters_b"]
CTRL = ["seeding", "incentive", "bridge_outreach"]
MECHS = {"A": "credibility erodes with promised waiting cohorts",
         "B": "incentive expectations: promised members leave when incentive falls below memory",
         "C": "cross-community relationships built by introductions carry word of mouth"}
N_SUB = 2
WF = 0

_PER = ["L", "MU", "MI", "X", "DU", "DI", "W1", "W2", "W3", "V1", "V2", "V3", "K"]
STATE = [s + "a" for s in _PER] + [s + "b" for s in _PER] + ["E", "R"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.array(([1e4] * 12 + [1.0]) * 2 + [2.0, 1.0])

R_MAX = 1.5
PHI_HALF = 0.7
PHI_2 = 2.0 / 2.7
TAU_K = 50.0

PARAMS = [
    ("N_a", 400.0, 60.0, 3000.0, True),
    ("N_b", 350.0, 40.0, 3000.0, True),
    ("s_a", 0.003, 1e-4, 1.0, True),
    ("s_b", 0.0009, 1e-5, 1.0, True),
    ("q", 0.015, 1e-3, 1.5, True),
    ("o", 0.0015, 1e-5, 0.1, True),
    ("tau_on", 9.0, 1.0, 60.0, True),
    ("c_L", 0.015, 1e-4, 0.5, True),
    ("k_X", 0.05, 0.005, 1.5, True),
    ("m0", 0.5, 0.0, 1.0, False),
    ("k_conv", 0.05, 0.0, 0.7, False),
    ("k_A", 0.1, 0.0, 3.0, False),
    ("k_B", 0.05, 0.0, 0.7, False),
    ("tau_E", 25.0, 1.0, 500.0, True),
    ("tau_D", 10.0, 1.0, 1000.0, True),
    ("k_inc", 1.0, 0.01, 10.0, True),
    ("f_Ia", 0.4, 0.02, 0.98, False),
    ("f_Ib", 0.4, 0.02, 0.98, False),
    ("c_off", 0.03, 0.0, 0.5, False),
    ("q_x", 0.01, 0.0, 1.0, False),
    ("tau_R", 30.0, 1.0, 1000.0, True),
    ("s_xa", 0.0005, 1e-6, 1.0, True),
    ("s_xb", 0.001, 1e-6, 1.0, True),
]
if WF:
    PARAMS = PARAMS + [("F_w", 500.0, 150.0, 5000.0, True), ("h_Q", 1.0, 1e-3, 100.0, True)]


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


def _community(M, s, N, fI, drive, th, inc, phi, g, ch, mech):
    L, MU, MI, X, DU, DI, W1, W2, W3, V1, V2, V3, K = s
    A = L + MU + MI + X
    poolU = _pos(M, N * (1.0 - fI) - L - MU - W1 - W2 - W3 - DU, 1.0)
    poolI = _pos(M, N * fI - MI - X - V1 - V2 - V3 - DI, 1.0)
    base = drive + th["q"] * A / N + th["o"]
    if "A" in mech:
        base = base * K
    rhoU = base
    rhoI = base * th["k_inc"] * phi
    newU = rhoU / (1.0 + rhoU / R_MAX) * poolU
    newI = rhoI / (1.0 + rhoI / R_MAX) * poolI
    k = 3.0 * g / th["tau_on"]
    conv = th["k_conv"] * inc * L
    oL = th["c_L"] * L
    oMU = ch * MU
    oMI = ch * MI
    oX = th["k_X"] * X
    dW1 = newU - k * W1
    dW2 = k * (W1 - W2)
    dW3 = k * (W2 - W3)
    dV1 = newI - k * V1
    dV2 = k * (V1 - V2)
    dV3 = k * (V2 - V3)
    dL = k * W3 - oL - conv
    dMU = conv - oMU
    dMI = k * V3 - oMI
    dX = -oX
    dDU = oL + oMU - DU / th["tau_D"]
    dDI = oMI + oX - DI / th["tau_D"]
    if "A" in mech:
        dK = (1.0 - K) / TAU_K - th["k_A"] * K * (V1 + V2 + V3) / N
    else:
        dK = 0.0 * K
    return [dL, dMU, dMI, dX, dDU, dDI, dW1, dW2, dW3, dV1, dV2, dV3, dK], A, W1 + W2 + W3 + V1 + V2 + V3


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    seed, inc, br = uu
    n = len(_PER)
    sa, sb, E, R = s[:n], s[n:2 * n], s[2 * n], s[2 * n + 1]
    phi = inc / (inc + PHI_HALF)
    ch = th["c_L"] + th["c_off"] * (1.0 - phi / PHI_2)
    dE = 0.0 * E
    if "B" in mech:
        ch = ch + th["k_B"] * _pos(M, E - inc, 0.05)
        dE = (inc - E) / th["tau_E"]
    Aa = sa[0] + sa[1] + sa[2] + sa[3]
    Ab = sb[0] + sb[1] + sb[2] + sb[3]
    loc = seed * _pos(M, 1.0 - br, 0.01)
    dra = th["s_a"] * loc + th["s_xa"] * seed * br
    drb = th["s_b"] * loc + th["s_xb"] * seed * br
    dR = 0.0 * R
    if "C" in mech:
        dra = dra + th["q_x"] * R * Ab / th["N_b"]
        drb = drb + th["q_x"] * R * Aa / th["N_a"]
        dR = (seed * br / 10.0 - R) / th["tau_R"]
    if WF:
        Q = sum(sa[6:12]) + sum(sb[6:12])
        Fon = _pos(M, th["F_w"] - Aa - Ab, 1.0)
        g = Fon / (Fon + th["h_Q"] * Q + 1e-9)
    else:
        g = 1.0
    da, _, _ = _community(M, sa, th["N_a"], th["f_Ia"], dra, th, inc, phi, g, ch, mech)
    db, _, _ = _community(M, sb, th["N_b"], th["f_Ib"], drb, th, inc, phi, g, ch, mech)
    return _pack(M, da + db + [dE, dR])


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    n = len(_PER)
    return _pack(M, [s[0] + s[1] + s[2] + s[3], s[n] + s[n + 1] + s[n + 2] + s[n + 3]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    a = M.max(y[0], 0.0)
    b = M.max(y[1], 0.0)
    m0 = th["m0"]
    z = 0.0 * a + 0.0 * m0
    one = z + 1.0
    return _pack(M, [(1.0 - m0) * a, z, z, m0 * a, z, z, z, z, z, z, z, z, one,
                     (1.0 - m0) * b, z, z, m0 * b, z, z, z, z, z, z, z, z, one, z, z])
