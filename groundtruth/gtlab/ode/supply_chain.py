"""Grey-box ODE: supply_chain. NUMPY + math only (ships verbatim in submissions).

Structure
- Supplier stock in two classes (S1 primary, S2 second). Production effort releases new
  material (capacity-limited), product mix selects the class of NEW production and the
  requested dispatch split. Orders withdraw only available stock (smooth min).
- Rush share of primary-line departures bypasses treatment straight to the conveyor; the
  rest of class 1 and all of class 2 go through treatment, whose rate depends on an
  activity level that decays without maintenance; maintenance also takes treatment time.
- Conveyor = 3-stage delay chain -> receiving terminal (staffed by receiving effort) ->
  retail stock, sold to final customers at a fixed demand rate.
- shipments = arrivals at the retailer + secondary-grade overflow.
Mechanisms: A congested transport + rework loop (rework returns to primary intake,
return-space overflow leaves as secondary grade), B machine heat/wear (slow, reset by
maintenance), C adaptive production commitment (production follows an EMA of orders).
"""
import math

import numpy as np

FAMILY = "supply_chain"
OBS = ["shipments", "inventory_supplier", "inventory_retail"]
CTRL = ["order_quantity", "lead_time_buy", "product_mix", "production_effort",
        "receiving_effort", "maintenance"]
MECHS = {
    "A": "congested transport and rework loop",
    "B": "machine heat/wear lowers treatment rate, reset by maintenance",
    "C": "adaptive production commitments follow an EMA of orders",
}
N_SUB = 2

STATE = ["S1", "S2", "TQ", "C1", "C2", "C3", "G", "R", "act", "RW", "cong", "heat", "commit"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e6)
STATE_HI[8] = 1.0
STATE_HI[10] = 50.0
STATE_HI[11] = 1.0
STATE_HI[12] = 100.0

PARAMS = [
    ("pmax", 20.0, 0.2, 1000.0, True),
    ("scap", 600.0, 10.0, 1e5, True),
    ("sh1", 0.6, 0.02, 0.98, False),
    ("k_disp", 0.5, 0.01, 5.0, True),
    ("rush0", 0.5, 0.0, 1.0, False),
    ("rush_k", -0.4, -1.0, 1.0, False),
    ("mu_t", 25.0, 0.2, 2000.0, True),
    ("tau_act", 60.0, 3.0, 5000.0, True),
    ("k_maint", 0.1, 1e-3, 5.0, True),
    ("maint_time", 0.4, 0.0, 0.95, False),
    ("act0", 1.0, 0.05, 1.0, False),
    ("tau_c", 3.0, 0.5, 50.0, True),
    ("mu_r", 20.0, 0.2, 2000.0, True),
    ("rcap", 500.0, 10.0, 1e5, True),
    ("dem", 10.0, 0.05, 1000.0, True),
    ("rew", 0.1, 0.0, 0.8, False),          # A
    ("cong_k", 1.0, 0.0, 20.0, False),      # A
    ("rw_cap", 30.0, 0.5, 5000.0, True),    # A
    ("tau_cg", 20.0, 1.0, 1000.0, True),    # A
    ("tau_heat", 200.0, 5.0, 1e4, True),    # B
    ("heat_k", 0.5, 0.0, 0.95, False),      # B
    ("tau_com", 100.0, 3.0, 5000.0, True),  # C
    ("com_k", 0.5, 0.0, 3.0, False),        # C
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
    oq, ltb, mix, pe, re, mt = uu
    S1, S2, TQ, C1, C2, C3, G, R, act, RW, cg, ht, cm = s
    prod = pe * th["pmax"] * _pos(M, 1.0 - (S1 + S2) / th["scap"], 1e-3)
    if "C" in mech:
        prod = prod * _pos(M, 1.0 + th["com_k"] * (cm / 40.0 - 0.5), 1e-3)
    kd = th["k_disp"]
    d1 = _pos(M, _smin(M, oq * (1.0 - mix), kd * S1, 0.01), 1e-4)
    d2 = _pos(M, _smin(M, oq * mix, kd * S2, 0.01), 1e-4)
    rs = M.min(M.max(th["rush0"] + th["rush_k"] * ltb, 0.0), 1.0)
    trate = th["mu_t"] * act * (1.0 - th["maint_time"] * mt)
    if "B" in mech:
        trate = trate * (1.0 - th["heat_k"] * ht)
    treat = _pos(M, _smin(M, TQ, trate, 0.01), 1e-4)
    tc = th["tau_c"]
    if "A" in mech:
        tc = tc * (1.0 + 0.2 * cg)
    arr = C3 / tc
    recv = _pos(M, _smin(M, G, re * th["mu_r"] * _pos(M, 1.0 - R / th["rcap"], 1e-3), 0.01), 1e-4)
    rew_in = 0.0
    second = 0.0
    if "A" in mech:
        rflow = treat * M.min(th["rew"] * (1.0 + th["cong_k"] * cg), 0.95)
        rew_in = rflow * _pos(M, 1.0 - RW / th["rw_cap"], 1e-3)
        rew_in = M.min(rew_in, rflow)
        second = rflow - rew_in
    return prod, d1, d2, rs, treat, tc, arr, recv, rew_in, second


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    oq, ltb, mix, pe, re, mt = uu
    S1, S2, TQ, C1, C2, C3, G, R, act, RW, cg, ht, cm = s
    prod, d1, d2, rs, treat, tc, arr, recv, rew_in, second = _alg(M, s, uu, th, mech)
    good = treat - rew_in - second
    rw_out = RW / 5.0
    d = [0.0] * _NS
    d[0] = prod * (1.0 - mix) - d1
    d[1] = prod * mix - d2
    d[2] = d1 * (1.0 - rs) + d2 + rw_out - treat
    d[3] = d1 * rs + good - C1 / tc
    d[4] = (C1 - C2) / tc
    d[5] = (C2 - C3) / tc
    d[6] = arr - recv
    d[7] = recv - _smin(M, th["dem"], 0.5 * R, 0.01)
    d[8] = -act / th["tau_act"] + th["k_maint"] * mt * (1.0 - act)
    if "A" in mech:
        d[9] = rew_in - rw_out
        d[10] = ((C1 + C2 + C3) / (3.0 * th["tau_c"] * th["mu_t"] + 1e-6) * 5.0 - cg) / th["tau_cg"]
    if "B" in mech:
        d[11] = (pe / 1.5 - ht) / th["tau_heat"] - mt * ht / 20.0
    if "C" in mech:
        d[12] = (oq - cm) / th["tau_com"]
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    prod, d1, d2, rs, treat, tc, arr, recv, rew_in, second = _alg(M, s, uu, th, mech)
    return _pack(M, [recv + second, s[0] + s[1], s[7]])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    sup = M.max(y[1], 0.0)
    ret = M.max(y[2], 0.0)
    z = 0.0 * sup
    # internal buffers and conveyors start empty; stocks split into fixed class shares
    return _pack(M, [th["sh1"] * sup, (1.0 - th["sh1"]) * sup, z, z, z, z, z, ret,
                     th["act0"] + z, z, z, z, z])
