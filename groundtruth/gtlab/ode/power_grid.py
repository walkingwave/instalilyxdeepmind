"""Grey-box ODE: power_grid. NUMPY + math only (ships verbatim in submissions).

Structure
- Thermostatic cooling loads: a ring of K "off/warming" stages and K "on/cooling" stages
  (Erlang ring => damped oscillatory rebound after synchronisation). Price lengthens the
  off period (lower duty cycle) and a filtered setpoint `sp` produces a switching burst when
  price moves (price up -> running loads switch off early; price down -> warm loads switch on).
- Base (non-thermostatic) demand, price elastic, plus a decaying anomaly fixed by the
  initial observed load.
- Supply: renewables delivered through the interconnector (curtailed at ic * capacity),
  reserve battery (power + SOC limits, charging draws grid power), conventional units:
  a slow scheduler tracking net demand and a governor with droop around it.
- Frequency deviation: swing-like balance of supply - demand.
Mechanisms: A interconnector thermal derating, B reserve thermal limit, C load
heterogeneity (backward dispersion inside the ring).
"""
import math

import numpy as np

FAMILY = "power_grid"
OBS = ["load", "frequency", "renewable_share"]
CTRL = ["price_signal", "reserve_dispatch", "charging_allowance", "interconnector"]
MECHS = {
    "A": "interconnector thermal derating (capacity falls with EMA of flow)",
    "B": "reserve thermal/duration limit (reserve power falls with EMA of use)",
    "C": "thermal heterogeneity (dispersion damps TCL synchronisation)",
}
N_SUB = 2
K = 4          # stages per half of the thermostat ring
P_REF = 0.8    # reference price at reset

STATE = ([f"off{i}" for i in range(K)] + [f"on{i}" for i in range(K)] +
         ["sp", "dem_anom", "ren_anom", "soc", "g_sched", "g_conv", "dfreq", "th_ic", "th_res"])
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, np.inf)
STATE_HI[:2 * K] = 1.0
STATE_LO[2 * K + 1] = -1e5       # demand anomaly
STATE_LO[2 * K + 2] = -1e5       # renewable anomaly
STATE_HI[2 * K + 3] = 1.0        # soc
STATE_HI[2 * K + 4] = 1e5
STATE_HI[2 * K + 5] = 1e5
STATE_LO[2 * K + 6] = -5.0       # frequency deviation, Hz
STATE_HI[2 * K + 6] = 5.0
STATE_HI[2 * K + 7] = 2.0
STATE_HI[2 * K + 8] = 2.0

PARAMS = [
    # thermostatic population
    ("tau_off", 14.0, 4.0, 200.0, True),
    ("tau_on", 9.0, 4.0, 200.0, True),
    ("k_price", 0.5, 0.0, 4.0, False),       # price lengthens off period
    ("k_shift", 0.3, 0.0, 3.0, False),       # setpoint-shift switching burst
    ("tau_set", 3.0, 0.5, 100.0, True),
    ("p_tcl", 300.0, 5.0, 5000.0, True),
    # base demand
    ("base_load", 500.0, 5.0, 10000.0, True),
    ("elast", 0.1, 0.0, 0.45, False),
    ("tau_dem", 20.0, 1.0, 3000.0, True),
    # supply
    ("ren_avail", 200.0, 1.0, 5000.0, True),
    ("tau_ren", 30.0, 1.0, 3000.0, True),
    ("ic_cap", 300.0, 5.0, 5000.0, True),
    ("res_pmax", 150.0, 5.0, 1000.0, True),
    ("res_energy", 2000.0, 20.0, 1e5, True),
    ("charge_max", 30.0, 0.5, 1000.0, True),
    ("tau_sched", 8.0, 1.0, 200.0, True),
    ("tau_gov", 2.0, 1.0, 30.0, True),
    ("gmax", 3000.0, 20.0, 20000.0, True),
    ("k_droop", 1000.0, 20.0, 50000.0, True),
    ("inertia", 2.0, 1.0, 50.0, True),
    ("damp_rel", 0.2, 0.0, 2.0, False),
    ("f_nom", 50.0, 45.0, 65.0, False),
    # mechanisms
    ("tau_ic", 300.0, 10.0, 5000.0, True),    # A
    ("derate", 0.5, 0.0, 0.95, False),        # A
    ("tau_rth", 60.0, 5.0, 3000.0, True),     # B
    ("rth_k", 0.6, 0.0, 0.95, False),         # B
    ("disp", 0.15, 0.0, 0.8, False),          # C
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
def _tau_off_eff(th, p):
    kp = th["k_price"]
    return th["tau_off"] * (1.0 + kp * p) / (1.0 + kp * P_REF)


def _algebra(M, s, u, th, mech):
    """Shared algebraic quantities (flows, supply, demand) given state list s."""
    p, rd, ca, ic = u[0], u[1], u[2], u[3]
    ontot = s[K] + s[K + 1] + s[K + 2] + s[K + 3]
    soc = s[2 * K + 3]
    charge = th["charge_max"] * ca * (1.0 - soc)
    demand = th["base_load"] * (1.0 - th["elast"] * (p - P_REF)) + s[2 * K + 1] + th["p_tcl"] * ontot + charge
    demand = _pos(M, demand, 1.0)
    pm = th["res_pmax"]
    if "B" in mech:
        pm = pm * (1.0 - th["rth_k"] * M.min(s[2 * K + 8], 1.0))
    pm = pm * _smin(M, 1.0, 20.0 * soc, 0.05)
    pr = _pos(M, _smin(M, rd, pm, 1.0), 0.01)
    cap = ic * th["ic_cap"]
    if "A" in mech:
        cap = cap * (1.0 - th["derate"] * M.min(s[2 * K + 7], 1.0))
    ren = _pos(M, th["ren_avail"] + s[2 * K + 2], 1.0)
    ren_used = _pos(M, _smin(M, ren, cap, 2.0), 0.01)
    return demand, charge, pr, ren_used


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    p = uu[0]
    demand, charge, pr, ren_used = _algebra(M, s, uu, th, mech)
    r_off = K / _tau_off_eff(th, p)
    r_on = K / th["tau_on"]
    sh = th["k_shift"] * (p - s[2 * K])
    sh_up = _pos(M, sh, 1e-3)       # price rose: running loads switch off
    sh_dn = _pos(M, -sh, 1e-3)      # price fell: warm idle loads switch on
    off = s[:K]
    on = s[K:2 * K]
    d = [0.0] * _NS
    out_on = [sh_up * on[i] * (i + 1) / K for i in range(K)]
    out_off = [sh_dn * off[i] * (i + 1) / K for i in range(K)]
    burst_to_off = out_on[0] + out_on[1] + out_on[2] + out_on[3]
    burst_to_on = out_off[0] + out_off[1] + out_off[2] + out_off[3]
    # forward circulation
    for i in range(K):
        inflow = (r_on * on[K - 1] + burst_to_off) if i == 0 else r_off * off[i - 1]
        d[i] = inflow - r_off * off[i] - out_off[i]
    for i in range(K):
        inflow = (r_off * off[K - 1] + burst_to_on) if i == 0 else r_on * on[i - 1]
        d[K + i] = inflow - r_on * on[i] - out_on[i]
    if "C" in mech:
        bo = th["disp"] * r_off
        bn = th["disp"] * r_on
        for i in range(1, K):
            fo = bo * off[i]
            d[i] -= fo
            d[i - 1] += fo
            fn = bn * on[i]
            d[K + i] -= fn
            d[K + i - 1] += fn
    d[2 * K] = (p - s[2 * K]) / th["tau_set"]
    d[2 * K + 1] = -s[2 * K + 1] / th["tau_dem"]
    d[2 * K + 2] = -s[2 * K + 2] / th["tau_ren"]
    d[2 * K + 3] = (charge - pr) / th["res_energy"]
    net = demand - ren_used - pr
    d[2 * K + 4] = (net - s[2 * K + 4]) / th["tau_sched"]
    target = s[2 * K + 4] - th["k_droop"] * s[2 * K + 6]
    target = _smin(M, _pos(M, target, 1.0), th["gmax"], 1.0)
    d[2 * K + 5] = (target - s[2 * K + 5]) / th["tau_gov"]
    supply = s[2 * K + 5] + ren_used + pr
    kd = th["k_droop"]
    d[2 * K + 6] = (supply - demand - th["damp_rel"] * kd * s[2 * K + 6]) / (th["inertia"] * kd)
    if "A" in mech:
        d[2 * K + 7] = (ren_used / th["ic_cap"] - s[2 * K + 7]) / th["tau_ic"]
    if "B" in mech:
        d[2 * K + 8] = (pr / th["res_pmax"] - s[2 * K + 8]) / th["tau_rth"]
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    demand, charge, pr, ren_used = _algebra(M, s, uu, th, mech)
    freq = th["f_nom"] + s[2 * K + 6]
    share = ren_used / (ren_used + s[2 * K + 5] + pr + 1e-6)
    return _pack(M, [demand, freq, share])


def x0(y0, th, mech):
    M = _PY if np.ndim(y0) == 1 else _NP
    y = y0.tolist() if M is _PY else [y0[..., j] for j in range(y0.shape[-1])]
    load = _pos(M, y[0], 1.0)
    share = M.min(M.max(y[2], 0.0), 0.99)
    toff = _tau_off_eff(th, P_REF)
    cyc = toff + th["tau_on"]
    s = [toff / (K * cyc)] * K + [th["tau_on"] / (K * cyc)] * K
    ontot = th["tau_on"] / cyc
    s.append(P_REF + 0.0 * load)
    s.append(load - th["base_load"] - th["p_tcl"] * ontot)
    ren0 = share * load
    s.append(ren0 - th["ren_avail"])
    s.append(1.0 + 0.0 * load)
    g0 = _pos(M, load - ren0, 1.0)
    s.append(g0)
    s.append(g0)
    s.append(M.min(M.max(y[1] - th["f_nom"], -2.0), 2.0))
    s.append(0.0 * load)
    s.append(0.0 * load)
    return _pack(M, s)
