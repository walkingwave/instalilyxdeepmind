"""Grey-box ODE: reservoir. NUMPY + math only (ships verbatim in submissions).

Structure (tick = 1 day)
- Season: a harmonic oscillator (cos, sin) with a fixed phase at reset drives river
  inflow, nutrient load and light. Inflow = seasonal base + decaying anomaly from the
  initial reading (+ delayed return flow under B).
- Volume integrator: dV = inflow - outflow - evaporation - spill(V > vmax).
- outflow = smooth min(release + irrigation, fouled outlet capacity, available water).
- Two water layers: surface pollutant index PS (algae), deep pollutant index PD (anoxic
  release driven by an oxygen deficit OD from decomposition). Aeration mixes layers and
  re-oxygenates. Outlet quality = exp(-((1-w)*PS + w*PD)), w = withdrawal depth.
Mechanisms: A fouling biomass restricting discharge (removed by aeration and flushing),
B groundwater/irrigation return of water + contaminants after a delay, C sediment store
remobilised into the surface by aeration.
"""
import math

import numpy as np

FAMILY = "reservoir"
OBS = ["level", "inflow", "outflow", "quality"]
CTRL = ["release_rate", "irrigation_allocation", "withdrawal_depth", "aeration"]
MECHS = {
    "A": "fouling biomass restricts actual discharge",
    "B": "groundwater/irrigated-land return of water and contaminants after a delay",
    "C": "deposited sediment remobilised by aeration",
}
N_SUB = 1

STATE = ["V", "cs", "sn", "ain", "NU", "AL", "PS", "PD", "OD", "BF", "R1", "R2", "SD"]
_NS = len(STATE)
STATE_LO = np.zeros(_NS)
STATE_HI = np.full(_NS, 1e6)
STATE_LO[1] = -1.5
STATE_HI[1] = 1.5
STATE_LO[2] = -1.5
STATE_HI[2] = 1.5
STATE_LO[3] = -1e4
STATE_HI[4] = 1e3
STATE_HI[5] = 1e3
STATE_HI[6] = 50.0
STATE_HI[7] = 50.0
STATE_HI[8] = 1.0
STATE_HI[9] = 1.0
STATE_HI[12] = 1e3

PARAMS = [
    ("i0", 5.0, 0.05, 500.0, True),
    ("i_amp", 0.5, 0.0, 0.99, False),
    ("phase", 0.0, -3.2, 3.2, False),
    ("period", 365.0, 250.0, 450.0, False),
    ("tau_ain", 10.0, 1.0, 1000.0, True),
    ("vmax", 1000.0, 20.0, 1e6, True),
    ("evap", 0.001, 0.0, 0.02, False),
    ("omax", 20.0, 0.5, 1000.0, True),
    ("nu_in", 0.1, 1e-3, 10.0, True),
    ("g_al", 0.15, 1e-3, 0.9, True),
    ("al_m", 0.05, 1e-3, 0.9, True),
    ("mix0", 0.01, 0.0, 0.5, False),
    ("aer_mix", 0.1, 0.0, 0.9, False),
    ("q_al", 0.3, 0.0, 30.0, False),
    ("ps_dec", 0.02, 1e-3, 0.5, True),
    ("od_k", 0.3, 0.0, 5.0, False),
    ("od_re", 0.05, 0.0, 0.9, False),
    ("pd_k", 0.05, 0.0, 2.0, False),
    ("flush", 1.0, 0.0, 20.0, False),
    ("q0_deep", 1.2, 0.2, 5.0, True),      # reference profile: deep/surface pollutant ratio
    ("bf_k", 0.5, 0.0, 10.0, False),       # A
    ("bf_rm", 0.05, 0.0, 1.0, False),      # A
    ("foul_k", 0.5, 0.0, 0.95, False),     # A
    ("ret_f", 0.3, 0.0, 1.0, False),       # B
    ("tau_ret", 20.0, 2.0, 300.0, True),   # B
    ("ret_c", 0.5, 0.0, 20.0, False),      # B
    ("sd_k", 0.5, 0.0, 10.0, False),       # C
    ("rem_k", 0.05, 0.0, 0.9, False),      # C
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
    rel, irr, wd, aer = uu
    V, cs, sn, ain, NU, AL, PS, PD, OD, BF, R1, R2, SD = s
    base = th["i0"] * (1.0 + th["i_amp"] * cs)
    ret = (2.0 * R2 / th["tau_ret"]) if "B" in mech else 0.0
    inflow = _pos(M, base + ain + ret, 1e-3)
    cap = th["omax"]
    if "A" in mech:
        cap = cap * (1.0 - th["foul_k"] * BF)
    req = rel + irr
    out = _smin(M, _smin(M, req, cap, 0.05), 0.5 * V, 0.05)
    out = _pos(M, out, 1e-4)
    qual = M.exp(-((1.0 - wd) * PS + wd * PD))
    return base, ret, inflow, out, qual


def f(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    rel, irr, wd, aer = uu
    V, cs, sn, ain, NU, AL, PS, PD, OD, BF, R1, R2, SD = s
    base, ret, inflow, out, qual = _alg(M, s, uu, th, mech)
    w = 2.0 * math.pi / th["period"] if M is _PY else 2.0 * np.pi / th["period"]
    Vs = V + 1.0
    req = rel + irr + 1e-6
    out_surf = out * ((1.0 - wd) * rel + irr) / req
    out_deep = out - out_surf
    light = 1.0 + 0.5 * sn
    grow = th["g_al"] * light * AL * NU / (NU + 1.0)
    mix = th["mix0"] + th["aer_mix"] * aer
    d = [0.0] * _NS
    d[0] = inflow - out - th["evap"] * V - _pos(M, V - th["vmax"], 0.01 * th["vmax"])
    d[1] = -w * sn
    d[2] = w * cs
    d[3] = -ain / th["tau_ain"]
    d[4] = th["nu_in"] * base - grow - NU * inflow / Vs
    d[5] = grow - th["al_m"] * AL - AL * out_surf / Vs + 0.001
    fl = th["flush"] * 100.0 / Vs
    dps = 0.01 * th["q_al"] * AL - PS * (th["ps_dec"] + fl * out_surf / 100.0) + mix * (PD - PS)
    dpd = th["pd_k"] * OD - PD * (0.2 * th["ps_dec"] + fl * out_deep / 100.0) + mix * (PS - PD)
    d[8] = th["od_k"] * 0.01 * th["al_m"] * AL * (1.0 - OD) - OD * (0.01 + th["od_re"] * aer)
    if "A" in mech:
        d[9] = th["bf_k"] * 0.01 * AL * (1.0 - BF) - th["bf_rm"] * (aer + out / th["omax"]) * BF
    if "B" in mech:
        kr = 2.0 / th["tau_ret"]
        d[10] = th["ret_f"] * irr - kr * R1
        d[11] = kr * (R1 - R2)
        dps = dps + th["ret_c"] * ret / Vs
    if "C" in mech:
        rem = th["rem_k"] * aer * SD
        d[12] = th["sd_k"] * 0.01 * th["al_m"] * AL - rem
        dps = dps + rem * 10.0 / Vs
    d[6] = dps
    d[7] = dpd
    return _pack(M, d)


def h(x, u, th, mech):
    M, s, uu = _unpack(x, u)
    base, ret, inflow, out, qual = _alg(M, s, uu, th, mech)
    return _pack(M, [s[0], inflow, out, qual])


def x0(y0, th, mech):
    M, y = _yunpack(y0)
    ph = th["phase"]
    cs = math.cos(ph) if M is _PY else np.cos(ph)
    sn = math.sin(ph) if M is _PY else np.sin(ph)
    base = th["i0"] * (1.0 + th["i_amp"] * cs)
    q = M.min(M.max(y[3], 1e-3), 0.999)
    ps = -M.log(q)
    z = 0.0 * y[0]
    return _pack(M, [M.max(y[0], 0.0), cs + z, sn + z, M.max(y[1], 0.0) - base, 1.0 + z, 1.0 + z,
                     ps, ps * th["q0_deep"], 0.2 + z, 0.1 + z, z, z, 0.5 + z])
