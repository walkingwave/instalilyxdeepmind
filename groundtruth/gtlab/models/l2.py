"""L2: stable Hammerstein-Wiener linear state space (PLAN section 5).

    x_{t+1} = A x_t + B phi(u~_{t-d}),   v_t = C x_{t+1} + D phi(u~_{t-d}) + c,   x_0 = E g(y0) + e0
    y_t = clip(g^-1(v_t))
A is block diagonal: real 1x1 modes r and 2x2 rotations r [[cos th, -sin th], [sin th, cos th]]
with r = r_max * sigmoid(rho) < 1 (stable by construction), th = pi * sigmoid(vartheta).

Init: the fitted L1 (every L1 bank/observable is a real mode) plus n_cplx complex modes whose
(r, th) come from a small numpy PO-MOESP (N4SID-family) estimate of A on the transformed data;
if that fails, default frequencies. Then least_squares(loss='cauchy') on open-loop physical
residuals / sigma with a wall-clock deadline (best-so-far kept).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from gtlab.models import common as C
from gtlab.models.l1 import L1
from gtlab.runtime import infer as rt


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _logit(a):
    a = np.clip(a, 1e-9, 1 - 1e-9)
    return np.log(a / (1 - a))


class _Deadline(Exception):
    pass


# ----------------------------------------------------------------------------- N4SID (PO-MOESP)
def moesp_eigs(pairs, n, i=10):
    """Eigenvalues of A from PO-MOESP on [(Uin [T, nu], Yout [T, ny])...] (several runs)."""
    nu = pairs[0][0].shape[1]
    ny = pairs[0][1].shape[1]
    blocks = []
    for Uin, Y in pairs:
        T = Uin.shape[0]
        N = T - 2 * i + 1
        if N < 5:
            continue

        def hank(M, start):
            return np.vstack([M[start + k:start + k + N].T for k in range(i)])
        blocks.append(np.vstack([hank(Uin, i), hank(Uin, 0), hank(Y, 0), hank(Y, i)]))
    if not blocks:
        raise ValueError("runs too short for N4SID")
    H = np.hstack(blocks)
    H = H / np.sqrt(H.shape[1])
    Rq = np.linalg.qr(H.T, mode="r")
    L = Rq.T
    a = i * nu
    b = a + i * (nu + ny)
    L32 = L[b:, a:b]
    Us, s, _ = np.linalg.svd(L32, full_matrices=False)
    n = int(min(n, len(s)))
    G = Us[:, :n] * np.sqrt(s[:n])
    A = np.linalg.pinv(G[:-ny]) @ G[ny:]
    return np.linalg.eigvals(A), s


class L2(C.DevModel):
    kind = "l2"

    def __init__(self, spec, n_cplx=2, extra_real=0, r_max=1.0, l1=None, l1_cfg=None,
                 max_nfev=40, time_budget_s=None, n4sid_i=10, seed=0, reg=0.03, anchor=True, **cfg):
        super().__init__(spec, **cfg)
        # anchor: x0 = pinv(C) (v0 - c - D phi(u0)), the minimum-norm hidden state consistent with
        # the observed initial; no free E / e0 (they over-fit from a handful of resets).
        self.anchor = bool(anchor)
        self.n_cplx = int(n_cplx)
        self.extra_real = int(extra_real)
        self.r_max = float(r_max)
        self.l1 = l1
        self.l1_cfg = dict(l1_cfg or {"hist": False})
        self.max_nfev = int(max_nfev)
        self.time_budget_s = time_budget_s
        self.n4sid_i = int(n4sid_i)
        self.seed = seed
        self.reg = float(reg)

    # ---------------------------------------------------------------- structure
    @property
    def n(self):
        return self.n_real + 2 * self.n_cplx

    def _feats(self, U):
        return rt.phi(rt.delay_u(rt.norm_u(self.meta, U), self.delays), self.phi)

    def _sizes(self):
        n, p, nf = self.n, self.spec.p, self.nf
        return [("rho", self.n_real + self.n_cplx), ("th", self.n_cplx), ("B", n * nf), ("C", p * n),
                ("D", p * nf), ("c", p), ("E", n * p), ("e0", n)]

    def _unpack(self, x):
        n, p, nf = self.n, self.spec.p, self.nf
        out, i = {}, 0
        for name, sz in self._sizes():
            out[name] = x[i:i + sz]
            i += sz
        out["B"] = out["B"].reshape(n, nf)
        out["C"] = out["C"].reshape(p, n)
        out["D"] = out["D"].reshape(p, nf)
        out["E"] = out["E"].reshape(n, p)
        out["r"] = self.r_max * _sig(out["rho"])
        out["theta"] = np.pi * _sig(out["th"])
        return out

    def _pack(self, P):
        return np.concatenate([np.ravel(P[k]) for k, _ in self._sizes()])

    def A_dense(self, P):
        n = self.n
        A = np.zeros((n, n))
        nr = self.n_real
        for i in range(nr):
            A[i, i] = P["r"][i]
        for k in range(self.n_cplx):
            r, th = P["r"][nr + k], P["theta"][k]
            i = nr + 2 * k
            A[i:i + 2, i:i + 2] = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return A

    def _states(self, P, F, v0):
        """X [T, n] (state after each tick), via scalar / complex lfilter per mode."""
        BF = F @ P["B"].T
        if self.anchor:
            x0 = np.linalg.pinv(P["C"]) @ (v0 - P["c"] - P["D"] @ F[0])
        else:
            x0 = P["E"] @ v0 + P["e0"]
        T = F.shape[0]
        X = np.empty((T, self.n))
        nr = self.n_real
        for i in range(nr):
            lam = P["r"][i]
            X[:, i] = lfilter([1.0], [1.0, -lam], BF[:, i], zi=[lam * x0[i]])[0]
        for k in range(self.n_cplx):
            i = nr + 2 * k
            mu = P["r"][nr + k] * np.exp(1j * P["theta"][k])
            w0 = x0[i] + 1j * x0[i + 1]
            w = lfilter([1.0], [1.0, -mu], BF[:, i] + 1j * BF[:, i + 1], zi=[mu * w0])[0]
            X[:, i] = w.real
            X[:, i + 1] = w.imag
        return X

    def _sim_v(self, P, F, v0):
        X = self._states(P, F, v0)
        return X @ P["C"].T + F @ P["D"].T + P["c"]

    # ---------------------------------------------------------------- init
    def _init_from_l1(self, l1, cplx):
        p, nf, K = self.spec.p, self.nf, l1.K
        n = self.n
        rng = np.random.default_rng(self.seed)
        P = {"B": np.zeros((n, nf)), "C": np.zeros((p, n)), "D": np.zeros((p, nf)), "c": l1.c.copy(),
             "E": np.zeros((n, p)), "e0": np.zeros(n)}
        r = np.zeros(self.n_real + self.n_cplx)
        idx = 0
        for k in range(K):
            for j in range(p):
                a = l1.a[k, j]
                r[idx] = a
                P["B"][idx] = (1 - a) * l1.W[k, :, j]
                P["C"][j, idx] = 1.0
                P["E"][idx] = l1.E[k, j]
                P["e0"][idx] = l1.b[k, j]
                idx += 1
        for e in range(self.extra_real):
            r[idx] = 0.9
            P["B"][idx] = 1e-3 * rng.standard_normal(nf)
            P["C"][:, idx] = 1e-3 * rng.standard_normal(p)
            idx += 1
        th = np.zeros(self.n_cplx)
        for k in range(self.n_cplx):
            rk, tk = cplx[k]
            r[self.n_real + k] = rk
            th[k] = tk
            i = self.n_real + 2 * k
            P["B"][i:i + 2] = 1e-2 * rng.standard_normal((2, nf))
            P["C"][:, i:i + 2] = 1e-2 * rng.standard_normal((p, 2))
        P["rho"] = _logit(np.clip(r / self.r_max, 1e-4, 1 - 1e-6))
        P["th"] = _logit(np.clip(th / np.pi, 1e-4, 1 - 1e-4))
        return P

    def _cplx_init(self, data):
        default = [(0.98, 2 * np.pi / 60), (0.995, 2 * np.pi / 300), (0.95, 2 * np.pi / 20),
                   (0.99, 2 * np.pi / 150)]
        default = (default * ((self.n_cplx // len(default)) + 1))[:self.n_cplx]
        if self.n_cplx == 0:
            return [], "none"
        try:
            keep = [f for f in range(self.nf) if np.ptp(np.concatenate([D["F"][:, f] for D in data])) > 0]
            pairs = [(D["F"][:, keep], D["V"]) for D in data]
            order = min(2 * self.n_cplx + self.spec.p + 2, 16)
            eig, _ = moesp_eigs(pairs, order, i=self.n4sid_i)
            cand = [(abs(z), abs(np.angle(z))) for z in eig
                    if np.imag(z) > 1e-4 and 0.5 < abs(z) < 1.2 and abs(np.angle(z)) > 1e-3]
            cand.sort(key=lambda t: -t[0])
            got = [(min(r, 0.999 * self.r_max), th) for r, th in cand[:self.n_cplx]]
            src = f"n4sid({len(got)})"
            return got + default[len(got):], src
        except Exception as e:
            return default, f"default ({type(e).__name__})"

    # ---------------------------------------------------------------- fit
    def fit(self, runs):
        with C.single_thread():
            return self._fit(runs)

    def _fit(self, runs):
        t0 = time.time()
        self._prepare(runs)
        budget = self.time_budget_s
        if self.l1 is None:
            cfg = dict(self.l1_cfg)
            if budget is not None:
                cfg.setdefault("time_budget_s", 0.35 * budget)
            self.l1 = L1(self.spec, clip=self.clip, sigma=self.sigma, **cfg).fit(runs)
        l1 = self.l1
        self.tr, self.phi, self.delays = l1.tr, l1.phi, list(l1.delays)
        self.n_real = l1.K * self.spec.p + self.extra_real
        Tref = float(np.mean([r.T for r in runs]))
        data = [{"F": self._feats(r.U), "V": rt.g_fwd(r.Y, self.tr), "v0": rt.g_fwd(r.y0, self.tr),
                 "Y": r.Y, "w": C.run_weights(r.T, Tref)} for r in runs]
        self.nf = data[0]["F"].shape[1]
        cplx, src = self._cplx_init(data)
        P0 = self._init_from_l1(l1, cplx)
        x0 = self._pack(P0)
        lo, hi, sig = self.lo, self.hi, self.sigma
        best = {"cost": np.inf, "x": x0}
        # ridge toward the L1-derived init on the linear maps (not on poles / offsets)
        reg_mask = np.concatenate([np.full(sz, 0.0 if k in ("rho", "th", "c") else 1.0)
                                   for k, sz in self._sizes()])
        deadline = None if budget is None else t0 + budget

        def resid(x, track=True):
            P = self._unpack(x)
            res = []
            for D in data:
                y = rt.g_inv(self._sim_v(P, D["F"], D["v0"]), self.tr)
                y = np.clip(np.where(np.isfinite(y), y, 1e12), lo, hi)
                res.append(((y - D["Y"]) / sig * D["w"][:, None]).ravel())
            if self.reg > 0:
                res.append(np.sqrt(self.reg) * (x - x0) * reg_mask)
            r = np.concatenate(res)
            r = np.where(np.isfinite(r), r, 1e6)
            if track:
                c = 0.5 * float(np.sum(np.log1p(r * r)))
                if c < best["cost"]:
                    best["cost"], best["x"] = c, x.copy()
                if deadline is not None and time.time() > deadline:
                    raise _Deadline()
            return r

        cost0 = 0.5 * float(np.sum(np.log1p(resid(x0, track=False) ** 2)))
        best["cost"] = cost0
        status = "init"
        if self.max_nfev > 0 and (deadline is None or time.time() < deadline - 1):
            try:
                sol = least_squares(resid, x0, loss="cauchy", f_scale=1.0, max_nfev=self.max_nfev,
                                    x_scale="jac", method="trf")
                status = f"lsq:{sol.status}"
                if sol.cost < best["cost"]:
                    best["cost"], best["x"] = sol.cost, sol.x
            except _Deadline:
                status = "deadline"
            except Exception as e:
                status = f"error:{type(e).__name__}"
        self.P = self._unpack(best["x"])
        self.info = {"n": self.n, "n_real": self.n_real, "n_cplx": self.n_cplx, "cplx_init": src,
                     "cost0": cost0, "cost": best["cost"], "status": status,
                     "r": self.P["r"].tolist(), "theta": self.P["theta"].tolist(),
                     "delays": self.delays, "fit_s": round(time.time() - t0, 1)}
        return self

    # ---------------------------------------------------------------- rollout / export
    def raw_rollout(self, y0, U):
        F = self._feats(U)
        return rt.g_inv(self._sim_v(self.P, F, rt.g_fwd(y0, self.tr)), self.tr)

    def export(self):
        P = self.P
        return {"kind": "l2", "tr": self.tr, "phi": self.phi, "delays": [int(d) for d in self.delays],
                "A": self.A_dense(P).tolist(), "B": P["B"].tolist(), "C": P["C"].tolist(),
                "D": P["D"].tolist(), "c": P["c"].tolist(), "E": P["E"].tolist(), "e0": P["e0"].tolist(),
                "x0_from_y0": bool(self.anchor)}
