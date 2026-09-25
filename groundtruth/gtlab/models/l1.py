"""L1: multi-timescale Hammerstein lag (PLAN section 5).

Per observable j, K banks in transformed space v:
    h_t        = ah_j h_{t-1} + (1 - ah_j) * ||u~_t - u~_rec||_rms          (history state, optional)
    z^k_{t+1}  = a_kj z^k_t + (1 - a_kj) * (1 + gamma_kj h_t) * (phi(u~_{t-d}) @ W_k)_j
    v_t        = c_j + sum_k z^k_{t+1};     z^k_0 = E_k[j] . g(y0) + b_kj
    y_t        = clip(g^-1(v_t))
Stable by construction (0 < a < 1). Observables are independent given the shared features, so
each is fitted separately: ridge init on a timescale grid (linear given a), then
least_squares(loss='cauchy') on the open-loop physical residual / sigma.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from gtlab.models import common as C
from gtlab.runtime import infer as rt

TIMESCALE_SETS = {
    1: [(0.9,), (0.97,), (0.99,), (0.7,), (0.995,)],
    2: [(0.7, 0.98), (0.5, 0.95), (0.9, 0.995), (0.3, 0.9)],
    3: [(0.5, 0.95, 0.995), (0.3, 0.9, 0.99), (0.7, 0.97, 0.998), (0.2, 0.8, 0.98), (0.8, 0.99, 0.999)],
}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def _col_tr(tr, j):
    return {"kind": [tr["kind"][j]], "s": [tr["s"][j]], "mu": [tr["mu"][j]], "sd": [tr["sd"][j]],
            "eps": tr.get("eps", 1e-3)}


class L1(C.DevModel):
    kind = "l1"

    def __init__(self, spec, K=3, hist=True, pairs="auto", delay_grid=(0, 1, 2, 4, 8), delays=None,
                 max_nfev=60, time_budget_s=None, anchor=True, **cfg):
        super().__init__(spec, **cfg)
        self.K = int(K)
        # anchor: the initial state is tied to the observed initial, z0_k = (v0_j - c)/K + b_k, so
        # v_0 = v0_j + sum_k b_k; only K small offsets are free instead of a K x p matrix E.
        self.anchor = bool(anchor)
        self.hist = bool(hist)
        self.phi = C.phi_spec(spec.m, pairs=pairs)
        self.delay_grid = tuple(delay_grid) if delays is None else (None,)
        self.delays = list(delays) if delays is not None else [0] * spec.m
        self.max_nfev = int(max_nfev)
        self.time_budget_s = time_budget_s

    # ---------------------------------------------------------------- data prep
    def _feats(self, U, delays=None):
        d = self.delays if delays is None else delays
        return rt.phi(rt.delay_u(rt.norm_u(self.meta, U), d), self.phi)

    def _dist(self, U):
        return np.sqrt(np.mean((rt.norm_u(self.meta, U) - self.urec) ** 2, axis=1))

    def _prep_runs(self, runs, delays):
        Tref = float(np.mean([r.T for r in runs]))
        out = []
        for r in runs:
            out.append({"F": self._feats(r.U, delays), "V": rt.g_fwd(r.Y, self.tr),
                        "v0": rt.g_fwd(r.y0, self.tr), "Y": r.Y, "w": C.run_weights(r.T, Tref),
                        "d": self._dist(r.U), "T": r.T})
        return out

    # ---------------------------------------------------------------- linear init
    def _linear(self, data, j, avec):
        p = self.spec.p
        Xs, ys = [], []
        for D in data:
            T, nf = D["F"].shape
            tt = np.arange(1, T + 1)
            cols = []
            for a in avec:
                cols.append(C.lfilt(a, D["F"], np.zeros(nf)))
            if self.anchor:
                # v_t = c + sum_k a_k^t (v0_j - c)/K + sum_k lfilt(a_k, F) W_k
                #     = v0_j D_t + c (1 - D_t) + ...,  D_t = mean_k a_k^t
                Dt = np.mean([a ** tt for a in avec], axis=0)
                cols.append((1.0 - Dt)[:, None])
                target = D["V"][:, j] - D["v0"][j] * Dt
            else:
                for a in avec:
                    cols.append((a ** tt)[:, None] * D["v0"][None, :])
                for a in avec:
                    cols.append((a ** tt)[:, None])
                cols.append(np.ones((T, 1)))
                target = D["V"][:, j]
            X = np.hstack(cols) * D["w"][:, None]
            Xs.append(X)
            ys.append(target * D["w"])
        X, y = np.vstack(Xs), np.concatenate(ys)
        beta = C.ridge(X, y)
        sse = float(np.sum((X @ beta - y) ** 2))
        K, nf = len(avec), data[0]["F"].shape[1]
        W = beta[:K * nf].reshape(K, nf)
        if self.anchor:
            E = np.zeros((K, p)); E[:, j] = 1.0 / K
            b = np.zeros(K)
            c = beta[-1]
        else:
            E = beta[K * nf:K * nf + K * p].reshape(K, p)
            b = beta[K * nf + K * p:K * nf + K * p + K]
            c = beta[-1]
        return sse, {"a": np.array(avec, float), "W": W, "E": E, "b": b, "c": float(c)}

    def _best_linear(self, data, j):
        best = None
        for avec in TIMESCALE_SETS[self.K]:
            sse, par = self._linear(data, j, avec)
            if best is None or sse < best[0]:
                best = (sse, par)
        return best

    # ---------------------------------------------------------------- param packing (one obs)
    def _pack(self, par):
        parts = [_logit(par["a"]), par["W"].ravel(), [par["c"]], par["E"].ravel(), par["b"]]
        if self.hist:
            parts += [par["gamma"], [_logit(par["ah"])]]
        return np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in parts])

    def _unpack(self, x, nf):
        K, p = self.K, self.spec.p
        i = 0
        a = _sig(x[i:i + K]); i += K
        W = x[i:i + K * nf].reshape(K, nf); i += K * nf
        c = x[i]; i += 1
        E = x[i:i + K * p].reshape(K, p); i += K * p
        b = x[i:i + K]; i += K
        par = {"a": a, "W": W, "c": c, "E": E, "b": b}
        if self.hist:
            par["gamma"] = x[i:i + K]; i += K
            par["ah"] = float(_sig(x[i])); i += 1
        return par

    def _bounds(self, nf):
        K, p = self.K, self.spec.p
        lo = [np.full(K, -4.0), np.full(K * nf, -np.inf), [-np.inf], np.full(K * p, -np.inf), np.full(K, -np.inf)]
        hi = [np.full(K, 9.2), np.full(K * nf, np.inf), [np.inf], np.full(K * p, np.inf), np.full(K, np.inf)]
        if self.anchor:
            lo[4] = np.full(K, -1.0); hi[4] = np.full(K, 1.0)
        if self.hist:
            lo += [np.full(K, -0.95), [_logit(0.5)]]
            hi += [np.full(K, 5.0), [_logit(0.9995)]]
        return np.concatenate([np.asarray(v, float) for v in lo]), np.concatenate([np.asarray(v, float) for v in hi])

    # ---------------------------------------------------------------- per-obs simulation (fast)
    def _sim_obs(self, par, F, v0, dist):
        T = F.shape[0]
        Q = F @ par["W"].T                       # [T, K]
        if self.hist:
            h = C.lfilt(par["ah"], dist, 0.0)
            Q = Q * (1.0 + np.outer(h, par["gamma"]))
        if self.anchor:
            z0 = (v0[par["j"]] - par["c"]) / self.K + par["b"]   # [K]
        else:
            z0 = par["E"] @ v0 + par["b"]            # [K]
        v = np.full(T, par["c"])
        for k in range(self.K):
            v = v + C.lfilt(par["a"][k], Q[:, k], z0[k])
        return v

    def _resid_obs(self, x, data, j, nf, trj, lo, hi, sig):
        par = self._unpack(x, nf)
        par["j"] = j
        res = []
        for D in data:
            v = self._sim_obs(par, D["F"], D["v0"], D["d"])
            y = rt.g_inv(v[:, None], trj)[:, 0]
            y = np.clip(np.where(np.isfinite(y), y, 1e12), lo, hi)
            res.append((y - D["Y"][:, j]) / sig * D["w"])
        r = np.concatenate(res)
        return np.where(np.isfinite(r), r, 1e6)

    # ---------------------------------------------------------------- fit
    def fit(self, runs):
        with C.single_thread():
            return self._fit(runs)

    def _fit(self, runs):
        t_start = time.time()
        self._prepare(runs)
        self.tr = C.auto_transform(self.spec, runs)
        self.urec = rt.norm_u(self.meta, np.array([[self.meta["recovery"][c] for c in self.spec.controls]]))[0]
        p, m = self.spec.p, self.spec.m
        # delay choice (one global delay for all controls) by the linear-stage SSE
        if self.delay_grid != (None,):
            best_d, best_sse, sse0 = 0, None, None
            max_T = min(r.T for r in runs)
            for dly in self.delay_grid:
                if dly >= max_T // 2:
                    continue
                data = self._prep_runs(runs, [dly] * m)
                sse = sum(self._best_linear(data, j)[0] for j in range(p))
                if dly == 0:
                    sse0 = sse
                if best_sse is None or sse < best_sse:
                    best_d, best_sse = dly, sse
            if sse0 is not None and best_sse > 0.99 * sse0:
                best_d = 0
            self.delays = [int(best_d)] * m
        data = self._prep_runs(runs, self.delays)
        nf = data[0]["F"].shape[1]
        K = self.K
        self.a = np.zeros((K, p)); self.W = np.zeros((K, nf, p)); self.E = np.zeros((K, p, p))
        self.b = np.zeros((K, p)); self.c = np.zeros(p)
        self.gamma = np.zeros((K, p)); self.ah = np.full(p, 0.99)
        lo_all, hi_all = self.lo, self.hi
        self.info = {"delays": self.delays, "obs": {}}
        for j in range(p):
            sse, par = self._best_linear(data, j)
            if self.hist:
                par["gamma"] = np.zeros(K)
                par["ah"] = 0.99
            x0 = self._pack(par)
            blo, bhi = self._bounds(nf)
            x0 = np.clip(x0, blo + 1e-9, bhi - 1e-9)
            trj = _col_tr(self.tr, j)
            args = (data, j, nf, trj, lo_all[j], hi_all[j], float(self.sigma[j]))
            cost0 = 0.5 * float(np.sum(np.log1p(self._resid_obs(x0, *args) ** 2)))
            nfev = self.max_nfev
            if self.time_budget_s is not None:
                left = self.time_budget_s - (time.time() - t_start)
                if left < 1.0:
                    nfev = 0
            if nfev > 0:
                try:
                    sol = least_squares(self._resid_obs, x0, args=args, loss="cauchy", f_scale=1.0,
                                        bounds=(blo, bhi), max_nfev=nfev, x_scale="jac", method="trf")
                    x = sol.x if sol.cost <= cost0 else x0
                    cost = min(sol.cost, cost0)
                except Exception as e:           # keep the linear init
                    x, cost = x0, cost0
                    self.info["obs"][self.spec.observables[j] + "_err"] = repr(e)
            else:
                x, cost = x0, cost0
            par = self._unpack(x, nf)
            if self.anchor:                      # E is unused by the anchored simulation; pin it
                par["E"] = np.zeros((K, p)); par["E"][:, j] = 1.0 / K
            self.a[:, j] = par["a"]; self.W[:, :, j] = par["W"]; self.c[j] = par["c"]
            self.E[:, j, :] = par["E"]; self.b[:, j] = par["b"]
            if self.hist:
                self.gamma[:, j] = par["gamma"]; self.ah[j] = par["ah"]
            self.info["obs"][self.spec.observables[j]] = {"cost0": cost0, "cost": float(cost)}
        self.info["fit_s"] = round(time.time() - t_start, 2)
        return self

    # ---------------------------------------------------------------- rollout / export
    def raw_rollout(self, y0, U):
        F = self._feats(U)
        v0 = rt.g_fwd(y0, self.tr)
        dist = self._dist(U)
        V = np.empty((U.shape[0], self.spec.p))
        for j in range(self.spec.p):
            par = {"a": self.a[:, j], "W": self.W[:, :, j].copy(), "c": self.c[j], "E": self.E[:, j, :],
                   "b": self.b[:, j], "j": j}
            if self.hist:
                par["gamma"] = self.gamma[:, j]; par["ah"] = self.ah[j]
            V[:, j] = self._sim_obs(par, F, v0, dist)
        return rt.g_inv(V, self.tr)

    def export(self):
        b_out = self.b - self.c[None, :] / self.K if self.anchor else self.b
        blob = {"kind": "l1", "tr": self.tr, "phi": self.phi, "delays": [int(d) for d in self.delays],
                "a": self.a.tolist(), "W": self.W.tolist(), "E": self.E.tolist(), "b": b_out.tolist(),
                "c": self.c.tolist(), "hist": None}
        if self.hist:
            blob["hist"] = {"ah": self.ah.tolist(), "gamma": self.gamma.tolist(), "urec": self.urec.tolist()}
        return blob
