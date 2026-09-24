"""L0a persistence and L0b relax-to-learned-equilibrium (PLAN section 5).

L0b: in transformed space v, v_{t+1} = a_j v_t + (1 - a_j) * (phi(u_t) @ W)_j with v_0 = g(y0).
a_j picked per observable on a grid, W_j by ridge regression (linear given a_j).
"""
from __future__ import annotations

import numpy as np

from gtlab.models import common as C
from gtlab.runtime import infer as rt


class L0a(C.DevModel):
    kind = "l0a"

    def fit(self, runs):
        self._prepare(runs)
        return self

    def raw_rollout(self, y0, U):
        return np.tile(np.asarray(y0, float), (U.shape[0], 1))

    def export(self):
        return {"kind": "l0a"}


A_GRID = tuple(1.0 - np.geomspace(0.6, 0.0007, 18))


class L0b(C.DevModel):
    kind = "l0b"

    def __init__(self, spec, a_grid=A_GRID, pairs="auto", delays=None, sq=True, **cfg):
        super().__init__(spec, **cfg)
        self.a_grid = a_grid
        self.phi = C.phi_spec(spec.m, sq=sq, pairs=pairs)
        self.delays = list(delays) if delays else [0] * spec.m

    def _feats(self, U):
        return rt.phi(rt.delay_u(rt.norm_u(self.meta, U), self.delays), self.phi)

    def fit(self, runs):
        self._prepare(runs)
        self.tr = C.auto_transform(self.spec, runs)
        p = self.spec.p
        Tref = float(np.mean([r.T for r in runs]))
        data = []
        for r in runs:
            F = self._feats(r.U)
            V = rt.g_fwd(r.Y, self.tr)
            v0 = rt.g_fwd(r.y0, self.tr)
            data.append((F, V, v0, C.run_weights(r.T, Tref)))
        nf = data[0][0].shape[1]
        self.a = np.zeros(p)
        self.W = np.zeros((nf, p))
        for j in range(p):
            best = None
            for a in self.a_grid:
                Xs, ys = [], []
                for F, V, v0, w in data:
                    T = F.shape[0]
                    Phi = C.lfilt(a, F, np.zeros(nf))
                    decay = a ** np.arange(1, T + 1) * v0[j]
                    Xs.append(Phi * w[:, None])
                    ys.append((V[:, j] - decay) * w)
                X, y = np.vstack(Xs), np.concatenate(ys)
                Wj = C.ridge(X, y)
                sse = float(np.sum((X @ Wj - y) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, a, Wj)
            self.a[j] = best[1]
            self.W[:, j] = best[2]
        return self

    def raw_rollout(self, y0, U):
        Q = self._feats(U) @ self.W
        v0 = rt.g_fwd(y0, self.tr)
        V = np.column_stack([C.lfilt(self.a[j], Q[:, j], v0[j]) for j in range(self.spec.p)])
        return rt.g_inv(V, self.tr)

    def export(self):
        return {"kind": "l0b", "tr": self.tr, "phi": self.phi, "delays": [int(d) for d in self.delays],
                "a": self.a.tolist(), "W": self.W.tolist()}
