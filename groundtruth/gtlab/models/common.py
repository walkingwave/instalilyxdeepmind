"""Dev-side helpers shared by the model ladder (transforms, features, clip ranges, docs).

The numeric definitions of transforms / features / clipping live in gtlab.runtime.infer so the
shipped runtime and the dev models cannot drift apart.
"""
from __future__ import annotations

import numpy as np

from gtlab.runtime import infer as rt


# ----------------------------------------------------------------------------- spec-level meta
def hard_vectors(spec):
    lo, hi = [], []
    for o in spec.observables:
        a, b = spec.output_bounds(o)
        lo.append(0.0 if a is None else float(a))
        hi.append(None if b is None else float(b))
    return lo, hi


CLIP_MARGIN = 1.0


def soft_clip(spec, runs, margin=None):
    """[min - margin*range, max + margin*range] of all observed data (incl. y0), per observable,
    intersected with the hard bounds. margin defaults to CLIP_MARGIN=1.0 (PLAN said 0.2, but on
    mocks the 4000-step truth of integrating observables (queues, stocks) leaves the short-run
    data range by several x, so a tight clip costs more than it protects)."""
    margin = CLIP_MARGIN if margin is None else margin
    hlo, hhi = hard_vectors(spec)
    if not runs:
        return list(hlo), list(hhi)
    Y = np.concatenate([np.vstack([r.y0[None, :], r.Y]) for r in runs], axis=0)
    mn, mx = np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
    rng = np.maximum(mx - mn, 1e-9 + 1e-3 * np.abs(mx))
    lo = mn - margin * rng
    hi = mx + margin * rng
    clo = [float(max(l, h)) if h is not None else float(l) for l, h in zip(lo, hlo)]
    chi = [float(min(u, h)) if h is not None else float(u) for u, h in zip(hi, hhi)]
    return clo, chi


def make_doc(spec, blob, runs=None, clip=None, info=None):
    hlo, hhi = hard_vectors(spec)
    if clip is None:
        clip = soft_clip(spec, runs or [])
    return {
        "format": rt.FORMAT,
        "system": spec.id,
        "observables": list(spec.observables),
        "controls": list(spec.controls),
        "bounds": {c: [float(spec.bounds[c][0]), float(spec.bounds[c][1])] for c in spec.controls},
        "recovery": {c: float(spec.recovery.get(c, spec.bounds[c][0])) for c in spec.controls},
        "hard_lo": hlo, "hard_hi": hhi,
        "clip_lo": list(clip[0]), "clip_hi": list(clip[1]),
        "model": blob,
        "info": info or {},
    }


def meta_doc(spec, clip=None):
    """A doc without a model: enough for norm_u / clip_vectors."""
    return make_doc(spec, {"kind": "l0a"}, clip=clip)


# ----------------------------------------------------------------------------- transforms
def auto_transform(spec, runs, overrides=None, eps=1e-3):
    """Per-observable transform by the PLAN rule; mu/sd standardize the transformed data."""
    overrides = overrides or {}
    Y = np.concatenate([np.vstack([r.y0[None, :], r.Y]) for r in runs], axis=0)
    kinds, s = [], []
    for j, o in enumerate(spec.observables):
        y = Y[:, j]
        lo, hi = spec.output_bounds(o)
        k = overrides.get(o)
        if k is None:
            q01, q99 = np.quantile(y, 0.01), np.quantile(y, 0.99)
            if hi == 1.0 and lo == 0.0:
                k = "logit"
            elif np.min(y) >= 0 and q99 > 20 * max(q01, 1e-12) and q99 > 0:
                k = "log1p"
            else:
                k = "affine"
        if k == "log1p":
            pos = y[y > 0]
            med = float(np.median(pos)) if pos.size else 1.0
            sc = max(0.05 * med, 1e-3 * float(np.max(np.abs(y))), 1e-9)
        elif k == "logit":
            sc = 1.0
        else:
            sc = max(float(np.std(y)), 1e-3 * float(np.max(np.abs(y))), 1e-9)
        kinds.append(k)
        s.append(float(sc))
    tr = {"kind": kinds, "s": s, "mu": [0.0] * len(kinds), "sd": [1.0] * len(kinds), "eps": eps}
    G = rt.g_fwd(Y, tr)
    tr["mu"] = [float(v) for v in G.mean(axis=0)]
    tr["sd"] = [float(max(v, 1e-6)) for v in G.std(axis=0)]
    return tr


def phi_spec(m, sq=True, pairs="auto", const=True):
    if pairs == "auto":
        pairs = [[i, j] for i in range(m) for j in range(i + 1, m)] if m <= 4 else []
    return {"sq": bool(sq), "pairs": [list(map(int, p)) for p in (pairs or [])], "const": bool(const)}


def lfilt(a, x, z0):
    """z_{t+1} = a z_t + (1-a) x_t with z_{-1}... returns [z_1..z_T]. Vectorised over columns of x
    only when a is scalar. x [T] or [T, c]."""
    from scipy.signal import lfilter
    x = np.asarray(x, float)
    zi = np.atleast_1d(a * np.asarray(z0, float))
    if x.ndim == 1:
        return lfilter([1.0 - a], [1.0, -a], x, zi=zi)[0]
    zi = np.broadcast_to(zi, (x.shape[1],))[None, :].copy()
    return lfilter([1.0 - a], [1.0, -a], x, axis=0, zi=zi)[0]


def run_weights(T, Tref=None, early=50, early_w=2.0):
    """Residual weights (sqrt, multiply residuals): every run carries the weight of a Tref-tick
    run, so runs count equally while a typical tick keeps weight ~1 (cauchy f_scale=1 stays in
    sigma units); the first `early` ticks (the reset transient, present in every eval episode)
    are up-weighted."""
    Tref = T if Tref is None else Tref
    w = np.ones(T)
    w[:min(early, T)] = early_w
    w = w / w.mean() * (float(Tref) / T)
    return np.sqrt(w)


def ridge(X, y, lam=1e-4):
    XtX = X.T @ X
    reg = lam * (np.trace(XtX) / max(1, XtX.shape[0]) + 1e-12)
    return np.linalg.solve(XtX + reg * np.eye(XtX.shape[0]), X.T @ y)


# ----------------------------------------------------------------------------- dev base class
from gtlab.models.base import Model  # noqa: E402


class DevModel(Model):
    """Shared dev plumbing: clip range, physical rollout = finalize(raw rollout)."""

    kind = "dev"

    def __init__(self, spec, clip=None, sigma=None, **cfg):
        super().__init__(spec, **cfg)
        self.clip = clip
        self.sigma = None if sigma is None else np.asarray(sigma, float)
        self.meta = meta_doc(spec, clip)
        self.info = {}

    def _prepare(self, runs):
        if self.clip is None:
            self.clip = soft_clip(self.spec, runs)
            self.meta = meta_doc(self.spec, self.clip)
        if self.sigma is None:
            from gtlab.metric import sigma_proxy
            self.sigma = sigma_proxy(runs)
        self.lo, self.hi = rt.clip_vectors(self.meta)

    def raw_rollout(self, y0, U):
        raise NotImplementedError

    def rollout(self, y0, U):
        y0 = np.asarray(y0, float)
        lo, hi = rt.clip_vectors(self.meta)
        return rt.finalize(self.raw_rollout(y0, np.atleast_2d(np.asarray(U, float))), y0, lo, hi)

    def doc(self):
        return make_doc(self.spec, self.export(), clip=(self.meta["clip_lo"], self.meta["clip_hi"]),
                        info=self.info)


def single_thread():
    """BLAS thread limit for fitting: multi-threaded OpenBLAS is ~10-30x slower on the small
    SVDs inside least_squares on this 2-CPU box."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except Exception:
        import contextlib
        return contextlib.nullcontext()
