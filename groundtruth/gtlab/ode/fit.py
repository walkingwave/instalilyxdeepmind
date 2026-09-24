"""Grey-box ODE fitter (dev side; scipy allowed).

fit_ode(mod, runs, sigma, mech, ...) -> (theta_vec, info)
    least_squares(loss='cauchy') on open-loop rollout residuals (y_hat - y)/sigma, in a
    transformed parameter space z in [0,1]^P (log-scaled for log params), Latin-hypercube
    multistart around the PARAMS init, early starts on truncated runs, polish on full data.
    The Jacobian is forward differences evaluated in ONE batched rollout: family f/h/x0
    accept states of shape (B, n) and theta values of shape (B,), so all P perturbed
    parameter vectors x all runs integrate together.
fit_all_mechs(mod, runs, val_runs, sigma, ...) -> dict
    fits {AB, AC, BC}; picks by gtlab.metric.robust_score on val runs.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from gtlab import metric
from gtlab.ode import core

PAIRS = ("AB", "AC", "BC")


# ---------------------------------------------------------------- param transforms
def _pinfo(mod):
    lo = np.array([p[2] for p in mod.PARAMS], float)
    hi = np.array([p[3] for p in mod.PARAMS], float)
    lg = np.array([bool(p[4]) for p in mod.PARAMS])
    init = np.array([p[1] for p in mod.PARAMS], float)
    return lo, hi, lg, init


def to_z(mod, vec):
    lo, hi, lg, _ = _pinfo(mod)
    vec = np.asarray(vec, float)
    llo, lhi = _slog(lo), _slog(hi)
    span = np.where(lg, lhi - llo, hi - lo)
    span = np.where(span > 0, span, 1.0)
    num = np.where(lg, _slog(vec) - llo, vec - lo)
    return np.clip(num / span, 0.0, 1.0)


def _slog(v):
    return np.log(np.maximum(np.asarray(v, float), 1e-300))


def from_z(mod, z):
    lo, hi, lg, _ = _pinfo(mod)
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    llo, lhi = _slog(lo), _slog(hi)
    return np.where(lg, np.exp(llo + z * (lhi - llo)), lo + z * (hi - lo))


# ---------------------------------------------------------------- batched rollout
def rollout_batch(mod, Y0, U, TH, mech, n_sub):
    """Y0 [B,p]; U [T,B,m]; TH dict name -> float or [B] array. Returns Y [T,B,p].
    Same arithmetic as gtlab.ode.core.rollout (RK4, clip after every stage)."""
    mech = frozenset(mech)
    x = core._clip(mod, np.asarray(mod.x0(Y0, TH, mech), float))
    T = U.shape[0]
    Y = np.empty((T, Y0.shape[0], len(mod.OBS)))
    dt = 1.0 / n_sub
    f = mod.f
    clip = core._clip
    for t in range(T):
        u = U[t]
        for _ in range(n_sub):
            k1 = f(x, u, TH, mech)
            k2 = f(clip(mod, x + 0.5 * dt * k1), u, TH, mech)
            k3 = f(clip(mod, x + 0.5 * dt * k2), u, TH, mech)
            k4 = f(clip(mod, x + dt * k3), u, TH, mech)
            x = clip(mod, x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0)
        Y[t] = mod.h(x, u, TH, mech)
    return Y


def _stack(runs, T_max=None):
    Ts = [min(r.T, T_max) if T_max else r.T for r in runs]
    T = max(Ts)
    R = len(runs)
    p = runs[0].Y.shape[1]
    m = runs[0].U.shape[1]
    U = np.empty((T, R, m))
    Y = np.zeros((T, R, p))
    mask = np.zeros((T, R), bool)
    for i, (r, Ti) in enumerate(zip(runs, Ts)):
        U[:Ti, i] = r.U[:Ti]
        U[Ti:, i] = r.U[Ti - 1]
        Y[:Ti, i] = r.Y[:Ti]
        mask[:Ti, i] = True
    Y0 = np.stack([np.asarray(r.y0, float) for r in runs])
    return Y0, U, Y, mask


class _Budget(Exception):
    pass


class _Problem:
    def __init__(self, mod, runs, sigma, mech, n_sub, T_max=None, fd_step=1e-4,
                 deadline=None, head_weight=1.0, head_len=50, free=None):
        self.mod = mod
        self.mech = frozenset(mech)
        self.n_sub = n_sub
        self.Y0, self.U, self.Y, self.mask = _stack(runs, T_max)
        self.R = len(runs)
        self.sigma = np.asarray(sigma, float)
        self.names = [p[0] for p in mod.PARAMS]
        self.fd = fd_step
        self.deadline = deadline
        w = np.ones(self.mask.shape)
        if head_weight != 1.0:
            w[:head_len] = head_weight
        self.w = np.sqrt(w)[self.mask]           # [N] residual weights
        self.free = np.arange(len(self.names)) if free is None else np.asarray(free)
        self.best = (np.inf, None)
        self.nev = 0
        self._cache = (None, None)

    def _thetas_to_TH(self, V):
        """V [K, P] natural params -> dict of [K*R] arrays."""
        return {n: np.repeat(V[:, j], self.R) for j, n in enumerate(self.names)}

    def _resid_many(self, Z):
        """Z [K, P] full z-vectors -> residual matrix [K, N]."""
        K = Z.shape[0]
        V = from_z(self.mod, Z)
        TH = self._thetas_to_TH(V)
        Y0 = np.tile(self.Y0, (K, 1))
        U = np.tile(self.U, (1, K, 1))
        with np.errstate(all="ignore"):
            Yh = rollout_batch(self.mod, Y0, U, TH, self.mech, self.n_sub)   # [T, K*R, p]
        Yh = Yh.reshape(Yh.shape[0], K, self.R, -1)
        E = (Yh - self.Y[:, None]) / self.sigma                                # [T,K,R,p]
        E = np.transpose(E, (1, 0, 2, 3))[:, self.mask]                        # [K, N, p]
        E = E * self.w[None, :, None]
        E = E.reshape(K, -1)
        return np.where(np.isfinite(E), np.clip(E, -1e4, 1e4), 1e4)

    def full_z(self, zf, base):
        z = base.copy()
        z[self.free] = zf
        return z

    def fun(self, zf, base):
        if self.deadline is not None and time.time() > self.deadline and self.best[1] is not None:
            raise _Budget()
        key = zf.tobytes()
        if self._cache[0] == key:
            return self._cache[1]
        r = self._resid_many(self.full_z(zf, base)[None])[0]
        self.nev += 1
        cost = float(np.sum(np.log1p(r * r)))
        if cost < self.best[0]:
            self.best = (cost, self.full_z(zf, base))
        self._cache = (key, r)
        return r

    def jac(self, zf, base):
        if self.deadline is not None and time.time() > self.deadline and self.best[1] is not None:
            raise _Budget()
        r0 = self.fun(zf, base)
        P = len(zf)
        steps = np.where(zf + self.fd <= 1.0, self.fd, -self.fd)
        Z = np.repeat(self.full_z(zf, base)[None], P, axis=0)
        Z[np.arange(P), self.free] += steps
        Rm = self._resid_many(Z)
        return ((Rm - r0[None]) / steps[:, None]).T

    def cost(self, z):
        r = self._resid_many(np.asarray(z)[None])[0]
        return float(np.sum(np.log1p(r * r)))


def _lhs_starts(z_init, n, spread, seed):
    P = len(z_init)
    starts = [z_init.copy()]
    if n > 1:
        s = qmc.LatinHypercube(d=P, seed=seed).random(n - 1)
        if spread is None:
            Z = s
        else:
            Z = z_init[None] + spread * (2 * s - 1)
        starts.extend(np.clip(Z, 0.0, 1.0))
    return starts


def fit_ode(mod, runs, sigma, mech, n_starts=8, max_nfev=60, n_sub=None, seed=0,
            time_budget=None, early_T=200, early_nfev=None, n_polish=2, spread=0.3,
            theta0=None, head_weight=1.0, free=None, verbose=False):
    """Fit theta (natural units, PARAMS order) for one mechanism pair.

    runs: list of gtlab.data.Run (fit uses r.y0, r.U, r.Y noisy). sigma: [p] physical.
    Early phase: n_starts starts on runs truncated to early_T ticks with early_nfev evals;
    polish phase: best n_polish on full runs with max_nfev evals. time_budget in seconds.
    free: optional list of param names to fit (others held at theta0/init).
    Returns (theta_vec, info) with info = {cost, z, nfev, time, mech, starts: [...]}.
    """
    t0 = time.time()
    n_sub = int(n_sub or getattr(mod, "N_SUB", 4))
    deadline = None if time_budget is None else t0 + float(time_budget)
    _, _, _, init = _pinfo(mod)
    base = to_z(mod, init if theta0 is None else theta0)
    names = [p[0] for p in mod.PARAMS]
    free_idx = None if free is None else np.array([names.index(n) for n in free])
    fidx = np.arange(len(names)) if free_idx is None else free_idx
    early_nfev = early_nfev or max(10, max_nfev // 3)
    starts = _lhs_starts(base[fidx], n_starts, spread, seed)
    max_T = max(r.T for r in runs)
    use_early = early_T is not None and early_T < max_T and n_starts > n_polish
    results = []
    if use_early:
        prob = _Problem(mod, runs, sigma, mech, n_sub, T_max=early_T,
                        deadline=None if deadline is None else t0 + 0.5 * float(time_budget),
                        head_weight=head_weight, free=free_idx)
        for i, zs in enumerate(starts):
            prob.best = (np.inf, None)
            try:
                res = least_squares(prob.fun, zs, jac=prob.jac, bounds=(0.0, 1.0), loss="cauchy",
                                    f_scale=1.0, max_nfev=early_nfev, args=(base,), x_scale=1.0)
                zf = res.x
            except _Budget:
                zf = prob.best[1][fidx]
            c = prob.cost(prob.full_z(zf, base))
            results.append((c, zf))
            if verbose:
                print(f"  early start {i}: cost {c:.1f}  ({time.time() - t0:.0f}s)")
            if deadline is not None and time.time() > t0 + 0.5 * float(time_budget):
                break
        results.sort(key=lambda t: t[0])
        cands = [zf for _, zf in results[:n_polish]]
    else:
        cands = starts
    prob = _Problem(mod, runs, sigma, mech, n_sub, deadline=deadline, head_weight=head_weight,
                    free=free_idx)
    final = []
    for i, zs in enumerate(cands):
        prob.best = (np.inf, None)
        try:
            res = least_squares(prob.fun, zs, jac=prob.jac, bounds=(0.0, 1.0), loss="cauchy",
                                f_scale=1.0, max_nfev=max_nfev, args=(base,), x_scale=1.0)
            zf = res.x
        except _Budget:
            zf = prob.best[1][fidx] if prob.best[1] is not None else zs
        zfull = prob.full_z(zf, base)
        c = prob.cost(zfull)
        final.append((c, zfull))
        if verbose:
            print(f"  polish {i}: cost {c:.1f}  ({time.time() - t0:.0f}s)")
        if deadline is not None and time.time() > deadline:
            break
    final.sort(key=lambda t: t[0])
    cbest, zbest = final[0]
    theta = from_z(mod, zbest)
    info = {"cost": cbest, "z": zbest, "time": time.time() - t0, "mech": "".join(sorted(mech)),
            "n_sub": n_sub, "early": [c for c, _ in results], "polish": [c for c, _ in final]}
    return theta, info


def predict_runs(mod, theta_vec, mech, runs, n_sub=None):
    n_sub = int(n_sub or getattr(mod, "N_SUB", 4))
    th = core.theta_dict(mod, theta_vec)
    return [core.rollout(mod, r.y0, r.U, th, frozenset(mech), n_sub=n_sub) for r in runs]


def val_score(mod, theta_vec, mech, runs, sigma, n_sub=None, use_truth=False):
    preds = predict_runs(mod, theta_vec, mech, runs, n_sub)
    Yh = np.concatenate(preds, axis=0)
    Yt = np.concatenate([(r.Ytrue if use_truth and r.Ytrue is not None else r.Y) for r in runs], axis=0)
    return metric.robust_score(Yh, Yt, sigma)


def fit_all_mechs(mod, runs, val_runs, sigma, pairs=PAIRS, time_budget=None, verbose=False, **kw):
    """Fit each mechanism pair and select by robust_score on val_runs (noisy Y).
    time_budget (s) is split evenly across pairs. Returns
    {"best": pair, "theta": vec, "results": {pair: {"theta","info","val"}}}."""
    out = {}
    per = None if time_budget is None else float(time_budget) / len(pairs)
    for pr in pairs:
        th, info = fit_ode(mod, runs, sigma, frozenset(pr), time_budget=per, verbose=verbose, **kw)
        v = val_score(mod, th, pr, val_runs, sigma) if val_runs else -info["cost"]
        out[pr] = {"theta": th, "info": info, "val": v}
        if verbose:
            print(f"mech {pr}: train cost {info['cost']:.1f}  val {v:.4f}  ({info['time']:.0f}s)")
    best = max(out, key=lambda k: out[k]["val"])
    return {"best": best, "theta": out[best]["theta"], "results": out}
