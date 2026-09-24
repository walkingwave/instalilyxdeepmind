"""Exact competition metric + sigma proxy."""
from __future__ import annotations

import numpy as np


def score(Yhat, Ytrue, sigma):
    """mean over ticks and observables of 1/(1+|e|/sigma). Yhat/Ytrue [T,p], sigma [p]."""
    e = np.abs(np.asarray(Yhat, float) - np.asarray(Ytrue, float))
    return float(np.mean(1.0 / (1.0 + e / np.asarray(sigma, float)[None, :])))


def score_per_obs(Yhat, Ytrue, sigma):
    e = np.abs(np.asarray(Yhat, float) - np.asarray(Ytrue, float))
    return np.mean(1.0 / (1.0 + e / np.asarray(sigma, float)[None, :]), axis=0)


def sigma_proxy(runs, floor_rel=1e-3):
    """Per-observable std across every collected observation, floored."""
    Y = np.concatenate([r.Y for r in runs], axis=0)
    s = Y.std(axis=0)
    floor = np.maximum(floor_rel * np.abs(Y.mean(axis=0)), 1e-9)
    # noise floor from first differences on the whole pool
    d = np.concatenate([np.diff(r.Y, axis=0) for r in runs if r.T > 2], axis=0)
    noise = np.median(np.abs(d - np.median(d, axis=0)), axis=0) * 1.4826 / np.sqrt(2)
    return np.maximum(s, np.maximum(floor, 3 * noise))


def robust_score(Yhat, Ytrue, sigma, mults=(0.5, 1.0, 2.0)):
    """Selection criterion: average score over sigma multipliers (sigma is unknown)."""
    return float(np.mean([score(Yhat, Ytrue, np.asarray(sigma) * k) for k in mults]))
