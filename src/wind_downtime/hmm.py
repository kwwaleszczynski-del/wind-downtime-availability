"""State segmentation with a hidden Markov model, learned without labels.

Four hidden states - running normally, idling in low wind, derated, stopped - emit three features per
10-minute interval, each relative to the turbine's own normal-behaviour curves (``curves.py``):

    f1 = (P - P^(u)) / P_rated                                  power shortfall
    f2 = (pitch - pitch^(u)) / (pitch_park - pitch^(u))         0 = normal pitch, 1 = parked
    f3 = (rpm - rpm^(u)) / rpm_rated                            rotor speed below normal

with diagonal Gaussian emissions (a missing feature is left out of the likelihood). The parameters start from
physical values (``INIT``) and are refined on the farm's own data by Baum-Welch; the means are shrunk towards
their starting values so that the states keep their meaning. Decoding is Viterbi: a single noisy interval in
the middle of a stop cannot end it, because leaving a state costs probability. Idle is reported as normal.

Usage:
    model = fit_hmm(fleet, curves, rows=fleet.years() == 2019)
    states = hmm_states(fleet, curves, model)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wind_downtime.detect import DERATE, NORMAL, STOP, _finish, runs

NAMES = ("normal", "idle", "derate", "stop")
TO_STATE = np.array([NORMAL, NORMAL, DERATE, STOP], dtype=np.int8)
INIT = {  # means and standard deviations of (f1, f2, f3)
    "mean": np.array([[0.0, 0.0, 0.0], [0.0, 0.5, -0.9], [-0.25, 0.1, 0.0], [-0.3, 1.0, -1.0]]),
    "sd": np.array([[0.05, 0.03, 0.05], [0.05, 0.3, 0.2], [0.1, 0.08, 0.08], [0.3, 0.1, 0.1]]),
}
STAY = np.array([0.995, 0.98, 0.95, 0.98])
# pseudo-observations pulling each state's mean towards INIT: the derate state is held almost fixed, otherwise
# it drifts to absorb the many small transients (start-ups, gusts) and stops meaning "limited"
SHRINK = np.array([2e3, 2e3, 1e9, 1e5])
VAR_SHRINK = np.array([1e3, 1e3, 1e9, 1e3])
MIN_STAY = 0.9
MIN_DERATE = 3  # intervals: shorter derate segments are transients (start-up, partial stops)
SD_FLOOR = 0.02


@dataclass
class HMM:
    mean: np.ndarray  # (state, feature)
    sd: np.ndarray
    trans: np.ndarray  # (state, state)
    start: np.ndarray
    loglik: list


def features(fleet, curves) -> np.ndarray:
    rated = fleet.rated[None, :]
    f1 = (fleet.power - curves.power(fleet.ws)) / rated
    p_norm = curves.pitch(fleet.ws)
    f2 = np.clip((fleet.pitch - p_norm) / np.maximum(curves.park_pitch[None, :] - p_norm, 20.0), -0.5, 1.5)
    f3 = np.clip((fleet.rpm - curves.rpm(fleet.ws)) / curves.rated_rpm[None, :], -1.5, 0.5)
    return np.stack([np.clip(f1, -1.2, 0.5), f2, f3], axis=-1)  # (time, turbine, feature)


def log_emission(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(time, turbine, state) log densities; NaN features contribute nothing."""
    z = (x[..., None, :] - mean) / sd
    ll = -0.5 * z**2 - np.log(sd) - 0.5 * np.log(2 * np.pi)
    return np.nansum(np.where(np.isnan(z), 0.0, ll), axis=-1)


def _probs(logb):
    b = np.exp(logb - logb.max(axis=-1, keepdims=True))
    return np.maximum(b, 1e-300)


def forward_backward(b, trans, start):
    """Scaled forward-backward over (time, sequence, state) emission probabilities."""
    t_len, n, k = b.shape
    alpha = np.empty_like(b)
    c = np.empty((t_len, n))
    a = start[None, :] * b[0]
    c[0] = a.sum(-1)
    alpha[0] = a / c[0][:, None]
    for t in range(1, t_len):
        a = (alpha[t - 1] @ trans) * b[t]
        c[t] = a.sum(-1)
        alpha[t] = a / c[t][:, None]
    beta = np.ones_like(b)
    xi = np.zeros((k, k))
    for t in range(t_len - 2, -1, -1):
        nb = b[t + 1] * beta[t + 1]
        beta[t] = (nb @ trans.T) / c[t + 1][:, None]
        xi += trans * np.einsum("ni,nj->ij", alpha[t], nb / c[t + 1][:, None])
    gamma = alpha * beta
    gamma /= gamma.sum(-1, keepdims=True)
    return gamma, xi, np.log(c).sum()


def fit_hmm(fleet, curves, rows=None, iterations: int = 8) -> HMM:
    x = features(fleet, curves)
    if rows is not None:
        x = x[rows]
    valid = np.isfinite(x[..., 0])
    # the idle state starts from where this farm actually idles: not producing below cut-in, not parked
    # (Senvion idles with the blades at 45 deg, Siemens at 0 deg with the rotor turning)
    f = fleet if rows is None else fleet.subset(rows)
    idle = ((f.power <= 0.02 * f.rated[None, :]) & (f.ws < curves.cut_in[None, :])
            & (f.pitch < curves.park_pitch[None, :] - 15.0))
    init = {k: v.copy() for k, v in INIT.items()}
    if idle.sum() > 100:
        init["mean"][1, 1:] = np.nanmedian(x[idle][:, 1:], axis=0)
    trans = np.full((4, 4), 0.0)
    for i, s in enumerate(STAY):
        trans[i] = (1 - s) / 3
        trans[i, i] = s
    model = HMM(init["mean"].copy(), init["sd"].copy(), trans, np.full(4, 0.25), [])
    for _ in range(iterations):
        logb = log_emission(x, model.mean, model.sd)
        logb[~valid] = 0.0
        gamma, xi, ll = forward_backward(_probs(logb), model.trans, model.start)
        model.loglik.append(float(ll))
        w = np.where(valid[..., None], gamma, 0.0)
        for f in range(x.shape[-1]):
            ok = np.isfinite(x[..., f])
            wf = np.where(ok[..., None], w, 0.0)
            xf = np.where(ok, x[..., f], 0.0)[..., None]
            n_k = wf.sum((0, 1))
            mu = ((wf * xf).sum((0, 1)) + SHRINK * init["mean"][:, f]) / (n_k + SHRINK)
            var = ((wf * (xf - mu) ** 2).sum((0, 1)) + VAR_SHRINK * init["sd"][:, f] ** 2) / (n_k + VAR_SHRINK)
            model.mean[:, f], model.sd[:, f] = mu, np.maximum(np.sqrt(var), SD_FLOOR)
        trans = xi / xi.sum(1, keepdims=True)
        stay = np.maximum(np.diag(trans), MIN_STAY)
        off = trans * (1 - np.eye(4))
        trans = off / off.sum(1, keepdims=True) * (1 - stay)[:, None] + np.diag(stay)
        model.trans = trans
        model.start = gamma[0].mean(0)
    return model


def viterbi(logb, trans, start) -> np.ndarray:
    t_len, n, k = logb.shape
    lt = np.log(trans)
    delta = np.log(np.maximum(start, 1e-12))[None, :] + logb[0]
    back = np.empty((t_len, n, k), dtype=np.int8)
    for t in range(1, t_len):
        cand = delta[:, :, None] + lt[None]
        back[t] = cand.argmax(1)
        delta = cand.max(1) + logb[t]
    path = np.empty((t_len, n), dtype=np.int8)
    path[-1] = delta.argmax(-1)
    idx = np.arange(n)
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, idx, path[t]]
    return path


def hmm_states(fleet, curves, model: HMM) -> np.ndarray:
    x = features(fleet, curves)
    logb = log_emission(x, model.mean, model.sd)
    logb[~np.isfinite(x[..., 0])] = 0.0
    path = viterbi(logb, model.trans, model.start)
    s = TO_STATE[path]
    derate = s == DERATE
    for i in range(derate.shape[1]):
        a, b = runs(derate[:, i])
        for k0, k1 in zip(a, b):
            if k1 - k0 < MIN_DERATE:
                derate[k0:k1, i] = False
    return _finish(fleet, s == STOP, derate)
