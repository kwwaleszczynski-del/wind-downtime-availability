"""Supervised detector: gradient boosting on per-interval features, trained on logged states.

All features are relative to the turbine's own normal-behaviour curves and rated values, so a model trained
on one turbine type can be applied to another (the transfer test: Senvion -> Siemens):

* the interval itself: power and expected power / rated, shortfall, wind speed above cut-in, pitch beyond
  normal and relative to the parking angle, rotor speed relative to rated and to normal, temperature;
* context: the same signals 10 and 30 minutes before and after, and the spread of power over an hour;
* the farm: share of the other turbines that are still where power is expected, their median shortfall.

Labels are the reference states (normal, derate, stop); intervals without data are left out.

Usage:
    X = interval_features(fleet, curves)
    model = train_gbm([(X_a, ref_a.state, rows_a), (X_b, ref_b.state, rows_b)])
    states = gbm_states(fleet, model, X)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from wind_downtime.detect import DERATE, NO_DATA, STOP, _finish, runs

MIN_DERATE = 3
NORMAL_SAMPLE = 0.15  # share of normal intervals kept for training (stops and derates are all kept)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    return out


def _loo_median(v: np.ndarray) -> np.ndarray:
    out = np.full(v.shape, np.nan)
    for i in range(v.shape[1]):
        others = np.delete(v, i, axis=1)
        ok = np.isfinite(others).any(axis=1)
        out[ok, i] = np.nanmedian(others[ok], axis=1)
    return out


def interval_features(fleet, curves) -> tuple[np.ndarray, list[str]]:
    rated = fleet.rated[None, :]
    rated_rpm = curves.rated_rpm[None, :]
    p_hat = curves.power(fleet.ws)
    pn, phn = fleet.power / rated, p_hat / rated
    short = phn - pn
    pitch_dev = (fleet.pitch - curves.pitch(fleet.ws)) / 30.0
    pitch_park = fleet.pitch / curves.park_pitch[None, :]
    rpm_n = fleet.rpm / rated_rpm
    rpm_dev = (fleet.rpm - curves.rpm(fleet.ws)) / rated_rpm
    still = ((pn <= 0.02) & (phn >= 0.05)).astype(float)
    still[np.isnan(pn) | np.isnan(phn)] = np.nan
    cols = {
        "p": pn, "p_hat": phn, "short": short, "ws_above_cut_in": fleet.ws - curves.cut_in[None, :],
        "pitch_dev": pitch_dev, "pitch_park": pitch_park, "rpm": rpm_n, "rpm_dev": rpm_dev, "temp": fleet.temp,
    }
    for name in ("p", "short", "pitch_park", "rpm"):
        for k in (1, 3, -1, -3):
            cols[f"{name}_{'lag' if k > 0 else 'lead'}{abs(k)}"] = _shift(cols[name], k)
    stack = np.stack([_shift(pn, k) for k in range(-3, 4)])
    cols["p_spread_1h"] = np.nanstd(stack, axis=0)
    cols["farm_still"] = _loo_median(still)
    cols["farm_short"] = _loo_median(short)
    names = list(cols)
    return np.stack([cols[n] for n in names], axis=-1).astype(np.float32), names


def training_rows(X: np.ndarray, state: np.ndarray, rows: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    sel = rows[:, None] & (state != NO_DATA) & np.isfinite(X[..., 0])
    keep = sel & ((state != 0) | (rng.random(state.shape) < NORMAL_SAMPLE))
    y = state[keep]
    w = np.where(y == 0, 1.0 / NORMAL_SAMPLE, 1.0)
    return X[keep], y, w


def train_gbm(parts, seed: int = 0) -> HistGradientBoostingClassifier:
    """``parts``: iterable of (X, reference state, row mask) per dataset."""
    rng = np.random.default_rng(seed)
    xs, ys, ws = zip(*(training_rows(X, s, r, rng) for X, s, r in parts))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    return model


def gbm_states(fleet, model, X: np.ndarray) -> np.ndarray:
    t, n, f = X.shape
    flat = X.reshape(-1, f)
    ok = np.isfinite(flat[:, 0])
    pred = np.zeros(t * n, dtype=np.int8)
    pred[ok] = model.predict(flat[ok])
    pred = pred.reshape(t, n)
    derate = pred == DERATE
    for i in range(n):
        a, b = runs(derate[:, i])
        for k0, k1 in zip(a, b):
            if k1 - k0 < MIN_DERATE:
                derate[k0:k1, i] = False
    return _finish(fleet, pred == STOP, derate)
