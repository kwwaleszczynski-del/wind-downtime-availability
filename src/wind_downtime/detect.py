"""Signal-only detectors of stops and deratings, interval by interval (no status codes).

Every detector returns a (time, turbine) int8 array of states (index into ``schema.STATES``). Missing and frozen
data are decided first and identically for all detectors (``no_data``): a row that is absent, has no power or
wind speed, or whose power, wind speed and rotor speed repeat exactly for 30 minutes or more (a stuck
data link).

==============  ===========================================================================================
detector        rule
==============  ===========================================================================================
``rules``       stop: power <= 0 at a wind speed >= 3.5 m/s (the usual first check); no derate
``residual``    the turbine's own power curve at its anemometer: stop when power <= 2 % of rated where 5 %
                or more is expected; derate when power is short by > 5 % of rated *and* held flat (a
                setpoint), or short by > 25 %, for 20 minutes - until the shortfall falls below 3 % for 20
                minutes (hysteresis)
``signature``   pitch and rotor speed: stop when the rotor is (nearly) still and the blades are feathered to
                the parking angle - at any wind speed, so stops during low wind count too; derate when the
                blades are pitched > 3 deg beyond the normal pitch at that wind speed, the rotor turns at
                speed and power is short by > 5 % of rated (same hysteresis)
==============  ===========================================================================================

The HMM and gradient boosting detectors are in ``hmm.py`` and ``ml.py``.

Usage:
    curves = fit_curves(fleet)
    states = {name: fn(fleet, curves) for name, fn in DETECTORS.items()}
"""
from __future__ import annotations

import numpy as np

from wind_downtime.curves import Curves
from wind_downtime.schema import STATES

NORMAL, DERATE, STOP, NO_DATA = range(4)
assert STATES == ("normal", "derate", "stop", "no_data")
FROZEN_N = 3


def frozen(x: np.ndarray, n: int = FROZEN_N) -> np.ndarray:
    """True where a column repeats the same value exactly for at least ``n`` consecutive rows."""
    same = np.zeros(x.shape, dtype=bool)
    same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
    out = np.zeros(x.shape, dtype=bool)
    for i in range(x.shape[1]):
        s = same[:, i]
        start = np.flatnonzero(s & ~np.r_[False, s[:-1]])  # first repeat of a run
        end = np.flatnonzero(s & ~np.r_[s[1:], False]) + 1
        for a, b in zip(start, end):
            if b - a + 1 >= n:
                out[a - 1:b, i] = True
    return out


def no_data(fleet) -> np.ndarray:
    absent = ~fleet.present | np.isnan(fleet.power) | np.isnan(fleet.ws)
    stuck = frozen(fleet.power) & frozen(fleet.ws) & (frozen(fleet.rpm) | np.isnan(fleet.rpm))
    return absent | stuck


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of the True runs of a 1-D mask."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def hysteresis(enter: np.ndarray, stay: np.ndarray, n_enter: int = 2, n_exit: int = 2) -> np.ndarray:
    """Two-threshold switch per column.

    An episode is a run of ``stay`` rows (``enter`` implies ``stay``; gaps shorter than ``n_exit`` rows are
    bridged). It switches on at the start of the first streak of ``n_enter`` consecutive ``enter`` rows and
    stays on to the end of the run.
    """
    enter = enter.astype(bool)
    stay = stay.astype(bool) | enter
    out = np.zeros(enter.shape, dtype=bool)
    for i in range(enter.shape[1]):
        st = stay[:, i].copy()
        gs, ge = runs(~st)
        for a, b in zip(gs, ge):
            if b - a < n_exit and a > 0 and b < len(st):
                st[a:b] = True
        es, ee = runs(enter[:, i])
        es = es[(ee - es) >= n_enter]
        if not len(es):
            continue
        rs, re = runs(st)
        j = np.searchsorted(es, rs)
        for a, b, k in zip(rs, re, j):
            if k < len(es) and es[k] < b:
                out[es[k]:b, i] = True
    return out


def _finish(fleet, stop, derate) -> np.ndarray:
    out = np.full(fleet.power.shape, NORMAL, dtype=np.int8)
    out[derate] = DERATE
    out[stop] = STOP
    out[no_data(fleet)] = NO_DATA
    return out


def rules(fleet, curves: Curves | None = None) -> np.ndarray:
    stop = (fleet.power <= 0.0) & (fleet.ws >= 3.5)
    return _finish(fleet, stop, np.zeros_like(stop))


def plateau(fleet, tol: float = 0.015) -> np.ndarray:
    """Power held at a constant level below rated over three consecutive intervals: a setpoint at work.

    Free-running power follows the wind and moves by several per cent of rated from one 10-minute mean to the
    next; under a power limit it stays within ``tol`` of rated.
    """
    p, rated = fleet.power, fleet.rated[None, :]
    stack = np.stack([np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0)])
    stack[0, 0], stack[2, -1] = np.nan, np.nan
    spread = np.max(stack, axis=0) - np.min(stack, axis=0)
    return (spread < tol * rated) & (p > 0.05 * rated) & (p < 0.97 * rated)


def residual(fleet, curves: Curves) -> np.ndarray:
    rated = fleet.rated[None, :]
    p_hat = curves.power(fleet.ws)
    stop = (fleet.power <= 0.02 * rated) & (p_hat >= 0.05 * rated)
    short = (p_hat - fleet.power) / rated
    producing = fleet.power > 0.02 * rated
    enter = producing & ((short > 0.05) & plateau(fleet) | (short > 0.25))
    derate = hysteresis(enter, (short > 0.03) & producing)
    return _finish(fleet, stop, derate & ~stop)


def parked(fleet, curves: Curves) -> np.ndarray:
    """Rotor (nearly) still with the blades at the parking angle."""
    slow = fleet.rpm < 0.15 * curves.rated_rpm[None, :]
    feathered = fleet.pitch >= curves.park_pitch[None, :] - 15.0
    return slow & feathered


def signature(fleet, curves: Curves) -> np.ndarray:
    rated = fleet.rated[None, :]
    p_hat = curves.power(fleet.ws)
    slow = fleet.rpm < 0.15 * curves.rated_rpm[None, :]
    stop = parked(fleet, curves) & (fleet.power <= 0.02 * rated)
    # without a pitch signal fall back to a still rotor where power is expected
    stop |= np.isnan(fleet.pitch) & slow & (fleet.power <= 0.02 * rated) & (p_hat >= 0.05 * rated)
    pitch_dev = fleet.pitch - curves.pitch(fleet.ws)
    at_speed = fleet.rpm >= 0.8 * curves.rpm(fleet.ws)
    short = (p_hat - fleet.power) / rated
    producing = fleet.power > 0.02 * rated
    enter = (pitch_dev > 3.0) & at_speed & (short > 0.05) & producing
    stay = (pitch_dev > 1.5) & at_speed & (short > 0.02) & producing
    derate = hysteresis(enter, stay)
    return _finish(fleet, stop, derate & ~stop)


DETECTORS = {"rules": rules, "residual": residual, "signature": signature}
