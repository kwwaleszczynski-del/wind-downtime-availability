"""Per-turbine normal-behaviour curves against nacelle wind speed: power, pitch and rotor speed.

Robust binned medians over the intervals that look like normal operation (producing, not obviously limited),
so that they can be learned from the same unlabelled data the detectors run on. Each curve is made monotone
where physics says so (power and rotor speed rise with wind speed) and extended flat beyond the last bin.

Usage:
    curves = fit_curves(fleet, rows=fleet.years() == 2019)
    p_exp = curves.power(fleet.ws)          # (time, turbine) expected power at the own anemometer
    pitch_dev = fleet.pitch - curves.pitch(fleet.ws)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN = 0.5
WS_GRID = np.arange(0.0, 30.01, BIN)
MIN_COUNT = 12  # intervals per bin (2 hours)


@dataclass
class Curves:
    grid: np.ndarray
    p: np.ndarray  # (bin, turbine) median power
    pitch_: np.ndarray  # (bin, turbine) median pitch
    rpm_: np.ndarray  # (bin, turbine) median rotor speed
    rated: np.ndarray
    rated_rpm: np.ndarray  # 99th percentile of rotor speed while producing
    cut_in: np.ndarray  # lowest wind speed bin with median power > 2 % of rated
    park_pitch: np.ndarray  # pitch of a parked (feathered) rotor

    def _interp(self, table, ws):
        """Look up a (bin, turbine) table at (time, turbine) wind speeds."""
        ws = np.asarray(ws, dtype=float)
        out = np.full(ws.shape, np.nan)
        for i in range(table.shape[1]):
            ok = ~np.isnan(table[:, i])
            if ok.sum() >= 2:
                out[:, i] = np.interp(ws[:, i], self.grid[ok], table[ok, i])
        out[np.isnan(ws)] = np.nan
        return out

    def power(self, ws):
        """Expected power at the given (time, turbine) wind speeds; 0 below the first valid bin."""
        out = self._interp(self.p, ws)
        ws = np.asarray(ws, dtype=float)
        low = ws < self.cut_in[None, :] - BIN
        out[low] = 0.0
        out[ws > 25.0] = 0.0
        return out

    def pitch(self, ws):
        return self._interp(self.pitch_, ws)

    def rpm(self, ws):
        return self._interp(self.rpm_, ws)


def _binned_median(ws, values, mask):
    out = np.full((len(WS_GRID), ws.shape[1]), np.nan)
    b = np.round(ws / BIN).astype(float)
    for i in range(ws.shape[1]):
        ok = mask[:, i] & np.isfinite(values[:, i]) & np.isfinite(ws[:, i])
        if not ok.any():
            continue
        bi, v = b[ok, i].astype(int), values[ok, i]
        bi = np.clip(bi, 0, len(WS_GRID) - 1)
        order = np.argsort(bi, kind="stable")
        bi, v = bi[order], v[order]
        edges = np.flatnonzero(np.r_[True, bi[1:] != bi[:-1], True])
        for s, e in zip(edges[:-1], edges[1:]):
            if e - s >= MIN_COUNT:
                out[bi[s], i] = np.median(v[s:e])
    return out


def fit_curves(fleet, rows=None, exclude=None) -> Curves:
    """Learn the curves on ``rows`` (bool over time); ``exclude`` (time, turbine) masks known abnormal records.

    Pass 1 takes every producing record; pass 2 drops records more than 15 % of rated below the pass-1 power
    curve (stops, deratings, icing), which keeps the medians on normal operation even when a turbine spent a
    long time limited.
    """
    rows = np.ones(len(fleet.time), dtype=bool) if rows is None else rows
    ws, p, pitch, rpm = (np.where(rows[:, None], a, np.nan) for a in (fleet.ws, fleet.power, fleet.pitch, fleet.rpm))
    base = np.isfinite(ws) & np.isfinite(p)
    if exclude is not None:
        base &= ~exclude
    producing = base & (p > 0.02 * fleet.rated[None, :])
    rated = fleet.rated
    curve1 = _binned_median(ws, p, producing)
    keep = producing.copy()
    tmp = Curves(WS_GRID, _monotone(curve1), curve1, curve1, rated, rated, np.zeros_like(rated), rated)
    keep &= p >= tmp._interp(tmp.p, ws) - 0.15 * rated[None, :]
    pc = _monotone(_binned_median(ws, p, keep))
    pitch_c = _binned_median(ws, pitch, keep)
    rpm_c = _monotone(_binned_median(ws, rpm, keep))
    rated_rpm = np.array([np.nanpercentile(np.where(keep[:, i], rpm[:, i], np.nan), 99)
                          if keep[:, i].any() else np.nan for i in range(fleet.n)])
    cut_in = np.array([WS_GRID[np.flatnonzero(pc[:, i] > 0.02 * rated[i])[0]] if (pc[:, i] > 0.02 * rated[i]).any()
                       else 3.5 for i in range(fleet.n)])
    # a parked rotor: the highest pitch seen while not producing (90 deg on most turbines, 77 deg at the
    # Siemens farm); a high percentile, because low-wind idling (e.g. 45 deg) is far more common than stops
    idle = base & (p <= 0.0) & np.isfinite(pitch)
    park = np.array([np.nanpercentile(pitch[idle[:, i], i], 99) if idle[:, i].sum() > 50 else 85.0
                     for i in range(fleet.n)])
    return Curves(WS_GRID, pc, pitch_c, rpm_c, rated.astype(float), rated_rpm, cut_in, park)


def _monotone(table):
    """Running maximum over wind speed bins (NaN bins kept NaN)."""
    out = table.copy()
    for i in range(table.shape[1]):
        col = out[:, i]
        ok = ~np.isnan(col)
        col[ok] = np.maximum.accumulate(col[ok])
    return out
