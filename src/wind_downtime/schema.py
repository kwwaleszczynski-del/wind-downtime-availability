"""The generic input format, the IEC 61400-26 categories, and the array view of a fleet used by every method.

Input: one long table, one row per turbine and 10-minute interval (Parquet or CSV).

==============  =========================================================  ==========
column          meaning                                                    required
==============  =========================================================  ==========
timestamp       start of the 10-minute interval, UTC                       yes
turbine_id      any string                                                 yes
power           mean active power [kW]                                     yes
wind_speed      mean nacelle wind speed [m/s]                              yes
rotor_speed     mean rotor speed [rpm]                                     yes
pitch           mean blade pitch angle [deg]                               yes
nacelle_dir     nacelle position / yaw angle [deg]                         yes
ambient_temp    ambient (or nacelle outside) temperature [degC]            yes
status_code     code active in the interval (dominant one), any string     no
==============  =========================================================  ==========

Missing values are allowed (NaN). Rated power comes from a turbine table or a single ``--rated-kw``.

Usage:
    long = read_long(Path("scada.parquet"))
    fleet = to_fleet(long, rated_kw=2050)
    fleet.power.shape          # (n_intervals, n_turbines) on a complete 10-minute grid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STEP = pd.Timedelta("10min")
HOURS = 1 / 6  # one interval in hours
SIGNALS = ("power", "wind_speed", "rotor_speed", "pitch", "nacelle_dir", "ambient_temp")
REQUIRED = ("timestamp", "turbine_id") + SIGNALS

# Interval states found by the detectors; categories (below) say why.
STATES = ("normal", "derate", "stop", "no_data")

# Information categories of IEC 61400-26-1 (time-based availability), as used here. The standard's tree is
# finer (e.g. "out of environmental specification" covers low wind, high wind, icing and temperature); the
# exact definitions and their contractual use should be checked against the standard and the contract.
CATEGORIES = {
    "full_performance": "Full performance",
    "partial_performance": "Partial performance (technical derating)",
    "technical_standby": "Technical standby",
    "environmental": "Out of environmental specification",
    "requested": "Requested shutdown or curtailment (grid operator or owner)",
    "electrical": "Out of electrical specification (grid)",
    "scheduled_maintenance": "Scheduled maintenance",
    "planned_corrective": "Planned corrective action",
    "forced_outage": "Forced outage",
    "no_info": "Information unavailable",
}
CATEGORY_LIST = list(CATEGORIES)
CAT = {c: i for i, c in enumerate(CATEGORY_LIST)}
STOP_CATEGORIES = ("technical_standby", "environmental", "requested", "electrical", "scheduled_maintenance",
                   "planned_corrective", "forced_outage")
DERATE_CATEGORIES = ("partial_performance", "requested")


def read_long(path: Path) -> pd.DataFrame:
    """Read a long table (Parquet or CSV), check the columns and normalise the types."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    return validate_long(df)


def validate_long(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}; required: {list(REQUIRED)}")
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    if (ts.dt.second != 0).any() or (ts.dt.minute % 10 != 0).any():
        raise ValueError("timestamps must be on the 10-minute grid (interval start, UTC)")
    out["timestamp"] = ts
    out["turbine_id"] = out["turbine_id"].astype(str)
    for c in SIGNALS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "status_code" in out:
        code = out["status_code"]
        out["status_code"] = code.where(code.isna(), code.astype(str).str.replace(r"\.0$", "", regex=True))
    dup = out.duplicated(["turbine_id", "timestamp"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated (turbine_id, timestamp) rows")
    return out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


@dataclass
class Fleet:
    """Signals as (n_intervals, n_turbines) arrays on a complete 10-minute grid; NaN where nothing was logged."""

    time: pd.DatetimeIndex
    turbines: np.ndarray
    rated: np.ndarray  # kW per turbine
    power: np.ndarray
    ws: np.ndarray
    rpm: np.ndarray
    pitch: np.ndarray
    ndir: np.ndarray
    temp: np.ndarray
    present: np.ndarray  # bool: a row existed for this turbine and interval
    code: np.ndarray | None = None  # object array of status codes (None where absent)
    name: str = "fleet"
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.turbines)

    def __len__(self) -> int:
        return len(self.time)

    def subset(self, rows: np.ndarray) -> "Fleet":
        """Fleet restricted to a contiguous block of rows (bool mask or slice) of the time axis."""
        idx = np.flatnonzero(rows) if isinstance(rows, np.ndarray) and rows.dtype == bool else rows
        sl = slice(idx[0], idx[-1] + 1) if not isinstance(idx, slice) else idx
        arrays = {k: getattr(self, k)[sl] for k in ("power", "ws", "rpm", "pitch", "ndir", "temp", "present")}
        extra = {k: (v[sl] if isinstance(v, np.ndarray) and v.shape[:1] == (len(self.time),) else v)
                 for k, v in self.extra.items()}
        return Fleet(time=self.time[sl], turbines=self.turbines, rated=self.rated, name=self.name,
                     code=None if self.code is None else self.code[sl], extra=extra, **arrays)

    def years(self) -> np.ndarray:
        return np.asarray(self.time.year)


def to_fleet(long: pd.DataFrame, rated_kw=None, name: str = "fleet") -> Fleet:
    """Long table -> Fleet. ``rated_kw`` is a number, a {turbine_id: kW} mapping, or None (99th percentile)."""
    long = validate_long(long)
    t0, t1 = long["timestamp"].min(), long["timestamp"].max()
    grid = pd.date_range(t0, t1, freq=STEP, name="timestamp")
    turbines = np.array(sorted(long["turbine_id"].unique()))
    idx = pd.MultiIndex.from_frame(long[["timestamp", "turbine_id"]])

    def wide(col):
        s = pd.Series(long[col].to_numpy(), index=idx)
        return s.unstack("turbine_id").reindex(index=grid, columns=turbines).to_numpy()

    present = pd.Series(True, index=idx).unstack("turbine_id").reindex(index=grid, columns=turbines)
    present = present.fillna(False).to_numpy(dtype=bool)
    arrays = {dst: wide(src).astype(float) for src, dst in
              (("power", "power"), ("wind_speed", "ws"), ("rotor_speed", "rpm"), ("pitch", "pitch"),
               ("nacelle_dir", "ndir"), ("ambient_temp", "temp"))}
    if rated_kw is None:
        rated = np.nanpercentile(arrays["power"], 99.5, axis=0).round(-1)
    elif np.isscalar(rated_kw):
        rated = np.full(len(turbines), float(rated_kw))
    else:
        rated = np.array([float(rated_kw[t]) for t in turbines])
    code = None
    if "status_code" in long and long["status_code"].notna().any():
        code = wide("status_code").astype(object)
        code[pd.isna(code)] = None
    return Fleet(time=grid, turbines=turbines, rated=rated, present=present, code=code, name=name, **arrays)


def fleet_to_long(fleet: Fleet) -> pd.DataFrame:
    """Inverse of ``to_fleet`` (only rows that were present)."""
    t_idx, i_idx = np.nonzero(fleet.present)
    out = pd.DataFrame({
        "timestamp": fleet.time[t_idx], "turbine_id": fleet.turbines[i_idx],
        "power": fleet.power[t_idx, i_idx], "wind_speed": fleet.ws[t_idx, i_idx],
        "rotor_speed": fleet.rpm[t_idx, i_idx], "pitch": fleet.pitch[t_idx, i_idx],
        "nacelle_dir": fleet.ndir[t_idx, i_idx], "ambient_temp": fleet.temp[t_idx, i_idx],
    })
    if fleet.code is not None:
        out["status_code"] = fleet.code[t_idx, i_idx]
    return out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
