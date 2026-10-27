"""Reference labels from what the operators logged: the yardstick for every signal-only method.

For each 10-minute interval and turbine the reference gives a ``state`` (normal, derate, stop, no_data) and
an IEC 61400-26 ``category`` (index into ``schema.CATEGORY_LIST``, -1 where the logs do not say).

Greenbyte farms (Kelmarsh, Penmanshiel). The status logs carry an IEC category per event (the operator's
mapping in Greenbyte, ``GREENBYTE_IEC``). An interval is a **stop** when `Stop` events cover at least half of
it; its category is the one with the longest overlap. It is a **derate** when a power-reduction warning is
active for at least half of it *and* the limit binds: the turbine produced at least 5 % of rated power less
than its normal curve at the measured wind speed (a curtailment instruction during low wind costs nothing and
leaves no trace in the signals). Code 9000 *P output externally reduced* is an external instruction
(``requested``); the other reductions (gearbox, temperature) are technical (``partial_performance``).

Planned corrective action is not a Greenbyte category. It is derived here, as an explicit assumption: an
on-site manual stop (*Scheduled Maintenance* in Greenbyte) that starts more than an hour and less than
``CORRECTIVE_WINDOW`` after a forced outage of the same turbine ended is the planned follow-up repair; one that
starts within the hour continues the forced outage.

Hill of Towie. The operator published the shutdown duration of every interval (seconds out of 600, from
SCADA and operator logs); >= 300 s is a stop. Why: the controller's stop timers (grid stop -> electrical,
environmental stop -> environmental, turbine error -> forced outage, each >= 300 s), else the alarm log, which
describes only 12 codes (icing, high and low wind -> environmental; cable untwisting, pitch lubrication ->
technical standby). Anything else stays unknown (-1). A derate is an active power setpoint below 97 % of rated
that binds: the turbine produces within 10 % of rated of the setpoint while its normal curve at the measured
wind speed is at least 5 % of rated above it.

Usage:
    ref = reference(ds, curves)
    ref.state, ref.category      # (time, turbine) int8 arrays
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_downtime.schema import CAT, CATEGORY_LIST, STATES

GREENBYTE_IEC = {
    "Full Performance": "full_performance",
    "Partial Performance": "partial_performance",
    "Technical Standby": "technical_standby",
    "Out of Environmental Specification": "environmental",
    "Requested Shutdown": "requested",
    "Out of Electrical Specification": "electrical",
    "Scheduled Maintenance": "scheduled_maintenance",
    "Forced outage": "forced_outage",
}
CURTAILMENT_CODES = {9000}  # "P output externally reduced"
DERATE_PATTERN = r"P red|externally reduced|[Rr]educed power|[Pp]ower reduction|[Pp]ower limit"
HOT_ALARMS = {1005: "environmental", 8000: "environmental", 8210: "environmental", 8230: "environmental",
              8234: "environmental", 8235: "environmental", 8236: "environmental", 102: "environmental",
              10105: "technical_standby", 3130: "technical_standby"}
CORRECTIVE_WINDOW = pd.Timedelta("3D")
REPAIR_GAP = pd.Timedelta("1h")
BINDING = 0.05  # fraction of rated power a limit must cost to count as a derate
HALF = 300.0  # seconds
S, D, N, ND = (STATES.index(s) for s in ("normal", "derate", "stop", "no_data"))


@dataclass
class Reference:
    state: np.ndarray  # (time, turbine) int8, index into schema.STATES
    category: np.ndarray  # (time, turbine) int8, index into schema.CATEGORY_LIST or -1
    source: str


def overlap_seconds(time: pd.DatetimeIndex, turbines, events: pd.DataFrame, n_cat: int) -> np.ndarray:
    """Seconds of each interval covered by events of each category: array (time, turbine, category).

    ``events``: turbine_id, start, end, cat (int). Overlapping events of the same category are counted once
    per second only approximately (their seconds add up, capped at 600 per interval and category).
    """
    out = np.zeros((len(time), len(turbines), n_cat), dtype=np.float32)
    if events.empty:
        return out
    t0 = time[0].as_unit("ns").value if hasattr(time[0], "as_unit") else time[0].value
    step = 600 * 10**9
    col = {t: i for i, t in enumerate(turbines)}
    ev = events[events["turbine_id"].isin(col) & events["end"].notna() & (events["end"] > events["start"])]
    s = (pd.DatetimeIndex(ev["start"]).as_unit("ns").asi8 - t0)
    e = (pd.DatetimeIndex(ev["end"]).as_unit("ns").asi8 - t0)
    k0, k1 = np.floor_divide(s, step), np.floor_divide(e - 1, step)
    k0c, k1c = np.clip(k0, 0, len(time) - 1), np.clip(k1, 0, len(time) - 1)
    keep = (k1 >= 0) & (k0 < len(time))
    s, e, k0c, k1c = s[keep], e[keep], k0c[keep], k1c[keep]
    tid = ev["turbine_id"].map(col).to_numpy()[keep]
    cat = ev["cat"].to_numpy()[keep]
    reps = (k1c - k0c + 1).astype(np.int64)
    k = np.repeat(k0c, reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))
    lo = np.maximum(np.repeat(s, reps), k * step)
    hi = np.minimum(np.repeat(e, reps), (k + 1) * step)
    sec = np.clip((hi - lo) / 1e9, 0, 600).astype(np.float32)
    np.add.at(out, (k, np.repeat(tid, reps), np.repeat(cat, reps)), sec)
    return np.minimum(out, 600.0)


def derive_corrective(ev: pd.DataFrame) -> pd.Series:
    """Re-label scheduled-maintenance stops that follow a forced outage (see module docstring)."""
    cats = ev["category"].copy()
    for _, g in ev.sort_values("start").groupby("turbine_id"):
        fo_end = g["end"].where(g["category"].eq("forced_outage")).ffill().shift(1)
        sm = g["category"].eq("scheduled_maintenance") & fo_end.notna()
        gap = g["start"] - fo_end
        cats.loc[g.index[sm & (gap <= REPAIR_GAP)]] = "forced_outage"
        cats.loc[g.index[sm & (gap > REPAIR_GAP) & (gap <= CORRECTIVE_WINDOW)]] = "planned_corrective"
    return cats


def _finish(fleet, is_stop, stop_cat, is_derate, derate_cat) -> tuple[np.ndarray, np.ndarray]:
    """Combine stop and derate masks into state and category; missing data overrides the state only."""
    nodata = ~fleet.present | np.isnan(fleet.power) | np.isnan(fleet.ws)
    state = np.full(fleet.power.shape, S, dtype=np.int8)
    # running or idling for lack of wind: available, booked as full performance (as the detectors do)
    category = np.full(fleet.power.shape, CAT["full_performance"], dtype=np.int8)
    state[is_derate] = D
    category[is_derate] = derate_cat[is_derate]
    state[is_stop] = N
    category[is_stop] = stop_cat[is_stop]
    state[nodata] = ND
    category[nodata & ~is_stop] = CAT["no_info"]
    return state, category


def binding_limit(fleet, curves) -> np.ndarray:
    return curves.power(fleet.ws) - fleet.power > BINDING * fleet.rated[None, :]


def greenbyte_reference(ds, curves) -> Reference:
    fleet, ev = ds.fleet, ds.events.copy()
    ev["category"] = ev["iec_category"].map(GREENBYTE_IEC)
    stops = ev[ev["status"].eq("Stop") & ev["category"].notna()].copy()
    stops["category"] = derive_corrective(stops)
    derates = ev[ev["status"].ne("Stop") & (ev["message"].fillna("").str.contains(DERATE_PATTERN)
                                           | ev["category"].eq("partial_performance"))].copy()
    derates["category"] = np.where(derates["code"].isin(CURTAILMENT_CODES), "requested", "partial_performance")
    n_cat = len(CATEGORY_LIST)
    stop_sec = overlap_seconds(fleet.time, fleet.turbines, stops.assign(cat=stops["category"].map(CAT)), n_cat)
    der_sec = overlap_seconds(fleet.time, fleet.turbines, derates.assign(cat=derates["category"].map(CAT)), n_cat)
    is_stop = stop_sec.sum(-1) >= HALF
    is_derate = (der_sec.sum(-1) >= HALF) & binding_limit(fleet, curves)
    state, category = _finish(fleet, is_stop, np.argmax(stop_sec, -1), is_derate, np.argmax(der_sec, -1))
    return Reference(state, category, "greenbyte status log")


def hot_reference(ds, curves) -> Reference:
    fleet, ev, x = ds.fleet, ds.events, ds.extra
    known = ev[ev["code"].isin(HOT_ALARMS)].copy()
    known["cat"] = known["code"].map(HOT_ALARMS).map(CAT)
    alarm_sec = overlap_seconds(fleet.time, fleet.turbines, known, len(CATEGORY_LIST))
    is_stop = np.nan_to_num(x["shutdown_s"]) >= HALF
    stop_cat = np.where(alarm_sec.max(-1) > 0, np.argmax(alarm_sec, -1), -1).astype(np.int8)
    # the controller's own stop flags take precedence over the alarm codes
    for flag, cat in (("error_s", "forced_outage"), ("env_stop_s", "environmental"), ("grid_stop_s", "electrical")):
        stop_cat[np.nan_to_num(x[flag]) >= HALF] = CAT[cat]
    # a setpoint limit binds when the turbine runs at the setpoint while the wind would give clearly more
    sp, rated = np.nan_to_num(x["setpoint"], nan=np.inf), fleet.rated[None, :]
    is_derate = ((sp < 0.97 * rated) & (fleet.power >= sp - 0.1 * rated)
                 & (curves.power(fleet.ws) >= sp + BINDING * rated))
    state, category = _finish(fleet, is_stop, stop_cat, is_derate, np.full(fleet.power.shape, CAT["requested"]))
    return Reference(state, category, "operator shutdown duration + controller flags + alarm log")


def reference(ds, curves) -> Reference:
    return hot_reference(ds, curves) if ds.name == "hill_of_towie" else greenbyte_reference(ds, curves)
