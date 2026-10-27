"""Why did it stop? Cause classification of events into IEC 61400-26 categories.

Three ways, in order of preference in practice:

1. **Status codes + a mapping table** (``classify_by_codes``): the dominant code of the event looked up in a
   user-supplied ``code -> category`` table. The usual way when the controller logs are available and someone
   has mapped them; events without a mapped code fall through to 2 or 3.
2. **Transparent rules on signals** (``classify_rules``), no training:

   ======================================================  =============================================
   condition (checked top to bottom)                       category
   ======================================================  =============================================
   derate, at least half of the farm limited at once       requested (curtailment)
   derate, otherwise                                       partial performance (technical derating)
   stop, 90th percentile of the farm's wind >= 18 m/s      environmental (high wind)
   stop, <= 2 degC and power already short by >= 5 % of    environmental (icing)
   rated in the 2 hours before
   stop, at least half of the farm stopped within 20 min   electrical (grid) if at most 20 minutes,
                                                           otherwise requested
   stop, at most 30 minutes                                technical standby
   stop, starts Mon-Fri 07-17 local time, at least 1 h     planned corrective action if the turbine came
                                                           back from a stop of >= 1 h in the 3 days before,
                                                           otherwise scheduled maintenance
   any other stop                                          forced outage
   ======================================================  =============================================

3. **Gradient boosting on the same event features** (``train_event_model``), trained on logged events.

Events with missing data are always *information unavailable*.

Usage:
    feats = event_features(fleet, curves, events, states)
    feats["category"] = classify_rules(feats)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from wind_downtime.detect import DERATE, NO_DATA, STOP, parked
from wind_downtime.schema import CAT, CATEGORY_LIST, STATES

LOCAL_TZ = "Europe/London"
# thresholds read off the logged events of the training periods (never the test periods)
GRID_TRIP_H = 1 / 3  # farm-wide stops of up to 20 minutes are grid trips, longer ones requested shutdowns
HIGH_WIND = 18.0  # m/s, 90th percentile of the farm's 10-minute wind speed during the stop
# Share of each rule verdict that matched the logged category on the Senvion training events of the study
# (results/rule_precision.csv); reported as the confidence of a rule-based category in the event table.
# Single-turbine derates scored 0: at Penmanshiel they were all external instructions (code 9000).
RULE_PRECISION = {
    "electrical": 0.46,
    "environmental": 0.69,
    "forced_outage": 0.33,
    "partial_performance": 0.0,
    "planned_corrective": 0.17,
    "requested": 0.61,
    "scheduled_maintenance": 0.8,
    "technical_standby": 0.74,
}
WORK_HOURS = (7, 17)
FEATURES = ["is_stop", "duration_h", "log_duration", "hour", "weekday", "work_hours", "ws_mean", "ws_p90",
            "ws_start", "high_wind_frac", "temp_min", "temp_mean", "pre_short", "farm_sync", "farm_frac",
            "parked_frac", "pitch_rel", "rpm_rel", "p_level", "p_flat", "prev_gap_h", "prev_dur_h",
            "prev_work_hours", "month_sin", "month_cos"]


def _farm_state_fraction(states: np.ndarray, code: int) -> np.ndarray:
    """(time, turbine) share of the *other* turbines in state ``code`` (stops count for derate as well)."""
    hit = (states == code) | ((code == DERATE) & (states == STOP))
    valid = states != NO_DATA
    n_other = valid.sum(1, keepdims=True) - valid
    return np.where(n_other > 0, (hit.sum(1, keepdims=True) - hit) / np.maximum(n_other, 1), np.nan)


def event_features(fleet, curves, events: pd.DataFrame, states: np.ndarray, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """One row of features per stop or derate event (``events`` from ``to_events``)."""
    ev = events[events["state"].isin(["stop", "derate"])].copy()
    if ev.empty:
        return ev.assign(**{c: pd.Series(dtype=float) for c in FEATURES})
    rated = fleet.rated
    farm_ws = np.nanmedian(np.where(fleet.present, fleet.ws, np.nan), axis=1)
    p_hat = curves.power(fleet.ws)
    short = (p_hat - fleet.power) / rated[None, :]
    park = parked(fleet, curves)
    frac = {STOP: _farm_state_fraction(states, STOP), DERATE: _farm_state_fraction(states, DERATE)}
    start_local = pd.DatetimeIndex(ev["start"]).tz_convert(tz)
    rows = []
    for (_, e), t_loc in zip(ev.iterrows(), start_local):
        i, k0, k1 = int(e["col"]), int(e["k0"]), int(e["k1"])
        code = STOP if e["state"] == "stop" else DERATE
        sl = slice(k0, k1)
        ws = farm_ws[sl]
        temp = fleet.temp[sl, i]
        pre = slice(max(k0 - 12, 0), k0)
        pre_ok = fleet.power[pre, i] > 0.02 * rated[i]
        sync_win = slice(max(k0 - 2, 0), min(k0 + 3, len(fleet.time)))
        p = fleet.power[sl, i] / rated[i]
        rows.append({
            "is_stop": float(code == STOP),
            "hour": t_loc.hour + t_loc.minute / 60.0, "weekday": float(t_loc.weekday()),
            "work_hours": float(t_loc.weekday() < 5 and WORK_HOURS[0] <= t_loc.hour < WORK_HOURS[1]),
            "ws_mean": np.nanmean(ws) if np.isfinite(ws).any() else np.nan,
            "ws_p90": np.nanpercentile(ws, 90) if np.isfinite(ws).any() else np.nan,
            "ws_start": farm_ws[max(k0 - 1, 0)],
            "high_wind_frac": float(np.mean(np.nan_to_num(ws) >= 20.0)),
            "temp_min": np.nanmin(temp) if np.isfinite(temp).any() else np.nan,
            "temp_mean": np.nanmean(temp) if np.isfinite(temp).any() else np.nan,
            "pre_short": np.nanmean(short[pre, i][pre_ok]) if pre_ok.any() else np.nan,
            "farm_sync": np.nanmax(frac[code][sync_win, i]) if np.isfinite(frac[code][sync_win, i]).any() else np.nan,
            "farm_frac": np.nanmean(frac[code][sl, i]) if np.isfinite(frac[code][sl, i]).any() else np.nan,
            "parked_frac": float(np.mean(park[sl, i])),
            "pitch_rel": np.nanmean(fleet.pitch[sl, i]) / curves.park_pitch[i] if np.isfinite(fleet.pitch[sl, i]).any() else np.nan,
            "rpm_rel": np.nanmean(fleet.rpm[sl, i]) / curves.rated_rpm[i] if np.isfinite(fleet.rpm[sl, i]).any() else np.nan,
            "p_level": np.nanmedian(p) if np.isfinite(p).any() else np.nan,
            "p_flat": np.nanstd(p) if np.isfinite(p).any() else np.nan,
            "month_sin": np.sin(2 * np.pi * t_loc.month / 12), "month_cos": np.cos(2 * np.pi * t_loc.month / 12),
        })
    feats = pd.DataFrame(rows, index=ev.index)
    feats["log_duration"] = np.log10(ev["duration_h"])
    # the turbine's previous stop of at least an hour: how long ago it ended, how long it was, when it began
    prev = []
    stops = ev.assign(work_hours=feats["work_hours"])
    last = {}
    for idx, e in stops.sort_values("k0").iterrows():
        p = last.get(e["col"])
        prev.append((idx, (e["k0"] - p[0]) / 6.0 if p else 9999.0, p[1] if p else 0.0, p[2] if p else 0.0))
        if e["state"] == "stop" and e["duration_h"] >= 1.0:
            last[e["col"]] = (e["k1"], e["duration_h"], e["work_hours"])
    pg = pd.DataFrame(prev, columns=["idx", "prev_gap_h", "prev_dur_h", "prev_work_hours"]).set_index("idx")
    feats = feats.join(pg)
    return pd.concat([ev, feats], axis=1)


def classify_rules(f: pd.DataFrame) -> pd.Series:
    out = pd.Series("forced_outage", index=f.index, dtype=object)
    stop = f["is_stop"].eq(1)
    farm = f["farm_sync"].fillna(0).ge(0.5) | f["farm_frac"].fillna(0).ge(0.5)
    planned_time = stop & f["work_hours"].eq(1) & f["duration_h"].ge(1.0)
    after_stop = f["prev_gap_h"].le(72) & f["prev_dur_h"].ge(1.0)
    rules = [  # later rules override earlier ones, so the list runs from the weakest to the strongest
        (planned_time, "scheduled_maintenance"),
        (planned_time & after_stop, "planned_corrective"),
        (stop & f["duration_h"].le(0.5), "technical_standby"),
        (stop & farm & f["duration_h"].le(GRID_TRIP_H), "electrical"),
        (stop & farm & f["duration_h"].gt(GRID_TRIP_H), "requested"),
        (stop & f["temp_min"].le(2.0) & f["pre_short"].ge(0.05), "environmental"),
        (stop & f["ws_p90"].ge(HIGH_WIND), "environmental"),
        (~stop, "partial_performance"),
        (~stop & farm, "requested"),
    ]
    for mask, cat in rules:
        out[mask.fillna(False).to_numpy()] = cat
    return out


def train_event_model(feats: pd.DataFrame, labels: pd.Series, seed: int = 0) -> HistGradientBoostingClassifier:
    ok = labels.notna()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                           l2_regularization=1.0, class_weight="balanced", random_state=seed)
    model.fit(feats.loc[ok, FEATURES].to_numpy(dtype=float), labels[ok].to_numpy())
    return model


def classify_model(model, feats: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Predicted category and its probability (the confidence reported in the event table)."""
    if feats.empty:
        return pd.Series(dtype=object), pd.Series(dtype=float)
    proba = model.predict_proba(feats[FEATURES].to_numpy(dtype=float))
    cls = model.classes_[proba.argmax(1)]
    return pd.Series(cls, index=feats.index), pd.Series(proba.max(1), index=feats.index)


def classify_by_codes(events: pd.DataFrame, code: np.ndarray, mapping: dict) -> pd.Series:
    """Dominant mapped status code within each event -> category (None where nothing maps)."""
    out = pd.Series([None] * len(events), index=events.index, dtype=object)
    for idx, e in events.iterrows():
        vals = code[int(e["k0"]):int(e["k1"]), int(e["col"])]
        cats = [mapping.get(str(v)) for v in vals if v is not None]
        cats = [c for c in cats if c is not None]
        if cats:
            out[idx] = max(set(cats), key=cats.count)
    return out


def event_label(events: pd.DataFrame, category: np.ndarray) -> pd.Series:
    """Reference category of each event: the most common known category over its intervals."""
    out = pd.Series(None, index=events.index, dtype=object)
    for idx, e in events.iterrows():
        c = category[int(e["k0"]):int(e["k1"]), int(e["col"])]
        c = c[c >= 0]
        if len(c):
            out[idx] = CATEGORY_LIST[np.bincount(c, minlength=len(CATEGORY_LIST)).argmax()]
    return out


def paint(shape, events: pd.DataFrame, categories: pd.Series, states: np.ndarray) -> np.ndarray:
    """(time, turbine) category array from event categories; normal and missing intervals filled in."""
    out = np.full(shape, CAT["full_performance"], dtype=np.int8)
    out[states == STATES.index("no_data")] = CAT["no_info"]
    for idx, e in events.iterrows():
        cat = categories.get(idx)
        if cat is not None and isinstance(cat, str):
            out[int(e["k0"]):int(e["k1"]), int(e["col"])] = CAT[cat]
    return out
