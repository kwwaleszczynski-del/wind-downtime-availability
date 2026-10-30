"""The tool for your own data: long-format SCADA in, event table and availability KPIs out.

Steps: normal-behaviour curves learned from the data itself -> detector (default: pitch / rotor speed
signature) -> events -> cause: your status-code mapping where it covers the event, the transparent signal rules
otherwise -> expected power (default: running neighbours; WARP with a layout) -> lost energy per event ->
time- and energy-based availability per turbine and year and for the farm.

Outputs in ``--out``:

* ``events.csv``: turbine_id, start, end (UTC), duration_h, state (stop / derate / no_data), category,
  category_source (status code / signal rules), confidence, lost_kwh, expected_kwh, wind_mean, status_code
* ``kpi_turbine_year.csv`` and ``kpi_farm_year.csv``: hours and lost MWh per category, time and energy
  availability under the chosen contract
* ``intervals.parquet``: state and category of every 10-minute interval

Usage:
    run_pipeline(Path("scada.parquet"), Path("out"), rated_kw=2050, mapping=Path("codes.csv"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wind_downtime import availability as av
from wind_downtime.classify import RULE_PRECISION, classify_by_codes, classify_rules, event_features, paint
from wind_downtime.curves import fit_curves
from wind_downtime.detect import DETECTORS
from wind_downtime.energy import expected_power, lost_energy
from wind_downtime.events import to_events
from wind_downtime.hmm import fit_hmm, hmm_states
from wind_downtime.schema import CATEGORIES, CATEGORY_LIST, HOURS, STATES, read_long, to_fleet
from wind_downtime.sources import layout_from_latlon

log = logging.getLogger(__name__)


def read_mapping(path: Path) -> dict:
    """CSV with columns code, category (category keys as in ``schema.CATEGORIES``)."""
    m = pd.read_csv(path, dtype=str)
    bad = sorted(set(m["category"]) - set(CATEGORIES))
    if bad:
        raise ValueError(f"unknown categories in {path.name}: {bad}; allowed: {list(CATEGORIES)}")
    return dict(zip(m["code"].str.replace(r"\.0$", "", regex=True), m["category"]))


def read_turbines(path: Path) -> tuple[dict, pd.DataFrame | None]:
    t = pd.read_csv(path)
    t["turbine_id"] = t["turbine_id"].astype(str)
    rated = dict(zip(t["turbine_id"], t["rated_kw"].astype(float)))
    if {"lat", "lon", "hub_height_m", "rotor_diameter_m"} <= set(t.columns):
        return rated, layout_from_latlon(t)
    if {"x_m", "y_m", "hub_height_m", "rotor_diameter_m"} <= set(t.columns):
        return rated, t.set_index("turbine_id")[["x_m", "y_m", "hub_height_m", "rotor_diameter_m", "rated_kw"]]
    return rated, None


def run_pipeline(path: Path, out: Path, rated_kw=None, turbines: Path | None = None, mapping: Path | None = None,
                 detector: str = "signature", expected: str = "neighbours", contract: str = "oem_warranty",
                 allowance_h: float | None = None, tz: str = "Europe/Warsaw") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    layout = None
    if turbines is not None:
        rated_kw, layout = read_turbines(turbines)
    fleet = to_fleet(read_long(path), rated_kw=rated_kw)
    log.info("%d turbines, %d intervals (%s - %s)", fleet.n, len(fleet.time), fleet.time[0], fleet.time[-1])
    curves = fit_curves(fleet)
    if detector == "hmm":
        states = hmm_states(fleet, curves, fit_hmm(fleet, curves))
    else:
        states = DETECTORS[detector](fleet, curves)
    events = to_events(states, fleet.time, fleet.turbines)
    sd = events[events["state"].isin(["stop", "derate"])]
    feats = event_features(fleet, curves, sd, states, tz=tz)
    cats = pd.Series(None, index=events.index, dtype=object)
    source = pd.Series(None, index=events.index, dtype=object)
    if mapping is not None and fleet.code is not None:
        by_code = classify_by_codes(sd, fleet.code, read_mapping(mapping))
        cats.loc[by_code.index] = by_code
        source.loc[by_code.dropna().index] = "status code"
    rule = classify_rules(feats) if len(feats) else pd.Series(dtype=object)
    todo = cats.loc[sd.index].isna()
    cats.loc[todo[todo].index] = rule.loc[todo[todo].index]
    source.loc[todo[todo].index] = "signal rules"
    nd = events["state"].eq("no_data")
    cats[nd], source[nd] = "no_info", "missing data"
    conf = pd.Series(np.where(source.eq("signal rules"), cats.map(RULE_PRECISION), 1.0), index=events.index)

    if expected == "warp" and layout is None:
        raise ValueError("--expected warp needs a turbine table with the layout (--turbines)")
    normal = (states == STATES.index("normal")) & (fleet.power > 0.02 * fleet.rated[None, :])
    p_hat = expected_power(fleet, curves, normal, expected, layout=layout)
    lost = lost_energy(fleet, p_hat, states)
    exp_kwh = np.where(np.isfinite(p_hat), np.maximum(p_hat, 0), 0) * HOURS

    table = events[["turbine_id", "start", "end", "duration_h", "state"]].copy()
    table["category"], table["category_source"], table["confidence"] = cats, source, conf.round(2)
    table["lost_kwh"] = [lost[k0:k1, c].sum() for c, k0, k1 in zip(events["col"], events["k0"], events["k1"])]
    table["expected_kwh"] = [exp_kwh[k0:k1, c].sum() for c, k0, k1 in zip(events["col"], events["k0"], events["k1"])]
    table["wind_mean"] = [np.nanmean(fleet.ws[k0:k1, c]) if np.isfinite(fleet.ws[k0:k1, c]).any() else np.nan
                          for c, k0, k1 in zip(events["col"], events["k0"], events["k1"])]
    if fleet.code is not None:
        table["status_code"] = [_dominant(fleet.code[k0:k1, c]) for c, k0, k1
                                in zip(events["col"], events["k0"], events["k1"])]
    table.round(3).to_csv(out / "events.csv", index=False)

    category = paint(states.shape, events, cats, states)
    actual = np.where(fleet.power > 0, fleet.power, 0.0) * HOURS
    years = fleet.time.year
    uy = sorted(set(years))
    groups = np.searchsorted(uy, years)[:, None] * fleet.n + np.arange(fleet.n)[None, :]
    names = [f"{y}|{t}" for y in uy for t in fleet.turbines]
    totals = av.category_totals(category, lost, actual, groups, names)
    allowance = av.MAINTENANCE_ALLOWANCE_H[contract] if allowance_h is None else allowance_h
    rows = []
    for grp, t in totals.groupby("group", sort=False):
        y, tid = grp.split("|")
        rows.append({"year": int(y), "turbine_id": tid, **_kpi_row(t, contract, allowance, 1.0)})
    kt = pd.DataFrame(rows)
    kt.round(5).to_csv(out / "kpi_turbine_year.csv", index=False)
    farm_rows = []
    for y in uy:
        t = totals[totals["group"].str.startswith(f"{y}|")].groupby("category", as_index=False)[
            ["hours", "lost_kwh", "actual_kwh"]].sum()
        farm_rows.append({"year": y, **_kpi_row(t, contract, allowance, float(fleet.n))})
    kf = pd.DataFrame(farm_rows)
    kf.round(5).to_csv(out / "kpi_farm_year.csv", index=False)
    pd.DataFrame({"timestamp": np.repeat(fleet.time, fleet.n), "turbine_id": np.tile(fleet.turbines, len(fleet.time)),
                  "state": np.array(STATES)[states.ravel()], "category": np.array(CATEGORY_LIST)[category.ravel()],
                  "lost_kwh": lost.ravel()}).to_parquet(out / "intervals.parquet")
    log.info("%d events written to %s", len(table), out)
    return {"events": table, "kpi_turbine_year": kt, "kpi_farm_year": kf}


def _kpi_row(totals: pd.DataFrame, contract: str, allowance: float, n_turbine_years: float) -> dict:
    k = av.kpis(totals, av.CONTRACTS[contract], allowance, n_turbine_years)
    t = totals.set_index("category")
    row = {"time_availability": k["time_availability"], "energy_availability": k["energy_availability"],
           "contract": contract}
    for c in CATEGORY_LIST:
        row[f"hours_{c}"] = t.loc[c, "hours"] if c in t.index else 0.0
    for c in CATEGORY_LIST:
        row[f"lost_mwh_{c}"] = t.loc[c, "lost_kwh"] / 1000 if c in t.index else 0.0
    return row


def _dominant(codes: np.ndarray):
    vals = [c for c in codes if c is not None]
    return max(dict.fromkeys(vals), key=vals.count) if vals else None  # a tie goes to the code seen first
