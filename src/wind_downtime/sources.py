"""Adapters from the three open datasets to the generic long format, plus what each offers as a reference.

=============  ==========================  ======================================================================
farm           turbines                    reference for validation
=============  ==========================  ======================================================================
Kelmarsh       6 x Senvion MM92 (2.05 MW)  Greenbyte status log with IEC 61400-26 categories; Greenbyte lost
                                           production to downtime and curtailment
Penmanshiel    14 x Senvion MM82 (2.05 MW) the same (also the power setpoint and lost production to grid
                                           curtailment)
Hill of Towie  21 x Siemens SWT-2.3-VS-82  shutdown duration per interval published by the operator (SCADA and
                                           operator logs), alarm log with 12 described codes, controller flags
                                           (time ready, running, turbine error), active power setpoint
=============  ==========================  ======================================================================

Each loader returns a ``Dataset``: the Fleet arrays (only generic signals), the turbine layout, the event log
(if any) and extra reference arrays that the detectors never see.

Usage:
    ds = load_dataset("penmanshiel", [2018, 2020], Path("data"))
    ds.fleet.power.shape, ds.events.head(), ds.extra["gb_lost_downtime"].shape
"""
from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from wind_power_curve.cleaning import consistent_wind_speed
from wind_power_curve.scada import read_status_csv

from wind_downtime.schema import STEP, Fleet, to_fleet

EARTH_RADIUS_M = 6_371_000.0
KELMARSH_COD = pd.Timestamp("2016-04-15", tz="UTC")
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")
RATED_KW = {"kelmarsh": 2050.0, "penmanshiel": 2050.0, "hill_of_towie": 2300.0}
GREENBYTE_COLUMNS = {
    "Power (kW)": "power",
    "Wind speed (m/s)": "ws_scada",
    "Wind speed Sensor 1 (m/s)": "ws1",
    "Wind speed Sensor 2 (m/s)": "ws2",
    "Rotor speed (RPM)": "rotor_speed",
    "Nacelle position (°)": "nacelle_dir",
    "Nacelle ambient temperature (°C)": "ambient_temp",
    "Blade angle (pitch position) A (°)": "pitch_a",
    "Blade angle (pitch position) B (°)": "pitch_b",
    "Blade angle (pitch position) C (°)": "pitch_c",
    "Energy Export (kWh)": "energy_kwh",
    "Turbine Power setpoint (kW)": "setpoint",
    "Lost Production to Downtime (kWh)": "gb_lost_downtime",
    "Lost Production to Curtailment (Total) (kWh)": "gb_lost_curtailment",
    "Lost Production to Curtailment (Grid) (kWh)": "gb_lost_curtailment_grid",
}
EXTRA_COLUMNS = ("energy_kwh", "setpoint", "gb_lost_downtime", "gb_lost_curtailment", "gb_lost_curtailment_grid",
                 "shutdown_s", "ready_s", "running_s", "error_s", "env_stop_s", "grid_stop_s", "com_stop_s")
log = logging.getLogger(__name__)


@dataclass
class Dataset:
    name: str
    fleet: Fleet
    layout: pd.DataFrame  # turbine_id -> x_m, y_m, hub_height_m, rotor_diameter_m, rated_kw
    events: pd.DataFrame  # turbine_id, start, end, status, code, message, iec_category (may be empty)
    extra: dict = field(default_factory=dict)  # (time, turbine) reference arrays

    def years(self) -> list[int]:
        return sorted(set(self.fleet.time.year))


def kelmarsh_dir() -> Path:
    return Path(os.environ.get("KELMARSH_DATA_DIR", "data/kelmarsh"))


def layout_from_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """turbine_id, Latitude, Longitude, hub height, rotor diameter, rated -> metres east/north of the centre."""
    lat0, lon0 = np.deg2rad(df["lat"].mean()), np.deg2rad(df["lon"].mean())
    lat, lon = np.deg2rad(df["lat"]), np.deg2rad(df["lon"])
    out = df[["turbine_id", "hub_height_m", "rotor_diameter_m", "rated_kw"]].copy()
    out["x_m"] = EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0)
    out["y_m"] = EARTH_RADIUS_M * (lat - lat0)
    return out.set_index("turbine_id").sort_index()


def _greenbyte_static(path: Path, width: int) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["Title"])
    num = df["Title"].str.extract(r"(\d+)$", expand=False).astype(int)
    return layout_from_latlon(pd.DataFrame({
        "turbine_id": [f"T{n:0{width}d}" for n in num],
        "lat": df["Latitude"].to_numpy(), "lon": df["Longitude"].to_numpy(),
        "hub_height_m": df["Hub Height (m)"].to_numpy(), "rotor_diameter_m": df["Rotor Diameter (m)"].to_numpy(),
        "rated_kw": df["Rated power (kW)"].to_numpy(dtype=float)}))


def _wide_extra(long: pd.DataFrame, fleet: Fleet) -> dict:
    idx = pd.MultiIndex.from_frame(long[["timestamp", "turbine_id"]])
    out = {}
    for c in EXTRA_COLUMNS:
        if c in long:
            s = pd.Series(long[c].to_numpy(dtype=float), index=idx)
            out[c] = s.unstack("turbine_id").reindex(index=fleet.time, columns=fleet.turbines).to_numpy()
    return out


# --------------------------------------------------------------------------------------------- Kelmarsh

def load_kelmarsh(years: list[int], data_dir: Path) -> Dataset:
    """Kelmarsh via the ``wind_power_curve`` reader (its Parquet cache or the raw archives)."""
    from wind_power_curve.scada import load

    kdir = kelmarsh_dir() if data_dir is None else data_dir
    scada, status = load(sorted(years), kdir)
    df = scada.reset_index()
    df["wind_speed"] = consistent_wind_speed(scada).to_numpy()
    df = df[df["timestamp"] >= KELMARSH_COD]  # commissioning data before COD are not operation
    long = pd.DataFrame({
        "timestamp": df["timestamp"], "turbine_id": "T" + df["turbine"].astype(str),
        "power": df["power_kw"], "wind_speed": df["wind_speed"],
        "rotor_speed": df["rotor_rpm"], "pitch": df["pitch"], "nacelle_dir": df["nacelle_pos"],
        "ambient_temp": df["temp_c"], "energy_kwh": df["energy_kwh"],
        "gb_lost_downtime": df["gb_lost_downtime_kwh"], "gb_lost_curtailment": df["gb_lost_curtailment_kwh"],
    })
    fleet = to_fleet(long, rated_kw=RATED_KW["kelmarsh"], name="kelmarsh")
    events = status.assign(turbine_id="T" + status["turbine"].astype(str)).drop(columns="turbine")
    events = events[events["end"].isna() | (events["end"] > KELMARSH_COD)]
    layout = _greenbyte_static(kdir / "raw" / "Kelmarsh_WT_static.csv", 1)
    return Dataset("kelmarsh", fleet, layout, events, _wide_extra(long, fleet))


# ------------------------------------------------------------------------------------------ Penmanshiel

def read_greenbyte_turbine(fh: io.TextIOBase, turbine_id: str) -> pd.DataFrame:
    """One Greenbyte 10-minute export -> long rows with the generic signals and the reference columns."""
    text = fh.read()
    lines = text.splitlines()
    skip = next(i for i, line in enumerate(lines[:60]) if line.startswith("# Date and time"))
    df = pd.read_csv(io.StringIO(text), skiprows=skip, na_values=["NaN"],
                     usecols=lambda c: c == "# Date and time" or c in GREENBYTE_COLUMNS)
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["# Date and time"]).dt.tz_localize("UTC"),
                        "turbine_id": turbine_id})
    for src, dst in GREENBYTE_COLUMNS.items():
        out[dst] = df[src].to_numpy(dtype=float) if src in df else np.nan
    out["pitch"] = out[["pitch_a", "pitch_b", "pitch_c"]].mean(axis=1)
    # the mean of the two anemometers where both report (the SCADA signal changed definition at Kelmarsh)
    out["wind_speed"] = out[["ws1", "ws2"]].mean(axis=1, skipna=False).fillna(out["ws_scada"])
    return out.drop(columns=["pitch_a", "pitch_b", "pitch_c", "ws1", "ws2"])


def _read_penmanshiel_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scada, status = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            m = re.search(r"_Penmanshiel_(\d+)_", name)
            if not m or not name.endswith(".csv"):
                continue
            tid = f"T{int(m.group(1)):02d}"
            with z.open(name) as raw:
                fh = io.TextIOWrapper(raw, encoding="utf-8-sig")
                if name.startswith("Turbine_Data"):
                    scada.append(read_greenbyte_turbine(fh, tid))
                elif name.startswith("Status"):
                    status.append(read_status_csv(fh, 0).drop(columns="turbine").assign(turbine_id=tid))
    return pd.concat(scada, ignore_index=True), pd.concat(status, ignore_index=True)


def load_penmanshiel(years: list[int], data_dir: Path) -> Dataset:
    raw, proc = data_dir / "raw" / "penmanshiel", data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    longs, events = [], []
    for y in sorted(years):
        lp, ep = proc / f"penmanshiel_{y}.parquet", proc / f"penmanshiel_status_{y}.parquet"
        if not (lp.exists() and ep.exists()):
            archives = sorted(raw.glob(f"Penmanshiel_SCADA_{y}_WT*.zip"))
            if not archives:
                raise FileNotFoundError(f"no Penmanshiel archive for {y} in {raw}; run 'fetch' first")
            parts = [_read_penmanshiel_zip(a) for a in archives]
            pd.concat([p[0] for p in parts], ignore_index=True).to_parquet(lp)
            pd.concat([p[1] for p in parts], ignore_index=True).to_parquet(ep)
            log.info("penmanshiel %d cached", y)
        longs.append(pd.read_parquet(lp))
        events.append(pd.read_parquet(ep))
    long = pd.concat(longs, ignore_index=True).drop_duplicates(["turbine_id", "timestamp"])
    fleet = to_fleet(long, rated_kw=RATED_KW["penmanshiel"], name="penmanshiel")
    layout = _greenbyte_static(raw / "Penmanshiel_WT_static.csv", 2)
    ev = pd.concat(events, ignore_index=True).drop_duplicates()
    return Dataset("penmanshiel", fleet, layout, ev, _wide_extra(long, fleet))


# ---------------------------------------------------------------------------------------- Hill of Towie

HOT_STATION_OFFSET = 2304509  # StationId = turbine number + offset
HOT_SIGNALS = {
    "tblSCTurbine": {"wtc_AcWindSp_mean": "wind_speed", "wtc_MainSRpm_mean": "rotor_speed",
                     "wtc_PitcPosA_mean": "pitch_a", "wtc_PitcPosB_mean": "pitch_b", "wtc_PitcPosC_mean": "pitch_c",
                     "wtc_NacelPos_mean": "nacelle_dir", "wtc_PowerRef_endvalue": "setpoint"},
    "tblSCTurGrid": {"wtc_ActPower_mean": "power"},
    "tblSCTurTemp": {"wtc_AmbieTmp_mean": "ambient_temp"},
    "tblSCTurFlag": {"wtc_ScReToOp_timeon": "ready_s", "wtc_ScInOper_timeon": "running_s",
                     "wtc_ScTurSto_timeon": "error_s", "wtc_ScEnvSto_timeon": "env_stop_s",
                     "wtc_ScGrdSto_timeon": "grid_stop_s", "wtc_ScComSto_timeon": "com_stop_s"},
}


def _read_hot_table(z: zipfile.ZipFile, table: str, cols: dict) -> pd.DataFrame:
    parts = []
    for name in sorted(n for n in z.namelist() if n.startswith(table + "_")):
        with z.open(name) as fh:
            df = pd.read_csv(fh, usecols=lambda c: c in ("TimeStamp", "StationId", *cols))
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # TimeStamp is the end of the 10-minute period
    df["timestamp"] = pd.to_datetime(df.pop("TimeStamp"), utc=True) - STEP
    df["turbine_id"] = "T" + (df.pop("StationId") - HOT_STATION_OFFSET).astype(int).map("{:02d}".format)
    df = df.rename(columns=cols).drop_duplicates(["turbine_id", "timestamp"])
    return df.set_index(["turbine_id", "timestamp"])


def load_hill_of_towie(years: list[int], data_dir: Path) -> Dataset:
    raw, proc = data_dir / "raw" / "hill_of_towie", data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    longs, alarms = [], []
    for y in sorted(years):
        lp, ap = proc / f"hill_of_towie_{y}.parquet", proc / f"hill_of_towie_alarms_{y}.parquet"
        if not (lp.exists() and ap.exists()):
            with zipfile.ZipFile(raw / f"{y}_subset.zip") as z:
                tables = [_read_hot_table(z, t, c) for t, c in HOT_SIGNALS.items()]
                alarm = pd.concat([pd.read_csv(z.open(n)) for n in sorted(z.namelist())
                                   if n.startswith("tblAlarmLog_")], ignore_index=True)
            df = tables[0].join(tables[1:], how="outer").reset_index()
            df["pitch"] = df[["pitch_a", "pitch_b", "pitch_c"]].mean(axis=1)
            df = df.drop(columns=["pitch_a", "pitch_b", "pitch_c"])
            df = df[df["timestamp"].dt.year == y]
            sd = pd.read_csv(zipfile.ZipFile(raw / "Hill_of_Towie_ShutdownDuration.zip").open("ShutdownDuration.csv"))
            sd["timestamp"] = pd.to_datetime(sd["TimeStamp_StartFormat"], utc=True)
            sd = sd[sd["timestamp"].dt.year == y].rename(columns={"TurbineName": "turbine_id",
                                                                 "ShutdownDuration": "shutdown_s"})
            df = df.merge(sd[["turbine_id", "timestamp", "shutdown_s"]], on=["turbine_id", "timestamp"], how="left")
            df.to_parquet(lp)
            alarm.to_parquet(ap)
            log.info("hill of towie %d cached: %d rows, %d alarms", y, len(df), len(alarm))
        longs.append(pd.read_parquet(lp))
        alarms.append(pd.read_parquet(ap))
    long = pd.concat(longs, ignore_index=True)
    fleet = to_fleet(long, rated_kw=RATED_KW["hill_of_towie"], name="hill_of_towie")
    meta = pd.read_csv(raw / "Hill_of_Towie_turbine_metadata.csv", encoding="utf-8-sig")
    layout = layout_from_latlon(pd.DataFrame({
        "turbine_id": meta["Turbine Name"], "lat": meta["Latitude"], "lon": meta["Longitude"],
        "hub_height_m": meta["Hub Height (m)"], "rotor_diameter_m": meta["Rotor Diameter (m)"],
        "rated_kw": meta["Rated power (kW)"].astype(float)}))
    events = hot_alarm_events(pd.concat(alarms, ignore_index=True), raw / "Hill_of_Towie_alarms_description.csv")
    return Dataset("hill_of_towie", fleet, layout, events, _wide_extra(long, fleet))


def hot_alarm_events(alarm: pd.DataFrame, description_csv: Path) -> pd.DataFrame:
    """Alarm log -> the event table layout of the Greenbyte logs (message from the 12 described codes)."""
    desc = pd.read_csv(description_csv).set_index("Alarm Code")
    ev = pd.DataFrame({
        "turbine_id": "T" + (alarm["StationNr"] - HOT_STATION_OFFSET).astype(int).map("{:02d}".format),
        "start": pd.to_datetime(alarm["TimeOn"], utc=True),
        "end": pd.to_datetime(alarm["TimeOff"], utc=True, errors="coerce"),
        "code": alarm["Alarmcode"].astype(int),
    })
    ev["message"] = ev["code"].map(desc["Description"].str.strip(" :"))
    stopping = ev["code"].map(desc["Stopping"])
    ev["status"] = np.where(stopping.eq(1), "Stop", np.where(stopping.eq(0), "Informational", "Unknown"))
    ev["iec_category"] = None
    return ev


LOADERS = {"kelmarsh": load_kelmarsh, "penmanshiel": load_penmanshiel, "hill_of_towie": load_hill_of_towie}


def load_dataset(farm: str, years: list[int], data_dir: Path) -> Dataset:
    if farm == "kelmarsh":
        return load_kelmarsh(years, kelmarsh_dir())
    return LOADERS[farm](years, data_dir)


def export_long(ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An open dataset in the generic long format with a status code per interval, plus its code mapping.

    The code of an interval is that of the `Stop` event covering at least half of it (Greenbyte farms);
    the mapping table is the operator's own IEC category of each code. Used for the end-to-end demo of the
    ``run`` command.
    """
    from wind_downtime.reference import GREENBYTE_IEC
    from wind_downtime.schema import fleet_to_long

    f = ds.fleet
    code = np.full(f.power.shape, None, dtype=object)
    ev = ds.events[ds.events["status"].eq("Stop") & ds.events["end"].notna()].copy()
    ev["category"] = ev["iec_category"].map(GREENBYTE_IEC)
    ev = ev[ev["category"].notna()]
    t0 = f.time[0]
    col = {t: i for i, t in enumerate(f.turbines)}
    for _, e in ev.sort_values("start").iterrows():
        if e["turbine_id"] not in col:
            continue
        k0 = int(np.ceil(((e["start"] - t0) / STEP) - 0.5))
        k1 = int(np.floor(((e["end"] - t0) / STEP) - 0.5)) + 1
        if k1 > k0:
            code[max(k0, 0):max(min(k1, len(f.time)), 0), col[e["turbine_id"]]] = str(int(e["code"]))
    f.code = code
    long = fleet_to_long(f)
    mapping = (ev.assign(code=ev["code"].astype(int).astype(str))
               .groupby("code")["category"].agg(lambda s: s.mode().iloc[0]).reset_index())
    f.code = None
    return long, mapping
