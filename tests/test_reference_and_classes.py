import numpy as np
import pandas as pd
import pytest

from wind_downtime import availability as av
from wind_downtime.classify import classify_by_codes, classify_rules, event_features, event_label, paint
from wind_downtime.curves import fit_curves
from wind_downtime.detect import signature
from wind_downtime.events import to_events
from wind_downtime.reference import derive_corrective, greenbyte_reference, overlap_seconds
from wind_downtime.schema import CAT, CATEGORY_LIST, STATES, to_fleet
from wind_downtime.sources import Dataset

T = pd.date_range("2021-01-01", periods=12, freq="10min", tz="UTC")


def test_overlap_seconds():
    ev = pd.DataFrame({"turbine_id": ["A", "A", "B"],
                       "start": pd.to_datetime(["2021-01-01 00:05", "2021-01-01 00:15", "2021-01-01 01:00"], utc=True),
                       "end": pd.to_datetime(["2021-01-01 00:12", "2021-01-01 00:45", "2021-01-01 01:00"], utc=True),
                       "cat": [1, 2, 3]})
    sec = overlap_seconds(T, ["A", "B"], ev, 4)
    assert sec[0, 0, 1] == 300 and sec[1, 0, 1] == 120
    assert sec[1, 0, 2] == 300 and sec[2, 0, 2] == 600 and sec[4, 0, 2] == 300
    assert sec[:, 1].sum() == 0  # zero-length event


def test_corrective_is_derived_from_the_previous_forced_outage():
    t = pd.to_datetime(["2021-01-01 00:00", "2021-01-01 05:30", "2021-01-02 09:00", "2021-01-02 10:00",
                        "2021-01-20 09:00"], utc=True)
    ev = pd.DataFrame({"turbine_id": "A", "start": t, "end": t + pd.Timedelta("5h"),
                       "category": ["forced_outage", "scheduled_maintenance", "forced_outage",
                                    "scheduled_maintenance", "scheduled_maintenance"]})
    ev.loc[2, "end"] = ev.loc[2, "start"] + pd.Timedelta("30min")
    out = derive_corrective(ev)
    # 30 min after a forced outage ended: still the repair; 30 min after the second one: also the repair;
    # 18 days later: scheduled maintenance
    assert out.tolist() == ["forced_outage", "forced_outage", "forced_outage", "forced_outage",
                            "scheduled_maintenance"]
    ev.loc[3, "start"] = ev.loc[2, "end"] + pd.Timedelta("1D")
    assert derive_corrective(ev)[3] == "planned_corrective"


def test_greenbyte_reference_on_synthetic_logs(farm):
    df, truth = farm
    fleet = to_fleet(df, rated_kw=2000)
    a, b = truth[(0, "stop")][0]
    events = pd.DataFrame({
        "turbine_id": ["WTG1", "WTG4"], "start": [fleet.time[a], fleet.time[351]],
        "end": [fleet.time[b - 1] + pd.Timedelta("10min"), fleet.time[386] + pd.Timedelta("10min")],
        "status": ["Stop", "Warning"], "code": [3110, 9000],
        "message": ["Frequency converter error", "P output externally reduced"],
        "iec_category": ["Forced outage", "Partial Performance"]})
    ds = Dataset("synthetic", fleet, pd.DataFrame(), events)
    ref = greenbyte_reference(ds, fit_curves(fleet))
    assert (ref.state[a:b, 0] == STATES.index("stop")).all()
    assert (ref.category[a:b, 0] == CAT["forced_outage"]).all()
    der = ref.state[351:387, 3] == STATES.index("derate")
    assert der.mean() > 0.4  # only where the cap binds
    assert (ref.category[351:387, 3][der] == CAT["requested"]).all()
    i, g0, g1 = truth["gap"]
    assert (ref.state[g0:g1, i] == STATES.index("no_data")).all()


@pytest.fixture(scope="module")
def classified(farm):
    df, truth = farm
    fleet = to_fleet(df, rated_kw=2000)
    curves = fit_curves(fleet)
    st = signature(fleet, curves)
    ev = to_events(st, fleet.time, fleet.turbines, which=("stop", "derate"))
    feats = event_features(fleet, curves, ev, st, tz="UTC")
    feats["category"] = classify_rules(feats)
    return fleet, st, feats, truth


def test_rules_tell_farm_stops_from_single_stops(classified):
    fleet, st, feats, truth = classified
    a, _ = truth[(0, "farm_stop")][0]
    farm_ev = feats[(feats["k0"] <= a + 1) & (feats["k1"] > a) & (feats["state"] == "stop")]
    assert len(farm_ev) == 4 and (farm_ev["category"] == "requested").all()  # 3 h, all turbines
    a, _ = truth[(3, "derate")][0]
    der = feats[(feats["col"] == 3) & (feats["state"] == "derate") & (feats["k1"] > a)].iloc[0]
    assert der["category"] == "partial_performance"
    assert feats["farm_sync"].between(0, 1).all()


def test_rules_table():
    f = pd.DataFrame({
        "is_stop": [1, 1, 1, 1, 1, 1, 0, 0], "duration_h": [5, 0.3, 2, 0.3, 3, 4, 2, 2],
        "work_hours": [0, 0, 1, 0, 1, 0, 0, 0], "farm_sync": [0, 0, 0, 0.8, 0, 0, 0, 0.9],
        "farm_frac": [0, 0, 0, 0.8, 0, 0, 0, 0.9], "ws_p90": [8, 8, 8, 8, 8, 21, 8, 8],
        "temp_min": [5, 5, 5, 5, 5, 5, 5, 5], "pre_short": [0, 0, 0, 0, 0, 0, 0, 0],
        "prev_gap_h": [9999, 9999, 9999, 9999, 10, 9999, 9999, 9999], "prev_dur_h": [0, 0, 0, 0, 6, 0, 0, 0]})
    assert classify_rules(f).tolist() == ["forced_outage", "technical_standby", "scheduled_maintenance",
                                          "electrical", "planned_corrective", "environmental",
                                          "partial_performance", "requested"]


def test_codes_mapping_and_labels():
    ev = pd.DataFrame({"col": [0, 0], "k0": [0, 5], "k1": [4, 8], "state": ["stop", "stop"]})
    code = np.array([[None], ["21"], ["21"], ["3500"], [None], ["999"], [None], [None]], dtype=object)
    cats = classify_by_codes(ev, code, {"21": "forced_outage", "3500": "electrical"})
    assert cats.iloc[0] == "forced_outage" and pd.isna(cats.iloc[1])
    category = np.full((8, 1), CAT["full_performance"], dtype=np.int8)
    category[0:4, 0] = [CAT["electrical"], CAT["electrical"], CAT["forced_outage"], -1]
    assert event_label(ev, category).tolist() == ["electrical", "full_performance"]
    st = np.zeros((8, 1), dtype=np.int8)
    painted = paint((8, 1), ev, pd.Series(["forced_outage", "requested"]), st)
    assert painted[:, 0].tolist() == [CAT["forced_outage"]] * 4 + [CAT["full_performance"]] + [CAT["requested"]] * 3


def test_availability_kpis():
    # one turbine-year: 8000 h available, 500 h forced outage, 200 h grid, 60 h maintenance
    tot = pd.DataFrame({"category": CATEGORY_LIST, "hours": 0.0, "lost_kwh": 0.0, "actual_kwh": 0.0})
    t = tot.set_index("category")
    t.loc["full_performance", ["hours", "actual_kwh"]] = [8000, 5e6]
    t.loc["forced_outage", ["hours", "lost_kwh"]] = [500, 4e5]
    t.loc["electrical", ["hours", "lost_kwh"]] = [200, 1e5]
    t.loc["scheduled_maintenance", ["hours", "lost_kwh"]] = [60, 3e4]
    t = t.reset_index()
    iec = av.kpis(t, av.CONTRACTS["iec_operational"])
    assert iec["time_availability"] == pytest.approx(8200 / 8760)
    assert iec["energy_availability"] == pytest.approx(1 - 4.3e5 / 5.53e6)
    oem = av.kpis(t, av.CONTRACTS["oem_warranty"], allowance_h=60)
    assert oem["time_availability"] == pytest.approx(8000 / 8500)  # grid excluded, maintenance excused
    assert oem["energy_availability"] == pytest.approx(1 - 4e5 / 5.43e6)
    cat = np.array([[CAT["electrical"], CAT["no_info"]]], dtype=np.int8)
    assert av.remap(cat, av.SCENARIOS["grid and curtailment stops counted as turbine faults"]).tolist() == [
        [CAT["forced_outage"], CAT["no_info"]]]


def test_category_totals():
    cat = np.array([[0, 8], [8, 8]], dtype=np.int8)
    lost = np.array([[0.0, 100.0], [50.0, 25.0]])
    act = np.array([[300.0, 0.0], [0.0, 0.0]])
    groups = np.array([[0, 1], [0, 1]])
    tot = av.category_totals(cat, lost, act, groups, ["a", "b"]).set_index(["group", "category"])
    assert tot.loc[("a", "forced_outage"), "hours"] == pytest.approx(1 / 6)
    assert tot.loc[("b", "forced_outage"), "lost_kwh"] == 125.0
    assert tot.loc[("a", "full_performance"), "actual_kwh"] == 300.0


def test_hill_of_towie_reference_from_flags(farm):
    from wind_downtime.reference import hot_reference

    df, truth = farm
    fleet = to_fleet(df, rated_kw=2000)
    shape = fleet.power.shape
    extra = {k: np.zeros(shape) for k in ("shutdown_s", "error_s", "env_stop_s", "grid_stop_s")}
    extra["setpoint"] = np.full(shape, 2000.0)
    a, b = truth[(0, "stop")][0]
    extra["shutdown_s"][a:b, 0] = 600
    extra["error_s"][a:b, 0] = 600
    c, d = truth[(0, "farm_stop")][0]
    extra["shutdown_s"][c:d, :] = 600
    extra["grid_stop_s"][c:d, :] = 600
    # a setpoint that binds (turbine 3 runs at it in good wind) and one that does not (turbine 2 far below it)
    good = np.flatnonzero(fleet.ws[:, 3] > 11)[:12]
    extra["setpoint"][good, 3] = 1200.0
    fleet.power[good, 3] = 1190.0
    alarms = pd.DataFrame(columns=["turbine_id", "start", "end", "code"])
    ref = hot_reference(Dataset("hot", fleet, pd.DataFrame(), alarms, extra), fit_curves(fleet))
    assert (ref.state[a:b, 0] == STATES.index("stop")).all() and (ref.category[a:b, 0] == CAT["forced_outage"]).all()
    assert (ref.category[c:d, :] == CAT["electrical"]).all()
    assert (ref.state[good, 3] == STATES.index("derate")).mean() > 0.8
    assert (ref.category[good, 3][ref.state[good, 3] == STATES.index("derate")] == CAT["requested"]).all()
