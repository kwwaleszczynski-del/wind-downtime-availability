import numpy as np
import pandas as pd
import pytest

from wind_downtime.curves import fit_curves
from wind_downtime.detect import DERATE, NO_DATA, NORMAL, STOP, frozen, hysteresis, no_data, residual, rules, signature
from wind_downtime.events import interval_scores, match_events, to_events
from wind_downtime.hmm import fit_hmm, hmm_states
from wind_downtime.schema import fleet_to_long, read_long, to_fleet, validate_long


@pytest.fixture(scope="module")
def fleet(farm):
    return to_fleet(farm[0], rated_kw=2000)


@pytest.fixture(scope="module")
def curves(fleet):
    return fit_curves(fleet)


def test_long_format_is_checked(farm):
    df, _ = farm
    with pytest.raises(ValueError, match="missing columns"):
        validate_long(df.drop(columns="pitch"))
    bad = df.copy()
    bad.loc[0, "timestamp"] = bad.loc[0, "timestamp"] + pd.Timedelta("3min")
    with pytest.raises(ValueError, match="10-minute grid"):
        validate_long(bad)
    with pytest.raises(ValueError, match="duplicated"):
        validate_long(pd.concat([df, df.iloc[:1]]))


def test_fleet_round_trip(farm, fleet, tmp_path):
    df, truth = farm
    assert fleet.power.shape == (20 * 144, 4)
    i, a, b = truth["gap"]
    assert not fleet.present[a:b, i].any() and np.isnan(fleet.power[a:b, i]).all()
    back = fleet_to_long(fleet)
    assert len(back) == len(df)
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    assert np.allclose(to_fleet(read_long(path), rated_kw=2000).power, fleet.power, equal_nan=True)


def test_curves_recover_the_turbine(curves):
    assert np.allclose(curves.rated_rpm, 15.0, atol=0.5)
    assert np.allclose(curves.park_pitch, 90.0, atol=3)
    assert (curves.cut_in <= 4.0).all()
    from conftest import power_curve
    ws = np.array([[6.0] * 4, [9.0] * 4, [14.0] * 4])
    assert np.allclose(curves.power(ws), power_curve(ws), rtol=0.06)


def test_frozen_and_missing(farm, fleet):
    _, truth = farm
    nd = no_data(fleet)
    i, a, b = truth["frozen"]
    assert nd[a:b, i].all()
    i, a, b = truth["gap"]
    assert nd[a:b, i].all()
    assert nd.mean() < 0.01
    x = np.array([[1.0], [2.0], [2.0], [2.0], [3.0], [3.0]])
    assert frozen(x).ravel().tolist() == [False, True, True, True, False, False]


def test_hysteresis():
    enter = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=bool)[:, None]
    stay = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0], dtype=bool)[:, None]
    on = hysteresis(enter, stay, n_enter=2, n_exit=2).ravel()
    # on from the first 2-row streak; single rows without "stay" (7, 9) do not end the episode
    assert on.tolist() == [False] + [True] * 10 + [False]
    enter = np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool)[:, None]
    stay = np.array([1, 1, 1, 0, 0, 1, 0], dtype=bool)[:, None]
    # two rows without "stay" end it; a single "enter" row does not start a new one
    assert hysteresis(enter, stay).ravel().tolist() == [True, True, True, False, False, False, False]


def _hit(states, i, a, b, code):
    return np.mean(states[a:b, i] == code)


def test_detectors_find_the_injected_stops(farm, fleet, curves):
    _, truth = farm
    sig, rul, res = signature(fleet, curves), rules(fleet, curves), residual(fleet, curves)
    a, b = truth[(0, "stop")][0]
    for s in (sig, rul, res):
        assert _hit(s, 0, a, b, STOP) > 0.9
    # a stop during low wind: only the pitch / rotor speed signature sees it
    a, b = truth[(2, "stop_low_wind")][0]
    assert _hit(sig, 2, a, b, STOP) > 0.9
    assert _hit(rul, 2, a, b, STOP) < 0.2
    # low-wind idling (blades at 45 deg) is not a stop for the signature detector
    idle = (fleet.ws < 2.5) & (fleet.pitch < 50) & fleet.present
    assert np.mean(sig[idle] == STOP) < 0.02
    a, b = truth[(3, "derate")][0]
    assert _hit(res, 3, a, b, DERATE) > 0.5
    assert _hit(sig, 3, a, b, DERATE) > 0.5
    assert np.mean(sig[:, 3] == DERATE) < 0.05


def test_hmm_segments_stops(farm, fleet, curves):
    _, truth = farm
    model = fit_hmm(fleet, curves, iterations=4)
    s = hmm_states(fleet, curves, model)
    a, b = truth[(0, "stop")][0]
    assert _hit(s, 0, a, b, STOP) > 0.9
    a, b = truth[(0, "farm_stop")][0]
    assert _hit(s, 1, a, b, STOP) > 0.9
    assert np.mean(s == NORMAL) > 0.9
    assert np.isfinite(model.loglik).all()


def test_events_and_matching(fleet, curves):
    ref = np.zeros((20, 1), dtype=np.int8)
    ref[2:6, 0] = STOP
    ref[10:12, 0] = STOP
    ref[15:18, 0] = DERATE
    det = np.zeros_like(ref)
    det[3:7, 0] = STOP
    det[7, 0] = 0
    det[8, 0] = STOP  # a false stop, merged with nothing (gap of 1 normal row to 3:7 -> merged)
    det[18, 0] = NO_DATA
    t = pd.date_range("2021-01-01", periods=20, freq="10min", tz="UTC")
    rev, dev = to_events(ref, t, ["A"]), to_events(det, t, ["A"])
    assert rev[rev["state"] == "stop"]["duration_h"].tolist() == pytest.approx([4 / 6, 2 / 6])
    assert len(dev[dev["state"] == "stop"]) == 1  # 3:7 and 8 merged across one normal interval
    m = match_events(rev, dev, "stop")
    assert m["recall"] == pytest.approx(0.5) and m["precision"] == pytest.approx(1.0)
    assert m["duration_bias_h"] == pytest.approx((6 - 4) / 6)
    sc = {d["state"]: d for d in interval_scores(ref, det)}
    assert sc["stop"]["recall"] == pytest.approx(3 / 6)
    assert sc["derate"]["recall"] == 0.0


def test_plateau_marks_a_power_setpoint():
    from wind_downtime.detect import plateau

    t = pd.date_range("2021-01-01", periods=8, freq="10min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "turbine_id": "A", "wind_speed": 12.0, "rotor_speed": 15.0, "pitch": 5.0,
                       "nacelle_dir": 0.0, "ambient_temp": 5.0,
                       "power": [1500, 1700, 1600, 1200, 1205, 1198, 1202, 1800]})
    f = to_fleet(df, rated_kw=2000)
    assert plateau(f)[:, 0].tolist() == [False, False, False, False, True, True, False, False]
