import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from wind_downtime.curves import fit_curves
from wind_downtime.detect import signature
from wind_downtime.energy import expected_power, lost_energy, neighbour_wind
from wind_downtime.fetch import RemoteFile, extract_members
from wind_downtime.ml import gbm_states, interval_features, train_gbm
from wind_downtime.pipeline import run_pipeline
from wind_downtime.schema import STATES, to_fleet


@pytest.fixture(scope="module")
def setup(farm):
    df, truth = farm
    fleet = to_fleet(df, rated_kw=2000)
    curves = fit_curves(fleet)
    states = signature(fleet, curves)
    return fleet, curves, states, truth


def test_neighbour_wind_and_lost_energy(setup):
    fleet, curves, states, truth = setup
    normal = states == 0
    u = neighbour_wind(fleet, normal, np.ones(len(fleet.time), dtype=bool))
    ok = normal & np.isfinite(u)
    assert np.median(np.abs(u[ok] / fleet.ws[ok] - 1)) < 0.05
    a, b = truth[(0, "stop")][0]
    from conftest import power_curve
    truth_kwh = power_curve(fleet.ws[a:b, 0]).sum() / 6
    for m in ("nacelle", "neighbours"):
        p_hat = expected_power(fleet, curves, normal, m)
        lost = lost_energy(fleet, p_hat, states)
        assert lost[a:b, 0].sum() == pytest.approx(truth_kwh, rel=0.1)
        assert lost[states == 0].sum() == 0.0


def test_gbm_detector_learns_the_states(setup):
    fleet, curves, states, truth = setup
    X, names = interval_features(fleet, curves)
    assert X.shape == fleet.power.shape + (len(names),)
    model = train_gbm([(X, states, np.ones(len(fleet.time), dtype=bool))])
    pred = gbm_states(fleet, model, X)
    a, b = truth[(0, "stop")][0]
    assert np.mean(pred[a:b, 0] == STATES.index("stop")) > 0.9


def test_pipeline_end_to_end(farm, tmp_path):
    df, truth = farm
    df = df.copy()
    df["status_code"] = None
    a, b = truth[(0, "stop")][0]
    t = sorted(df["timestamp"].unique())
    sel = (df["turbine_id"] == "WTG1") & df["timestamp"].between(t[a], t[b - 1])
    df.loc[sel, "status_code"] = "3110"
    path = tmp_path / "scada.parquet"
    df.to_parquet(path)
    mapping = tmp_path / "codes.csv"
    pd.DataFrame({"code": ["3110"], "category": ["forced_outage"]}).to_csv(mapping, index=False)
    out = run_pipeline(path, tmp_path / "out", rated_kw=2000, mapping=mapping, tz="UTC")
    ev = out["events"]
    coded = ev[ev["category_source"] == "status code"]
    assert len(coded) == 1 and coded["category"].iloc[0] == "forced_outage"
    assert set(ev["state"]) >= {"stop", "derate", "no_data"}
    assert (ev.loc[ev["state"] == "no_data", "category"] == "no_info").all()
    k = out["kpi_turbine_year"]
    assert k["time_availability"].between(0.9, 1.0).all()
    assert (tmp_path / "out" / "events.csv").exists() and (tmp_path / "out" / "intervals.parquet").exists()
    with pytest.raises(ValueError, match="unknown categories"):
        pd.DataFrame({"code": ["1"], "category": ["broken"]}).to_csv(mapping, index=False)
        run_pipeline(path, tmp_path / "out2", rated_kw=2000, mapping=mapping)


class FakeSession:
    """Serves byte ranges of an in-memory file, like Zenodo does."""

    def __init__(self, blob: bytes):
        self.blob, self.calls = blob, 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        lo, hi = map(int, headers["Range"].split("=")[1].split("-"))
        r = type("R", (), {})()
        r.content, r.status_code = self.blob[lo:hi + 1], 206
        r.raise_for_status = lambda: None
        return r


def test_remote_zip_members(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("tblSCTurbine_2021_01.csv", "a,b\n1,2\n" * 1000)
        z.writestr("tblSCTurPress_2021_01.csv", "x\n" * 5000)
        z.writestr("tblAlarmLog_2021_01.csv", "TimeOn,TimeOff,StationNr,Alarmcode\n")
    blob = buf.getvalue()
    sess = FakeSession(blob)
    f = RemoteFile(sess, "u", len(blob))
    f.seek(-4, 2)
    assert f.read(4) == blob[-4:]
    import re
    dest = extract_members(sess, {"url": "u", "size": len(blob)}, re.compile(r"^(tblSCTurbine|tblAlarmLog)_"),
                           tmp_path / "sub.zip")
    with zipfile.ZipFile(dest) as z:
        assert sorted(z.namelist()) == ["tblAlarmLog_2021_01.csv", "tblSCTurbine_2021_01.csv"]
        assert z.read("tblSCTurbine_2021_01.csv").startswith(b"a,b\n1,2")
