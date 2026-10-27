"""A small synthetic wind farm with known stops, deratings and gaps (no network, no real data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RATED = 2000.0
RATED_RPM = 15.0
PARK = 90.0


def power_curve(ws):
    ws = np.asarray(ws, dtype=float)
    p = RATED / (1 + np.exp(-(ws - 8.0) / 1.2))
    return np.where((ws < 3.0) | (ws > 25.0), 0.0, p)


def pitch_curve(ws):
    return np.clip((np.asarray(ws) - 11.5) * 2.5, 0.0, 30.0)


def rpm_curve(ws):
    return np.clip(6 + 0.9 * np.asarray(ws), 0, RATED_RPM)


def make_farm(days: int = 20, n: int = 4, seed: int = 1) -> tuple[pd.DataFrame, dict]:
    """Long table plus the truth: dict of (turbine, kind) -> list of (start index, end index)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-03-01", periods=days * 144, freq="10min", tz="UTC")
    base = 7.0 + 5.5 * np.sin(np.arange(len(t)) / 300.0) + np.cumsum(rng.normal(0, 0.05, len(t)))
    base = np.clip(base, 0.5, 22.0)
    ws = base[:, None] * (1 + rng.normal(0, 0.03, (len(t), n)))
    power = power_curve(ws) * (1 + rng.normal(0, 0.02, ws.shape))
    pitch = pitch_curve(ws) + rng.normal(0, 0.3, ws.shape)
    rpm = rpm_curve(ws) + rng.normal(0, 0.1, ws.shape)
    # low-wind idling: blades at 45 deg, rotor slow, no power
    idle = ws < 3.0
    power[idle], pitch[idle], rpm[idle] = 0.0, 45.0, 1.0
    truth = {}

    def stop(i, a, b, kind="stop"):
        power[a:b, i], pitch[a:b, i], rpm[a:b, i] = -2.0, PARK, 0.0
        truth.setdefault((i, kind), []).append((a, b))

    def derate(i, a, b, cap=0.6 * RATED):
        over = power[a:b, i] > cap
        power[a:b, i] = np.minimum(power[a:b, i], cap)
        pitch[a:b, i] += np.where(over, 6.0, 0.0)
        truth.setdefault((i, "derate"), []).append((a, b))

    hi = np.flatnonzero(base > 9.0)
    lo = np.flatnonzero(base < 2.5)
    stop(0, hi[100], hi[100] + 30)                    # 5 h stop in good wind
    stop(1, hi[400], hi[400] + 6)                     # 1 h stop
    if len(lo) > 50:
        stop(2, lo[10], lo[10] + 24, "stop_low_wind")  # maintenance during low wind
    derate(3, hi[200], hi[200] + 36)                  # 6 h derating in good wind
    for i in range(n):                                # farm-wide stop of 3 h (grid / curtailment)
        stop(i, hi[600], hi[600] + 18, "farm_stop")
    df = pd.DataFrame({
        "timestamp": np.repeat(t, n), "turbine_id": np.tile([f"WTG{i + 1}" for i in range(n)], len(t)),
        "power": power.ravel(), "wind_speed": ws.ravel(), "rotor_speed": rpm.ravel(), "pitch": pitch.ravel(),
        "nacelle_dir": np.tile(220.0 + rng.normal(0, 5, (len(t), 1)), (1, n)).ravel(),
        "ambient_temp": 8.0 + rng.normal(0, 1, len(t) * n),
    })
    # a gap of 2 h on turbine 2 and a frozen data link of 1 h on turbine 1
    gap = (df["turbine_id"] == "WTG2") & df["timestamp"].isin(t[1000:1012])
    df = df[~gap].reset_index(drop=True)
    fz = (df["turbine_id"] == "WTG1") & df["timestamp"].isin(t[1500:1506])
    for c in ("power", "wind_speed", "rotor_speed"):
        df.loc[fz, c] = df.loc[fz, c].iloc[0]
    truth["gap"] = (1, 1000, 1012)
    truth["frozen"] = (0, 1500, 1506)
    return df, truth


@pytest.fixture(scope="session")
def farm():
    return make_farm()
