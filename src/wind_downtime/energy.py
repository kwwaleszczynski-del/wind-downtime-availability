"""Expected power while a turbine is stopped or limited, and the energy lost.

Three references for the power a turbine *would* have produced:

``nacelle``     the turbine's own power curve at its own nacelle anemometer. Simple and always available, but
                a stopped rotor no longer slows the air at the nacelle, so the anemometer reads high exactly
                when it is needed.
``neighbours``  the turbine's own curve at a wind speed taken from the running neighbours: the median of
                their anemometers times a per-sector transfer ratio learned in normal operation (12 sectors of
                the neighbours' mean nacelle direction). No layout needed.
``warp``        WARP (Wake-Adjusted Reference Power) from
                github.com/kwwaleszczynski-del/wind-wake-aware-verification: the free stream from the
                unwaked running turbines, carried to the turbine through a PyWake wake model and a fitted
                site factor. Needs the layout.

Lost energy of an interval = max(expected - actual, 0) x 10 minutes, for intervals in a stop or derate state.

Usage:
    p_hat = expected_power(fleet, curves, normal, method="neighbours", base=fleet.years() == 2019)
    lost_kwh = lost_energy(fleet, p_hat, states)
"""
from __future__ import annotations

import logging

import numpy as np

from wind_downtime.schema import HOURS, STATES

N_SECTORS = 12
log = logging.getLogger(__name__)


def circular_mean_others(angles: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(time, turbine) circular mean of the *other* turbines' angles (degrees)."""
    rad = np.deg2rad(np.where(valid, angles, np.nan))
    s, c = np.nan_to_num(np.sin(rad)), np.nan_to_num(np.cos(rad))
    cnt = (valid & np.isfinite(rad)).astype(float)
    so, co, no = s.sum(1, keepdims=True) - s, c.sum(1, keepdims=True) - c, cnt.sum(1, keepdims=True) - cnt
    out = np.rad2deg(np.arctan2(so, co)) % 360.0
    out[no == 0] = np.nan
    return out


def median_others(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = np.where(valid, values, np.nan)
    out = np.full(v.shape, np.nan)
    for i in range(v.shape[1]):
        others = np.delete(v, i, axis=1)
        ok = np.isfinite(others).any(axis=1)
        out[ok, i] = np.nanmedian(others[ok], axis=1)
    return out


def neighbour_wind(fleet, normal: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Wind speed at each turbine from its running neighbours, with per-sector transfer ratios."""
    u_nb = median_others(fleet.ws, normal)
    theta = circular_mean_others(fleet.ndir, normal)
    sector = np.floor(np.nan_to_num(theta) / (360.0 / N_SECTORS)).astype(int) % N_SECTORS
    ratio = np.ones((N_SECTORS, fleet.n))
    fit = normal & base[:, None] & (u_nb >= 4.0) & (u_nb <= 15.0) & np.isfinite(theta)
    for i in range(fleet.n):
        for s in range(N_SECTORS):
            sel = fit[:, i] & (sector[:, i] == s)
            if sel.sum() >= 50:
                ratio[s, i] = np.median(fleet.ws[sel, i] / u_nb[sel, i])
    return u_nb * ratio[sector, np.arange(fleet.n)[None, :]]


def wake_tables(lay, rated_kw: float, k: float = 0.035, cache=None) -> np.ndarray:
    """Single-wake deficit tables of wake_aware for one wake expansion rate, shape (1, j, i, wd, ws)."""
    from wake_aware.wake import pair_tables

    if cache is not None and cache.exists():
        return np.load(cache)
    tables = pair_tables(lay["x_m"].to_numpy(), lay["y_m"].to_numpy(), lay["hub_height_m"].to_numpy(),
                         float(lay["rotor_diameter_m"].iloc[0]), rated_kw, k_values=[k]).astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, tables)
    return tables


def warp_expected(fleet, layout, normal: np.ndarray, base: np.ndarray, k: float = 0.035,
                  sigma_deg: float = 3.0, cache=None) -> np.ndarray:
    """Expected power from WARP (wake_aware), with the wake parameters calibrated at Kelmarsh.

    Without a turbulence signal and with no k(TI) slope, WARP only ever looks up the wake expansion rate
    ``k``: only that slice of the single-wake tables is computed (PyWake, Bastankhah Gaussian wake) and
    stands in for the whole k grid.
    """
    from wake_aware.data import Farm
    from wake_aware.model import WarpParams, fit_warp, predict
    from wake_aware.wake import K_GRID, smooth_direction

    lay = layout.loc[fleet.turbines]
    one = smooth_direction(wake_tables(lay, float(fleet.rated[0]), k, cache).astype(float), sigma_deg)
    tables = np.broadcast_to(one, (len(K_GRID),) + one.shape[1:])  # every k looks up the same slice
    nan = np.full(fleet.power.shape, np.nan)
    farm = Farm(time=fleet.time, turbines=fleet.turbines, x=lay["x_m"].to_numpy(), y=lay["y_m"].to_numpy(),
                hub_height=lay["hub_height_m"].to_numpy(), diameter=float(lay["rotor_diameter_m"].iloc[0]),
                rated_kw=float(fleet.rated[0]), power=fleet.power, ws=fleet.ws, ws_scada=fleet.ws, ws_std=nan,
                rho=np.full(fleet.power.shape, 1.225), nacelle=fleet.ndir, vane=nan, normal=normal,
                operating=fleet.rpm > 3.0, p_ratio=nan, p_curve=nan)
    params = WarpParams(k_ref=k, sigma_deg=sigma_deg, theta0=0.0)
    model = fit_warp(farm, base, params, tables)
    return predict(farm, model).p_hat


def expected_power(fleet, curves, normal: np.ndarray, method: str, base=None, layout=None,
                   cache=None) -> np.ndarray:
    base = np.ones(len(fleet.time), dtype=bool) if base is None else base
    if method == "nacelle":
        return curves.power(fleet.ws)
    if method == "neighbours":
        u = neighbour_wind(fleet, normal, base)
        # where no neighbour runs (farm-wide stops) fall back to the own anemometer
        return curves.power(np.where(np.isfinite(u), u, fleet.ws))
    if method == "warp":
        p = warp_expected(fleet, layout, normal, base, cache=cache)
        return np.where(np.isfinite(p), p, curves.power(fleet.ws))
    raise ValueError(method)


def lost_energy(fleet, p_hat: np.ndarray, states: np.ndarray) -> np.ndarray:
    """kWh lost per interval in stop or derate states (0 elsewhere, NaN where nothing can be said)."""
    limited = np.isin(states, [STATES.index("stop"), STATES.index("derate")])
    actual = np.nan_to_num(fleet.power, nan=0.0)
    lost = np.maximum(p_hat - np.maximum(actual, 0.0), 0.0) * HOURS
    return np.where(limited, lost, 0.0)
