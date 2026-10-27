"""The comparison study on the three open farms: detection, classification, lost energy, availability.

Protocol (nothing is tuned on the test periods; the label-free parts - normal-behaviour curves, HMM - are
learned on each farm's own unlabelled data, as they would be in use):

=============  =========================  ===========================  =======================================
farm           labelled training           test                         baseline for neighbour ratios / WARP
=============  =========================  ===========================  =======================================
Kelmarsh       2016 (from COD) - 2019      2020-2021                    2017
Penmanshiel    2018                        2020                         2018
Hill of Towie  none (Siemens: transfer)    Jul-Dec 2021                 Jan-Jun 2021
               Jan-Jun 2021 for the
               in-farm comparison only
=============  =========================  ===========================  =======================================

The supervised detector and the event classifier are trained on the Senvion farms only and applied to all
three test sets; on Hill of Towie a model trained on its own first half-year is the in-farm yardstick.

Usage:
    python -m wind_downtime study --data-dir data --results results
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wind_downtime import availability as av
from wind_downtime.classify import (classify_model, classify_rules, event_features, event_label, paint,
                                    train_event_model)
from wind_downtime.curves import fit_curves
from wind_downtime.detect import DETECTORS, STOP
from wind_downtime.energy import expected_power, lost_energy, neighbour_wind
from wind_downtime.events import interval_scores, match_events, to_events
from wind_downtime.hmm import fit_hmm, hmm_states
from wind_downtime.ml import gbm_states, interval_features, train_gbm
from wind_downtime.reference import reference
from wind_downtime.schema import CAT, HOURS, STATES
from wind_downtime.sources import load_dataset

log = logging.getLogger(__name__)
FARMS = {
    "kelmarsh": dict(years=[2016, 2017, 2018, 2019, 2020, 2021], train=lambda t: t.year <= 2019,
                     test=lambda t: t.year >= 2020, base=lambda t: t.year == 2017, tz="Europe/London"),
    "penmanshiel": dict(years=[2018, 2020], train=lambda t: t.year == 2018, test=lambda t: t.year == 2020,
                        base=lambda t: t.year == 2018, tz="Europe/London"),
    "hill_of_towie": dict(years=[2021], train=lambda t: t.month <= 6, test=lambda t: t.month >= 7,
                          base=lambda t: t.month <= 6, tz="Europe/London"),
}
SENVION = ("kelmarsh", "penmanshiel")
LABEL = {"kelmarsh": "Kelmarsh", "penmanshiel": "Penmanshiel", "hill_of_towie": "Hill of Towie"}
METHODS = ["rules", "residual", "signature", "hmm", "gbm"]
METHOD_LABEL = {"rules": "Rule: power <= 0 at >= 3.5 m/s", "residual": "Power curve residual + hysteresis",
                "signature": "Pitch / rotor speed signature", "hmm": "HMM segmentation (unsupervised)",
                "gbm": "Gradient boosting (trained on Senvion)", "gbm_in_farm": "Gradient boosting (in-farm)"}
EXPECTED = ["nacelle", "neighbours", "warp"]


@dataclass
class Farm:
    name: str
    ds: object
    curves: object
    ref: object
    X: np.ndarray
    train: np.ndarray
    test: np.ndarray
    base: np.ndarray
    states: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)


def _rows(fleet, fn) -> np.ndarray:
    return np.asarray(fn(fleet.time))


def prepare(name: str, data_dir: Path) -> Farm:
    cfg = FARMS[name]
    ds = load_dataset(name, cfg["years"], data_dir)
    curves = fit_curves(ds.fleet)
    ref = reference(ds, curves)
    X, _ = interval_features(ds.fleet, curves)
    f = ds.fleet
    return Farm(name, ds, curves, ref, X, _rows(f, cfg["train"]), _rows(f, cfg["test"]), _rows(f, cfg["base"]))


def detect_all(farms: dict) -> None:
    gbm = train_gbm([(F.X, F.ref.state, F.train) for n, F in farms.items() if n in SENVION])
    for name, F in farms.items():
        t0 = time.time()
        f = F.ds.fleet
        for m, fn in DETECTORS.items():
            F.states[m] = fn(f, F.curves)
        F.states["hmm"] = hmm_states(f, F.curves, fit_hmm(f, F.curves, rows=F.test))
        F.states["gbm"] = gbm_states(f, gbm, F.X)
        if name not in SENVION:
            own = train_gbm([(F.X, F.ref.state, F.train)])
            F.states["gbm_in_farm"] = gbm_states(f, own, F.X)
        log.info("%s: detectors done in %.0f s", name, time.time() - t0)


def detection_metrics(farms: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ivl, evt = [], []
    for name, F in farms.items():
        f, test = F.ds.fleet, F.test
        ref_ev = to_events(F.ref.state[test], f.time[test], f.turbines)
        for m, st in F.states.items():
            for d in interval_scores(F.ref.state[test], st[test]):
                ivl.append({"farm": name, "method": m, **d})
            det_ev = to_events(st[test], f.time[test], f.turbines)
            for s in ("stop", "derate"):
                evt.append({"farm": name, "method": m, **match_events(ref_ev, det_ev, s)})
    return pd.DataFrame(ivl), pd.DataFrame(evt)


# ------------------------------------------------------------------------------------- classification

def _events_with_features(F: Farm, states: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    f = F.ds.fleet.subset(rows)
    st = states[rows]
    ev = to_events(st, f.time, f.turbines, which=("stop", "derate"))
    feats = event_features(f, F.curves, ev, st, tz=FARMS[F.name]["tz"])
    feats["label"] = event_label(feats, F.ref.category[rows])
    return feats


def classification(farms: dict) -> tuple:
    """Event classifier trained on Senvion reference events; scored on oracle and detected events."""
    train = pd.concat([_events_with_features(F, F.ref.state, F.train) for n, F in farms.items() if n in SENVION])
    model = train_event_model(train, train["label"])
    # how often each rule's verdict is right on the training events: the confidence the tool reports
    train["rules"] = classify_rules(train)
    ok = train["label"].notna()
    prec = (train.loc[ok, "rules"] == train.loc[ok, "label"]).groupby(train.loc[ok, "rules"]).agg(["mean", "size"])
    prec = prec.rename(columns={"mean": "precision", "size": "n_events"})
    rows, conf, painted = [], [], {}
    for name, F in farms.items():
        oracle = _events_with_features(F, F.ref.state, F.test)
        oracle["rules"] = classify_rules(oracle)
        oracle["gbm"], oracle["confidence"] = classify_model(model, oracle)
        for m in ("rules", "gbm"):
            ok = oracle["label"].notna()
            w = oracle.loc[ok, "duration_h"]
            hit = oracle.loc[ok, m] == oracle.loc[ok, "label"]
            rows.append({"farm": name, "classifier": m, "segmentation": "reference events", "n_events": int(ok.sum()),
                         "accuracy_events": hit.mean(), "accuracy_time": (w * hit).sum() / w.sum()})
            conf.append(pd.crosstab(oracle.loc[ok, "label"], oracle.loc[ok, m]).stack().rename("n").reset_index()
                        .rename(columns={"label": "true", m: "predicted"}).assign(farm=name, classifier=m))
        # end to end: detected events, painted back onto the time axis
        for det in ("signature", "gbm"):
            ev = _events_with_features(F, F.states[det], F.test)
            ev["rules"] = classify_rules(ev)
            ev["gbm"], ev["confidence"] = classify_model(model, ev)
            for m in ("rules", "gbm"):
                cat = paint(F.ref.state[F.test].shape, ev, ev[m], F.states[det][F.test])
                painted[(name, det, m)] = cat
                ref_cat, ref_state = F.ref.category[F.test], F.ref.state[F.test]
                sel = np.isin(ref_state, [STATES.index("stop"), STATES.index("derate")]) & (ref_cat >= 0)
                rows.append({"farm": name, "classifier": m, "segmentation": f"detected ({det})",
                             "n_events": len(ev), "accuracy_events": np.nan,
                             "accuracy_time": float(np.mean(cat[sel] == ref_cat[sel])) if sel.any() else np.nan})
            painted[(name, det, "none")] = paint(F.ref.state[F.test].shape, ev,
                                                 pd.Series(np.where(ev["state"] == "stop", "forced_outage",
                                                                    "partial_performance"), index=ev.index),
                                                 F.states[det][F.test])
    return pd.DataFrame(rows), pd.concat(conf, ignore_index=True), painted, model, prec


# --------------------------------------------------------------------------------------- lost energy

def expected_all(farms: dict, data_dir: Path) -> None:
    for name, F in farms.items():
        f = F.ds.fleet
        normal = (F.states["signature"] == 0) & (f.power > 0.02 * f.rated[None, :])
        for m in EXPECTED:
            t0 = time.time()
            F.expected[m] = expected_power(f, F.curves, normal, m, base=F.base, layout=F.ds.layout,
                                           cache=data_dir / "processed" / f"wake_tables_{name}.npy")
            log.info("%s: expected power %s in %.0f s", name, m, time.time() - t0)


def masking_test(farms: dict, days: int = 1, seed: int = 0) -> pd.DataFrame:
    """Hide a running turbine's power for a day and predict its energy from each reference."""
    rng = np.random.default_rng(seed)
    out = []
    for name, F in farms.items():
        f = F.ds.fleet
        k = 144 * days
        idx = np.flatnonzero(F.test)
        normal = F.ref.state == 0
        for _ in range(400):
            i = rng.integers(f.n)
            s = rng.integers(idx[0], idx[-1] - k)
            sl = slice(s, s + k)
            if normal[sl, i].mean() < 0.95:
                continue
            ok = normal[sl, i]
            actual = np.nansum(f.power[sl, i][ok]) * HOURS
            if actual < 2000:  # kWh
                continue
            row = {"farm": name, "turbine": f.turbines[i], "start": f.time[s], "actual_kwh": actual}
            for m in EXPECTED:
                row[m] = np.nansum(F.expected[m][sl, i][ok]) * HOURS / actual - 1.0
            out.append(row)
    return pd.DataFrame(out)


def stop_bias(farms: dict) -> pd.DataFrame:
    """During logged stops: does the stopped turbine's own anemometer read high, and what does it cost?"""
    out = []
    for name, F in farms.items():
        f = F.ds.fleet
        normal = (F.states["signature"] == 0) & (f.power > 0.02 * f.rated[None, :])
        u_nb = neighbour_wind(f, normal, F.base)
        stop = (F.ref.state == STOP) & F.test[:, None] & np.isfinite(u_nb) & (u_nb >= 4.0) & (u_nb <= 15.0)
        run = (F.ref.state == 0) & normal & F.test[:, None] & np.isfinite(u_nb) & (u_nb >= 4.0) & (u_nb <= 15.0)
        row = {"farm": name, "stop_intervals": int(stop.sum()),
               "ws_ratio_stopped": float(np.nanmedian(f.ws[stop] / u_nb[stop])),
               "ws_ratio_running": float(np.nanmedian(f.ws[run] / u_nb[run]))}
        all_stop = (F.ref.state == STOP) & F.test[:, None]
        for m in EXPECTED:
            row[f"lost_{m}_mwh"] = float(np.nansum(np.maximum(F.expected[m][all_stop], 0)) * HOURS / 1000)
        if "gb_lost_downtime" in F.ds.extra:
            gb = F.ds.extra["gb_lost_downtime"]
            row["lost_operator_mwh"] = float(np.nansum(np.maximum(gb[all_stop], 0)) / 1000)
        out.append(row)
    return pd.DataFrame(out)


def lost_energy_errors(farms: dict, ref_method: str = "warp") -> pd.DataFrame:
    """Lost energy from detected states vs from logged states, with the same expected power."""
    out = []
    for name, F in farms.items():
        f, test = F.ds.fleet, F.test
        p_hat = F.expected[ref_method]
        lost_ref = lost_energy(f, p_hat, F.ref.state)[test]
        for m, st in F.states.items():
            lost_det = lost_energy(f, p_hat, st)[test]
            row = {"farm": name, "method": m, "ref_mwh": lost_ref.sum() / 1000, "det_mwh": lost_det.sum() / 1000}
            for s in ("stop", "derate"):
                c = STATES.index(s)
                r = lost_ref[F.ref.state[test] == c].sum()
                d = lost_det[st[test] == c].sum()
                row[f"{s}_ref_mwh"], row[f"{s}_det_mwh"] = r / 1000, d / 1000
                row[f"{s}_error"] = d / r - 1 if r > 0 else np.nan
            row["error"] = row["det_mwh"] / row["ref_mwh"] - 1 if row["ref_mwh"] > 0 else np.nan
            out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------- availability

def _turbine_year_groups(f, rows) -> tuple[np.ndarray, list]:
    years = f.time[rows].year
    uy = sorted(set(years))
    g = (np.searchsorted(uy, years)[:, None] * f.n + np.arange(f.n)[None, :])
    names = [f"{y}|{t}" for y in uy for t in f.turbines]
    return g, names


def availability_study(farms: dict, painted: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Availability per turbine-year from the reference and from each pipeline; misclassification scenarios."""
    rows, scen, totals_out = [], [], []
    for name, F in farms.items():
        f, test = F.ds.fleet, F.test
        g, names = _turbine_year_groups(f, test)
        p_hat = F.expected["warp"]
        actual = np.where(f.power > 0, f.power, 0.0)[test] * HOURS
        ref_cat = F.ref.category[test]
        if name not in SENVION:  # causes partly unknown: compare downtime only (every stop unavailable)
            ref_cat = np.where(F.ref.state[test] == STOP, CAT["forced_outage"], ref_cat)
        lost_ref = lost_energy(f, p_hat, F.ref.state)[test]
        pipelines = {("reference", "logs"): (ref_cat, lost_ref)}
        for det in ("signature", "gbm"):
            lost_det = lost_energy(f, p_hat, F.states[det])[test]
            for cls in ("none", "rules", "gbm"):
                pipelines[(det, cls)] = (painted[(name, det, cls)], lost_det)
        for (det, cls), (cat, lost) in pipelines.items():
            if name not in SENVION and cls != "logs":
                cat = np.where(F.states[det][test] == STOP, CAT["forced_outage"], cat)
            tot = av.category_totals(cat, lost, actual, g, names)
            tot["farm"], tot["detector"], tot["classifier"] = name, det, cls
            totals_out.append(tot)
            for contract, roles in av.CONTRACTS.items():
                for grp, t in tot.groupby("group"):
                    k = av.kpis(t, roles, av.MAINTENANCE_ALLOWANCE_H[contract])
                    y, tid = grp.split("|")
                    rows.append({"farm": name, "year": int(y), "turbine": tid, "detector": det, "classifier": cls,
                                 "contract": contract, **k})
        if name in SENVION:
            for sname, mapping in av.SCENARIOS.items():
                cat = av.remap(ref_cat, mapping)
                tot = av.category_totals(cat, lost_ref, actual, g, names)
                for contract, roles in av.CONTRACTS.items():
                    base = av.kpis(av.category_totals(ref_cat, lost_ref, actual, np.zeros_like(g), ["all"]), roles,
                                   av.MAINTENANCE_ALLOWANCE_H[contract], n_turbine_years=len(names))
                    alt = av.kpis(tot.assign(group="all").groupby("category", as_index=False)[["hours", "lost_kwh", "actual_kwh"]].sum(),
                                  roles, av.MAINTENANCE_ALLOWANCE_H[contract], n_turbine_years=len(names))
                    scen.append({"farm": name, "scenario": sname, "contract": contract,
                                 "time_pp": 100 * (alt["time_availability"] - base["time_availability"]),
                                 "energy_pp": 100 * (alt["energy_availability"] - base["energy_availability"]),
                                 "reference_time": base["time_availability"], "reference_energy": base["energy_availability"]})
    kpi = pd.DataFrame(rows)
    return kpi, pd.DataFrame(scen), pd.concat(totals_out, ignore_index=True)


def availability_errors(kpi: pd.DataFrame) -> pd.DataFrame:
    ref = kpi[kpi["detector"] == "reference"].set_index(["farm", "year", "turbine", "contract"])
    out = []
    for (det, cls), g in kpi[kpi["detector"] != "reference"].groupby(["detector", "classifier"]):
        g = g.set_index(["farm", "year", "turbine", "contract"])
        for col in ("time_availability", "energy_availability"):
            d = 100 * (g[col] - ref.loc[g.index, col])
            for (farm, contract), x in d.groupby(level=["farm", "contract"]):
                out.append({"farm": farm, "contract": contract, "detector": det, "classifier": cls, "metric": col,
                            "bias_pp": x.mean(), "mae_pp": x.abs().mean(), "max_abs_pp": x.abs().max(),
                            "turbine_years": len(x)})
    return pd.DataFrame(out)


def run(data_dir: Path, results: Path) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    farms = {}
    for name in FARMS:
        t0 = time.time()
        farms[name] = prepare(name, data_dir)
        log.info("%s prepared in %.0f s", name, time.time() - t0)
    detect_all(farms)
    ivl, evt = detection_metrics(farms)
    ivl.to_csv(results / "detection_intervals.csv", index=False)
    evt.to_csv(results / "detection_events.csv", index=False)
    cls, conf, painted, event_model, prec = classification(farms)
    prec.to_csv(results / "rule_precision.csv")
    cls.to_csv(results / "classification.csv", index=False)
    conf.to_csv(results / "classification_confusion.csv", index=False)
    expected_all(farms, data_dir)
    mask = masking_test(farms)
    mask.to_csv(results / "lost_energy_masking.csv", index=False)
    bias = stop_bias(farms)
    bias.to_csv(results / "lost_energy_stops.csv", index=False)
    lerr = lost_energy_errors(farms)
    lerr.to_csv(results / "lost_energy_detection.csv", index=False)
    kpi, scen, totals = availability_study(farms, painted)
    kpi.to_csv(results / "availability_turbine_years.csv", index=False)
    scen.to_csv(results / "availability_scenarios.csv", index=False)
    totals.to_csv(results / "category_totals.csv", index=False)
    aerr = availability_errors(kpi)
    aerr.to_csv(results / "availability_errors.csv", index=False)
    return {"farms": farms, "painted": painted, "event_model": event_model}
