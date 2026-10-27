"""Charts for the README and the interactive dashboard (docs/index.html) from the study results.

``dashboard_data`` runs right after the study (it needs the per-interval states); ``build`` only reads
``results/`` and can be re-run on its own.

Usage:
    python -m wind_downtime report
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wind_downtime.classify import event_label  # noqa: E402
from wind_downtime.events import to_events  # noqa: E402
from wind_downtime.schema import CATEGORIES, CATEGORY_LIST, STATES  # noqa: E402

log = logging.getLogger(__name__)
COLORS = {
    "full_performance": "#9aa5b1", "partial_performance": "#e3a33b", "technical_standby": "#8e7cc3",
    "environmental": "#3c9ad1", "requested": "#2f9e6e", "electrical": "#1d6b8f",
    "scheduled_maintenance": "#c9a227", "planned_corrective": "#d9772b", "forced_outage": "#c8423b",
    "no_info": "#cfd5db", "unknown": "#6b7480",
}
TIMELINE = {"reference": None, "signature": "rules", "gbm": "gbm", "hmm": None}
# a stretch of each test period worth looking at first
WINDOWS = {"kelmarsh": ["2021-01-20", "2021-03-31"], "penmanshiel": ["2020-09-01", "2020-10-31"],
           "hill_of_towie": ["2021-10-01", "2021-12-01"]}
METHOD_ORDER = ["rules", "residual", "signature", "hmm", "gbm", "gbm_in_farm"]
METHOD_SHORT = {"rules": "Rule P<=0", "residual": "Residual", "signature": "Pitch/rpm", "hmm": "HMM",
                "gbm": "GBM (Senvion)", "gbm_in_farm": "GBM (in-farm)"}
FARM_LABEL = {"kelmarsh": "Kelmarsh", "penmanshiel": "Penmanshiel", "hill_of_towie": "Hill of Towie"}
FARM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots of the validated categorical palette


def _ms(ts) -> list:
    return (pd.DatetimeIndex(ts).as_unit("ms").asi8).tolist()


def dashboard_data(out: dict, results: Path) -> Path:
    """Events per turbine for the timeline and category totals for the calculator -> results/dashboard.json."""
    farms, painted = out["farms"], out["painted"]
    data = {"categories": CATEGORY_LIST + ["unknown"],
            "category_labels": [CATEGORIES[c] for c in CATEGORY_LIST] + ["Stop, cause not logged"],
            "colors": [COLORS[c] for c in CATEGORY_LIST + ["unknown"]], "farms": {}}
    unknown = len(CATEGORY_LIST)
    for name, F in farms.items():
        f, test = F.ds.fleet, F.test
        time = f.time[test]
        farm = {"label": FARM_LABEL[name], "turbines": f.turbines.tolist(), "timeline": {},
                "period": [str(time[0].date()), str(time[-1].date())], "window": WINDOWS[name]}
        for method, cls in TIMELINE.items():
            st = F.ref.state[test] if method == "reference" else F.states[method][test]
            ev = to_events(st, time, f.turbines)
            if method == "reference":
                cat = event_label(ev, F.ref.category[test])
            elif cls is not None:
                cat = event_label(ev, painted[(name, method, cls)])
            else:
                cat = pd.Series(np.where(ev["state"] == "derate", "partial_performance", "forced_outage"), index=ev.index)
            if method != "reference":  # the logs may know why a turbine without data was down
                cat = cat.where(ev["state"] != "no_data", "no_info")
            cat = cat.fillna("no_info")
            code = cat.map({c: i for i, c in enumerate(CATEGORY_LIST)}).fillna(unknown).astype(int)
            farm["timeline"][method] = {
                "turbine": ev["col"].astype(int).tolist(), "start": _ms(ev["start"]), "end": _ms(ev["end"]),
                "state": ev["state"].map({s: i for i, s in enumerate(STATES)}).astype(int).tolist(),
                "category": code.tolist(),
            }
        data["farms"][name] = farm
    path = results / "dashboard.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


# ----------------------------------------------------------------------------------------- charts

def _style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5})


def fig_detection(evt: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    farms = list(FARM_LABEL)
    for ax, state in zip(axes, ("stop", "derate")):
        d = evt[evt["state"] == state]
        methods = [m for m in METHOD_ORDER if m in set(d["method"])]
        w = 0.8 / len(farms)
        for j, farm in enumerate(farms):
            x = d[d["farm"] == farm].set_index("method").reindex(methods)
            ax.bar(np.arange(len(methods)) + (j - 1) * w, x["f1"].fillna(0), w, label=FARM_LABEL[farm],
                   color=FARM_COLORS[j])
        ax.set_xticks(np.arange(len(methods)), [METHOD_SHORT[m] for m in methods], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{state.capitalize()} events: F1 (overlap match)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    axes[0].set_ylim(0, 1.12)
    axes[1].set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_scenarios(scen: pd.DataFrame, path: Path) -> None:
    _style()
    d = scen[scen["contract"] == "oem_warranty"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    names = list(dict.fromkeys(d["scenario"]))
    w = 0.38
    for j, farm in enumerate(["kelmarsh", "penmanshiel"]):
        x = d[d["farm"] == farm].set_index("scenario").reindex(names)
        ax.barh(np.arange(len(names)) + (j - 0.5) * w, x["time_pp"], w, label=FARM_LABEL[farm],
                color=FARM_COLORS[:2][j])
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("change of contractual time-based availability [percentage points]")
    ax.set_title("What a misclassification does to a warranty availability figure (test years)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_lost_energy(mask: pd.DataFrame, stops: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    farms = list(FARM_LABEL)
    for j, farm in enumerate(farms):
        d = mask[mask["farm"] == farm]
        for k, m in enumerate(("nacelle", "neighbours", "warp")):
            v = 100 * d[m].dropna()
            ax.boxplot(v, positions=[k * 4 + j], widths=0.7, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": FARM_COLORS[j], "alpha": 0.8},
                       medianprops={"color": "k"})
    ax.set_xticks([1, 5, 9], ["own anemometer", "neighbours", "WARP"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("error of one day's energy [%]")
    ax.set_title("Masked running days: predicted vs actual energy")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    for j, farm in enumerate(farms):
        r = stops[stops["farm"] == farm].iloc[0]
        vals = [r["lost_nacelle_mwh"], r["lost_neighbours_mwh"], r["lost_warp_mwh"]]
        ax.bar(np.arange(3) * 4 + j, np.array(vals) / r["lost_warp_mwh"], 0.8, color=FARM_COLORS[j],
               label=FARM_LABEL[farm])
    ax.set_xticks([1, 5, 9], ["own anemometer", "neighbours", "WARP"])
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("expected energy in logged stops / WARP")
    ax.set_title("Logged stops: what each reference says was lost")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(conf: pd.DataFrame, path: Path) -> None:
    _style()
    d = conf[conf["farm"].isin(["kelmarsh", "penmanshiel"])]
    cats = [c for c in CATEGORY_LIST if c in set(d["true"]) | set(d["predicted"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cls in zip(axes, ("rules", "gbm")):
        m = d[d["classifier"] == cls].pivot_table(index="true", columns="predicted", values="n", aggfunc="sum")
        m = m.reindex(index=cats, columns=cats).fillna(0)
        share = m.div(m.sum(1).replace(0, np.nan), axis=0)
        ax.imshow(share.fillna(0), cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(m.to_numpy()):
            if v:
                ax.text(c, r, int(v), ha="center", va="center", fontsize=7.5,
                        color="white" if share.iloc[r, c] > 0.55 else "black")
        short = [c.replace("_", " ").replace("scheduled maintenance", "sched. maint.")
                 .replace("planned corrective", "planned corr.").replace("partial performance", "partial perf.")
                 .replace("technical standby", "tech. standby") for c in cats]
        ax.set_xticks(range(len(cats)), short, rotation=40, ha="right")
        ax.set_yticks(range(len(cats)), short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("logged")
        ax.set_title({"rules": "Transparent rules", "gbm": "Gradient boosting"}[cls] + " - logged events, Senvion test years")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build(results: Path, docs: Path) -> None:
    img = docs / "img"
    img.mkdir(parents=True, exist_ok=True)
    evt = pd.read_csv(results / "detection_events.csv")
    fig_detection(evt, img / "detection.png")
    fig_scenarios(pd.read_csv(results / "availability_scenarios.csv"), img / "scenarios.png")
    fig_lost_energy(pd.read_csv(results / "lost_energy_masking.csv"), pd.read_csv(results / "lost_energy_stops.csv"),
                    img / "lost_energy.png")
    fig_confusion(pd.read_csv(results / "classification_confusion.csv"), img / "confusion.png")
    from wind_downtime.dashboard import write_dashboard

    write_dashboard(results, docs / "index.html")
    log.info("report written to %s", docs)
