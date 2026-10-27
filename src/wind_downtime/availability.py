"""Time-based (IEC 61400-26-1) and energy-based (IEC 61400-26-2) availability, and what misclassification costs.

A *contract* says, for every category, whether its time counts as available, unavailable or is excluded
(removed from both sides, e.g. grid outages under most O&M contracts):

```
A_time   = T_available / (T_available + T_unavailable)
A_energy = 1 - E_lost,unavailable / (E_actual + E_lost)          (excluded periods removed from both)
```

Two presets, both illustrative - the real definitions are in the contract and in the standard, which should
be checked before any number here is used:

* ``iec_operational``: the IEC 61400-26-1 information model read from the operator's side: every operative
  state is available (generating, technical standby, out of environmental or electrical specification,
  requested shutdown); scheduled maintenance, planned corrective action and forced outage are unavailable;
  time without information is excluded.
* ``oem_warranty``: a typical manufacturer's availability guarantee: grid, environment and owner or grid
  operator requests are excused (excluded); scheduled maintenance is excused up to an allowance of hours per
  turbine and year; missing information counts as available.

Usage:
    totals = category_totals(category, lost_kwh, actual_kwh, groups, group_names)
    kpi = kpis(totals[totals["group"] == group_names[0]], CONTRACTS["oem_warranty"], allowance_h=60)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wind_downtime.schema import CAT, CATEGORY_LIST, HOURS

AVAILABLE, UNAVAILABLE, EXCLUDED = "available", "unavailable", "excluded"
CONTRACTS = {
    "iec_operational": {
        "full_performance": AVAILABLE, "partial_performance": AVAILABLE, "technical_standby": AVAILABLE,
        "environmental": AVAILABLE, "requested": AVAILABLE, "electrical": AVAILABLE,
        "scheduled_maintenance": UNAVAILABLE, "planned_corrective": UNAVAILABLE, "forced_outage": UNAVAILABLE,
        "no_info": EXCLUDED,
    },
    "oem_warranty": {
        "full_performance": AVAILABLE, "partial_performance": AVAILABLE, "technical_standby": AVAILABLE,
        "environmental": EXCLUDED, "requested": EXCLUDED, "electrical": EXCLUDED,
        "scheduled_maintenance": UNAVAILABLE, "planned_corrective": UNAVAILABLE, "forced_outage": UNAVAILABLE,
        "no_info": AVAILABLE,
    },
}
MAINTENANCE_ALLOWANCE_H = {"iec_operational": 0.0, "oem_warranty": 60.0}

# Misclassifications whose effect on contractual availability is quantified
SCENARIOS = {
    "every stop counted as a turbine fault": {c: "forced_outage" for c in
                                              ("technical_standby", "environmental", "requested", "electrical",
                                               "scheduled_maintenance", "planned_corrective")},
    "grid and curtailment stops counted as turbine faults": {"requested": "forced_outage",
                                                             "electrical": "forced_outage"},
    "missing data counted as downtime": {"no_info": "forced_outage"},
    "maintenance counted as forced outage": {"scheduled_maintenance": "forced_outage",
                                             "planned_corrective": "forced_outage"},
    "environmental stops counted as faults": {"environmental": "forced_outage"},
}


def category_totals(category: np.ndarray, lost_kwh: np.ndarray, actual_kwh: np.ndarray, groups: np.ndarray,
                    group_names) -> pd.DataFrame:
    """Hours, lost and actual energy per (group, category). ``groups``: (time, turbine) integer group index."""
    rows = []
    for g, name in enumerate(group_names):
        sel = groups == g
        if not sel.any():
            continue
        cat = category[sel]
        cat = np.where(cat < 0, CAT["forced_outage"], cat)  # unknown stops: the conservative choice
        hours = np.bincount(cat, minlength=len(CATEGORY_LIST)) * HOURS
        lost = np.bincount(cat, weights=np.nan_to_num(lost_kwh[sel]), minlength=len(CATEGORY_LIST))
        act = np.bincount(cat, weights=np.nan_to_num(np.maximum(actual_kwh[sel], 0)), minlength=len(CATEGORY_LIST))
        for c, cname in enumerate(CATEGORY_LIST):
            rows.append({"group": name, "category": cname, "hours": hours[c], "lost_kwh": lost[c], "actual_kwh": act[c]})
    return pd.DataFrame(rows)


def kpis(totals: pd.DataFrame, contract: dict, allowance_h: float = 0.0, n_turbine_years: float = 1.0) -> dict:
    """Availability from category totals of one group (one turbine-year or a whole farm-year)."""
    t = totals.set_index("category")
    role = t.index.map(contract)
    hours, lost, act = t["hours"], t["lost_kwh"], t["actual_kwh"]
    unavail_h = hours[role == UNAVAILABLE].sum()
    avail_h = hours[role == AVAILABLE].sum()
    # a maintenance allowance moves scheduled maintenance hours from unavailable to excluded
    sm = hours.get("scheduled_maintenance", 0.0) if contract.get("scheduled_maintenance") == UNAVAILABLE else 0.0
    excused = min(sm, allowance_h * n_turbine_years)
    unavail_h -= excused
    lost_un = lost[role == UNAVAILABLE].sum()
    if sm > 0 and excused > 0:
        lost_un -= lost.get("scheduled_maintenance", 0.0) * excused / sm
    keep = role != EXCLUDED
    potential = act[keep].sum() + lost[keep].sum()
    return {"time_availability": avail_h / (avail_h + unavail_h) if avail_h + unavail_h > 0 else np.nan,
            "energy_availability": 1.0 - lost_un / potential if potential > 0 else np.nan,
            "unavailable_h": unavail_h, "available_h": avail_h, "lost_unavailable_mwh": lost_un / 1000.0,
            "lost_total_mwh": lost.sum() / 1000.0, "potential_mwh": potential / 1000.0}


def remap(category: np.ndarray, mapping: dict) -> np.ndarray:
    out = category.copy()
    for src, dst in mapping.items():
        out[category == CAT[src]] = CAT[dst]
    return out
