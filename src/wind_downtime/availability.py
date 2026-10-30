"""Time-based (IEC 61400-26-1) and energy-based (IEC 61400-26-2) availability, and what misclassification costs.

A *contract* says, for every category, whether its time counts as available, unavailable or is excluded
(removed from both sides, e.g. grid outages under most O&M contracts):

```
A_time   = T_available / (T_available + T_unavailable)
A_energy = 1 - E_lost,unavailable / (E_actual + E_lost)          (excluded periods removed from both)
```

Two presets, both illustrative - the real definitions are in the contract and in the standard, which should
be checked before any number here is used (the presets and the formulas are those of the res-asset-kpi
library, ``res_asset_kpi.wind``; this module feeds it the integer category codes used in this package):

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
from res_asset_kpi import wind
from res_asset_kpi.wind import AVAILABLE, CONTRACTS, EXCLUDED, MAINTENANCE_ALLOWANCE_H, UNAVAILABLE  # noqa: F401

from wind_downtime.schema import CAT, CATEGORY_LIST, HOURS

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
# category code -> name; code -1 (a stop of unknown cause) -> None, which the library counts as a forced outage
_NAMES = np.array(CATEGORY_LIST + [None], dtype=object)


def category_totals(category: np.ndarray, lost_kwh: np.ndarray, actual_kwh: np.ndarray, groups: np.ndarray,
                    group_names) -> pd.DataFrame:
    """Hours, lost and actual energy per (group, category). ``groups``: (time, turbine) integer group index."""
    labels = np.asarray(group_names, dtype=object)[groups]
    return wind.category_totals(_NAMES[category], HOURS, lost_kwh, actual_kwh, groups=labels)


def kpis(totals: pd.DataFrame, contract: dict, allowance_h: float = 0.0, n_turbine_years: float = 1.0) -> dict:
    """Availability from category totals of one group (one turbine-year or a whole farm-year)."""
    a = wind.availability(totals, contract, allowance_h, n_turbine_years)
    return {"time_availability": a.time, "energy_availability": a.energy,
            "unavailable_h": a.unavailable_h, "available_h": a.available_h,
            "lost_unavailable_mwh": a.lost_unavailable_kwh / 1000.0, "lost_total_mwh": a.lost_total_kwh / 1000.0,
            "potential_mwh": a.potential_kwh / 1000.0}


def remap(category: np.ndarray, mapping: dict) -> np.ndarray:
    out = category.copy()
    for src, dst in mapping.items():
        out[category == CAT[src]] = CAT[dst]
    return out
