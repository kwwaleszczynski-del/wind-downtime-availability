"""From interval states to events, and the event-level comparison with a reference.

An event is a run of intervals of one turbine in the same non-normal state (stop, derate, no_data). Runs of
the same state separated by a single interval are merged (a restart attempt does not end a stop).

Matching: a detected and a reference event of the same state match when they share at least one interval.
Recall is the share of reference events matched, precision the share of detected events matched. Duration
error compares each reference event with the detected event that overlaps it most.

Usage:
    ev = to_events(states, fleet.time, fleet.turbines)
    m = match_events(to_events(ref.state, ...), ev, state="stop")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wind_downtime.detect import runs
from wind_downtime.schema import STATES, STEP

MERGE_GAP = 1


def to_events(states: np.ndarray, time: pd.DatetimeIndex, turbines, which=("derate", "stop", "no_data"),
              merge_gap: int = MERGE_GAP) -> pd.DataFrame:
    rows = []
    for name in which:
        code = STATES.index(name)
        for i, tid in enumerate(turbines):
            m = states[:, i] == code
            if merge_gap:
                gs, ge = runs(~m)
                for a, b in zip(gs, ge):
                    if b - a <= merge_gap and a > 0 and b < len(m) and (states[a:b, i] == 0).all():
                        m[a:b] = True
            s, e = runs(m)
            for a, b in zip(s, e):
                rows.append((tid, i, name, a, b))
    ev = pd.DataFrame(rows, columns=["turbine_id", "col", "state", "k0", "k1"])
    ev["start"] = time[ev["k0"].to_numpy()] if len(ev) else pd.Series(dtype="datetime64[ns, UTC]")
    ev["end"] = time[ev["k1"].to_numpy() - 1] + STEP if len(ev) else pd.Series(dtype="datetime64[ns, UTC]")
    ev["duration_h"] = (ev["k1"] - ev["k0"]) / 6.0
    return ev.sort_values(["col", "k0"]).reset_index(drop=True)


def _overlaps(a: pd.DataFrame, b: pd.DataFrame) -> list[tuple[int, int, int]]:
    """(index in a, index in b, shared intervals) for every overlapping pair on the same turbine."""
    out = []
    for col, ga in a.groupby("col"):
        gb = b[b["col"] == col]
        if gb.empty:
            continue
        b0, b1, bi = gb["k0"].to_numpy(), gb["k1"].to_numpy(), gb.index.to_numpy()
        order = np.argsort(b0)
        b0, b1, bi = b0[order], b1[order], bi[order]
        for ia, k0, k1 in zip(ga.index, ga["k0"], ga["k1"]):
            j = np.searchsorted(b0, k1, side="left")  # b events starting before a ends
            cand = np.arange(j)
            cand = cand[b1[cand] > k0]
            for c in cand:
                out.append((ia, bi[c], int(min(k1, b1[c]) - max(k0, b0[c]))))
    return out


def match_events(ref: pd.DataFrame, det: pd.DataFrame, state: str) -> dict:
    r, d = ref[ref["state"] == state], det[det["state"] == state]
    pairs = _overlaps(r, d)
    matched_r = {p[0] for p in pairs}
    matched_d = {p[1] for p in pairs}
    best = {}
    for ia, ib, n in pairs:
        if n > best.get(ia, (None, -1))[1]:
            best[ia] = (ib, n)
    err = np.array([d.loc[ib, "duration_h"] - r.loc[ia, "duration_h"] for ia, (ib, _) in best.items()])
    recall = len(matched_r) / len(r) if len(r) else np.nan
    precision = len(matched_d) / len(d) if len(d) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    return {"state": state, "n_ref": len(r), "n_det": len(d), "recall": recall, "precision": precision, "f1": f1,
            "duration_bias_h": float(np.median(err)) if len(err) else np.nan,
            "duration_mae_h": float(np.median(np.abs(err))) if len(err) else np.nan,
            "ref_hours": float(r["duration_h"].sum()), "det_hours": float(d["duration_h"].sum())}


def interval_scores(ref_state: np.ndarray, det_state: np.ndarray, rows=None) -> list[dict]:
    """Per-state precision, recall and F1 over intervals where neither side is missing data."""
    nd = STATES.index("no_data")
    valid = (ref_state != nd) & (det_state != nd)
    if rows is not None:
        valid &= rows[:, None]
    out = []
    for name in ("stop", "derate"):
        c = STATES.index(name)
        tp = np.sum(valid & (ref_state == c) & (det_state == c))
        fp = np.sum(valid & (ref_state != c) & (det_state == c))
        fn = np.sum(valid & (ref_state == c) & (det_state != c))
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        out.append({"state": name, "precision": p, "recall": r,
                    "f1": 2 * p * r / (p + r) if p + r > 0 else np.nan, "ref_intervals": int(tp + fn)})
    return out
