"""Command line interface.

Usage:
    python -m wind_downtime fetch --penmanshiel 2018 2020 --hill-of-towie 2021
    python -m wind_downtime study                     # the comparison on the open farms -> results/
    python -m wind_downtime report                    # charts and docs/index.html from results/
    python -m wind_downtime export penmanshiel --years 2020 --out demo   # an open farm in the long format
    python -m wind_downtime run scada.parquet --rated-kw 2050 --mapping codes.csv --out out/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("wind_downtime")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wind_downtime", description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download Penmanshiel and Hill of Towie from Zenodo")
    f.add_argument("--penmanshiel", type=int, nargs="*", default=[2018, 2020])
    f.add_argument("--hill-of-towie", type=int, nargs="*", default=[2021])

    s = sub.add_parser("study", help="run the comparison on the open farms")
    s.add_argument("--results", type=Path, default=Path("results"))
    s.add_argument("--docs", type=Path, default=Path("docs"))

    r = sub.add_parser("report", help="charts and dashboard from the study results")
    r.add_argument("--results", type=Path, default=Path("results"))
    r.add_argument("--docs", type=Path, default=Path("docs"))

    e = sub.add_parser("export", help="write an open farm in the long format + its code mapping (demo input)")
    e.add_argument("farm", choices=["kelmarsh", "penmanshiel"])
    e.add_argument("--years", type=int, nargs="+", required=True)
    e.add_argument("--out", type=Path, default=Path("demo"))

    u = sub.add_parser("run", help="detect, classify and price downtime on your own long-format SCADA table")
    u.add_argument("input", type=Path, help="Parquet or CSV in the long format (see README)")
    u.add_argument("--out", type=Path, default=Path("out"))
    u.add_argument("--rated-kw", type=float, help="rated power of all turbines (or use --turbines)")
    u.add_argument("--turbines", type=Path, help="CSV: turbine_id, rated_kw [, x_m, y_m | lat, lon, "
                                                  "hub_height_m, rotor_diameter_m] (layout enables WARP)")
    u.add_argument("--mapping", type=Path, help="CSV: code, category - your status codes to IEC categories")
    u.add_argument("--detector", choices=["signature", "residual", "hmm", "rules"], default="signature")
    u.add_argument("--expected", choices=["neighbours", "nacelle", "warp"], default="neighbours")
    u.add_argument("--contract", choices=["iec_operational", "oem_warranty"], default="oem_warranty")
    u.add_argument("--maintenance-allowance-h", type=float, help="excused scheduled maintenance per turbine-year")
    u.add_argument("--tz", default="Europe/Warsaw", help="local time zone for working-hour features")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    if a.cmd == "fetch":
        from wind_downtime.fetch import fetch

        for path in fetch(a.data_dir, a.penmanshiel, a.hill_of_towie):
            log.info("ready: %s", path)
    elif a.cmd == "study":
        from wind_downtime import report, study

        out = study.run(a.data_dir, a.results)
        report.dashboard_data(out, a.results)
        report.build(a.results, a.docs)
    elif a.cmd == "report":
        from wind_downtime import report

        report.build(a.results, a.docs)
    elif a.cmd == "export":
        from wind_downtime.sources import export_long, load_dataset

        long, mapping = export_long(load_dataset(a.farm, a.years, a.data_dir))
        a.out.mkdir(parents=True, exist_ok=True)
        long.to_parquet(a.out / f"{a.farm}_long.parquet")
        mapping.to_csv(a.out / f"{a.farm}_codes.csv", index=False)
        log.info("%d rows and %d mapped codes written to %s", len(long), len(mapping), a.out)
    elif a.cmd == "run":
        from wind_downtime.pipeline import run_pipeline

        run_pipeline(a.input, a.out, rated_kw=a.rated_kw, turbines=a.turbines, mapping=a.mapping,
                     detector=a.detector, expected=a.expected, contract=a.contract,
                     allowance_h=a.maintenance_allowance_h, tz=a.tz)
    return 0
