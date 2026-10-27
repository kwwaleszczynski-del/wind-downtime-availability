"""Download the open datasets used here from Zenodo.

* **Penmanshiel** (14 x Senvion MM82, CC BY 4.0): yearly Greenbyte archives, downloaded whole and checked
  against the published MD5 checksums. Record 16807304.
* **Hill of Towie** (21 x Siemens SWT-2.3-VS-82, CC BY 4.0): the yearly archives hold 13 tables, of which five
  are needed (``TABLES``). Zenodo serves byte ranges, so only those members are read from the remote zip
  (central directory first, then each member) and re-packed locally; the zip CRC of every member is checked on
  extraction. This saves about a quarter of the download. Record 22662930 (version 2.1.0).
* **Kelmarsh** (6 x Senvion MM92) is downloaded by the ``wind_power_curve`` package of the earlier project
  (``python -m wind_power_curve fetch``); point ``KELMARSH_DATA_DIR`` at its data folder.

Usage:
    python -m wind_downtime fetch --penmanshiel 2018 2020 --hill-of-towie 2021
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import requests
from wind_power_curve.zenodo import download_file

PENMANSHIEL_RECORD = "16807304"
HILL_OF_TOWIE_RECORD = "22662930"
PENMANSHIEL_STATIC = ("Penmanshiel_WT_static.csv",)
HILL_OF_TOWIE_STATIC = ("Hill_of_Towie_turbine_metadata.csv", "Hill_of_Towie_alarms_description.csv",
                        "Hill_of_Towie_ShutdownDuration.zip")
TABLES = ("tblSCTurbine", "tblSCTurGrid", "tblSCTurTemp", "tblSCTurFlag", "tblAlarmLog")
USER_AGENT = "wind-downtime-availability (open-source analysis)"
log = logging.getLogger(__name__)


def record_files(session: requests.Session, record: str) -> dict[str, dict]:
    """{file name: {"size", "md5", "url"}} of a Zenodo record."""
    r = session.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    r.raise_for_status()
    out = {}
    for f in r.json()["files"]:
        algo, _, digest = f["checksum"].partition(":")
        out[f["key"]] = {"size": f["size"], "md5": digest if algo == "md5" else None, "url": f["links"]["self"]}
    return out


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, session: requests.Session, url: str, size: int):
        self.session, self.url, self.size, self.pos = session, url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=300)
        r.raise_for_status()
        data = r.content
        if r.status_code != 206 or len(data) != n:
            raise IOError(f"range request failed ({r.status_code}, {len(data)} of {n} bytes)")
        buffer[:n] = data
        self.pos += n
        return n


def extract_members(session: requests.Session, info: dict, pattern: re.Pattern, dest: Path) -> Path:
    """Copy the members of a remote zip whose names match ``pattern`` into the local zip ``dest``."""
    if dest.exists():
        log.info("cached %s", dest.name)
        return dest
    remote = zipfile.ZipFile(io.BufferedReader(RemoteFile(session, info["url"], info["size"]), 8 << 20))
    members = [m for m in remote.infolist() if pattern.search(m.filename)]
    total = sum(m.compress_size for m in members)
    log.info("%s: %d of %d members, %.0f MB of %.0f MB", dest.name, len(members), len(remote.infolist()),
             total / 1e6, info["size"] / 1e6)
    tmp = dest.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for m in members:
            out.writestr(m.filename, remote.read(m))  # read() checks the member's CRC-32
            log.info("  %s", m.filename)
    tmp.replace(dest)
    return dest


def fetch_penmanshiel(years: list[int], data_dir: Path, session: requests.Session) -> list[Path]:
    raw = data_dir / "raw" / "penmanshiel"
    raw.mkdir(parents=True, exist_ok=True)
    files = record_files(session, PENMANSHIEL_RECORD)
    names = list(PENMANSHIEL_STATIC)
    for y in years:
        found = [n for n in files if re.fullmatch(rf"Penmanshiel_SCADA_{y}_WT\d+-\d+_\d+\.zip", n)]
        if not found:
            raise ValueError(f"Penmanshiel: no whole-year archives for {y}")
        names += sorted(found)
    return [download_file(session, files[n], raw / n) for n in names]


def fetch_hill_of_towie(years: list[int], data_dir: Path, session: requests.Session) -> list[Path]:
    raw = data_dir / "raw" / "hill_of_towie"
    raw.mkdir(parents=True, exist_ok=True)
    files = record_files(session, HILL_OF_TOWIE_RECORD)
    out = [download_file(session, files[n], raw / n) for n in HILL_OF_TOWIE_STATIC]
    pattern = re.compile(rf"^({'|'.join(TABLES)})_\d{{4}}_\d{{2}}\.csv$")
    for y in years:
        out.append(extract_members(session, files[f"{y}.zip"], pattern, raw / f"{y}_subset.zip"))
    return out


def fetch(data_dir: Path, penmanshiel: list[int], hill_of_towie: list[int]) -> list[Path]:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    paths = fetch_penmanshiel(penmanshiel, data_dir, session) if penmanshiel else []
    if hill_of_towie:
        paths += fetch_hill_of_towie(hill_of_towie, data_dir, session)
    return paths
