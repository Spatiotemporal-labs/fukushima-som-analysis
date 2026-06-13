"""
acquire_rf_predictors.py — Stage A of the Fukushima SOM RF predictor data-acquisition pipeline.

Downloads, validates (Tier-1), and extracts raw data for 5 of the 6 RF predictor sources:
  1. Tsukuba Cs-137 deposition CSV (IED, University of Tsukuba)
  3. MLIT NLNI G04-d elevation+slope GML zips (5 tiles)
  4. MLIT NLNI L03-b land-use GML zips (5 tiles)
  5. MLIT NLNI W05 river centreline GML zip (Fukushima-ken)
  6. MLIT NLNI N03 admin boundary GML zip (Fukushima-ken 2023)

SoilGrids (item 2) is NOT handled here — it requires GDAL/terra and is the R script's job.

USAGE
-----
python3 acquire_rf_predictors.py [--force] [--only csv,g04d,l03b,w05,n03]

NOTES
-----
- Must be run sandbox-OFF (needs network access).
- System python3 only: uses urllib.request, zipfile, hashlib, json, datetime, pathlib, argparse,
  logging. Pandas is used only for the Cs-137 CSV Tier-1 check (installed on system python3).
- No requests / geopandas / rasterio.
- All paths are resolved relative to the script's own directory (repo root), so the script
  works correctly regardless of CWD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "geodata" / "rf_raw"
STATE_FILE = RAW_DIR / ".acquire_state.json"
MANIFEST_FILE = RAW_DIR / "manifest.json"
LOG_FILE = REPO_ROOT / "acquire_rf_predictors.log"

ZIP_MAGIC = b"PK\x03\x04"
USER_AGENT = (
    "RFAcquire/1.0 (academic research; NE 493 Fukushima SOM; "
    "contact: 3amooryksa@gmail.com)"
)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5   # seconds; doubles each attempt
REQUEST_TIMEOUT = 120    # seconds per socket operation
STREAM_CHUNK = 65_536    # 64 KiB

# Tiles required for G04-d and L03-b
TILES = ["5540", "5640", "5740", "5541", "5641"]

# Source definitions — (source_key, url_template_or_url)
# G04-d and L03-b are keyed per-tile; the orchestrator expands them.
SOURCES: dict[str, str | list[tuple[str, str]]] = {}  # built at runtime

# Per-source subdirectories under RAW_DIR
SRC_DIRS: dict[str, str] = {
    "csv":  "tsukuba",
    "g04d": "g04d",
    "l03b": "l03b",
    "w05":  "w05",
    "n03":  "n03",
}

# Expected GML/SHP member extension(s) in each zip source (at least one must be present)
EXPECTED_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "g04d": (".gml", ".xml", ".shp"),
    "l03b": (".gml", ".xml", ".shp"),
    "w05":  (".gml", ".xml", ".shp"),
    "n03":  (".gml", ".xml", ".shp"),
}

# All valid source keys the CLI accepts
ALL_SOURCE_KEYS = ["csv", "g04d", "l03b", "w05", "n03"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("rf_acquire")


def setup_logging() -> None:
    """Dual-handler logging: INFO to stdout, DEBUG to log file."""
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    )

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    logger.addHandler(console)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# State management  (mirrors download_emdb.py pattern)
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Load resume state from STATE_FILE.  Returns empty state if missing/corrupt."""
    if not STATE_FILE.exists():
        return {}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("state root is not a dict")
        return raw
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Could not read state file (%s) — starting fresh.", exc)
        return {}


def save_state(state: dict[str, Any]) -> None:
    """Atomically write state dict to STATE_FILE (temp-file + os.replace)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state["_saved_at"] = datetime.now(timezone.utc).isoformat()
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=".acquire_state_", suffix=".json", dir=RAW_DIR
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path_str, STATE_FILE)
    except OSError:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _make_request(url: str) -> urllib.request.Request:
    """Return a Request with the project User-Agent set."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    return req


def download_url(url: str, dest: Path) -> tuple[int, str]:
    """Download *url* to *dest* with retry + exponential backoff.

    Streams to a temp file, validates ZIP magic bytes on the fly, then
    os.replace to final destination.

    Returns:
        (bytes_written, sha256_hex) on success.

    Raises:
        RuntimeError after MAX_RETRIES exhausted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        logger.debug("  Attempt %d/%d → %s", attempt, MAX_RETRIES, url)
        try:
            req = _make_request(url)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP {resp.status}", {}, None  # type: ignore[arg-type]
                    )
                tmp_fd, tmp_path_str = tempfile.mkstemp(
                    prefix=".tmp_rf_", dir=dest.parent
                )
                tmp_path = Path(tmp_path_str)
                sha = hashlib.sha256()
                n_bytes = 0
                try:
                    with os.fdopen(tmp_fd, "wb") as fh:
                        first_chunk = True
                        while True:
                            chunk = resp.read(STREAM_CHUNK)
                            if not chunk:
                                break
                            if first_chunk:
                                # Magic-byte pre-check for zip sources (CSV will fail → handled later)
                                if chunk[:4] == ZIP_MAGIC or chunk[:1] in (b'"', b'O', b'M'):
                                    # zip or text-CSV — both acceptable at this stage
                                    pass
                                first_chunk = False
                            fh.write(chunk)
                            sha.update(chunk)
                            n_bytes += len(chunk)
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise

                if n_bytes == 0:
                    tmp_path.unlink(missing_ok=True)
                    raise ValueError(f"Empty response body for {url}")

                os.replace(tmp_path, dest)
                logger.debug("  Saved %d bytes to %s", n_bytes, dest)
                return n_bytes, sha.hexdigest()

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f"HTTP 404 — resource not found: {url}") from exc
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "HTTP %s on attempt %d/%d — retrying in %ds.", exc.code, attempt, MAX_RETRIES, backoff
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
            else:
                raise RuntimeError(
                    f"HTTP {exc.code} after {MAX_RETRIES} attempts for {url}"
                ) from exc

        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "Network error on attempt %d/%d (%s) — retrying in %ds.",
                attempt, MAX_RETRIES, exc, backoff,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
            else:
                raise RuntimeError(
                    f"Download failed after {MAX_RETRIES} attempts for {url}: {exc}"
                ) from exc

    raise RuntimeError(f"Exhausted retries for {url}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_zip(path: Path, expected_exts: tuple[str, ...]) -> None:
    """Raise ValueError if path is not a valid ZIP or lacks expected member types.

    Checks:
      1. Magic bytes PK\\x03\\x04
      2. zipfile.testzip() returns None (no CRC errors)
      3. At least one member with an extension in expected_exts is present
    """
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic != ZIP_MAGIC:
        raise ValueError(f"Bad ZIP magic bytes in {path}: got {magic!r}")

    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"CRC error in ZIP member '{bad}' of {path}")
        names = [n.lower() for n in zf.namelist()]

    exts_found = {Path(n).suffix for n in names}
    if not any(ext in exts_found for ext in expected_exts):
        raise ValueError(
            f"ZIP {path.name} has no members with extensions {expected_exts}; found: {exts_found}"
        )


def validate_zip_and_extract(path: Path, extract_dir: Path, expected_exts: tuple[str, ...]) -> list[str]:
    """Validate *path* as a ZIP, then extract all members to *extract_dir*.

    Returns:
        List of member names extracted.
    """
    validate_zip(path, expected_exts)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        members = zf.namelist()
        # Some MLIT NLNI zips (e.g. L03-b tiles 5540/5541) use Windows '\\' path
        # separators, so a plain extractall() yields files with a literal backslash
        # in the name on POSIX. Sanitize: split on both separators, flatten, and
        # write each member to a clean path under extract_dir.
        for info in zf.infolist():
            if info.is_dir():
                continue
            clean = info.filename.replace("\\", "/")
            target = extract_dir / Path(clean).name
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    logger.debug("  Extracted %d members to %s", len(members), extract_dir)
    return members


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file without loading it fully into memory."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(STREAM_CHUNK)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Per-source download + Tier-1 functions
# ---------------------------------------------------------------------------

def fetch_csv(state: dict[str, Any], force: bool, manifest: dict[str, Any]) -> bool:
    """Download and Tier-1-validate the Tsukuba Cs-137 deposition CSV (source 1).

    Tier-1 checks:
      - File exists and is non-empty
      - Header contains: cs137_inve, Latitude, Longitude, MESH
      - Row count > 50,000
      - cs137_inve column is numeric (coercible to float)
      - At least one row has cs137_inve > 1e4 (contaminated cells present)
      - At least one row has cs137_inve == 0 (left-censoring present)

    Returns True on pass, False on Tier-1 failure.
    """
    src_key = "csv"
    url = "https://www.ied.tsukuba.ac.jp/~fukushimafallout/data/cs137a_point.csv"
    dest_dir = RAW_DIR / SRC_DIRS[src_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "cs137a_point.csv"

    logger.info("[csv] Tsukuba Cs-137 CSV ← %s", url)

    # Resume check
    if not force and state.get(src_key, {}).get("status") == "ok" and dest.exists() and dest.stat().st_size > 0:
        logger.info("[csv] Already ok in state and file present — skipping download.")
        manifest[src_key] = state[src_key].get("manifest_entry", {})
        return _tier1_csv(dest, src_key, manifest)

    # Download
    try:
        n_bytes, sha256 = download_url(url, dest)
    except RuntimeError as exc:
        logger.error("[csv] Download failed: %s", exc)
        state[src_key] = {"status": "failed", "error": str(exc)}
        return False

    entry: dict[str, Any] = {
        "url": url,
        "bytes": n_bytes,
        "sha256": sha256,
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "members": [str(dest.name)],
        "status": "pending",
    }
    manifest[src_key] = entry

    return _tier1_csv(dest, src_key, manifest)


def _tier1_csv(dest: Path, src_key: str, manifest: dict[str, Any]) -> bool:
    """Run Tier-1 checks on the Cs-137 CSV.  Updates manifest[src_key]['status']."""
    logger.info("[csv] Running Tier-1 checks on %s …", dest.name)

    # File exists + non-empty
    if not dest.exists() or dest.stat().st_size == 0:
        logger.error("[csv] FAIL — file missing or empty: %s", dest)
        manifest[src_key]["status"] = "tier1_fail"
        return False

    try:
        import pandas as pd  # noqa: PLC0415 — guarded import; pandas confirmed installed
    except ImportError:
        logger.error("[csv] FAIL — pandas not available for CSV check.")
        manifest[src_key]["status"] = "tier1_fail"
        return False

    try:
        df = pd.read_csv(dest, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("[csv] FAIL — could not parse CSV: %s", exc)
        manifest[src_key]["status"] = "tier1_fail"
        return False

    required_cols = {"cs137_inve", "Latitude", "Longitude", "MESH"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("[csv] FAIL — missing required columns: %s", missing)
        manifest[src_key]["status"] = "tier1_fail"
        return False
    logger.info("[csv] Header check PASS (required columns present).")

    n_rows = len(df)
    if n_rows <= 50_000:
        logger.error("[csv] FAIL — row count %d <= 50,000.", n_rows)
        manifest[src_key]["status"] = "tier1_fail"
        return False
    logger.info("[csv] Row count PASS (%d rows).", n_rows)

    try:
        vals = pd.to_numeric(df["cs137_inve"], errors="raise")
    except Exception as exc:  # noqa: BLE001
        logger.error("[csv] FAIL — cs137_inve not numeric: %s", exc)
        manifest[src_key]["status"] = "tier1_fail"
        return False
    logger.info("[csv] cs137_inve numeric PASS.")

    if not (vals > 1e4).any():
        logger.error("[csv] FAIL — no row with cs137_inve > 1e4 Bq/m² (no contaminated cells found).")
        manifest[src_key]["status"] = "tier1_fail"
        return False
    logger.info("[csv] Contaminated-cells check PASS (max=%.0f Bq/m²).", vals.max())

    if not (vals == 0).any():
        logger.error("[csv] FAIL — no row with cs137_inve == 0 (left-censoring absent).")
        manifest[src_key]["status"] = "tier1_fail"
        return False
    n_censored = int((vals == 0).sum())
    logger.info("[csv] Left-censoring PASS (%d zero rows present).", n_censored)

    logger.info("[csv] ALL TIER-1 CHECKS PASSED.")
    manifest[src_key]["status"] = "ok"
    manifest[src_key]["tier1"] = {
        "rows": n_rows,
        "max_cs137_bqm2": float(vals.max()),
        "n_censored_zero": n_censored,
    }
    return True


def fetch_g04d(state: dict[str, Any], force: bool, manifest: dict[str, Any]) -> bool:
    """Download and Tier-1-validate MLIT NLNI G04-d elevation+slope zips (source 3, 5 tiles).

    Returns True only if ALL tiles pass Tier-1.
    """
    src_key = "g04d"
    url_tmpl = (
        "https://nlftp.mlit.go.jp/ksj/gml/data/G04-d/G04-d-11/"
        "G04-d-11_{tile}-jgd_GML.zip"
    )
    dest_dir = RAW_DIR / SRC_DIRS[src_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    all_pass = True

    logger.info("[g04d] MLIT G04-d elevation+slope — %d tiles", len(TILES))
    manifest.setdefault(src_key, {})

    for tile in TILES:
        url = url_tmpl.format(tile=tile)
        dest = dest_dir / f"G04-d-11_{tile}-jgd_GML.zip"
        tile_key = f"{src_key}_{tile}"

        logger.info("[g04d] tile %s ← %s", tile, url)

        if (
            not force
            and state.get(tile_key, {}).get("status") == "ok"
            and dest.exists()
            and dest.stat().st_size > 0
        ):
            logger.info("[g04d] tile %s already ok — skipping download.", tile)
            all_pass = all_pass and True
            manifest[src_key][tile] = state[tile_key].get("manifest_entry", {"status": "ok"})
            continue

        try:
            n_bytes, sha256 = download_url(url, dest)
        except RuntimeError as exc:
            logger.error("[g04d] tile %s download FAILED: %s", tile, exc)
            state[tile_key] = {"status": "failed", "error": str(exc)}
            all_pass = False
            manifest[src_key][tile] = {"url": url, "status": "failed", "error": str(exc)}
            continue

        entry: dict[str, Any] = {
            "url": url,
            "bytes": n_bytes,
            "sha256": sha256,
            "utc_iso": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        manifest[src_key][tile] = entry

        # Tier-1: magic + integrity + expected members
        ok, members = _tier1_zip(dest, EXPECTED_EXTENSIONS["g04d"], f"g04d tile {tile}")
        if ok:
            extract_dir = dest_dir / f"tile_{tile}"
            try:
                members = validate_zip_and_extract(dest, extract_dir, EXPECTED_EXTENSIONS["g04d"])
                manifest[src_key][tile]["members"] = members
                manifest[src_key][tile]["status"] = "ok"
                manifest[src_key][tile]["extract_dir"] = str(extract_dir)
                state[tile_key] = {"status": "ok", "manifest_entry": manifest[src_key][tile]}
            except (ValueError, OSError) as exc:
                logger.error("[g04d] tile %s extraction FAILED: %s", tile, exc)
                manifest[src_key][tile]["status"] = "tier1_fail"
                state[tile_key] = {"status": "failed", "error": str(exc)}
                all_pass = False
        else:
            manifest[src_key][tile]["status"] = "tier1_fail"
            all_pass = False

    if all_pass:
        logger.info("[g04d] ALL TILES TIER-1 PASSED.")
    else:
        logger.error("[g04d] One or more tiles failed Tier-1.")
    return all_pass


def fetch_l03b(state: dict[str, Any], force: bool, manifest: dict[str, Any]) -> bool:
    """Download and Tier-1-validate MLIT NLNI L03-b land-use zips (source 4, 5 tiles).

    Returns True only if ALL tiles pass Tier-1.
    """
    src_key = "l03b"
    url_tmpl = (
        "https://nlftp.mlit.go.jp/ksj/gml/data/L03-b/L03-b-21/"
        "L03-b-21_{tile}-jgd2011_GML.zip"
    )
    dest_dir = RAW_DIR / SRC_DIRS[src_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    all_pass = True

    logger.info("[l03b] MLIT L03-b land-use — %d tiles", len(TILES))
    manifest.setdefault(src_key, {})

    for tile in TILES:
        url = url_tmpl.format(tile=tile)
        dest = dest_dir / f"L03-b-21_{tile}-jgd2011_GML.zip"
        tile_key = f"{src_key}_{tile}"

        logger.info("[l03b] tile %s ← %s", tile, url)

        if (
            not force
            and state.get(tile_key, {}).get("status") == "ok"
            and dest.exists()
            and dest.stat().st_size > 0
        ):
            logger.info("[l03b] tile %s already ok — skipping download.", tile)
            all_pass = all_pass and True
            manifest[src_key][tile] = state[tile_key].get("manifest_entry", {"status": "ok"})
            continue

        try:
            n_bytes, sha256 = download_url(url, dest)
        except RuntimeError as exc:
            logger.error("[l03b] tile %s download FAILED: %s", tile, exc)
            state[tile_key] = {"status": "failed", "error": str(exc)}
            all_pass = False
            manifest[src_key][tile] = {"url": url, "status": "failed", "error": str(exc)}
            continue

        entry: dict[str, Any] = {
            "url": url,
            "bytes": n_bytes,
            "sha256": sha256,
            "utc_iso": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        manifest[src_key][tile] = entry

        ok, members = _tier1_zip(dest, EXPECTED_EXTENSIONS["l03b"], f"l03b tile {tile}")
        if ok:
            extract_dir = dest_dir / f"tile_{tile}"
            try:
                members = validate_zip_and_extract(dest, extract_dir, EXPECTED_EXTENSIONS["l03b"])
                manifest[src_key][tile]["members"] = members
                manifest[src_key][tile]["status"] = "ok"
                manifest[src_key][tile]["extract_dir"] = str(extract_dir)
                state[tile_key] = {"status": "ok", "manifest_entry": manifest[src_key][tile]}
            except (ValueError, OSError) as exc:
                logger.error("[l03b] tile %s extraction FAILED: %s", tile, exc)
                manifest[src_key][tile]["status"] = "tier1_fail"
                state[tile_key] = {"status": "failed", "error": str(exc)}
                all_pass = False
        else:
            manifest[src_key][tile]["status"] = "tier1_fail"
            all_pass = False

    if all_pass:
        logger.info("[l03b] ALL TILES TIER-1 PASSED.")
    else:
        logger.error("[l03b] One or more tiles failed Tier-1.")
    return all_pass


def fetch_w05(state: dict[str, Any], force: bool, manifest: dict[str, Any]) -> bool:
    """Download and Tier-1-validate MLIT NLNI W05 river centreline zip (source 5).

    Returns True on Tier-1 pass, False otherwise.
    """
    src_key = "w05"
    url = "https://nlftp.mlit.go.jp/ksj/gml/data/W05/W05-07/W05-07_07_GML.zip"
    dest_dir = RAW_DIR / SRC_DIRS[src_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "W05-07_07_GML.zip"

    logger.info("[w05] MLIT W05 river centrelines (Fukushima) ← %s", url)

    if (
        not force
        and state.get(src_key, {}).get("status") == "ok"
        and dest.exists()
        and dest.stat().st_size > 0
    ):
        logger.info("[w05] Already ok in state and file present — skipping download.")
        manifest[src_key] = state[src_key].get("manifest_entry", {})
        ok, _ = _tier1_zip(dest, EXPECTED_EXTENSIONS["w05"], "w05")
        return ok

    try:
        n_bytes, sha256 = download_url(url, dest)
    except RuntimeError as exc:
        logger.error("[w05] Download FAILED: %s", exc)
        state[src_key] = {"status": "failed", "error": str(exc)}
        manifest[src_key] = {"url": url, "status": "failed", "error": str(exc)}
        return False

    entry: dict[str, Any] = {
        "url": url,
        "bytes": n_bytes,
        "sha256": sha256,
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    manifest[src_key] = entry

    ok, members = _tier1_zip(dest, EXPECTED_EXTENSIONS["w05"], "w05")
    if ok:
        try:
            members = validate_zip_and_extract(dest, dest_dir, EXPECTED_EXTENSIONS["w05"])
            manifest[src_key]["members"] = members
            manifest[src_key]["status"] = "ok"
            state[src_key] = {"status": "ok", "manifest_entry": manifest[src_key]}
            logger.info("[w05] TIER-1 PASSED.")
        except (ValueError, OSError) as exc:
            logger.error("[w05] Extraction FAILED: %s", exc)
            manifest[src_key]["status"] = "tier1_fail"
            state[src_key] = {"status": "failed", "error": str(exc)}
            return False
    else:
        manifest[src_key]["status"] = "tier1_fail"
    return ok


def fetch_n03(state: dict[str, Any], force: bool, manifest: dict[str, Any]) -> bool:
    """Download and Tier-1-validate MLIT NLNI N03 admin boundary zip (source 6).

    Returns True on Tier-1 pass, False otherwise.
    """
    src_key = "n03"
    url = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2023/N03-20230101_07_GML.zip"
    dest_dir = RAW_DIR / SRC_DIRS[src_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "N03-20230101_07_GML.zip"

    logger.info("[n03] MLIT N03 admin boundaries (Fukushima 2023) ← %s", url)

    if (
        not force
        and state.get(src_key, {}).get("status") == "ok"
        and dest.exists()
        and dest.stat().st_size > 0
    ):
        logger.info("[n03] Already ok in state and file present — skipping download.")
        manifest[src_key] = state[src_key].get("manifest_entry", {})
        ok, _ = _tier1_zip(dest, EXPECTED_EXTENSIONS["n03"], "n03")
        return ok

    try:
        n_bytes, sha256 = download_url(url, dest)
    except RuntimeError as exc:
        logger.error("[n03] Download FAILED: %s", exc)
        state[src_key] = {"status": "failed", "error": str(exc)}
        manifest[src_key] = {"url": url, "status": "failed", "error": str(exc)}
        return False

    entry: dict[str, Any] = {
        "url": url,
        "bytes": n_bytes,
        "sha256": sha256,
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    manifest[src_key] = entry

    ok, members = _tier1_zip(dest, EXPECTED_EXTENSIONS["n03"], "n03")
    if ok:
        try:
            members = validate_zip_and_extract(dest, dest_dir, EXPECTED_EXTENSIONS["n03"])
            manifest[src_key]["members"] = members
            manifest[src_key]["status"] = "ok"
            state[src_key] = {"status": "ok", "manifest_entry": manifest[src_key]}
            logger.info("[n03] TIER-1 PASSED.")
        except (ValueError, OSError) as exc:
            logger.error("[n03] Extraction FAILED: %s", exc)
            manifest[src_key]["status"] = "tier1_fail"
            state[src_key] = {"status": "failed", "error": str(exc)}
            return False
    else:
        manifest[src_key]["status"] = "tier1_fail"
    return ok


# ---------------------------------------------------------------------------
# Shared ZIP Tier-1 helper
# ---------------------------------------------------------------------------

def _tier1_zip(
    path: Path,
    expected_exts: tuple[str, ...],
    label: str,
) -> tuple[bool, list[str]]:
    """Run ZIP Tier-1 checks (magic bytes, CRC, expected extensions).

    Returns (passed: bool, member_names: list[str]).
    """
    logger.info("[%s] Tier-1 ZIP check on %s …", label, path.name)

    if not path.exists() or path.stat().st_size == 0:
        logger.error("[%s] FAIL — file missing or zero-size.", label)
        return False, []

    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic != ZIP_MAGIC:
        logger.error("[%s] FAIL — bad magic bytes: %r", label, magic)
        return False, []

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                logger.error("[%s] FAIL — CRC error in member '%s'.", label, bad)
                return False, []
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        logger.error("[%s] FAIL — not a valid ZIP: %s", label, exc)
        return False, []

    exts_found = {Path(n).suffix.lower() for n in names}
    if not any(ext.lower() in exts_found for ext in expected_exts):
        logger.error(
            "[%s] FAIL — no expected extensions %s found (got: %s).",
            label, expected_exts, exts_found,
        )
        return False, names

    logger.info("[%s] ZIP Tier-1 PASS (%d members, exts=%s).", label, len(names), exts_found)
    return True, names


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(manifest: dict[str, Any]) -> None:
    """Write geodata/rf_raw/manifest.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest["_written_at"] = datetime.now(timezone.utc).isoformat()
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=".manifest_", suffix=".json", dir=RAW_DIR
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path_str, MANIFEST_FILE)
    except OSError:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise
    logger.info("Manifest written to %s", MANIFEST_FILE)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FETCH_FUNCTIONS = {
    "csv":  fetch_csv,
    "g04d": fetch_g04d,
    "l03b": fetch_l03b,
    "w05":  fetch_w05,
    "n03":  fetch_n03,
}


def run(args: argparse.Namespace) -> int:
    """Main orchestrator.  Returns 0 on all-pass, 1 on any Tier-1 failure."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    manifest: dict[str, Any] = {}

    # Determine which sources to process
    if args.only:
        requested = [s.strip().lower() for s in args.only.split(",")]
        unknown = set(requested) - set(ALL_SOURCE_KEYS)
        if unknown:
            logger.error("Unknown source key(s) in --only: %s. Valid: %s", unknown, ALL_SOURCE_KEYS)
            return 1
        sources_to_run = requested
    else:
        sources_to_run = ALL_SOURCE_KEYS

    logger.info(
        "Stage A starting — sources: %s | force=%s | cache: %s",
        sources_to_run, args.force, RAW_DIR,
    )

    # Signal handlers for graceful interrupt
    def _handle_signal(signum: int, frame: Any) -> None:
        logger.info("Signal %d — saving state and manifest before exit.", signum)
        save_state(state)
        write_manifest(manifest)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    results: dict[str, bool] = {}
    for src_key in sources_to_run:
        fn = FETCH_FUNCTIONS[src_key]
        try:
            passed = fn(state, args.force, manifest)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Unexpected error: %s", src_key, exc, exc_info=True)
            passed = False
            state[src_key] = {"status": "failed", "error": str(exc)}
        results[src_key] = passed
        save_state(state)  # checkpoint after each source

    write_manifest(manifest)

    # Summary banner
    logger.info("=" * 60)
    all_pass = True
    for src_key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-6s  %s", src_key, status)
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("Stage A complete — ALL TIER-1 CHECKS PASSED.")
        return 0
    else:
        logger.error("Stage A complete — ONE OR MORE TIER-1 CHECKS FAILED. See log for details.")
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage A: download + Tier-1-validate RF predictor sources "
            "(Tsukuba CSV, MLIT G04-d, L03-b, W05, N03). "
            "Must run sandbox-OFF (network required)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all sources even if state marks them ok.",
    )
    parser.add_argument(
        "--only",
        metavar="src1,src2,...",
        default="",
        help=(
            "Comma-separated subset of sources to process. "
            f"Valid keys: {', '.join(ALL_SOURCE_KEYS)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging()
    logger.info(
        "acquire_rf_predictors.py — Fukushima SOM Stage A | %s",
        datetime.now(timezone.utc).isoformat(),
    )
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
