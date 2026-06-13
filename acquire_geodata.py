"""
acquire_geodata.py — DEM acquisition for Fukushima 3D terrain poster.

Downloads the GSI 10m DEM via EMDB GIS API (Phase A) or falls back to
SRTM 30m via R elevatr package (Phase B). Writes geodata/manifest.json
as a handoff contract for Session 3 (rayshader_3d.R).

Usage:
    python acquire_geodata.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBOX = {"xmin": 140.125, "xmax": 141.042, "ymin": 36.742, "ymax": 38.145}
FDNPP = {"lon": 141.032, "lat": 37.422, "exclusion_radius_km": 20}

# Candidate EMDB GIS endpoint — probed first; falls back if not a valid GeoTIFF
EMDB_DEM_PROBE_URL = "https://emdb.jaea.go.jp/emdb/dtjson/item_info"

DEM_DIR = Path("geodata/dem")
MANIFEST_PATH = Path("geodata/manifest.json")

# GeoTIFF magic bytes (little-endian and big-endian TIFF)
TIFF_MAGIC = (b"II\x2a\x00", b"MM\x00\x2a")

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5
REQUEST_TIMEOUT = 60

USER_AGENT = (
    "FukushimaDEM/1.0 (academic research; NE 493; "
    "contact: research@university.edu)"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("acquire_geodata")


def setup_logging() -> None:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S"))
    fh = logging.FileHandler("acquire_geodata.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_geotiff(path: Path) -> bool:
    """Return True if path exists, is non-empty, and has TIFF magic bytes."""
    try:
        if not path.exists() or path.stat().st_size < 4:
            return False
        with path.open("rb") as fh:
            header = fh.read(4)
        return header in TIFF_MAGIC
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(dem_path: Path, source: str, resolution_m: int) -> dict:
    """Build the manifest dict that Session 3 reads."""
    return {
        "dem_path": str(dem_path),
        "dem_source": source,
        "dem_resolution_m": resolution_m,
        "bbox": BBOX,
        "fdnpp": FDNPP,
        "som_clusters_path": "output/som_clusters.parquet",
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "EMDB GIS API probed first (Phase A); "
            "fell back to elevatr SRTM 30m if unavailable (Phase B)."
        ),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    """Atomically write manifest JSON using temp-file-then-os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_str = tempfile.mkstemp(prefix=".manifest_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_str, path)
    except Exception:
        try:
            os.unlink(tmp_str)
        except OSError:
            pass
        raise
    logger.info("Manifest written to %s", path)


# ---------------------------------------------------------------------------
# Phase A — EMDB probe and download
# ---------------------------------------------------------------------------

def probe_emdb_dem(session: requests.Session) -> str | None:
    """Probe the EMDB GIS API for a downloadable GeoTIFF DEM.

    Returns the DEM URL if EMDB responds with a valid GeoTIFF, else None.
    Note: EMDB may serve DEM as CSV (point data) rather than GeoTIFF raster.
    In that case this probe returns None and Phase B (elevatr) is used.
    """
    try:
        logger.info("Phase A — probing EMDB GIS API: %s", EMDB_DEM_PROBE_URL)
        resp = session.get(EMDB_DEM_PROBE_URL, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.info("EMDB probe returned HTTP %d — falling back to elevatr.", resp.status_code)
            return None
        # Validate response is a GeoTIFF (not JSON, CSV, or HTML)
        if resp.content[:4] not in TIFF_MAGIC:
            logger.info(
                "EMDB response is not a GeoTIFF (got %r) — falling back to elevatr.",
                resp.content[:4],
            )
            return None
        logger.info("EMDB DEM found at: %s", resp.url)
        return resp.url
    except requests.exceptions.RequestException as exc:
        logger.info("EMDB probe failed (%s) — falling back to elevatr.", exc)
        return None


def download_dem(url: str, output_path: Path, session: requests.Session) -> None:
    """Stream a GeoTIFF from url to output_path with retry + temp-file-then-replace.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Downloading DEM (attempt %d/%d): %s", attempt, MAX_RETRIES, url)
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()
                tmp_fd, tmp_str = tempfile.mkstemp(prefix=".dem_", suffix=".tif", dir=output_path.parent)
                tmp_path = Path(tmp_str)
                try:
                    with os.fdopen(tmp_fd, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                fh.write(chunk)
                    if not is_valid_geotiff(tmp_path):
                        tmp_path.unlink(missing_ok=True)
                        raise RuntimeError("Downloaded file is not a valid GeoTIFF.")
                    os.replace(tmp_str, output_path)
                    logger.info("DEM saved to %s", output_path)
                    return
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
        except requests.exceptions.Timeout:
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning("Timeout (attempt %d/%d) — retrying in %ds.", attempt, MAX_RETRIES, backoff)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
        except requests.exceptions.RequestException as exc:
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning("Request error (attempt %d/%d): %s — retrying in %ds.", attempt, MAX_RETRIES, exc, backoff)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)

    raise RuntimeError(f"All {MAX_RETRIES} retries exhausted downloading {url}")


# ---------------------------------------------------------------------------
# Phase B — elevatr fallback via Rscript
# ---------------------------------------------------------------------------

_ELEVATR_R_TEMPLATE = """\
suppressPackageStartupMessages({{
  library(elevatr)
  library(terra)
}})
bbox_df <- data.frame(
  x = c({xmin}, {xmax}),
  y = c({ymin}, {ymax})
)
dem <- get_elev_raster(
  locations = bbox_df,
  z         = 10,
  prj       = "EPSG:4326",
  src       = "aws"
)
dir.create(dirname("{out}"), recursive = TRUE, showWarnings = FALSE)
writeRaster(dem, "{out}", overwrite = TRUE)
cat("elevatr DEM written to {out}\\n")
"""


def run_elevatr_fallback(output_path: Path) -> None:
    """Write a temp R script and run it via Rscript to fetch SRTM 30m DEM.

    Raises:
        RuntimeError: If Rscript fails or output file is not produced.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r_script = _ELEVATR_R_TEMPLATE.format(
        xmin=BBOX["xmin"], xmax=BBOX["xmax"],
        ymin=BBOX["ymin"], ymax=BBOX["ymax"],
        out=str(output_path).replace("\\", "/"),
    )
    tmp_fd, tmp_r = tempfile.mkstemp(prefix="elevatr_", suffix=".R")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(r_script)
        logger.info("Phase B — running elevatr fallback via Rscript...")
        try:
            subprocess.run(
                ["Rscript", "--vanilla", tmp_r],
                check=True,
                capture_output=False,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"elevatr fallback failed — Rscript exited with code {exc.returncode}") from exc
        if not is_valid_geotiff(output_path):
            raise RuntimeError(
                f"elevatr fallback failed — output not a valid GeoTIFF: {output_path}"
            )
        logger.info("elevatr DEM saved to %s", output_path)
    finally:
        try:
            os.unlink(tmp_r)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main_logic(force: bool = False) -> None:
    """Main acquisition logic — idempotent unless --force is passed.

    Phase A: probe EMDB → download 10m GeoTIFF if available.
    Phase B: run elevatr Rscript fallback for 30m SRTM.
    Always writes/refreshes geodata/manifest.json.
    """
    dem_10m = DEM_DIR / "fukushima_10m.tif"
    dem_30m = DEM_DIR / "fukushima_30m.tif"

    # Idempotency — skip download if a valid DEM already exists
    if not force:
        for candidate, source, resolution in [
            (dem_10m, "EMDB-GSI-10m", 10),
            (dem_30m, "elevatr-SRTM-30m", 30),
        ]:
            if is_valid_geotiff(candidate):
                logger.info("Valid DEM already exists: %s — skipping download. Use --force to re-acquire.", candidate)
                manifest = build_manifest(candidate, source, resolution)
                write_manifest(manifest, MANIFEST_PATH)
                return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Phase A — EMDB probe
    dem_url = probe_emdb_dem(session)
    if dem_url is not None:
        download_dem(dem_url, dem_10m, session)
        manifest = build_manifest(dem_10m, "EMDB-GSI-10m", 10)
        write_manifest(manifest, MANIFEST_PATH)
        logger.info("Phase A complete — 10m DEM acquired from EMDB.")
        return

    # Phase B — elevatr fallback
    logger.info("Phase A unavailable — running Phase B (elevatr SRTM 30m).")
    run_elevatr_fallback(dem_30m)
    manifest = build_manifest(dem_30m, "elevatr-SRTM-30m", 30)
    write_manifest(manifest, MANIFEST_PATH)
    logger.info("Phase B complete — 30m DEM acquired via elevatr.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire Fukushima DEM for 3D terrain poster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid DEM already exists.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    main_logic(force=args.force)


if __name__ == "__main__":
    main()
