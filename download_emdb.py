"""
download_emdb.py — JAEA EMDB air dose rate dataset downloader.

Downloads ZIP archives from the JAEA Environmental Radioactivity Database
(EMDB) for all air dose rate items (content_code=1) across fiscal years
2011–2024. Supports resumable downloads via a JSON state file.

Usage:
    python download_emdb.py [--dry-run] [--year-start 2011] [--year-end 2024] [--delay 2.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://radioactivity.nra.go.jp/emdb"
ITEM_INFO_URL = f"{BASE_URL}/dtjson/item_info"
DOWNLOAD_URL_TEMPLATE = BASE_URL + "/download/zip/{item_code}/{fiscal_year}/{char_code}/{lang_code}"
CHAR_CODE = 1
LANG_CODE = 0
YEAR_START = 2011
YEAR_END = 2024
CONTENT_CODE = 1
DELAY_BETWEEN_REQUESTS = 2.0
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5
REQUEST_TIMEOUT = 60
DATASET_DIR = Path("dataset")
STATE_FILE = Path("download_state.json")
CLASS_CODE_NAMES: dict[int, str] = {
    2: "02_drive_survey",
    3: "03_walk_survey",
    4: "04_survey_meter",
    45: "45_airborne_monitoring",
    50: "50_background",
    60: "60_monitoring_post",
    78: "78_other",
}

ZIP_MAGIC = b"PK\x03\x04"
USER_AGENT = (
    "EMDB-Downloader/1.0 (academic research; NE 493; "
    "contact: research@university.edu)"
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("emdb")


def setup_logging() -> None:
    """Configure dual-handler logging: INFO to console, DEBUG to file."""
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    file_handler = logging.FileHandler("download.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Load download state from STATE_FILE.

    Returns:
        Dict with keys 'completed' (set), 'skipped_empty' (set),
        'failed' (dict mapping task_key -> error message).
    """
    if not STATE_FILE.exists():
        return {"completed": set(), "skipped_empty": set(), "failed": {}}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return {
            "completed": set(raw.get("completed", [])),
            "skipped_empty": set(raw.get("skipped_empty", [])),
            "failed": raw.get("failed", {}),
        }
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read state file: %s — starting fresh.", exc)
        return {"completed": set(), "skipped_empty": set(), "failed": {}}


def save_state(
    completed: set[str],
    skipped_empty: set[str],
    failed: dict[str, str],
) -> None:
    """Atomically write download state to STATE_FILE.

    Uses a temp file + os.replace to avoid partial writes on interruption.

    Args:
        completed: Set of task keys that finished successfully.
        skipped_empty: Set of task keys that returned no data (404 / empty).
        failed: Mapping of task key -> last error message.
    """
    payload = {
        "completed": sorted(completed),
        "skipped_empty": sorted(skipped_empty),
        "failed": failed,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".download_state_", suffix=".json", dir=STATE_FILE.parent
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, STATE_FILE)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Catalog and task list
# ---------------------------------------------------------------------------

def fetch_item_catalog() -> list[dict[str, Any]]:
    """Fetch item catalog from EMDB and filter for air dose rate items.

    Returns:
        List of item dicts where content_code == CONTENT_CODE.

    Raises:
        requests.HTTPError: If the catalog request fails.
        ValueError: If the response JSON is missing the 'data' key.
    """
    logger.info("Fetching item catalog from %s", ITEM_INFO_URL)
    response = requests.get(
        ITEM_INFO_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()
    if "data" not in payload:
        raise ValueError(f"Unexpected catalog response — missing 'data' key: {list(payload.keys())}")

    all_items: list[dict[str, Any]] = payload["data"]
    filtered = [item for item in all_items if item.get("content_code") == CONTENT_CODE]
    logger.info(
        "Catalog contains %d total items; %d match content_code=%d (air dose rate).",
        len(all_items),
        len(filtered),
        CONTENT_CODE,
    )
    return filtered


def build_task_list(
    items: list[dict[str, Any]],
    year_start: int,
    year_end: int,
) -> list[tuple[int, int, int, Path]]:
    """Build the full list of download tasks.

    Each task is a (item_code, class_code, fiscal_year, output_path) tuple.
    The output path follows the pattern:
        dataset/{class_code_name}/item_{item_code:03d}/{year}.zip

    Args:
        items: Filtered item list from fetch_item_catalog().
        year_start: First fiscal year to include (inclusive).
        year_end: Last fiscal year to include (inclusive).

    Returns:
        Ordered list of (item_code, class_code, fiscal_year, output_path).
    """
    tasks: list[tuple[int, int, int, Path]] = []
    for item in items:
        item_code: int = item["item_code"]
        class_code: int = item["class_code"]
        class_dir = CLASS_CODE_NAMES.get(class_code, f"{class_code:02d}_unknown")
        item_dir = DATASET_DIR / class_dir / f"item_{item_code:03d}"
        for year in range(year_start, year_end + 1):
            output_path = item_dir / f"{year}.zip"
            tasks.append((item_code, class_code, year, output_path))
    return tasks


# ---------------------------------------------------------------------------
# Single download
# ---------------------------------------------------------------------------

def download_one(
    item_code: int,
    fiscal_year: int,
    output_path: Path,
    session: requests.Session,
    delay: float = DELAY_BETWEEN_REQUESTS,
) -> str:
    """Download a single ZIP archive with retry logic.

    Streams the response body, validates the ZIP magic bytes, and writes
    to disk only if the file is valid. Skips files that already exist.

    Args:
        item_code: EMDB item identifier.
        fiscal_year: The fiscal year to download.
        output_path: Destination path for the ZIP file.
        session: Shared requests.Session for connection pooling.
        delay: Seconds to wait after a successful download.

    Returns:
        One of: "completed", "skipped", or "failed".
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.debug("Already exists, skipping: %s", output_path)
        return "skipped"

    url = DOWNLOAD_URL_TEMPLATE.format(
        item_code=item_code,
        fiscal_year=fiscal_year,
        char_code=CHAR_CODE,
        lang_code=LANG_CODE,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)

            # Handle rate limiting
            if response.status_code == 429:
                wait = 60
                logger.warning("HTTP 429 — rate limited; waiting %ds before retry.", wait)
                time.sleep(wait)
                continue

            # No data available for this item/year combination
            if response.status_code == 404:
                logger.debug("HTTP 404 — no data: item %03d year %d", item_code, fiscal_year)
                return "skipped"

            # Check Content-Length header for empty responses
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) == 0:
                logger.debug("Content-Length=0 — skipping: item %03d year %d", item_code, fiscal_year)
                return "skipped"

            response.raise_for_status()

            # Stream into a temp file, then validate before moving to final path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                prefix=".tmp_", suffix=".zip", dir=output_path.parent
            )
            tmp_path = Path(tmp_path_str)
            try:
                with os.fdopen(tmp_fd, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)

                # Validate ZIP magic bytes
                if tmp_path.stat().st_size == 0:
                    logger.debug("Empty body — skipping: item %03d year %d", item_code, fiscal_year)
                    tmp_path.unlink(missing_ok=True)
                    return "skipped"

                with tmp_path.open("rb") as fh:
                    header = fh.read(4)

                if header != ZIP_MAGIC:
                    logger.debug(
                        "No data for item %03d year %d (got %r instead of ZIP) — skipping.",
                        item_code,
                        fiscal_year,
                        header,
                    )
                    tmp_path.unlink(missing_ok=True)
                    return "skipped"

                os.replace(tmp_path, output_path)

            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

            time.sleep(delay)
            return "completed"

        except requests.exceptions.Timeout as exc:
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "Timeout on attempt %d/%d for item %03d year %d — retrying in %ds.",
                attempt, MAX_RETRIES, item_code, fiscal_year, backoff,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
            else:
                logger.error("All retries exhausted for item %03d year %d: %s", item_code, fiscal_year, exc)
                return "failed"

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if isinstance(status, int) and 500 <= status < 600:
                backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "HTTP %s on attempt %d/%d for item %03d year %d — retrying in %ds.",
                    status, attempt, MAX_RETRIES, item_code, fiscal_year, backoff,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                else:
                    logger.error("All retries exhausted for item %03d year %d: %s", item_code, fiscal_year, exc)
                    return "failed"
            else:
                logger.error("HTTP %s for item %03d year %d — not retrying.", status, item_code, fiscal_year)
                return "failed"

        except requests.exceptions.RequestException as exc:
            logger.error("Request error for item %03d year %d: %s", item_code, fiscal_year, exc)
            return "failed"

    return "failed"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    completed: set[str],
    skipped_empty: set[str],
    failed: dict[str, str],
    items: list[dict[str, Any]],
) -> None:
    """Write dataset/manifest.json summarising the completed download run.

    Args:
        completed: Set of completed task keys.
        skipped_empty: Set of skipped (empty/404) task keys.
        failed: Mapping of failed task keys to error messages.
        items: The filtered item list used for this run.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "JAEA EMDB",
        "url": "https://radioactivity.nra.go.jp/emdb/",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "encoding": "UTF-8",
        "content_code": CONTENT_CODE,
        "classification": "Air dose rate",
        "items_downloaded": len(items),
        "files_completed": len(completed),
        "files_skipped_empty": len(skipped_empty),
        "files_failed": len(failed),
    }
    manifest_path = DATASET_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Manifest written to %s", manifest_path)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Main download orchestrator.

    Fetches the item catalog, builds the task list, loads prior state,
    then iterates over tasks — skipping already-completed ones — and
    saves state periodically and on exit.

    Args:
        args: Parsed CLI arguments (dry_run, year_start, year_end, delay).
    """
    items = fetch_item_catalog()
    tasks = build_task_list(items, args.year_start, args.year_end)
    total = len(tasks)
    logger.info("Task list built: %d total downloads.", total)

    if args.dry_run:
        print(f"\n--- DRY RUN: {total} tasks ---")
        for item_code, class_code, fiscal_year, output_path in tasks:
            class_dir = CLASS_CODE_NAMES.get(class_code, f"{class_code:02d}_unknown")
            print(f"  item={item_code:03d}  year={fiscal_year}  -> {output_path}")
        print("--- End dry run ---\n")
        return

    state = load_state()
    completed: set[str] = state["completed"]
    skipped_empty: set[str] = state["skipped_empty"]
    failed: dict[str, str] = state["failed"]

    # Signal handler — save state and exit gracefully
    def _handle_signal(signum: int, frame: Any) -> None:
        logger.info("Signal %d received — saving state and exiting.", signum)
        save_state(completed, skipped_empty, failed)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    save_interval = 50  # persist state every N tasks
    processed = 0

    try:
        for idx, (item_code, class_code, fiscal_year, output_path) in enumerate(tasks, start=1):
            task_key = f"{item_code}_{fiscal_year}"
            class_dir = CLASS_CODE_NAMES.get(class_code, f"{class_code:02d}_unknown")

            # Skip tasks already resolved in a prior run
            if task_key in completed:
                logger.debug("[%d/%d] Already completed: %s", idx, total, task_key)
                continue
            if task_key in skipped_empty:
                logger.debug("[%d/%d] Previously skipped: %s", idx, total, task_key)
                continue

            logger.info(
                "[%d/%d] Downloading item %03d, year %d -> %s/item_%03d/%d.zip",
                idx, total, item_code, fiscal_year, class_dir, item_code, fiscal_year,
            )

            result = download_one(item_code, fiscal_year, output_path, session, delay=args.delay)

            if result == "completed":
                completed.add(task_key)
                failed.pop(task_key, None)
            elif result == "skipped":
                skipped_empty.add(task_key)
                failed.pop(task_key, None)
            else:
                failed[task_key] = f"Failed after {MAX_RETRIES} retries"

            processed += 1
            if processed % save_interval == 0:
                save_state(completed, skipped_empty, failed)
                logger.debug("State checkpoint saved (%d processed).", processed)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — saving state before exit.")
        save_state(completed, skipped_empty, failed)
        sys.exit(0)

    # Final state save
    save_state(completed, skipped_empty, failed)

    # Summary
    logger.info(
        "Download complete — %d completed, %d skipped (empty/404), %d failed.",
        len(completed),
        len(skipped_empty),
        len(failed),
    )
    if failed:
        logger.warning("Failed tasks: %s", ", ".join(sorted(failed.keys())))

    write_manifest(completed, skipped_empty, failed, items)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with attributes: dry_run, year_start, year_end, delay.
    """
    parser = argparse.ArgumentParser(
        description="Download JAEA EMDB air dose rate datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task list without downloading anything.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=YEAR_START,
        metavar="YEAR",
        help="First fiscal year to download (inclusive).",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=YEAR_END,
        metavar="YEAR",
        help="Last fiscal year to download (inclusive).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_BETWEEN_REQUESTS,
        metavar="SECONDS",
        help="Seconds to wait between successful requests (be polite).",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    setup_logging()
    args = parse_args()

    if args.year_start > args.year_end:
        logger.error(
            "--year-start (%d) must be <= --year-end (%d).",
            args.year_start,
            args.year_end,
        )
        sys.exit(1)

    logger.info(
        "EMDB downloader starting — years %d–%d, delay=%.1fs, dry_run=%s",
        args.year_start,
        args.year_end,
        args.delay,
        args.dry_run,
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
