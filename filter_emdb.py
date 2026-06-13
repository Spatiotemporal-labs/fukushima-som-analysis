"""
filter_emdb.py — Data filtration pipeline for JAEA EMDB air dose rate data.

Reads all ZIP archives from the dataset directory, applies six sequential
filter stages, and writes a SOM-ready Parquet file plus a JSON summary.

Usage:
    python filter_emdb.py [--dry-run] [--sensitivity] [--csv] [--verbose]
    python filter_emdb.py --dataset-dir dataset/ --output-dir output/
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FDNPP_LAT: float = 37.421389
FDNPP_LON: float = 141.032778
EARTH_RADIUS_KM: float = 6371.0

ENCODING_TRIAL_SEQUENCE: list[str] = ["utf-8-sig", "utf-8", "shift_jis", "cp932"]

EXPECTED_COLUMN_COUNT: int = 37

# Positional canonical column names (index → name).
# Indices 8-10, 16, 18-24, 26-28, 31, 33-35 are read in but not written to Parquet.
CANONICAL_COLUMNS: list[str] = [
    "category",               # 0
    "measurement_type",       # 1
    "organization",           # 2
    "mesh_code",              # 3
    "latitude",               # 4
    "longitude",              # 5
    "prefecture",             # 6
    "city",                   # 7
    "point_id",               # 8  — dropped at write
    "sampling_start_date",    # 9  — dropped at write
    "sampling_finish_date",   # 10 — dropped at write
    "correction_date",        # 11
    "value",                  # 12
    "statistical_error",      # 13
    "detection_lower_limit",  # 14
    "unit",                   # 15
    "sample_mass_kg",         # 16 — dropped at write
    "measurement_height_cm",  # 17
    "lower_depth_cm",         # 18 — dropped at write
    "_unused_19",             # 19 — dropped at write
    "_unused_20",             # 20 — dropped at write
    "_unused_21",             # 21 — dropped at write
    "_unused_22",             # 22 — dropped at write
    "_unused_23",             # 23 — dropped at write
    "_unused_24",             # 24 — dropped at write
    "sampling_point_state",   # 25
    "_unused_26",             # 26 — dropped at write
    "_unused_27",             # 27 — dropped at write
    "_unused_28",             # 28 — dropped at write
    "note",                   # 29
    "project_id",             # 30
    "mesh_id_basic",          # 31 — dropped at write
    "mesh_id_250m",           # 32
    "mesh_id_100m",           # 33 — dropped at write
    "mesh_id_50m",            # 34 — dropped at write
    "mesh_id_20m",            # 35 — dropped at write
    "distance_from_fdnpp_km", # 36
]

# Columns to drop before writing Parquet (indices listed in spec section 1.3)
COLUMNS_DROP_BEFORE_WRITE: set[str] = {
    "point_id",
    "sampling_start_date",
    "sampling_finish_date",
    "sample_mass_kg",
    "lower_depth_cm",
    "_unused_19",
    "_unused_20",
    "_unused_21",
    "_unused_22",
    "_unused_23",
    "_unused_24",
    "_unused_26",
    "_unused_27",
    "_unused_28",
    "mesh_id_basic",
    "mesh_id_100m",
    "mesh_id_50m",
    "mesh_id_20m",
}

# Ordered column list for Parquet output (spec section 2.1)
PARQUET_COLUMN_ORDER: list[str] = [
    "category",
    "measurement_type",
    "organization",
    "mesh_code",
    "latitude",
    "longitude",
    "prefecture",
    "city",
    "correction_date",
    "value",
    "statistical_error",
    "detection_lower_limit",
    "unit",
    "measurement_height_cm",
    "sampling_point_state",
    "note",
    "project_id",
    "mesh_id_250m",
    "distance_from_fdnpp_km",
    "year",
    "is_decontaminated",
    "height_anomalous",
    "source_item",
    "source_class",
]

# Measurement-type priority for tie-breaking in Stage 5
MEASUREMENT_TYPE_PRIORITY: dict[str, int] = {
    "Walk": 0,
    "Car": 1,
    "AirBorne": 2,
    "定点": 3,
}

NUMERIC_COERCE_COLUMNS: list[str] = [
    "latitude",
    "longitude",
    "value",
    "statistical_error",
    "detection_lower_limit",
    "measurement_height_cm",
    "distance_from_fdnpp_km",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("filter_emdb")


def setup_logging(verbose: bool = False, log_file: str = "filter.log") -> None:
    """Configure dual-handler logging: INFO to console, DEBUG to file.

    Args:
        verbose: When True, the console handler is set to DEBUG level.
        log_file: Path of the log file to write DEBUG output to.
    """
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: float = FDNPP_LAT,
    lon2: float = FDNPP_LON,
) -> np.ndarray | float:
    """Compute Haversine great-circle distance from (lat1, lon1) to (lat2, lon2).

    Args:
        lat1: Origin latitude(s) in decimal degrees.
        lon1: Origin longitude(s) in decimal degrees.
        lat2: Destination latitude in decimal degrees (FDNPP default).
        lon2: Destination longitude in decimal degrees (FDNPP default).

    Returns:
        Distance(s) in kilometres using Earth radius 6371.0 km.
    """
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _detect_encoding_and_read(raw_bytes: bytes, csv_name: str) -> tuple[str, list[str]]:
    """Try encodings in order and return (encoding_used, lines).

    Args:
        raw_bytes: Raw bytes of the CSV file.
        csv_name: Name used in log messages.

    Returns:
        Tuple of (encoding_name, list_of_text_lines).

    Raises:
        ValueError: If no encoding in the trial sequence succeeds.
    """
    for enc in ENCODING_TRIAL_SEQUENCE:
        try:
            text = raw_bytes.decode(enc)
            lines = text.splitlines()
            # Validate: find a non-empty line and check it is parseable
            for line in lines:
                if line.strip():
                    # Just test the first non-empty line decodes without replacement chars
                    _ = line.encode("utf-8")
                    break
            logger.debug("Encoding '%s' succeeded for %s", enc, csv_name)
            return enc, lines
        except (UnicodeDecodeError, UnicodeEncodeError):
            logger.debug("Encoding '%s' failed for %s", enc, csv_name)
            continue
    raise ValueError(f"No encoding succeeded for {csv_name}")


def _read_csv_from_bytes(
    raw_bytes: bytes,
    csv_name: str,
    item_dir_name: str,
    class_dir_name: str,
    distance_cutoff: float,
    stage0_counters: dict[str, int],
) -> pd.DataFrame | None:
    """Parse one CSV from raw bytes and apply pre-processing.

    Applies positional column renaming, numeric coercion, date parsing,
    Haversine fallback, computed columns, and Stage-1 distance filter inline.

    Args:
        raw_bytes: Raw content of the CSV file (binary).
        csv_name: Member name within the ZIP (for log messages).
        item_dir_name: e.g. ``item_090`` — injected as source_item.
        class_dir_name: e.g. ``03_walk_survey`` — injected as source_class.
        distance_cutoff: Maximum distance (km) to retain a row.
        stage0_counters: Mutable dict accumulating pre-processing counts.

    Returns:
        Processed DataFrame with distance filter already applied, or None
        if the file should be skipped entirely (wrong columns, undated, etc.).
    """
    # --- Encoding detection ---
    try:
        encoding_used, _lines = _detect_encoding_and_read(raw_bytes, csv_name)
    except ValueError as exc:
        logger.error("Encoding detection failed for %s: %s", csv_name, exc)
        stage0_counters["wrong_column_count"] += 1
        return None

    if encoding_used != ENCODING_TRIAL_SEQUENCE[0]:
        logger.warning(
            "Encoding fallback used for %s: tried %s, succeeded with %s",
            csv_name,
            ENCODING_TRIAL_SEQUENCE[0],
            encoding_used,
        )

    # --- Parse CSV ---
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            header=0,
            dtype=str,
            encoding=encoding_used,
            na_values=["", " "],
            keep_default_na=False,
            low_memory=False,
        )
    except Exception as exc:
        logger.error("CSV parse error for %s: %s", csv_name, exc)
        stage0_counters["wrong_column_count"] += 1
        return None

    # --- Column count check ---
    if len(df.columns) != EXPECTED_COLUMN_COUNT:
        logger.error(
            "Wrong column count in %s: expected %d, got %d (first header: %r)",
            csv_name,
            EXPECTED_COLUMN_COUNT,
            len(df.columns),
            df.columns[0] if len(df.columns) > 0 else "<empty>",
        )
        stage0_counters["wrong_column_count"] += 1
        return None

    # --- Empty CSV (no data rows) ---
    if df.empty:
        logger.warning("Empty CSV (no data rows): %s", csv_name)
        stage0_counters["empty_csv"] += 1
        return None

    # --- Positional rename ---
    df.columns = CANONICAL_COLUMNS

    # --- Entirely-undated file check (col 11) ---
    date_col_raw = df["correction_date"].dropna()
    if date_col_raw.empty:
        logger.warning("All correction_date values are empty in %s — skipping file", csv_name)
        stage0_counters["undated_files"] += 1
        return None

    # --- Numeric coercion ---
    for col in NUMERIC_COERCE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- mesh_id_250m normalisation ---
    df["mesh_id_250m"] = df["mesh_id_250m"].str.strip().replace("", pd.NA)

    # --- Category pre-filter ---
    non_air = df["category"].notna() & ~df["category"].str.contains(
        "空間線量率|Spatial dose rate", na=False
    )
    n_non_air = non_air.sum()
    if n_non_air > 0:
        logger.info(
            "Dropped %d non-air-dose rows from %s (item %s)",
            n_non_air,
            csv_name,
            item_dir_name,
        )
        stage0_counters["non_air_dose"] += int(n_non_air)
        df = df[~non_air].copy()

    if df.empty:
        return None

    # --- Date parsing ---
    try:
        df["correction_date"] = pd.to_datetime(
            df["correction_date"], format="mixed", dayfirst=False
        )
    except Exception:
        # Fallback: coerce row by row — failures become NaT
        df["correction_date"] = pd.to_datetime(
            df["correction_date"], errors="coerce", dayfirst=False
        )

    n_parse_fail = df["correction_date"].isna().sum()
    if n_parse_fail > 0:
        logger.warning(
            "%d rows dropped — correction_date parse failure in %s (item %s)",
            n_parse_fail,
            csv_name,
            item_dir_name,
        )
        stage0_counters["parse_failures"] += int(n_parse_fail)
        df = df[df["correction_date"].notna()].copy()

    if df.empty:
        return None

    # --- Missing coordinates → drop before Haversine ---
    no_coords = (
        df["latitude"].isna() | df["longitude"].isna()
    ) & df["distance_from_fdnpp_km"].isna()
    n_missing_coords = no_coords.sum()
    if n_missing_coords > 0:
        logger.warning(
            "%d rows dropped — no coords and no distance in %s (item %s)",
            n_missing_coords,
            csv_name,
            item_dir_name,
        )
        stage0_counters["missing_coords"] += int(n_missing_coords)
        df = df[~no_coords].copy()

    if df.empty:
        return None

    # --- Haversine fallback ---
    needs_haversine = df["distance_from_fdnpp_km"].isna()
    n_haversine = needs_haversine.sum()
    if n_haversine > 0:
        df.loc[needs_haversine, "distance_from_fdnpp_km"] = haversine_km(
            df.loc[needs_haversine, "latitude"].to_numpy(),
            df.loc[needs_haversine, "longitude"].to_numpy(),
        )
        logger.warning(
            "Haversine fallback used for %d rows in %s / %s",
            n_haversine,
            item_dir_name,
            csv_name,
        )
        stage0_counters["haversine_fallback"] += int(n_haversine)

    # --- Missing mesh_id_250m drop ---
    n_missing_mesh = df["mesh_id_250m"].isna().sum()
    if n_missing_mesh > 0:
        stage0_counters["missing_mesh_id"] += int(n_missing_mesh)
        df = df[df["mesh_id_250m"].notna()].copy()

    if df.empty:
        return None

    # --- Computed columns ---
    df["year"] = df["correction_date"].dt.year.astype("Int16")
    df["source_item"] = item_dir_name
    df["source_class"] = class_dir_name

    df["height_anomalous"] = (
        df["measurement_height_cm"].notna()
        & (df["measurement_height_cm"] != 100.0)
    )

    # Decontamination flag — handle item 335 mojibake gracefully
    # For item 335 string matching may be unreliable; we still attempt it
    # and fall back to False per spec section 1.2 Rule 3.
    try:
        is_decont = (
            df["sampling_point_state"].str.contains(
                "除染|decontamination", case=False, na=False
            )
            | df["note"].str.contains(
                "除染|decontamination", case=False, na=False
            )
        )
    except Exception:
        logger.warning(
            "is_decontaminated flag unreliable for item %s (mojibake) — defaulting to False",
            item_dir_name,
        )
        is_decont = pd.Series(False, index=df.index)

    df["is_decontaminated"] = is_decont

    # --- Stage 1: distance filter (applied per-chunk to control memory) ---
    df = df[df["distance_from_fdnpp_km"] <= distance_cutoff].copy()

    logger.debug(
        "Loaded %s/%s: %d rows after distance filter (≤ %.1f km)",
        item_dir_name,
        csv_name,
        len(df),
        distance_cutoff,
    )

    return df


# ---------------------------------------------------------------------------
# ZIP loading
# ---------------------------------------------------------------------------

def load_zip(
    zip_path: Path,
    item_dir_name: str,
    class_dir_name: str,
    distance_cutoff: float,
    stage0_counters: dict[str, int],
    per_file_counters: dict[str, dict[str, int]],
    skipped_zips: list[str],
    corrupted_zips: list[str],
) -> list[pd.DataFrame]:
    """Open one ZIP archive and return list of processed DataFrames (one per CSV).

    Args:
        zip_path: Filesystem path to the ZIP file.
        item_dir_name: Source item label injected into rows.
        class_dir_name: Source class label injected into rows.
        distance_cutoff: km cutoff for Stage 1 inline filter.
        stage0_counters: Mutable pre-processing counter dict.
        per_file_counters: Mutable per-file counter dict (keyed by csv_name).
        skipped_zips: List to append skipped ZIP paths to.
        corrupted_zips: List to append corrupted ZIP paths to.

    Returns:
        List of DataFrames (may be empty if ZIP was empty or all files failed).
    """
    try:
        zf_handle = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        logger.error("Corrupted ZIP (BadZipFile): %s", zip_path)
        corrupted_zips.append(str(zip_path))
        return []
    except OSError as exc:
        logger.error("Cannot open ZIP %s: %s", zip_path, exc)
        corrupted_zips.append(str(zip_path))
        return []

    with zf_handle as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            logger.warning("No CSV files in %s — skipped", zip_path)
            skipped_zips.append(str(zip_path))
            return []

        stage0_counters["csv_files_found"] += len(csv_names)
        results: list[pd.DataFrame] = []

        for csv_name in csv_names:
            raw_bytes = zf.read(csv_name)
            n_before = stage0_counters.get("_tmp_raw", 0)

            df = _read_csv_from_bytes(
                raw_bytes,
                csv_name,
                item_dir_name,
                class_dir_name,
                distance_cutoff,
                stage0_counters,
            )

            if df is not None and not df.empty:
                per_file_counters[csv_name] = {"rows": len(df)}
                results.append(df)

    return results


# ---------------------------------------------------------------------------
# Filter stages
# ---------------------------------------------------------------------------

def stage_2_detection_limit(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove rows at or below detection limit and rows with NaN value.

    Args:
        df: Combined DataFrame after Stage 1.

    Returns:
        Tuple of (filtered_df, n_dropped_below_limit, n_dropped_nan_value).
    """
    # NaN value rows
    nan_mask = df["value"].isna()
    n_nan_value = int(nan_mask.sum())
    df = df[~nan_mask].copy()

    # Below-or-at detection limit
    below_mask = (
        df["detection_lower_limit"].notna()
        & (df["value"] <= df["detection_lower_limit"])
    )
    n_below = int(below_mask.sum())
    df = df[~below_mask].copy()

    return df, n_below, n_nan_value


def stage_3_dose_threshold(
    df: pd.DataFrame, dose_threshold: float
) -> tuple[pd.DataFrame, int, int]:
    """Remove mesh cells whose earliest-date mean value is at or below threshold.

    Args:
        df: DataFrame after Stage 2.
        dose_threshold: Minimum initial dose rate in µSv/h (exclusive bound).

    Returns:
        Tuple of (filtered_df, n_meshes_dropped, n_meshes_in).
    """
    meshes_in = df["mesh_id_250m"].nunique()

    # Earliest date per mesh
    earliest_date_per_mesh = df.groupby("mesh_id_250m")["correction_date"].min()
    earliest_date_per_mesh.name = "earliest_date"

    # Rows at the earliest date; mean value per mesh
    earliest_values = (
        df.merge(earliest_date_per_mesh, on="mesh_id_250m")
        .query("correction_date == earliest_date")
        .groupby("mesh_id_250m")["value"]
        .mean()
    )

    passing_meshes = earliest_values[earliest_values > dose_threshold].index
    df = df[df["mesh_id_250m"].isin(passing_meshes)].copy()

    meshes_out = df["mesh_id_250m"].nunique()
    meshes_dropped = meshes_in - meshes_out
    return df, meshes_dropped, meshes_in


def stage_4_temporal_completeness(
    df: pd.DataFrame, min_temporal_points: int
) -> tuple[pd.DataFrame, int, int]:
    """Remove mesh cells with fewer than min_temporal_points distinct dates.

    Args:
        df: DataFrame after Stage 3.
        min_temporal_points: Minimum number of distinct correction_dates required.

    Returns:
        Tuple of (filtered_df, n_meshes_dropped, n_meshes_in).
    """
    meshes_in = df["mesh_id_250m"].nunique()

    date_counts = df.groupby("mesh_id_250m")["correction_date"].nunique()
    qualifying_meshes = date_counts[date_counts >= min_temporal_points].index
    df = df[df["mesh_id_250m"].isin(qualifying_meshes)].copy()

    meshes_out = df["mesh_id_250m"].nunique()
    meshes_dropped = meshes_in - meshes_out
    return df, meshes_dropped, meshes_in


def _resolve_measurement_type(group: pd.DataFrame) -> str:
    """Determine the winning measurement_type for a mixed-type mesh group.

    Args:
        group: Subset of rows belonging to one mesh_id_250m.

    Returns:
        Winning measurement_type string.
    """
    counts = group["measurement_type"].value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda t: MEASUREMENT_TYPE_PRIORITY.get(t, 99))


def stage_5_type_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Enforce a single measurement_type per mesh_id_250m.

    Logs each resolved conflict at WARNING level.

    Args:
        df: DataFrame after Stage 4.

    Returns:
        Tuple of (filtered_df, n_mixed_meshes, n_rows_dropped).
    """
    rows_in = len(df)

    type_counts_per_mesh = df.groupby("mesh_id_250m")["measurement_type"].nunique()
    mixed_meshes = type_counts_per_mesh[type_counts_per_mesh > 1].index

    n_mixed = len(mixed_meshes)

    if n_mixed > 0:
        # Log each mixed mesh
        mixed_subset = df[df["mesh_id_250m"].isin(mixed_meshes)]
        for mesh_id, grp in mixed_subset.groupby("mesh_id_250m"):
            type_row_counts = grp["measurement_type"].value_counts().to_dict()
            winner = _resolve_measurement_type(grp)
            dropped_types = {t: c for t, c in type_row_counts.items() if t != winner}
            logger.warning(
                "Mixed types in mesh %s: %s — winner=%s — dropping %s",
                mesh_id,
                type_row_counts,
                winner,
                dropped_types,
            )

    # Build winning-type map for ALL meshes (not just mixed; no-ops for pure meshes)
    winning_type: pd.Series = (
        df.groupby("mesh_id_250m", group_keys=False)
        .apply(_resolve_measurement_type)
    )

    df = df[df["measurement_type"] == df["mesh_id_250m"].map(winning_type)].copy()

    rows_dropped = rows_in - len(df)
    return df, n_mixed, rows_dropped


def stage_6_height_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Verify and recompute height_anomalous flag (non-destructive).

    Args:
        df: DataFrame after Stage 5.

    Returns:
        Same DataFrame with height_anomalous refreshed.
    """
    df["height_anomalous"] = (
        df["measurement_height_cm"].notna()
        & (df["measurement_height_cm"] != 100.0)
    )

    n_anomalous = int(df["height_anomalous"].sum())
    breakdown = (
        df[df["height_anomalous"]]
        .groupby("source_item")
        .size()
        .to_dict()
    )
    logger.info(
        "Stage 6 (height flag): anomalous=%d  (non-destructive)  breakdown=%s",
        n_anomalous,
        breakdown,
    )
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows based on key identity columns.

    Args:
        df: Combined DataFrame before Stage 1 group operations.

    Returns:
        Tuple of (deduplicated_df, n_dropped).
    """
    n_before = len(df)
    df = df.drop_duplicates(
        subset=[
            "mesh_id_250m",
            "correction_date",
            "value",
            "measurement_type",
            "source_item",
        ]
    ).copy()
    return df, n_before - len(df)


# ---------------------------------------------------------------------------
# Parquet output
# ---------------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the filtered DataFrame to a Parquet file.

    Applies the column order, type casts, and write settings from spec section 2.

    Args:
        df: Final filtered DataFrame.
        output_path: Destination .parquet file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop internal-only columns
    for col in COLUMNS_DROP_BEFORE_WRITE:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Reorder to Parquet spec
    df = df[[c for c in PARQUET_COLUMN_ORDER if c in df.columns]]

    # Convert correction_date from datetime64 → Python date objects for Arrow date32
    if pd.api.types.is_datetime64_any_dtype(df["correction_date"]):
        df = df.copy()
        df["correction_date"] = df["correction_date"].dt.date

    # Build PyArrow schema with required types
    dict_encoded_string = pa.dictionary(pa.int32(), pa.string())

    schema = pa.schema([
        pa.field("category",              dict_encoded_string,   nullable=False),
        pa.field("measurement_type",      dict_encoded_string,   nullable=False),
        pa.field("organization",          dict_encoded_string,   nullable=True),
        pa.field("mesh_code",             pa.string(),           nullable=True),
        pa.field("latitude",              pa.float32(),          nullable=False),
        pa.field("longitude",             pa.float32(),          nullable=False),
        pa.field("prefecture",            dict_encoded_string,   nullable=True),
        pa.field("city",                  pa.string(),           nullable=True),
        pa.field("correction_date",       pa.date32(),           nullable=False),
        pa.field("value",                 pa.float32(),          nullable=False),
        pa.field("statistical_error",     pa.float32(),          nullable=True),
        pa.field("detection_lower_limit", pa.float32(),          nullable=True),
        pa.field("unit",                  dict_encoded_string,   nullable=True),
        pa.field("measurement_height_cm", pa.float32(),          nullable=True),
        pa.field("sampling_point_state",  pa.string(),           nullable=True),
        pa.field("note",                  pa.string(),           nullable=True),
        pa.field("project_id",            dict_encoded_string,   nullable=True),
        pa.field("mesh_id_250m",          pa.string(),           nullable=False),
        pa.field("distance_from_fdnpp_km",pa.float32(),          nullable=False),
        pa.field("year",                  pa.int16(),            nullable=False),
        pa.field("is_decontaminated",     pa.bool_(),            nullable=False),
        pa.field("height_anomalous",      pa.bool_(),            nullable=False),
        pa.field("source_item",           dict_encoded_string,   nullable=False),
        pa.field("source_class",          dict_encoded_string,   nullable=False),
    ])

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    pq.write_table(
        table,
        output_path,
        compression="snappy",
        row_group_size=500_000,
    )
    logger.info("Parquet written to %s (%d rows)", output_path, len(df))


# ---------------------------------------------------------------------------
# filter_summary.json writer
# ---------------------------------------------------------------------------

def build_summary(
    params: dict[str, Any],
    file_disc: dict[str, Any],
    stage0: dict[str, Any],
    s1: dict[str, Any],
    s2: dict[str, Any],
    s3: dict[str, Any],
    s4: dict[str, Any],
    s5: dict[str, Any],
    s6: dict[str, Any],
    final_df: pd.DataFrame,
    per_item: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble filter_summary.json payload from accumulated statistics.

    Args:
        params: Pipeline parameters.
        file_disc: File discovery counters.
        stage0: Pre-processing counters.
        s1–s6: Per-stage counters.
        final_df: The final output DataFrame.
        per_item: Per-item summary dicts.

    Returns:
        Dict matching the schema from spec section 7.
    """
    year_vals = final_df["year"].dropna()
    dist_vals = final_df["distance_from_fdnpp_km"].dropna()
    value_vals = final_df["value"].dropna()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "file_discovery": file_disc,
        "stage_0_preprocessing": stage0,
        "stage_1_distance": s1,
        "stage_2_detection_limit": s2,
        "stage_3_dose_threshold": s3,
        "stage_4_temporal_completeness": s4,
        "stage_5_type_consistency": s5,
        "stage_6_height_flag": s6,
        "final_output": {
            "total_rows": len(final_df),
            "total_meshes": int(final_df["mesh_id_250m"].nunique()),
            "total_items": int(final_df["source_item"].nunique()),
            "year_range": [
                int(year_vals.min()) if not year_vals.empty else 0,
                int(year_vals.max()) if not year_vals.empty else 0,
            ],
            "distance_range_km": [
                float(dist_vals.min()) if not dist_vals.empty else 0.0,
                float(dist_vals.max()) if not dist_vals.empty else 0.0,
            ],
            "value_range_usvh": [
                float(value_vals.min()) if not value_vals.empty else 0.0,
                float(value_vals.max()) if not value_vals.empty else 0.0,
            ],
            "is_decontaminated_true": int(final_df["is_decontaminated"].sum()),
            "height_anomalous_true": int(final_df["height_anomalous"].sum()),
            "measurement_type_counts": (
                final_df["measurement_type"].value_counts().to_dict()
            ),
            "organization_counts": (
                final_df["organization"].value_counts().to_dict()
            ),
            "source_class_counts": (
                final_df["source_class"].value_counts().to_dict()
            ),
        },
        "per_item_breakdown": per_item,
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Write filter_summary.json to disk.

    Args:
        summary: Dict returned by build_summary().
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Filter summary written to %s", output_path)


# ---------------------------------------------------------------------------
# Sensitivity analysis comparison
# ---------------------------------------------------------------------------

def compute_sensitivity_comparison(
    df_010: pd.DataFrame,
    df_015: pd.DataFrame,
) -> dict[str, Any]:
    """Build sensitivity_comparison.json payload.

    Args:
        df_010: Final filtered DataFrame at threshold 0.10.
        df_015: Final filtered DataFrame at threshold 0.15.

    Returns:
        Comparison dict matching the schema from spec section 6.2.
    """
    meshes_010 = set(df_010["mesh_id_250m"].unique())
    meshes_015 = set(df_015["mesh_id_250m"].unique())

    both = meshes_010 & meshes_015
    only_010 = meshes_010 - meshes_015
    only_015 = meshes_015 - meshes_010

    if only_015:
        logger.warning(
            "Unexpected: %d meshes retained by T015 only (stricter threshold "
            "should never add meshes). Values: %s",
            len(only_015),
            list(only_015)[:10],
        )

    # Per-class row counts
    all_classes = sorted(
        set(df_010["source_class"].unique()) | set(df_015["source_class"].unique())
    )
    per_class: dict[str, dict[str, int]] = {}
    for cls in all_classes:
        per_class[cls] = {
            "T010": int((df_010["source_class"] == cls).sum()),
            "T015": int((df_015["source_class"] == cls).sum()),
        }

    # Distance-band retention
    bands = [(0, 20), (20, 40), (40, 60), (60, 80)]
    band_retention: dict[str, dict[str, int]] = {}
    for lo, hi in bands:
        key = f"{lo}-{hi}km"
        mask_010 = (df_010["distance_from_fdnpp_km"] >= lo) & (
            df_010["distance_from_fdnpp_km"] < hi
        )
        mask_015 = (df_015["distance_from_fdnpp_km"] >= lo) & (
            df_015["distance_from_fdnpp_km"] < hi
        )
        band_retention[key] = {
            "T010": int(mask_010.sum()),
            "T015": int(mask_015.sum()),
        }

    year_010 = df_010["year"].dropna()
    year_015 = df_015["year"].dropna()

    n_010 = len(df_010)
    n_015 = len(df_015)

    # Row overlap: rows in both sets identified by (mesh_id_250m, correction_date, value)
    key_cols = ["mesh_id_250m", "correction_date", "value"]
    merged = pd.merge(
        df_010[key_cols].drop_duplicates(),
        df_015[key_cols].drop_duplicates(),
        how="inner",
        on=key_cols,
    )
    row_overlap_fraction = len(merged) / max(n_010, 1)
    mesh_overlap_fraction = len(both) / max(len(meshes_010), 1)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_compared": [0.10, 0.15],
        "comparison": {
            "total_rows": {"T010": n_010, "T015": n_015},
            "total_meshes": {"T010": len(meshes_010), "T015": len(meshes_015)},
            "meshes_retained_by_T010_only": len(only_010),
            "meshes_retained_by_T015_only": len(only_015),
            "meshes_retained_by_both": len(both),
            "row_overlap_fraction": round(row_overlap_fraction, 4),
            "mesh_overlap_fraction": round(mesh_overlap_fraction, 4),
            "per_class_rows": per_class,
            "distance_band_retention": band_retention,
            "year_range": {
                "min_year": {
                    "T010": int(year_010.min()) if not year_010.empty else 0,
                    "T015": int(year_015.min()) if not year_015.empty else 0,
                },
                "max_year": {
                    "T010": int(year_010.max()) if not year_010.empty else 0,
                    "T015": int(year_015.max()) if not year_015.empty else 0,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def run_dry_run(dataset_dir: Path) -> None:
    """Discover all ZIPs and print a summary table without loading any data.

    Args:
        dataset_dir: Root directory containing class subdirectories.
    """
    all_zips = sorted(dataset_dir.glob("**/*.zip"))
    if not all_zips:
        print("No ZIP files found under", dataset_dir)
        return

    print(f"\nDry run — dataset directory: {dataset_dir}")
    print(f"Total ZIPs found: {len(all_zips)}\n")

    item_map: dict[str, dict[str, Any]] = {}
    total_csv = 0
    total_size = 0

    for zp in all_zips:
        class_dir = zp.parent.parent.name
        item_dir = zp.parent.name
        key = f"{class_dir}/{item_dir}"
        if key not in item_map:
            item_map[key] = {"zips": 0, "csv_count": 0, "size_bytes": 0}
        item_map[key]["zips"] += 1
        item_map[key]["size_bytes"] += zp.stat().st_size

        try:
            with zipfile.ZipFile(zp) as zf:
                n_csv = sum(1 for n in zf.namelist() if n.lower().endswith(".csv"))
        except zipfile.BadZipFile:
            n_csv = 0
        item_map[key]["csv_count"] += n_csv
        total_csv += n_csv
        total_size += zp.stat().st_size

    header = f"{'Item':<45} {'ZIPs':>6} {'CSVs':>6} {'Size (MB)':>12}"
    print(header)
    print("-" * len(header))
    for key, info in sorted(item_map.items()):
        size_mb = info["size_bytes"] / 1_048_576
        print(
            f"{key:<45} {info['zips']:>6} {info['csv_count']:>6} {size_mb:>11.1f}M"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<45} {len(all_zips):>6} {total_csv:>6} "
        f"{total_size / 1_048_576:>11.1f}M"
    )
    print("\nDry run complete — no data loaded.\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    distance_cutoff: float,
    dose_threshold: float,
    min_temporal_points: int,
    write_csv: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute the full filtration pipeline and return final DataFrame + summary.

    Stages 1–6 are applied in sequence. Stage 1 is applied inline per-chunk
    to control peak memory usage.

    Args:
        dataset_dir: Root dataset directory.
        output_dir: Directory to write output files to.
        distance_cutoff: Stage 1 distance threshold in km (inclusive).
        dose_threshold: Stage 3 initial dose rate threshold in µSv/h (exclusive).
        min_temporal_points: Stage 4 minimum distinct dates per mesh.
        write_csv: Whether to also write a CSV output file.

    Returns:
        Tuple of (final_filtered_df, summary_dict).
    """
    logger.info(
        "Pipeline start — dataset=%s  output=%s  dist<=%.1f km  "
        "dose>%.2f µSv/h  min_dates=%d",
        dataset_dir,
        output_dir,
        distance_cutoff,
        dose_threshold,
        min_temporal_points,
    )

    all_zips = sorted(dataset_dir.glob("**/*.zip"))
    n_total_zips = len(all_zips)
    logger.info("Discovered %d ZIP files", n_total_zips)

    # Accumulators
    stage0_counters: dict[str, int] = {
        "csv_files_found": 0,
        "empty_csv": 0,
        "wrong_column_count": 0,
        "undated_files": 0,
        "parse_failures": 0,
        "non_air_dose": 0,
        "missing_coords": 0,
        "missing_mesh_id": 0,
        "haversine_fallback": 0,
    }
    per_file_counters: dict[str, dict[str, int]] = {}
    skipped_zips: list[str] = []
    corrupted_zips: list[str] = []
    rows_loaded_raw: int = 0

    # Per-item accumulator: item_dir → {source_class, zips, rows_loaded, rows_after_distance}
    per_item_stats: dict[str, dict[str, Any]] = {}

    chunks: list[pd.DataFrame] = []

    for zip_path in all_zips:
        class_dir_name = zip_path.parent.parent.name
        item_dir_name = zip_path.parent.name

        dfs = load_zip(
            zip_path,
            item_dir_name,
            class_dir_name,
            distance_cutoff,
            stage0_counters,
            per_file_counters,
            skipped_zips,
            corrupted_zips,
        )

        for df_chunk in dfs:
            n_chunk = len(df_chunk)
            rows_loaded_raw += n_chunk
            if item_dir_name not in per_item_stats:
                per_item_stats[item_dir_name] = {
                    "source_class": class_dir_name,
                    "zips_processed": 0,
                    "rows_loaded": 0,
                    "rows_after_distance": 0,
                    "rows_in_final": 0,
                    "meshes_in_final": 0,
                }
            per_item_stats[item_dir_name]["rows_loaded"] += n_chunk
            per_item_stats[item_dir_name]["rows_after_distance"] += n_chunk
            chunks.append(df_chunk)

        if dfs or zip_path in skipped_zips or zip_path in corrupted_zips:
            if item_dir_name in per_item_stats:
                per_item_stats[item_dir_name]["zips_processed"] += 1

    if not chunks:
        logger.error("No data loaded from any ZIP. Exiting.")
        sys.exit(1)

    logger.info("Concatenating %d chunks into combined DataFrame...", len(chunks))
    df = pd.concat(chunks, ignore_index=True)
    logger.info("Combined DataFrame: %d rows", len(df))

    # --- Deduplication (before Stage 2 group ops) ---
    df, n_dupes = deduplicate(df)
    stage0_counters["duplicates"] = n_dupes
    logger.info("Deduplication: removed %d duplicate rows → %d rows remain", n_dupes, len(df))

    rows_after_preprocessing = len(df)

    stage0_summary = {
        "rows_loaded_raw": rows_loaded_raw,
        "rows_dropped_parse_failures": stage0_counters["parse_failures"],
        "rows_dropped_non_air_dose": stage0_counters["non_air_dose"],
        "rows_dropped_missing_coords": stage0_counters["missing_coords"],
        "rows_dropped_missing_mesh_id": stage0_counters["missing_mesh_id"],
        "rows_dropped_duplicates": n_dupes,
        "rows_after_preprocessing": rows_after_preprocessing,
        "haversine_fallback_rows": stage0_counters["haversine_fallback"],
    }

    # --- Stage 1 summary (applied inline per-chunk) ---
    s1_in = rows_loaded_raw
    s1_out = rows_after_preprocessing
    s1_dropped = s1_in - s1_out
    stage_1_summary = {
        "rows_in": s1_in,
        "rows_dropped": s1_dropped,
        "rows_out": s1_out,
        "pct_dropped": round(100.0 * s1_dropped / max(s1_in, 1), 2),
    }
    logger.info(
        "Stage 1 (distance <= %.1f km):     in=%d  dropped=%d  out=%d",
        distance_cutoff,
        s1_in,
        s1_dropped,
        s1_out,
    )

    # --- Stage 2 ---
    s2_in = len(df)
    df, n_below_limit, n_nan_val = stage_2_detection_limit(df)
    s2_out = len(df)
    s2_dropped = n_below_limit + n_nan_val
    stage_2_summary = {
        "rows_in": s2_in,
        "rows_dropped_below_limit": n_below_limit,
        "rows_dropped_nan_value": n_nan_val,
        "rows_out": s2_out,
        "pct_dropped": round(100.0 * s2_dropped / max(s2_in, 1), 2),
    }
    logger.info(
        "Stage 2 (detection limit):         in=%d  dropped=%d  out=%d",
        s2_in,
        s2_dropped,
        s2_out,
    )

    # --- Stage 3 ---
    s3_in = len(df)
    s3_meshes_in = df["mesh_id_250m"].nunique()
    df, s3_meshes_dropped, _ = stage_3_dose_threshold(df, dose_threshold)
    s3_out = len(df)
    s3_meshes_out = df["mesh_id_250m"].nunique()
    stage_3_summary = {
        "rows_in": s3_in,
        "meshes_in": s3_meshes_in,
        "meshes_dropped": s3_meshes_dropped,
        "meshes_out": s3_meshes_out,
        "rows_out": s3_out,
        "pct_rows_dropped": round(100.0 * (s3_in - s3_out) / max(s3_in, 1), 2),
        "pct_meshes_dropped": round(
            100.0 * s3_meshes_dropped / max(s3_meshes_in, 1), 2
        ),
    }
    logger.info(
        "Stage 3 (dose threshold > %.2f):   in=%d  meshes_dropped=%d  rows_out=%d",
        dose_threshold,
        s3_in,
        s3_meshes_dropped,
        s3_out,
    )

    # --- Stage 4 ---
    s4_in = len(df)
    s4_meshes_in = df["mesh_id_250m"].nunique()
    df, s4_meshes_dropped, _ = stage_4_temporal_completeness(df, min_temporal_points)
    s4_out = len(df)
    s4_meshes_out = df["mesh_id_250m"].nunique()
    stage_4_summary = {
        "rows_in": s4_in,
        "meshes_in": s4_meshes_in,
        "meshes_dropped": s4_meshes_dropped,
        "meshes_out": s4_meshes_out,
        "rows_out": s4_out,
        "pct_rows_dropped": round(100.0 * (s4_in - s4_out) / max(s4_in, 1), 2),
        "pct_meshes_dropped": round(
            100.0 * s4_meshes_dropped / max(s4_meshes_in, 1), 2
        ),
    }
    logger.info(
        "Stage 4 (temporal >= %d dates):    in=%d  meshes_dropped=%d  rows_out=%d",
        min_temporal_points,
        s4_in,
        s4_meshes_dropped,
        s4_out,
    )

    # --- Stage 5 ---
    s5_in = len(df)
    df, s5_mixed, s5_dropped = stage_5_type_consistency(df)
    s5_out = len(df)
    stage_5_summary = {
        "rows_in": s5_in,
        "mixed_meshes_found": s5_mixed,
        "rows_dropped": s5_dropped,
        "rows_out": s5_out,
        "pct_dropped": round(100.0 * s5_dropped / max(s5_in, 1), 2),
    }
    logger.info(
        "Stage 5 (type consistency):        in=%d  dropped=%d  out=%d",
        s5_in,
        s5_dropped,
        s5_out,
    )

    # --- Stage 6 ---
    df = stage_6_height_flag(df)
    stage_6_summary = {
        "rows_in": len(df),
        "height_anomalous_true": int(df["height_anomalous"].sum()),
        "rows_out": len(df),
    }

    # --- Per-item final counts ---
    final_item_counts = df.groupby("source_item").size().to_dict()
    final_item_meshes = (
        df.groupby("source_item")["mesh_id_250m"].nunique().to_dict()
    )
    for item, stats in per_item_stats.items():
        stats["rows_in_final"] = int(final_item_counts.get(item, 0))
        stats["meshes_in_final"] = int(final_item_meshes.get(item, 0))

    logger.info(
        "Final output: %d rows, %d meshes written to %s",
        len(df),
        df["mesh_id_250m"].nunique(),
        output_dir / "filtered_emdb.parquet",
    )

    # --- Write Parquet ---
    parquet_path = output_dir / "filtered_emdb.parquet"
    write_parquet(df.copy(), parquet_path)

    # --- Optionally write CSV ---
    if write_csv:
        csv_path = output_dir / "filtered_emdb.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_csv = df.drop(
            columns=[c for c in COLUMNS_DROP_BEFORE_WRITE if c in df.columns],
            errors="ignore",
        )
        df_csv = df_csv[[c for c in PARQUET_COLUMN_ORDER if c in df_csv.columns]]
        df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV written to %s", csv_path)

    # --- Build summary ---
    params: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "distance_cutoff_km": distance_cutoff,
        "dose_threshold_usvh": dose_threshold,
        "min_temporal_points": min_temporal_points,
    }
    file_disc: dict[str, Any] = {
        "total_zips_found": n_total_zips,
        "total_zips_skipped": len(skipped_zips),
        "total_zips_corrupted": len(corrupted_zips),
        "total_csv_files_found": stage0_counters["csv_files_found"],
        "total_csv_files_empty": stage0_counters["empty_csv"],
        "total_csv_files_wrong_columns": stage0_counters["wrong_column_count"],
        "total_csv_files_undated": stage0_counters["undated_files"],
    }

    summary = build_summary(
        params=params,
        file_disc=file_disc,
        stage0=stage0_summary,
        s1=stage_1_summary,
        s2=stage_2_summary,
        s3=stage_3_summary,
        s4=stage_4_summary,
        s5=stage_5_summary,
        s6=stage_6_summary,
        final_df=df,
        per_item=per_item_stats,
    )

    write_summary(summary, output_dir / "filter_summary.json")
    return df, summary


# ---------------------------------------------------------------------------
# Sensitivity analysis wrapper
# ---------------------------------------------------------------------------

def run_sensitivity(
    dataset_dir: Path,
    output_dir: Path,
    distance_cutoff: float,
    min_temporal_points: int,
    write_csv: bool,
) -> None:
    """Run the full pipeline at thresholds 0.10 and 0.15, then write comparison.

    All output goes to subdirectories; the top-level output_dir receives
    only sensitivity_comparison.json.

    Args:
        dataset_dir: Root dataset directory.
        output_dir: Parent directory for sensitivity subdirectories.
        distance_cutoff: Stage 1 distance threshold in km.
        min_temporal_points: Stage 4 minimum distinct dates per mesh.
        write_csv: Whether to also write CSV outputs.
    """
    thresholds = [0.10, 0.15]
    suffix_map = {0.10: "T010", 0.15: "T015"}
    results: dict[float, pd.DataFrame] = {}

    for threshold in thresholds:
        suffix = suffix_map[threshold]
        sub_dir = output_dir / f"sensitivity_{suffix}"
        logger.info(
            "=== Sensitivity run: dose_threshold=%.2f  output=%s ===",
            threshold,
            sub_dir,
        )
        df, _ = run_pipeline(
            dataset_dir=dataset_dir,
            output_dir=sub_dir,
            distance_cutoff=distance_cutoff,
            dose_threshold=threshold,
            min_temporal_points=min_temporal_points,
            write_csv=write_csv,
        )
        results[threshold] = df

    comparison = compute_sensitivity_comparison(results[0.10], results[0.15])
    comparison_path = output_dir / "sensitivity_comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Sensitivity comparison written to %s", comparison_path)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with all pipeline configuration attributes.
    """
    parser = argparse.ArgumentParser(
        description="Filter JAEA EMDB air dose rate data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset/",
        type=str,
        metavar="DIR",
        help="Root directory containing class subdirectories of ZIP archives.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        metavar="DIR",
        help="Directory to write filtered_emdb.parquet and filter_summary.json.",
    )
    parser.add_argument(
        "--distance-cutoff",
        default=80.0,
        type=float,
        metavar="KM",
        help="Stage 1: maximum distance from FDNPP in km (inclusive).",
    )
    parser.add_argument(
        "--dose-threshold",
        default=0.10,
        type=float,
        metavar="USVH",
        help="Stage 3: minimum initial dose rate in µSv/h (exclusive). "
             "Ignored when --sensitivity is set.",
    )
    parser.add_argument(
        "--min-temporal-points",
        default=4,
        type=int,
        metavar="N",
        help="Stage 4: minimum number of distinct measurement dates per mesh cell.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write output/filtered_emdb.csv (utf-8-sig, Excel-compatible).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files and report counts only; do not load or filter data.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run at thresholds 0.10 and 0.15 µSv/h and write comparison JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging to console.",
    )
    args = parser.parse_args()

    # Validation: cannot combine --sensitivity with explicit --dose-threshold
    # Detect if user explicitly passed --dose-threshold
    raw_argv = sys.argv[1:]
    user_set_threshold = any(
        a.startswith("--dose-threshold") for a in raw_argv
    )
    if args.sensitivity and user_set_threshold:
        parser.error("Cannot combine --sensitivity with --dose-threshold.")

    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Script entry point."""
    args = parse_args()

    setup_logging(verbose=args.verbose)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not dataset_dir.is_dir():
        logger.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    if args.dry_run:
        run_dry_run(dataset_dir)
        return

    if args.sensitivity:
        logger.info(
            "Sensitivity mode — running at dose thresholds 0.10 and 0.15 µSv/h"
        )
        run_sensitivity(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            distance_cutoff=args.distance_cutoff,
            min_temporal_points=args.min_temporal_points,
            write_csv=args.csv,
        )
    else:
        run_pipeline(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            distance_cutoff=args.distance_cutoff,
            dose_threshold=args.dose_threshold,
            min_temporal_points=args.min_temporal_points,
            write_csv=args.csv,
        )

    logger.info("filter_emdb.py complete.")


if __name__ == "__main__":
    main()
