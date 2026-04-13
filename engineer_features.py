"""engineer_features.py — Temporal feature engineering pipeline for Fukushima dose-rate analysis.

Reads the filtered EMDB Parquet file produced by filter_emdb.py, aggregates
measurements to daily medians per 250 m mesh cell, and computes a 10-feature
temporal vector for each cell suitable for SOM clustering.

Usage:
    python engineer_features.py [--dry-run] [--csv] [--verbose]
    python engineer_features.py --input output/filtered_emdb.parquet --output-dir output/
    python engineer_features.py --early-late-cutoff 2013-03-11 --verbose
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCIDENT_DATE: date = date(2011, 3, 11)
EARLY_LATE_CUTOFF: date = date(2013, 3, 11)  # 1 Cs-134 half-life post-accident
HALF_LIFE_CAP_YEARS: float = 100.0
LN_FLOOR: float = 1e-6
RECONTAM_RATIO_THRESHOLD: float = 0.5   # 50% increase triggers flag
RECONTAM_ABS_THRESHOLD: float = 0.05   # µSv/h absolute minimum increase
SLOPE_RATIO_GUARD: float = 5e-6        # per day — guard against division by tiny late_slope
MIN_POINTS_FOR_SLOPE: int = 2
DAYS_PER_YEAR: float = 365.25

# Expected mesh count in filtered_emdb.parquet
EXPECTED_MESH_COUNT: int = 138_768

# 10 SOM input features (for robust scaling + clipping)
SOM_FEATURE_COLUMNS: list[str] = [
    "total_decay_ratio",
    "log_linear_slope",
    "log_linear_r2",
    "early_slope",
    "late_slope",
    "residual_cv",
    "slope_ratio",
    "n_increases",
    "temporal_span_years",
]

# Expected columns in the input Parquet file
REQUIRED_INPUT_COLUMNS: list[str] = [
    "mesh_id_250m",
    "correction_date",
    "value",
    "distance_from_fdnpp_km",
    "latitude",
    "longitude",
    "measurement_type",
    "year",
    "is_decontaminated",
    "height_anomalous",
    "source_item",
    "source_class",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("engineer_features")


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure dual-handler logging: INFO/DEBUG to console, DEBUG to file.

    Args:
        verbose: When True, the console handler is set to DEBUG level.
        log_file: Path of the log file to write DEBUG output to. If None,
            no file handler is added.
    """
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Step 1: Load & categorical conversion
# ---------------------------------------------------------------------------

def load_filtered_data(path: Path) -> pd.DataFrame:
    """Load filtered_emdb.parquet and validate expected schema.

    Converts mesh_id_250m to a Categorical dtype, saving approximately 400 MB
    of RAM compared to a plain object column across 9.2 M rows.

    Args:
        path: Absolute path to filtered_emdb.parquet.

    Returns:
        DataFrame with mesh_id_250m as Categorical, correction_date as
        Python date objects (date32), and all required columns present.

    Raises:
        SystemExit: If the file is not found or required columns are missing.
    """
    if not path.is_file():
        logger.error("Input file not found: %s", path)
        sys.exit(1)

    logger.info("Loading %s ...", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Input is missing required columns: %s", missing)
        sys.exit(1)

    # Convert mesh_id_250m to categorical — critical for memory and groupby speed
    df["mesh_id_250m"] = df["mesh_id_250m"].astype("category")
    logger.info(
        "mesh_id_250m converted to Categorical: %d unique values",
        df["mesh_id_250m"].nunique(),
    )

    # Ensure correction_date is a proper date type (Arrow date32 reads as object
    # on some pandas/pyarrow combinations)
    if not pd.api.types.is_datetime64_any_dtype(df["correction_date"]):
        df["correction_date"] = pd.to_datetime(df["correction_date"])

    return df


# ---------------------------------------------------------------------------
# Step 2: Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw measurements to one median per (mesh_id_250m, correction_date).

    Computes per-mesh days_since_first and the natural log of the clipped
    median dose rate (ln_value), which linearises Cs radioactive decay.

    After aggregation the 9.2 M-row source frame is freed from memory.

    Args:
        df: Full filtered DataFrame as loaded by load_filtered_data().

    Returns:
        Daily-aggregated DataFrame sorted by (mesh_id_250m, correction_date)
        with columns: mesh_id_250m, correction_date, median_value,
        days_since_first, ln_value.
    """
    logger.info("Aggregating to daily medians per mesh ...")

    daily = (
        df.groupby(["mesh_id_250m", "correction_date"], observed=True)["value"]
        .median()
        .reset_index()
        .rename(columns={"value": "median_value"})
    )

    # Sort order is exploited by downstream .first()/.last() calls
    daily = daily.sort_values(["mesh_id_250m", "correction_date"]).reset_index(drop=True)

    # Re-categorise after reset_index (groupby may have dropped unused levels)
    daily["mesh_id_250m"] = daily["mesh_id_250m"].astype("category")

    # days_since_first — float32 is sufficient for OLS arithmetic at this scale
    first_date = daily.groupby("mesh_id_250m", observed=True)["correction_date"].transform("first")
    daily["days_since_first"] = (
        (daily["correction_date"] - first_date).dt.days.astype("float32")
    )

    # Natural log of clipped dose rate — linearises exponential decay
    daily["ln_value"] = np.log(
        np.clip(daily["median_value"].to_numpy(dtype="float64"), LN_FLOOR, None)
    ).astype("float32")

    logger.info(
        "Daily frame: %d rows, %d unique meshes",
        len(daily),
        daily["mesh_id_250m"].nunique(),
    )

    # Free the 9.2 M-row source frame
    del df
    gc.collect()

    return daily


# ---------------------------------------------------------------------------
# Step 3: Vectorized OLS via sufficient statistics
# ---------------------------------------------------------------------------

def _ols_from_stats(
    n: pd.Series,
    sum_t: pd.Series,
    sum_y: pd.Series,
    sum_ty: pd.Series,
    sum_t2: pd.Series,
    sum_y2: pd.Series,
    min_points: int = MIN_POINTS_FOR_SLOPE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply closed-form OLS via normal equations from aggregated sufficient statistics.

    No scipy, no apply(). Uses vectorised NumPy operations on full-length Series.

    The denominator guard (1e-12) handles degenerate cases where all measurements
    share the same timestamp (denom == 0), returning NaN slope/intercept.

    Args:
        n: Number of observations per mesh.
        sum_t: Sum of time offsets (days_since_first).
        sum_y: Sum of ln_value.
        sum_ty: Sum of t * y products.
        sum_t2: Sum of squared time offsets.
        sum_y2: Sum of squared ln_value.
        min_points: Minimum observations required; below this slope is NaN.

    Returns:
        Tuple of (slope, intercept, r2) Series indexed identically to input.
    """
    raw_denom = n * sum_t2 - sum_t ** 2
    denom = np.where(np.abs(raw_denom) < 1e-12, np.nan, raw_denom)

    slope = (n * sum_ty - sum_t * sum_y) / denom
    intercept = (sum_y - slope * sum_t) / n

    mean_y = sum_y / n
    ss_tot = sum_y2 - n * mean_y ** 2
    ss_tot = np.maximum(ss_tot, 1e-12)
    ss_res = (
        sum_y2
        - 2.0 * intercept * sum_y
        - 2.0 * slope * sum_ty
        + n * intercept ** 2
        + 2.0 * slope * intercept * sum_t
        + slope ** 2 * sum_t2
    )
    r2 = np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0)

    # Mask meshes with fewer than min_points measurements
    too_few = n < min_points
    slope = pd.Series(np.where(too_few, np.nan, slope), index=n.index)
    intercept = pd.Series(np.where(too_few, np.nan, intercept), index=n.index)
    r2 = pd.Series(np.where(too_few, np.nan, r2), index=n.index)

    return slope, intercept, r2


# Sufficient-statistics aggregation spec — shared by full-period and sub-period groupbys
_OLS_AGG_MAP: dict[str, Any] = {
    "t": ["count", "sum"],
    "y": "sum",
    "ty": "sum",
    "t2": "sum",
    "y2": "sum",
}
_OLS_AGG_COLUMNS: list[str] = ["n", "sum_t", "sum_y", "sum_ty", "sum_t2", "sum_y2"]


def _period_col(sub_stats: pd.DataFrame, col: str, period: str) -> pd.Series:
    """Return a sub-stats column for a given period, or a zero Series if absent.

    Args:
        sub_stats: Unstacked per-period sufficient-statistics DataFrame with a
            MultiIndex column level (stat_name, period).
        col: Sufficient-statistic column name (e.g. ``"n"``, ``"sum_t"``).
        period: Period label — ``"early"`` or ``"late"``.

    Returns:
        Float64 Series indexed by mesh_id_250m, or zeros when the period is
        entirely absent from the data (e.g. no measurements before the cutoff).
    """
    try:
        return sub_stats[(col, period)].astype("float64")
    except KeyError:
        return pd.Series(0.0, index=sub_stats.index)


def compute_ols_features(
    daily: pd.DataFrame,
    cutoff: date = EARLY_LATE_CUTOFF,
) -> pd.DataFrame:
    """Compute full-period and early/late-period OLS features for each mesh.

    Uses vectorised sufficient-statistics aggregation. No scipy dependency,
    no groupby apply(). The early period is [ACCIDENT_DATE, cutoff) and the
    late period is [cutoff, last measurement].

    Args:
        daily: Daily-aggregated frame from aggregate_daily().
        cutoff: Date splitting early and late OLS periods. Defaults to the
            module-level EARLY_LATE_CUTOFF constant (2013-03-11).

    Returns:
        DataFrame indexed by mesh_id_250m with columns:
        log_linear_slope, log_linear_r2, intercept, early_slope, late_slope.
    """
    logger.info("Computing OLS sufficient statistics (full period) ...")

    # Pre-compute product columns — float64 for numerical stability
    daily["t"] = daily["days_since_first"].astype("float64")
    daily["y"] = daily["ln_value"].astype("float64")
    daily["t2"] = daily["t"] ** 2
    daily["y2"] = daily["y"] ** 2
    daily["ty"] = daily["t"] * daily["y"]

    # Full-period sufficient statistics
    stats = daily.groupby("mesh_id_250m", observed=True).agg(_OLS_AGG_MAP)
    stats.columns = _OLS_AGG_COLUMNS

    slope_full, intercept_full, r2_full = _ols_from_stats(
        n=stats["n"].astype("float64"),
        sum_t=stats["sum_t"],
        sum_y=stats["sum_y"],
        sum_ty=stats["sum_ty"],
        sum_t2=stats["sum_t2"],
        sum_y2=stats["sum_y2"],
    )

    logger.info("Computing OLS sufficient statistics (early / late periods) ...")

    split_date = pd.Timestamp(cutoff)
    daily["period"] = np.where(
        daily["correction_date"] < split_date, "early", "late"
    )

    sub_stats = (
        daily.groupby(["mesh_id_250m", "period"], observed=True)
        .agg(_OLS_AGG_MAP)
    )
    sub_stats.columns = _OLS_AGG_COLUMNS
    sub_stats = sub_stats.unstack(level="period", fill_value=0)

    # Extract per-period columns safely — the period may be absent for some meshes
    for period in ("early", "late"):
        n_p = _period_col(sub_stats, "n", period)
        sum_t_p = _period_col(sub_stats, "sum_t", period)
        sum_y_p = _period_col(sub_stats, "sum_y", period)
        sum_ty_p = _period_col(sub_stats, "sum_ty", period)
        sum_t2_p = _period_col(sub_stats, "sum_t2", period)
        sum_y2_p = _period_col(sub_stats, "sum_y2", period)

        slope_p, _, _ = _ols_from_stats(
            n=n_p,
            sum_t=sum_t_p,
            sum_y=sum_y_p,
            sum_ty=sum_ty_p,
            sum_t2=sum_t2_p,
            sum_y2=sum_y2_p,
        )

        if period == "early":
            early_slope = slope_p
        else:
            late_slope = slope_p

    ols_df = pd.DataFrame(
        {
            "log_linear_slope": slope_full,
            "log_linear_r2": r2_full,
            "intercept": intercept_full,
            "early_slope": early_slope,
            "late_slope": late_slope,
        },
        index=stats.index,
    ).reset_index()

    logger.info(
        "OLS features computed for %d meshes (%.1f%% negative slope)",
        len(ols_df),
        100.0 * (ols_df["log_linear_slope"] < 0).sum() / max(len(ols_df), 1),
    )

    return ols_df


# ---------------------------------------------------------------------------
# Step 4: Level features
# ---------------------------------------------------------------------------

def compute_level_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute dose-rate level features for each mesh cell.

    Exploits the pre-sorted order (mesh, date) so that first() and last()
    map directly to chronologically first and last measurements.

    Args:
        daily: Daily-aggregated DataFrame sorted by (mesh_id_250m, correction_date).

    Returns:
        DataFrame with columns: mesh_id_250m, initial_dose_rate,
        final_dose_rate, max_dose_rate, max_dose_timing_days.
    """
    logger.info("Computing level features ...")

    g = daily.groupby("mesh_id_250m", observed=True)

    initial_dose_rate = g["median_value"].first()
    final_dose_rate = g["median_value"].last()
    max_dose_rate = g["median_value"].max()

    # max_dose_timing_days: days_since_first at the row where median_value hits the group max.
    # Use transform to broadcast the group max, then take the first matching days_since_first.
    group_max = g["median_value"].transform("max")
    is_max_row = daily["median_value"] == group_max
    max_timing = (
        daily.loc[is_max_row]
        .groupby("mesh_id_250m", observed=True)["days_since_first"]
        .first()
    )

    level_df = pd.DataFrame(
        {
            "initial_dose_rate": initial_dose_rate,
            "final_dose_rate": final_dose_rate,
            "max_dose_rate": max_dose_rate,
            "max_dose_timing_days": max_timing,
        }
    ).reset_index()

    return level_df


# ---------------------------------------------------------------------------
# Step 5: Shape features (fully vectorized)
# ---------------------------------------------------------------------------

def compute_shape_features(daily: pd.DataFrame, ols_df: pd.DataFrame) -> pd.DataFrame:
    """Compute shape features for each mesh using fully vectorised operations.

    No groupby apply(). Uses diff(), shift(), and merge() to derive:
    - n_increases: count of consecutive dose-rate increases
    - recontamination_flag: sustained large increase after apparent decay
    - residual_cv: coefficient of variation of OLS residuals

    Args:
        daily: Daily-aggregated DataFrame with t, y columns already present.
        ols_df: OLS feature DataFrame from compute_ols_features().

    Returns:
        DataFrame with columns: mesh_id_250m, n_increases,
        recontamination_flag, residual_cv.
    """
    logger.info("Computing shape features (vectorised) ...")

    g = daily.groupby("mesh_id_250m", observed=True)["median_value"]

    # --- n_increases ---
    diff = g.diff()
    daily["is_increase"] = diff > 0
    n_increases = (
        daily.groupby("mesh_id_250m", observed=True)["is_increase"]
        .sum()
        .rename("n_increases")
    )

    # --- recontamination_flag ---
    shifted = g.shift(1)
    pct_change = diff / shifted.clip(lower=LN_FLOOR)
    abs_change = diff

    daily["big_increase"] = (
        (pct_change > RECONTAM_RATIO_THRESHOLD) & (abs_change > RECONTAM_ABS_THRESHOLD)
    )
    next_also_up = (
        daily.groupby("mesh_id_250m", observed=True)["is_increase"]
        .shift(-1)
        .fillna(False)
    )
    daily["recontam_event"] = daily["big_increase"] & next_also_up
    recontamination_flag = (
        daily.groupby("mesh_id_250m", observed=True)["recontam_event"]
        .any()
        .rename("recontamination_flag")
    )

    # --- residual_cv ---
    # Merge OLS slope + intercept back to daily for vectorised residual computation
    ols_sub = ols_df[["mesh_id_250m", "intercept", "log_linear_slope"]].copy()
    daily_merged = daily.merge(ols_sub, on="mesh_id_250m", how="left")

    daily_merged["predicted"] = (
        daily_merged["intercept"] + daily_merged["log_linear_slope"] * daily_merged["t"]
    )
    daily_merged["residual"] = daily_merged["y"] - daily_merged["predicted"]

    resid_std = (
        daily_merged.groupby("mesh_id_250m", observed=True)["residual"]
        .std()
    )
    # Denominator is abs(mean(y)), NOT abs(mean(residual)) — OLS residuals have
    # mean ≈ 0 by construction, so dividing by mean(residual) produces infinity.
    mean_y = (
        daily_merged.groupby("mesh_id_250m", observed=True)["y"]
        .mean()
        .abs()
        .clip(lower=1e-8)
    )
    residual_cv = (resid_std / mean_y).rename("residual_cv")

    shape_df = pd.DataFrame(
        {
            "n_increases": n_increases,
            "recontamination_flag": recontamination_flag,
            "residual_cv": residual_cv,
        }
    ).reset_index()

    return shape_df


# ---------------------------------------------------------------------------
# Step 6: Derived features
# ---------------------------------------------------------------------------

def compute_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from already-merged level, OLS, and coverage data.

    Operates on the 138 K-row feature frame in place (by returning a modified copy).

    Args:
        features: Merged DataFrame containing initial_dose_rate, final_dose_rate,
            log_linear_slope, early_slope, late_slope, temporal_span_days, and
            has_early_data.

    Returns:
        DataFrame with four additional columns: total_decay_ratio,
        ecological_half_life_y, slope_ratio, temporal_span_years.
    """
    logger.info("Computing derived features ...")

    features = features.copy()

    # total_decay_ratio: ratio of final to initial dose rate
    features["total_decay_ratio"] = (
        features["final_dose_rate"]
        / features["initial_dose_rate"].clip(lower=LN_FLOOR)
    )

    # ecological_half_life_y: only meaningful for negative (decaying) slopes
    slope_vals = features["log_linear_slope"].to_numpy(dtype="float64")
    abs_slope = np.abs(slope_vals)
    abs_slope_safe = np.where(abs_slope < 1e-12, np.nan, abs_slope)
    half_life = np.where(
        slope_vals < 0,
        np.minimum(
            np.log(2) / abs_slope_safe / DAYS_PER_YEAR,
            HALF_LIFE_CAP_YEARS,
        ),
        np.nan,
    )
    features["ecological_half_life_y"] = half_life

    # slope_ratio: early / late — measures how much decay accelerated or slowed
    late_abs = features["late_slope"].abs()
    features["slope_ratio"] = np.where(
        late_abs > SLOPE_RATIO_GUARD,
        features["early_slope"] / features["late_slope"],
        1.0,
    )

    # temporal_span_years
    features["temporal_span_years"] = features["temporal_span_days"] / DAYS_PER_YEAR

    return features


# ---------------------------------------------------------------------------
# Step 7: Coverage features
# ---------------------------------------------------------------------------

def compute_coverage_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal coverage statistics for each mesh.

    Args:
        daily: Daily-aggregated DataFrame sorted by (mesh_id_250m, correction_date).

    Returns:
        DataFrame with columns: mesh_id_250m, n_measurement_dates,
        temporal_span_days, first_measurement_date, last_measurement_date.
    """
    logger.info("Computing coverage features ...")

    g = daily.groupby("mesh_id_250m", observed=True)["correction_date"]

    n_dates = g.count().rename("n_measurement_dates")
    first_date = g.first().rename("first_measurement_date")
    last_date = g.last().rename("last_measurement_date")
    span_days = (last_date - first_date).dt.days.rename("temporal_span_days")

    coverage_df = pd.DataFrame(
        {
            "n_measurement_dates": n_dates,
            "temporal_span_days": span_days,
            "first_measurement_date": first_date,
            "last_measurement_date": last_date,
        }
    ).reset_index()

    return coverage_df


# ---------------------------------------------------------------------------
# Step 8: Metadata features
# ---------------------------------------------------------------------------

def compute_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-mesh metadata from the original (pre-deletion) filtered frame.

    Args:
        df: Full filtered DataFrame (before aggregate_daily() deletes it).

    Returns:
        DataFrame with columns: mesh_id_250m, measurement_type,
        is_decontaminated, height_anomalous_pct, latitude, longitude,
        distance_from_fdnpp_km.
    """
    logger.info("Computing metadata features ...")

    g = df.groupby("mesh_id_250m", observed=True)

    # Mode of measurement_type — most frequent category per mesh
    mtype = (
        g["measurement_type"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .rename("measurement_type")
    )

    is_decontaminated = g["is_decontaminated"].any().rename("is_decontaminated")
    height_anomalous_pct = (
        g["height_anomalous"].mean().rename("height_anomalous_pct")
    )
    latitude = g["latitude"].median().rename("latitude")
    longitude = g["longitude"].median().rename("longitude")
    distance = g["distance_from_fdnpp_km"].median().rename("distance_from_fdnpp_km")

    meta_df = pd.DataFrame(
        {
            "measurement_type": mtype,
            "is_decontaminated": is_decontaminated,
            "height_anomalous_pct": height_anomalous_pct,
            "latitude": latitude,
            "longitude": longitude,
            "distance_from_fdnpp_km": distance,
        }
    ).reset_index()

    return meta_df


# ---------------------------------------------------------------------------
# Step 9: Imputation
# ---------------------------------------------------------------------------

def impute_nans(
    features: pd.DataFrame, som_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute NaN values in SOM feature columns using domain-aware strategies.

    Imputation rules:
    - early_slope: structural zero when has_early_data is False (mesh had no
      measurements before the EARLY_LATE_CUTOFF); otherwise column median.
    - slope_ratio: structural 1.0 when late_slope magnitude is at or below
      SLOPE_RATIO_GUARD (ratio is meaningless when denominator is near zero).
    - All remaining NaNs in SOM columns: column median.

    Args:
        features: Merged feature DataFrame with has_early_data and late_slope columns.
        som_cols: List of SOM feature column names to impute.

    Returns:
        Tuple of (imputed DataFrame, imputation_counts dict mapping column name
        to number of cells imputed).
    """
    logger.info("Imputing NaN values in SOM feature columns ...")

    features = features.copy()
    imputation_counts: dict[str, Any] = {}

    for col in som_cols:
        if col not in features.columns:
            logger.warning("SOM column '%s' not found in features — skipping imputation", col)
            continue

        n_nan_before = int(features[col].isna().sum())

        if col == "early_slope":
            # Structural imputation: no early data means slope is effectively 0
            no_early = ~features["has_early_data"]
            structural_mask = features[col].isna() & no_early
            features.loc[structural_mask, col] = 0.0
            n_structural = int(structural_mask.sum())

            # Remaining NaN (has_early_data but OLS failed) → column median
            remaining_mask = features[col].isna()
            if remaining_mask.any():
                median_val = float(features[col].median())
                features.loc[remaining_mask, col] = median_val
            else:
                median_val = float(features[col].median())

            imputation_counts[col] = {
                "n_nan_before": n_nan_before,
                "n_structural_zero": n_structural,
                "n_median": n_nan_before - n_structural,
                "median_used": median_val,
            }

        elif col == "slope_ratio":
            # Structural imputation: late_slope too small → ratio is 1.0
            late_small = features["late_slope"].abs() <= SLOPE_RATIO_GUARD
            structural_mask = features[col].isna() & late_small
            features.loc[structural_mask, col] = 1.0
            n_structural = int(structural_mask.sum())

            # Remaining NaN → column median
            remaining_mask = features[col].isna()
            if remaining_mask.any():
                median_val = float(features[col].median())
                features.loc[remaining_mask, col] = median_val
            else:
                median_val = float(features[col].median())

            imputation_counts[col] = {
                "n_nan_before": n_nan_before,
                "n_structural_one": n_structural,
                "n_median": n_nan_before - n_structural,
                "median_used": median_val,
            }

        else:
            # Generic: column median
            remaining_mask = features[col].isna()
            if remaining_mask.any():
                median_val = float(features[col].median())
                features.loc[remaining_mask, col] = median_val
            else:
                median_val = float(features[col].median()) if not features[col].isna().all() else 0.0

            imputation_counts[col] = {
                "n_nan_before": n_nan_before,
                "n_median": n_nan_before,
                "median_used": median_val,
            }

        n_nan_after = int(features[col].isna().sum())
        logger.debug(
            "  %s: %d NaN before → %d NaN after imputation",
            col,
            n_nan_before,
            n_nan_after,
        )

    return features, imputation_counts


# ---------------------------------------------------------------------------
# Step 10: Robust scaling
# ---------------------------------------------------------------------------

def normalize_features(
    features: pd.DataFrame, som_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply robust (IQR) scaling to SOM feature columns, then clip to [-3, 3].

    Robust scaling: z = (x - median) / IQR where IQR = Q75 - Q25.
    When IQR is zero (constant column), the scaled value is set to 0.0.
    Scaled columns are appended with a ``_z`` suffix.

    Args:
        features: Feature DataFrame with SOM columns already imputed.
        som_cols: List of SOM feature column names to scale.

    Returns:
        Tuple of (features DataFrame with _z columns appended,
        normalization_params dict mapping each column to its median and IQR).
    """
    logger.info("Applying robust (IQR) scaling to SOM features ...")

    features = features.copy()
    norm_params: dict[str, Any] = {}

    for col in som_cols:
        if col not in features.columns:
            logger.warning("SOM column '%s' not in features — skipping scaling", col)
            continue

        vals = features[col].astype("float64")
        median_val = float(vals.median())
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        iqr = q75 - q25

        if iqr < 1e-12:
            logger.warning(
                "Column '%s' has IQR ≈ 0 (%.2e) — scaled column set to 0.0", col, iqr
            )
            scaled = np.zeros(len(vals))
            iqr_used = float("nan")
        else:
            scaled = np.clip((vals.to_numpy() - median_val) / iqr, -3.0, 3.0)
            iqr_used = iqr

        features[f"{col}_z"] = scaled.astype("float32")
        norm_params[col] = {
            "median": median_val,
            "q25": q25,
            "q75": q75,
            "iqr": iqr_used,
        }
        logger.debug(
            "  %s: median=%.4f  IQR=%.4f  clipped_frac=%.2f%%",
            col,
            median_val,
            iqr_used if np.isfinite(iqr_used) else 0.0,
            100.0 * np.mean(np.abs(scaled) >= 3.0),
        )

    return features, norm_params


# ---------------------------------------------------------------------------
# Step 11: Assembly & output writers
# ---------------------------------------------------------------------------

def assemble_feature_matrix(
    ols_df: pd.DataFrame,
    level_df: pd.DataFrame,
    shape_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all per-mesh feature DataFrames into a single feature matrix.

    All frames are merged on mesh_id_250m using a left join anchored on ols_df
    (which contains every mesh that passed the OLS step). Row count is verified
    against EXPECTED_MESH_COUNT and logged as a warning if it differs.

    Args:
        ols_df: OLS feature frame.
        level_df: Level feature frame.
        shape_df: Shape feature frame.
        coverage_df: Coverage feature frame.
        meta_df: Metadata feature frame.

    Returns:
        Single merged DataFrame with one row per mesh_id_250m.
    """
    logger.info("Assembling feature matrix ...")

    features = ols_df.copy()
    for other in (level_df, shape_df, coverage_df, meta_df):
        features = features.merge(other, on="mesh_id_250m", how="left")

    n_rows = len(features)
    logger.info("Feature matrix: %d rows, %d columns", n_rows, features.shape[1])

    if n_rows != EXPECTED_MESH_COUNT:
        logger.warning(
            "Row count mismatch: expected %d, got %d",
            EXPECTED_MESH_COUNT,
            n_rows,
        )
    else:
        logger.info("Row count verified: %d (matches expected mesh count)", n_rows)

    return features


def write_parquet(features: pd.DataFrame, path: Path) -> None:
    """Write the feature matrix to a Parquet file with explicit PyArrow schema.

    Uses dictionary encoding for string columns, Snappy compression, and
    500 K-row row groups. Date columns are serialised as date32.

    Args:
        features: Assembled and normalised feature DataFrame.
        path: Destination .parquet file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing feature Parquet to %s ...", path)

    df = features.copy()

    # Convert date columns for Arrow serialisation
    for col in ("first_measurement_date", "last_measurement_date"):
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date

    dict_str = pa.dictionary(pa.int32(), pa.string())

    # Build schema dynamically — list fixed-type columns first, append _z columns
    fixed_fields: list[pa.Field] = [
        pa.field("mesh_id_250m",              pa.string(),    nullable=False),
        # OLS features
        pa.field("log_linear_slope",          pa.float64(),   nullable=True),
        pa.field("log_linear_r2",             pa.float64(),   nullable=True),
        pa.field("intercept",                 pa.float64(),   nullable=True),
        pa.field("early_slope",               pa.float64(),   nullable=True),
        pa.field("late_slope",                pa.float64(),   nullable=True),
        # Level features
        pa.field("initial_dose_rate",         pa.float32(),   nullable=True),
        pa.field("final_dose_rate",           pa.float32(),   nullable=True),
        pa.field("max_dose_rate",             pa.float32(),   nullable=True),
        pa.field("max_dose_timing_days",      pa.float32(),   nullable=True),
        # Shape features
        pa.field("n_increases",               pa.float64(),   nullable=True),
        pa.field("recontamination_flag",      pa.bool_(),     nullable=True),
        pa.field("residual_cv",               pa.float64(),   nullable=True),
        # Derived features
        pa.field("total_decay_ratio",         pa.float64(),   nullable=True),
        pa.field("ecological_half_life_y",    pa.float64(),   nullable=True),
        pa.field("slope_ratio",               pa.float64(),   nullable=True),
        pa.field("temporal_span_years",       pa.float64(),   nullable=True),
        pa.field("has_early_data",            pa.bool_(),     nullable=True),
        # Coverage features
        pa.field("n_measurement_dates",       pa.int64(),     nullable=True),
        pa.field("temporal_span_days",        pa.int64(),     nullable=True),
        pa.field("first_measurement_date",    pa.date32(),    nullable=True),
        pa.field("last_measurement_date",     pa.date32(),    nullable=True),
        # Metadata features
        pa.field("measurement_type",          dict_str,       nullable=True),
        pa.field("is_decontaminated",         pa.bool_(),     nullable=True),
        pa.field("height_anomalous_pct",      pa.float64(),   nullable=True),
        pa.field("latitude",                  pa.float32(),   nullable=True),
        pa.field("longitude",                 pa.float32(),   nullable=True),
        pa.field("distance_from_fdnpp_km",    pa.float32(),   nullable=True),
    ]

    # Append scaled _z columns in SOM_FEATURE_COLUMNS order
    z_fields = [
        pa.field(f"{col}_z", pa.float32(), nullable=True)
        for col in SOM_FEATURE_COLUMNS
        if f"{col}_z" in df.columns
    ]

    schema_fields = fixed_fields + z_fields

    # Only include fields whose columns are actually present in df
    schema = pa.schema([f for f in schema_fields if f.name in df.columns])

    # Reorder df columns to match schema
    present_cols = [f.name for f in schema if f.name in df.columns]
    extra_cols = [c for c in df.columns if c not in present_cols]
    df = df[present_cols + extra_cols]

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    pq.write_table(
        table,
        path,
        compression="snappy",
        row_group_size=500_000,
    )
    logger.info("Feature Parquet written: %d rows, %d columns", len(df), len(schema))


def write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write feature_summary.json to disk.

    Args:
        summary: Dict assembled by run_pipeline().
        path: Destination JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Feature summary written to %s", path)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    features: pd.DataFrame,
    norm_params: dict[str, Any],
) -> dict[str, Any]:
    """Execute post-pipeline sanity checks and return results for the summary.

    Checks performed:
    1. Output row count matches EXPECTED_MESH_COUNT.
    2. Percentage of meshes with positive log_linear_slope (should be minority).
    3. Decontaminated cells are present in the output.
    4. Normalised columns are roughly zero-centred (|mean_z| < 0.5).
    5. Log and warn if any pair of SOM features has |Pearson r| > 0.95.

    Args:
        features: Assembled, imputed, and normalised feature DataFrame.
        norm_params: Normalization parameter dict from normalize_features().

    Returns:
        Dict of sanity check results suitable for inclusion in feature_summary.json.
    """
    checks: dict[str, Any] = {}

    # Check 1: row count
    n_rows = len(features)
    checks["row_count"] = {
        "actual": n_rows,
        "expected": EXPECTED_MESH_COUNT,
        "passed": n_rows == EXPECTED_MESH_COUNT,
    }

    # Check 2: positive slope fraction
    pct_positive = float(
        100.0 * (features["log_linear_slope"] > 0).sum() / max(n_rows, 1)
    )
    checks["positive_slope_pct"] = {
        "value": round(pct_positive, 2),
        "note": "Expected minority of meshes to have increasing dose trend",
    }
    logger.info("Sanity check — positive slope: %.1f%% of meshes", pct_positive)

    # Check 3: decontaminated cells
    if "is_decontaminated" in features.columns:
        n_decontam = int(features["is_decontaminated"].sum())
        checks["decontaminated_cells"] = {
            "count": n_decontam,
            "passed": n_decontam > 0,
        }
        if n_decontam == 0:
            logger.warning("Sanity check FAILED — no decontaminated cells in output")
        else:
            logger.info("Sanity check — decontaminated cells: %d", n_decontam)

    # Check 4: normalised columns are roughly zero-centred
    z_col_stats: dict[str, Any] = {}
    for col in SOM_FEATURE_COLUMNS:
        z_col = f"{col}_z"
        if z_col in features.columns:
            mean_z = float(features[z_col].mean())
            std_z = float(features[z_col].std())
            centred_ok = abs(mean_z) < 0.5
            z_col_stats[col] = {
                "mean_z": round(mean_z, 4),
                "std_z": round(std_z, 4),
                "centred_ok": centred_ok,
            }
            if not centred_ok:
                logger.warning(
                    "Sanity check — %s_z mean=%.3f (expected |mean| < 0.5)", col, mean_z
                )
    checks["normalised_column_stats"] = z_col_stats

    # Check 5: correlation matrix
    z_cols = [f"{c}_z" for c in SOM_FEATURE_COLUMNS if f"{c}_z" in features.columns]
    if len(z_cols) >= 2:
        corr_matrix = features[z_cols].corr().round(4)
        corr_dict = corr_matrix.to_dict()

        high_corr_pairs: list[dict[str, Any]] = []
        for i, ci in enumerate(z_cols):
            for j, cj in enumerate(z_cols):
                if j <= i:
                    continue
                r = float(corr_matrix.loc[ci, cj])
                if abs(r) > 0.95:
                    high_corr_pairs.append({"col_a": ci, "col_b": cj, "r": round(r, 4)})
                    logger.warning(
                        "High correlation warning: %s ↔ %s  r=%.3f", ci, cj, r
                    )

        checks["correlation_matrix"] = corr_dict
        checks["high_correlation_pairs"] = high_corr_pairs
        logger.info(
            "Correlation matrix computed for %d SOM features; %d high-|r| pairs found",
            len(z_cols),
            len(high_corr_pairs),
        )

    return checks


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_feature_summary(
    params: dict[str, Any],
    features: pd.DataFrame,
    imputation_counts: dict[str, Any],
    norm_params: dict[str, Any],
    sanity_checks: dict[str, Any],
    timing: dict[str, float],
) -> dict[str, Any]:
    """Assemble the feature_summary.json payload.

    Args:
        params: Pipeline parameters (paths, cutoff dates, flags).
        features: Final assembled, imputed, and normalised feature DataFrame.
        imputation_counts: Per-column imputation counts from impute_nans().
        norm_params: Normalization parameters from normalize_features().
        sanity_checks: Sanity check results from run_sanity_checks().
        timing: Dict mapping step name to wall-clock seconds.

    Returns:
        Dict ready for JSON serialisation via write_summary().
    """
    per_feature_stats: dict[str, Any] = {}
    for col in SOM_FEATURE_COLUMNS:
        if col not in features.columns:
            continue
        vals = features[col].dropna()
        per_feature_stats[col] = {
            "mean": round(float(vals.mean()), 6) if not vals.empty else None,
            "std": round(float(vals.std()), 6) if not vals.empty else None,
            "min": round(float(vals.min()), 6) if not vals.empty else None,
            "max": round(float(vals.max()), 6) if not vals.empty else None,
            "nan_count": int(features[col].isna().sum()),
            "imputed_count": imputation_counts.get(col, {}).get("n_nan_before", 0),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "output_rows": len(features),
        "output_columns": features.shape[1],
        "per_feature_statistics": per_feature_stats,
        "imputation_counts": imputation_counts,
        "normalization_params": norm_params,
        "sanity_checks": sanity_checks,
        "step_timing_seconds": {k: round(v, 2) for k, v in timing.items()},
    }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    early_late_cutoff: date,
    write_csv: bool,
) -> None:
    """Orchestrate all feature engineering steps end to end.

    Steps executed in order:
    1. Load & categorical conversion
    2. Extract metadata features (before deleting source frame)
    3. Daily aggregation (frees source frame)
    4. Vectorised OLS (full + early/late)
    5. Level features
    6. Shape features
    7. Coverage features
    8. Derive has_early_data flag
    9. Assemble feature matrix
    10. Compute derived features
    11. Impute NaN values
    12. Robust scaling
    13. Sanity checks
    14. Write Parquet and JSON summary

    Args:
        input_path: Path to filtered_emdb.parquet.
        output_dir: Directory to write output files.
        early_late_cutoff: Date splitting early vs late OLS period.
        write_csv: If True, also write feature_matrix.csv (utf-8-sig).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # Step 1: Load
    t0 = time.perf_counter()
    df = load_filtered_data(input_path)
    timing["load"] = time.perf_counter() - t0
    n_input_rows = len(df)
    n_input_meshes = df["mesh_id_250m"].nunique()
    logger.info(
        "Input: %d rows, %d unique meshes", n_input_rows, n_input_meshes
    )

    # Step 2a (metadata) must run before df is deleted in aggregate_daily()
    t0 = time.perf_counter()
    meta_df = compute_metadata_features(df)
    timing["metadata"] = time.perf_counter() - t0

    # Step 2: Daily aggregation (also deletes df and gc.collect()s)
    t0 = time.perf_counter()
    daily = aggregate_daily(df)
    # df reference is now dead — do not use after this point
    timing["daily_aggregation"] = time.perf_counter() - t0

    # Step 3: OLS
    t0 = time.perf_counter()
    ols_df = compute_ols_features(daily, cutoff=early_late_cutoff)
    timing["ols"] = time.perf_counter() - t0

    # Step 4: Level features
    t0 = time.perf_counter()
    level_df = compute_level_features(daily)
    timing["level"] = time.perf_counter() - t0

    # Step 5: Shape features
    t0 = time.perf_counter()
    shape_df = compute_shape_features(daily, ols_df)
    timing["shape"] = time.perf_counter() - t0

    # Step 6: Coverage features
    t0 = time.perf_counter()
    coverage_df = compute_coverage_features(daily)
    timing["coverage"] = time.perf_counter() - t0

    # Free daily frame — no longer needed after feature extraction
    del daily
    gc.collect()

    # Step 7: Assemble
    t0 = time.perf_counter()
    features = assemble_feature_matrix(ols_df, level_df, shape_df, coverage_df, meta_df)
    timing["assemble"] = time.perf_counter() - t0

    # Derive has_early_data before compute_derived_features (it needs it for slope_ratio)
    features["has_early_data"] = features["early_slope"].notna()

    # Step 8: Derived features (operates on assembled frame)
    t0 = time.perf_counter()
    features = compute_derived_features(features)
    timing["derived"] = time.perf_counter() - t0

    # Step 9: Imputation
    t0 = time.perf_counter()
    features, imputation_counts = impute_nans(features, SOM_FEATURE_COLUMNS)
    timing["imputation"] = time.perf_counter() - t0

    # Step 10: Normalisation
    t0 = time.perf_counter()
    features, norm_params = normalize_features(features, SOM_FEATURE_COLUMNS)
    timing["normalization"] = time.perf_counter() - t0

    # Sanity checks
    t0 = time.perf_counter()
    sanity_checks = run_sanity_checks(features, norm_params)
    timing["sanity_checks"] = time.perf_counter() - t0

    # Write Parquet
    t0 = time.perf_counter()
    parquet_path = output_dir / "feature_matrix.parquet"
    write_parquet(features, parquet_path)
    timing["write_parquet"] = time.perf_counter() - t0

    # Optionally write CSV
    if write_csv:
        t0 = time.perf_counter()
        csv_path = output_dir / "feature_matrix.csv"
        features.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV written to %s", csv_path)
        timing["write_csv"] = time.perf_counter() - t0

    # Build and write summary
    params: dict[str, Any] = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "early_late_cutoff": str(early_late_cutoff),
        "accident_date": str(ACCIDENT_DATE),
        "half_life_cap_years": HALF_LIFE_CAP_YEARS,
        "ln_floor": LN_FLOOR,
        "recontam_ratio_threshold": RECONTAM_RATIO_THRESHOLD,
        "recontam_abs_threshold": RECONTAM_ABS_THRESHOLD,
        "slope_ratio_guard": SLOPE_RATIO_GUARD,
        "min_points_for_slope": MIN_POINTS_FOR_SLOPE,
        "som_feature_columns": SOM_FEATURE_COLUMNS,
        "n_input_rows": n_input_rows,
        "n_input_meshes": n_input_meshes,
    }

    summary = build_feature_summary(
        params=params,
        features=features,
        imputation_counts=imputation_counts,
        norm_params=norm_params,
        sanity_checks=sanity_checks,
        timing=timing,
    )
    write_summary(summary, output_dir / "feature_summary.json")

    total_time = sum(timing.values())
    logger.info(
        "Pipeline complete in %.1f s — %s",
        total_time,
        parquet_path,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_date(value: str) -> date:
    """Parse an ISO-8601 date string (YYYY-MM-DD) into a date object.

    Args:
        value: Date string in YYYY-MM-DD format.

    Returns:
        Corresponding date object.

    Raises:
        argparse.ArgumentTypeError: If the string is not a valid ISO-8601 date.
    """
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD format."
        ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feature engineering pipeline.

    Returns:
        Namespace with all pipeline configuration attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Temporal feature engineering pipeline for Fukushima dose-rate SOM analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="output/filtered_emdb.parquet",
        type=str,
        metavar="PARQUET",
        help="Path to filtered_emdb.parquet produced by filter_emdb.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        metavar="DIR",
        help="Directory to write feature_matrix.parquet and feature_summary.json.",
    )
    parser.add_argument(
        "--early-late-cutoff",
        default="2013-03-11",
        type=_parse_date,
        metavar="DATE",
        help=(
            "ISO-8601 date (YYYY-MM-DD) splitting early and late OLS periods. "
            "Default is one Cs-134 half-life after the accident."
        ),
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write output/feature_matrix.csv (utf-8-sig, Excel-compatible).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load input data, report shape and column info, then exit without computing features.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging to console.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Script entry point."""
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        verbose=args.verbose,
        log_file=output_dir / "feature_engineering.log",
    )

    logger.info("engineer_features.py starting")
    logger.info("  input       : %s", args.input)
    logger.info("  output_dir  : %s", output_dir)
    logger.info("  cutoff date : %s", args.early_late_cutoff)
    logger.info("  csv output  : %s", args.csv)
    logger.info("  dry run     : %s", args.dry_run)

    input_path = Path(args.input).resolve()

    if args.dry_run:
        df = load_filtered_data(input_path)
        logger.info("Dry run — input shape: %d rows x %d columns", *df.shape)
        logger.info("Unique meshes: %d", df["mesh_id_250m"].nunique())
        logger.info("Columns: %s", list(df.columns))
        logger.info("Date range: %s → %s",
                    df["correction_date"].min(), df["correction_date"].max())
        logger.info("Dry run complete. Exiting.")
        return

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        early_late_cutoff=args.early_late_cutoff,
        write_csv=args.csv,
    )

    logger.info("engineer_features.py complete.")


if __name__ == "__main__":
    main()
