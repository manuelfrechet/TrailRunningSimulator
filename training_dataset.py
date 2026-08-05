from __future__ import annotations

# -----------------------------------------------------------------------------
# Training dataset builder
# -----------------------------------------------------------------------------
# This module has one responsibility only:
#   - take multiple historical FIT-derived activity dataframes,
#   - add lightweight activity metadata,
#   - preserve the trajectory structure,
#   - and prepare one unified learning table.
#
# Version 1 is intentionally simple:
#   - no learning here,
#   - no simulation here,
#   - no model fitting here.
# -----------------------------------------------------------------------------

from typing import Any, Sequence

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Column ordering
# -----------------------------------------------------------------------------
# We keep the most useful columns near the front while preserving any remaining
# columns from the incoming activity dataframes.
# -----------------------------------------------------------------------------

PREFERRED_COLUMN_ORDER = [
    "activity_id",
    "activity_name",
    "sample_id",
    "timestamp",
    "time_from_start_s",
    "segment_duration_s",
    "distance_from_start_m",
    "segment_distance_m",
    "distance_delta_m",
    "altitude_m",
    "altitude_delta_m",
    "ascent_delta_m",
    "descent_delta_m",
    "ascent_cumul_from_start_m",
    "descent_cumul_from_start_m",
    "grade_pct",
    "heart_rate_bpm",
    "power",
    "cadence_spm",
    "speed_m_s",
    "step_length_m",
    "vertical_oscillation_mm",
    "stance_time_s",
    "accumulated_power",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _default_activity_name(activity_index: int) -> str:
    """
    Return a stable default name when the caller does not provide one.
    """
    return f"activity_{activity_index:03d}"


def _order_columns(columns: Sequence[str]) -> list[str]:
    """
    Keep preferred columns first, then preserve the remaining columns in their
    existing order.
    """
    columns = list(columns)
    preferred = [col for col in PREFERRED_COLUMN_ORDER if col in columns]
    remaining = [col for col in columns if col not in preferred]
    return preferred + remaining


def _sort_columns_for_trajectory(df: pd.DataFrame) -> list[str]:
    """
    Choose the most stable sort key for the activity trajectory.

    Priority:
      1) time_from_start_s
      2) timestamp
      3) original row order (no explicit sort key)
    """
    sort_columns: list[str] = []

    if "time_from_start_s" in df.columns:
        sort_columns.append("time_from_start_s")

    if "timestamp" in df.columns:
        # Keep timestamp as a tie-breaker when available.
        sort_columns.append("timestamp")

    return sort_columns


def _prepare_activity_frame(
    activity_df: pd.DataFrame,
    activity_id: int,
    activity_name: str,
) -> pd.DataFrame:
    """
    Add activity metadata and compute segment-level targets for one activity.

    The first row of each activity has no previous segment, so the segment
    duration/distance targets are left as NaN there.
    """
    if activity_df is None or activity_df.empty:
        return pd.DataFrame()

    df = activity_df.copy()

    # -------------------------------------------------------------------------
    # Standardize time and numeric fields we rely on for sorting and targets.
    # -------------------------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "time_from_start_s" in df.columns:
        df["time_from_start_s"] = pd.to_numeric(df["time_from_start_s"], errors="coerce")

    if "distance_from_start_m" in df.columns:
        df["distance_from_start_m"] = pd.to_numeric(df["distance_from_start_m"], errors="coerce")

    # -------------------------------------------------------------------------
    # Add activity metadata.
    # -------------------------------------------------------------------------
    df["activity_id"] = activity_id
    df["activity_name"] = activity_name
    df["sample_id"] = range(1, len(df) + 1)

    # -------------------------------------------------------------------------
    # Sort the trajectory so all target computations follow the same order.
    # -------------------------------------------------------------------------
    sort_columns = _sort_columns_for_trajectory(df)
    if sort_columns:
        df = df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    # Rebuild sample_id after sorting so it always matches trajectory order.
    df["sample_id"] = range(1, len(df) + 1)

    # -------------------------------------------------------------------------
    # Segment duration target
    # -------------------------------------------------------------------------
    # This is the first target we need for system identification:
    # duration of the current step relative to the previous step.
    # -------------------------------------------------------------------------
    if "time_from_start_s" in df.columns:
        df["segment_duration_s"] = df["time_from_start_s"].diff()
    elif "timestamp" in df.columns:
        df["segment_duration_s"] = df["timestamp"].diff().dt.total_seconds()
    else:
        df["segment_duration_s"] = np.nan

    # -------------------------------------------------------------------------
    # Segment distance target
    # -------------------------------------------------------------------------
    # This is useful for debugging and for later analysis of step size.
    # -------------------------------------------------------------------------
    if "distance_from_start_m" in df.columns:
        df["segment_distance_m"] = df["distance_from_start_m"].diff()
    else:
        df["segment_distance_m"] = np.nan

    # -------------------------------------------------------------------------
    # Final column ordering
    # -------------------------------------------------------------------------
    df = df[_order_columns(df.columns)]

    return df


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_training_dataset(
    activities: Sequence[pd.DataFrame],
    activity_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Build one unified learning dataframe from multiple historical activities.

    Parameters
    ----------
    activities:
        Sequence of activity dataframes already produced by the FIT feature
        pipeline.
    activity_names:
        Optional sequence of human-readable names, typically file names.
        If omitted, default names are generated.

    Returns
    -------
    pd.DataFrame
        One concatenated trajectory table with activity metadata and segment
        targets.
    """
    # -------------------------------------------------------------------------
    # Guard clauses
    # -------------------------------------------------------------------------
    if activities is None:
        return pd.DataFrame()

    activities = list(activities)

    if len(activities) == 0:
        return pd.DataFrame()

    if activity_names is not None and len(activity_names) != len(activities):
        raise ValueError("activity_names must have the same length as activities.")

    # -------------------------------------------------------------------------
    # Build one cleaned dataframe per activity
    # -------------------------------------------------------------------------
    prepared_frames: list[pd.DataFrame] = []

    for idx, activity_df in enumerate(activities, start=1):
        activity_name = (
            activity_names[idx - 1]
            if activity_names is not None
            else _default_activity_name(idx)
        )

        prepared = _prepare_activity_frame(
            activity_df=activity_df,
            activity_id=idx,
            activity_name=activity_name,
        )

        # Skip empty activities quietly. They do not contribute training rows.
        if not prepared.empty:
            prepared_frames.append(prepared)

    # -------------------------------------------------------------------------
    # Concatenate all activities into one learning table
    # -------------------------------------------------------------------------
    if not prepared_frames:
        return pd.DataFrame()

    dataset = pd.concat(prepared_frames, ignore_index=True)

    # -------------------------------------------------------------------------
    # Make sure the final dataset is easy to inspect and stable to use
    # -------------------------------------------------------------------------
    dataset = dataset[_order_columns(dataset.columns)]

    return dataset


def summarize_training_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """
    Return a compact summary of the training dataset.

    This is useful for the Streamlit UI and for sanity checks.
    """
    if dataset is None or dataset.empty:
        return {
            "n_activities": 0,
            "n_rows": 0,
            "n_rows_with_segment_duration": 0,
            "n_rows_missing_segment_duration": 0,
            "n_columns": 0,
        }

    summary: dict[str, Any] = {
        "n_activities": int(dataset["activity_id"].nunique()) if "activity_id" in dataset.columns else 0,
        "n_rows": int(len(dataset)),
        "n_rows_with_segment_duration": int(dataset["segment_duration_s"].notna().sum())
        if "segment_duration_s" in dataset.columns
        else 0,
        "n_rows_missing_segment_duration": int(dataset["segment_duration_s"].isna().sum())
        if "segment_duration_s" in dataset.columns
        else 0,
        "n_columns": int(dataset.shape[1]),
    }

    if "activity_name" in dataset.columns:
        summary["activity_names"] = sorted(
            [str(v) for v in dataset["activity_name"].dropna().unique().tolist()]
        )

    return summary


def activity_summary_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Return one summary row per activity.

    This is useful for debugging and for a quick overview in the UI.
    """
    if dataset is None or dataset.empty or "activity_id" not in dataset.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for activity_id, group in dataset.groupby("activity_id", sort=True):
        activity_name = (
            str(group["activity_name"].iloc[0])
            if "activity_name" in group.columns and not group["activity_name"].empty
            else f"activity_{activity_id:03d}"
        )

        row: dict[str, Any] = {
            "activity_id": int(activity_id),
            "activity_name": activity_name,
            "n_rows": int(len(group)),
            "n_rows_with_segment_duration": int(group["segment_duration_s"].notna().sum())
            if "segment_duration_s" in group.columns
            else 0,
            "total_duration_s": float(group["segment_duration_s"].sum(skipna=True))
            if "segment_duration_s" in group.columns
            else np.nan,
            "total_distance_m": float(group["distance_from_start_m"].max())
            if "distance_from_start_m" in group.columns and group["distance_from_start_m"].notna().any()
            else np.nan,
            "start_timestamp": group["timestamp"].min()
            if "timestamp" in group.columns
            else pd.NaT,
            "end_timestamp": group["timestamp"].max()
            if "timestamp" in group.columns
            else pd.NaT,
        }

        rows.append(row)

    return pd.DataFrame(rows)
  
