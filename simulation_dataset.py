from __future__ import annotations

# -----------------------------------------------------------------------------
# Simulation learning dataset builder
# -----------------------------------------------------------------------------
# This module builds a 50 m-aligned learning dataset from historical FIT-derived
# runner feature tables.
#
# Version 1 goal:
#   - resample each historical activity onto a fixed distance grid,
#   - interpolate the available numeric runner/terrain features,
#   - derive a meaningful segment_duration_s target on that grid.
# -----------------------------------------------------------------------------

from typing import Any, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Column ordering
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
# Helpers
# -----------------------------------------------------------------------------

def _default_activity_name(activity_index: int) -> str:
    return f"activity_{activity_index:03d}"


def _order_columns(columns: Sequence[str]) -> list[str]:
    columns = list(columns)
    preferred = [col for col in PREFERRED_COLUMN_ORDER if col in columns]
    remaining = [col for col in columns if col not in preferred]
    return preferred + remaining


def _build_target_distances(max_distance_m: float, segment_length_m: float) -> list[float]:
    if segment_length_m <= 0:
        raise ValueError("segment_length_m must be greater than 0")

    max_distance_m = max(0.0, float(max_distance_m))

    if max_distance_m == 0.0:
        return [0.0]

    targets = list(np.arange(0.0, max_distance_m, segment_length_m))
    if not targets or abs(targets[-1] - max_distance_m) > 1e-9:
        targets.append(max_distance_m)

    targets = sorted(set(round(float(t), 6) for t in targets))
    return targets


def _interpolate_numeric(
    source_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target_x: list[float],
) -> pd.Series:
    if x_column not in source_df.columns or y_column not in source_df.columns:
        return pd.Series([np.nan] * len(target_x), index=target_x, dtype="float64")

    source = source_df[[x_column, y_column]].copy()
    source[x_column] = pd.to_numeric(source[x_column], errors="coerce")
    source[y_column] = pd.to_numeric(source[y_column], errors="coerce")
    source = source.dropna(subset=[x_column, y_column])
    source = source.drop_duplicates(subset=[x_column], keep="last").sort_values(x_column)

    if source.empty:
        return pd.Series([np.nan] * len(target_x), index=target_x, dtype="float64")

    x_vals = source[x_column].to_numpy(dtype=float)
    y_vals = source[y_column].to_numpy(dtype=float)
    interp_vals = np.interp(np.asarray(target_x, dtype=float), x_vals, y_vals)

    return pd.Series(interp_vals, index=target_x, dtype="float64")


def _interpolate_datetime(
    source_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target_x: list[float],
) -> pd.Series:
    if x_column not in source_df.columns or y_column not in source_df.columns:
        return pd.Series([pd.NaT] * len(target_x), index=target_x)

    source = source_df[[x_column, y_column]].copy()
    source[x_column] = pd.to_numeric(source[x_column], errors="coerce")
    source[y_column] = pd.to_datetime(source[y_column], errors="coerce")
    source = source.dropna(subset=[x_column, y_column])
    source = source.drop_duplicates(subset=[x_column], keep="last").sort_values(x_column)

    if source.empty:
        return pd.Series([pd.NaT] * len(target_x), index=target_x)

    x_vals = source[x_column].to_numpy(dtype=float)
    y_vals = source[y_column].map(lambda v: v.value if pd.notna(v) else np.nan).to_numpy(dtype=float)

    interp_vals = np.interp(np.asarray(target_x, dtype=float), x_vals, y_vals)
    interp_vals = np.rint(interp_vals).astype("int64")

    return pd.to_datetime(interp_vals, unit="ns", errors="coerce")


def _ensure_distance_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that a usable cumulative distance axis exists.

    Priority:
      1) distance_from_start_m
      2) reconstructed from distance_delta_m
      3) synthetic row-order axis as a last resort
    """
    out = df.copy()

    if "distance_from_start_m" in out.columns:
        out["distance_from_start_m"] = pd.to_numeric(out["distance_from_start_m"], errors="coerce")

        if out["distance_from_start_m"].notna().any():
            return out

    # -------------------------------------------------------------------------
    # Reconstruct from delta distance if possible
    # -------------------------------------------------------------------------
    if "distance_delta_m" in out.columns:
        delta = pd.to_numeric(out["distance_delta_m"], errors="coerce").fillna(0.0)
        reconstructed = delta.cumsum()
        out["distance_from_start_m"] = reconstructed
        if out["distance_from_start_m"].notna().any():
            return out

    # -------------------------------------------------------------------------
    # Final fallback: synthetic axis from row order
    # -------------------------------------------------------------------------
    out["distance_from_start_m"] = np.arange(len(out), dtype=float)
    return out


def _prepare_activity_frame(
    activity_df: pd.DataFrame,
    activity_id: int,
    activity_name: str,
    segment_length_m: float,
) -> pd.DataFrame:
    if activity_df is None or activity_df.empty:
        return pd.DataFrame()

    df = activity_df.copy()

    # -------------------------------------------------------------------------
    # Standardize key axes
    # -------------------------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = _ensure_distance_axis(df)

    df["distance_from_start_m"] = pd.to_numeric(df["distance_from_start_m"], errors="coerce")
    df = df.dropna(subset=["distance_from_start_m"])

    if df.empty:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Sort trajectory and remove duplicate distances
    # -------------------------------------------------------------------------
    sort_columns: list[str] = ["distance_from_start_m"]
    if "time_from_start_s" in df.columns:
        df["time_from_start_s"] = pd.to_numeric(df["time_from_start_s"], errors="coerce")
        sort_columns.append("time_from_start_s")
    elif "timestamp" in df.columns:
        sort_columns.append("timestamp")

    df = df.sort_values(sort_columns, kind="mergesort")
    df = df.drop_duplicates(subset=["distance_from_start_m"], keep="last").reset_index(drop=True)

    max_distance_m = float(df["distance_from_start_m"].max())
    target_distances = _build_target_distances(max_distance_m, segment_length_m)

    out = pd.DataFrame({"distance_from_start_m": target_distances})

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    out["activity_id"] = activity_id
    out["activity_name"] = activity_name
    out["sample_id"] = range(1, len(out) + 1)

    # -------------------------------------------------------------------------
    # Interpolate core time axis
    # -------------------------------------------------------------------------
    if "time_from_start_s" in df.columns and df["time_from_start_s"].notna().any():
        out["time_from_start_s"] = _interpolate_numeric(
            df,
            "distance_from_start_m",
            "time_from_start_s",
            target_distances,
        )

    elif "timestamp" in df.columns and df["timestamp"].notna().any():
        interpolated_ts = _interpolate_datetime(
            df,
            "distance_from_start_m",
            "timestamp",
            target_distances,
        )
        out["timestamp"] = interpolated_ts

        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        if ts.notna().any():
            out["time_from_start_s"] = (ts - ts.iloc[0]).dt.total_seconds()
        else:
            out["time_from_start_s"] = np.arange(len(out), dtype=float)

    else:
        # Final fallback: keep the row order as a synthetic time axis.
        out["time_from_start_s"] = np.arange(len(out), dtype=float)

    # If timestamp exists, keep it too.
    if "timestamp" in df.columns:
        out["timestamp"] = _interpolate_datetime(
            df,
            "distance_from_start_m",
            "timestamp",
            target_distances,
        )

    # -------------------------------------------------------------------------
    # Interpolate every useful numeric column available in the source.
    # -------------------------------------------------------------------------
    ignored_columns = {
        "activity_id",
        "activity_name",
        "sample_id",
        "timestamp",
        "distance_from_start_m",
        "time_from_start_s",
        "segment_duration_s",
        "segment_distance_m",
        "distance_delta_m",
    }

    numeric_columns = [
        col
        for col in df.columns
        if col not in ignored_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in numeric_columns:
        out[col] = _interpolate_numeric(
            df,
            "distance_from_start_m",
            col,
            target_distances,
        )

    # -------------------------------------------------------------------------
    # Derived trajectory columns
    # -------------------------------------------------------------------------
    out["time_from_start_s"] = pd.to_numeric(out["time_from_start_s"], errors="coerce")
    out["segment_duration_s"] = out["time_from_start_s"].diff()
    out["segment_distance_m"] = out["distance_from_start_m"].diff()
    out["distance_delta_m"] = out["segment_distance_m"]

    # -------------------------------------------------------------------------
    # Remove rows that cannot be used for learning
    # -------------------------------------------------------------------------
    out = out.dropna(subset=["segment_duration_s"])

    # -------------------------------------------------------------------------
    # Derive terrain features consistently from the interpolated altitude
    # -------------------------------------------------------------------------
    if "altitude_m" in out.columns:
        out["altitude_delta_m"] = out["altitude_m"].diff()
        out["ascent_delta_m"] = out["altitude_delta_m"].clip(lower=0.0)
        out["descent_delta_m"] = (-out["altitude_delta_m"].clip(upper=0.0))
        out["ascent_cumul_from_start_m"] = out["ascent_delta_m"].fillna(0.0).cumsum()
        out["descent_cumul_from_start_m"] = out["descent_delta_m"].fillna(0.0).cumsum()

        segment_distance = out["segment_distance_m"].replace(0.0, np.nan)
        out["grade_pct"] = (out["altitude_delta_m"] / segment_distance) * 100.0
    else:
        out["altitude_delta_m"] = np.nan
        out["ascent_delta_m"] = np.nan
        out["descent_delta_m"] = np.nan
        out["ascent_cumul_from_start_m"] = np.nan
        out["descent_cumul_from_start_m"] = np.nan
        out["grade_pct"] = np.nan

    # -------------------------------------------------------------------------
    # Final ordering
    # -------------------------------------------------------------------------
    out = out[_order_columns(out.columns)]

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_simulation_dataset(
    activities: Sequence[pd.DataFrame],
    activity_names: Sequence[str] | None = None,
    segment_length_m: float = 50.0,
) -> pd.DataFrame:
    """
    Build a 50 m-aligned learning dataframe from multiple historical activities.
    """
    if activities is None:
        return pd.DataFrame()

    activities = list(activities)
    if len(activities) == 0:
        return pd.DataFrame()

    if activity_names is not None and len(activity_names) != len(activities):
        raise ValueError("activity_names must have the same length as activities.")

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
            segment_length_m=segment_length_m,
        )

        if not prepared.empty:
            prepared_frames.append(prepared)

    if not prepared_frames:
        return pd.DataFrame()

    dataset = pd.concat(prepared_frames, ignore_index=True)
    dataset = dataset[_order_columns(dataset.columns)]

    return dataset


def summarize_simulation_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """
    Compact summary of the 50 m-aligned learning dataset.
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

    return summary
    
