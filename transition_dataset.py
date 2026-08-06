from __future__ import annotations

# -----------------------------------------------------------------------------
# Transition dataset builder
# -----------------------------------------------------------------------------
# This module converts historical FIT-derived activity tables into a large
# transition dataset suitable for learning a dynamical system.
#
# Core idea:
#   For each historical activity, build a dense distance grid (default 1 m),
#   then extract rolling transitions over a fixed horizon (default 10 m):
#
#       state at distance d  ->  state at distance d + 10 m
#
# This produces many more training samples than the previous coarse approach.
# -----------------------------------------------------------------------------

from typing import Any, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_GRID_STEP_M = 1.0
DEFAULT_TRANSITION_HORIZON_M = 10.0

STATE_TARGET_COLUMNS = [
    "heart_rate_bpm",
    "power",
    "cadence_spm",
    "speed_m_s",
    "step_length_m",
    "vertical_oscillation_mm",
    "stance_time_s",
    "accumulated_power",
]

PREFERRED_COLUMN_ORDER = [
    "activity_id",
    "activity_name",
    "sample_id",
    "distance_from_start_m",
    "time_from_start_s",
    "timestamp",
    "transition_horizon_m",
    "segment_distance_m",
    "segment_duration_s",
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
    "next_heart_rate_bpm",
    "next_power",
    "next_cadence_spm",
    "next_speed_m_s",
    "next_step_length_m",
    "next_vertical_oscillation_mm",
    "next_stance_time_s",
    "next_accumulated_power",
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


def _validate_window_params(
    grid_step_m: float,
    transition_horizon_m: float,
) -> int:
    """
    Validate the rolling-window parameters.

    Returns the horizon in grid steps.
    """
    if grid_step_m <= 0:
        raise ValueError("grid_step_m must be greater than 0")
    if transition_horizon_m <= 0:
        raise ValueError("transition_horizon_m must be greater than 0")

    horizon_steps = int(round(transition_horizon_m / grid_step_m))
    if horizon_steps <= 0:
        raise ValueError("transition_horizon_m must be at least one grid step")

    reconstructed_horizon = horizon_steps * grid_step_m
    if not np.isclose(reconstructed_horizon, transition_horizon_m, atol=1e-9):
        raise ValueError(
            "transition_horizon_m must be an integer multiple of grid_step_m"
        )

    return horizon_steps


def _ensure_distance_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a usable cumulative distance axis exists.

    Priority:
      1) distance_from_start_m
      2) reconstructed from distance_delta_m
      3) synthetic row-order axis as a fallback
    """
    out = df.copy()

    if "distance_from_start_m" in out.columns:
        out["distance_from_start_m"] = pd.to_numeric(
            out["distance_from_start_m"], errors="coerce"
        )
        if out["distance_from_start_m"].notna().any():
            return out

    if "distance_delta_m" in out.columns:
        delta = pd.to_numeric(out["distance_delta_m"], errors="coerce").fillna(0.0)
        out["distance_from_start_m"] = delta.cumsum()
        if out["distance_from_start_m"].notna().any():
            return out

    out["distance_from_start_m"] = np.arange(len(out), dtype=float)
    return out


def _prepare_source_activity(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and sort the source activity trajectory before interpolation.
    """
    if activity_df is None or activity_df.empty:
        return pd.DataFrame()

    df = activity_df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = _ensure_distance_axis(df)
    df["distance_from_start_m"] = pd.to_numeric(df["distance_from_start_m"], errors="coerce")
    df = df.dropna(subset=["distance_from_start_m"])

    if df.empty:
        return pd.DataFrame()

    sort_columns: list[str] = ["distance_from_start_m"]
    if "time_from_start_s" in df.columns:
        df["time_from_start_s"] = pd.to_numeric(df["time_from_start_s"], errors="coerce")
        sort_columns.append("time_from_start_s")
    elif "timestamp" in df.columns:
        sort_columns.append("timestamp")

    df = df.sort_values(sort_columns, kind="mergesort")
    df = df.drop_duplicates(subset=["distance_from_start_m"], keep="last").reset_index(drop=True)

    # Shift to a zero-based origin so the transition dataset starts at 0 m / 0 s.
    df["distance_from_start_m"] = df["distance_from_start_m"] - float(df["distance_from_start_m"].iloc[0])

    if "time_from_start_s" in df.columns and df["time_from_start_s"].notna().any():
        df["time_from_start_s"] = pd.to_numeric(df["time_from_start_s"], errors="coerce")
        df["time_from_start_s"] = df["time_from_start_s"] - float(df["time_from_start_s"].iloc[0])

    return df


def _interpolate_numeric(
    source_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target_x: np.ndarray,
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
    interp_vals = np.interp(target_x.astype(float), x_vals, y_vals)

    return pd.Series(interp_vals, index=target_x, dtype="float64")


def _interpolate_datetime(
    source_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    target_x: np.ndarray,
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

    interp_vals = np.interp(target_x.astype(float), x_vals, y_vals)
    interp_vals = np.rint(interp_vals).astype("int64")

    return pd.to_datetime(interp_vals, unit="ns", errors="coerce")


def _build_dense_grid(
    source_df: pd.DataFrame,
    grid_step_m: float,
) -> pd.DataFrame:
    """
    Build a dense 1 m (or user-defined) distance grid for one historical activity.
    """
    if source_df is None or source_df.empty:
        return pd.DataFrame()

    max_distance_m = float(source_df["distance_from_start_m"].max())
    if max_distance_m <= 0:
        return pd.DataFrame()

    dense_distances = np.arange(0.0, max_distance_m + grid_step_m * 0.5, grid_step_m)

    dense = pd.DataFrame({"distance_from_start_m": dense_distances})

    # -------------------------------------------------------------------------
    # Interpolate time axis
    # -------------------------------------------------------------------------
    if "time_from_start_s" in source_df.columns and source_df["time_from_start_s"].notna().any():
        dense["time_from_start_s"] = _interpolate_numeric(
            source_df,
            "distance_from_start_m",
            "time_from_start_s",
            dense_distances,
        )
    elif "timestamp" in source_df.columns and source_df["timestamp"].notna().any():
        dense["timestamp"] = _interpolate_datetime(
            source_df,
            "distance_from_start_m",
            "timestamp",
            dense_distances,
        )
        ts = pd.to_datetime(dense["timestamp"], errors="coerce")
        if ts.notna().any():
            dense["time_from_start_s"] = (ts - ts.iloc[0]).dt.total_seconds()
        else:
            dense["time_from_start_s"] = np.arange(len(dense), dtype=float)
    else:
        dense["time_from_start_s"] = np.arange(len(dense), dtype=float)

    if "timestamp" in source_df.columns and source_df["timestamp"].notna().any():
        dense["timestamp"] = _interpolate_datetime(
            source_df,
            "distance_from_start_m",
            "timestamp",
            dense_distances,
        )

    # -------------------------------------------------------------------------
    # Interpolate useful numeric source columns
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
        "altitude_delta_m",
        "ascent_delta_m",
        "descent_delta_m",
        "ascent_cumul_from_start_m",
        "descent_cumul_from_start_m",
        "grade_pct",
        "next_heart_rate_bpm",
        "next_power",
        "next_cadence_spm",
        "next_speed_m_s",
        "next_step_length_m",
        "next_vertical_oscillation_mm",
        "next_stance_time_s",
        "next_accumulated_power",
        "transition_horizon_m",
    }

    numeric_columns = [
        col
        for col in source_df.columns
        if col not in ignored_columns and pd.api.types.is_numeric_dtype(source_df[col])
    ]

    for col in numeric_columns:
        dense[col] = _interpolate_numeric(
            source_df,
            "distance_from_start_m",
            col,
            dense_distances,
        )

    # -------------------------------------------------------------------------
    # Derive cumulative terrain load on the dense grid
    # -------------------------------------------------------------------------
    if "altitude_m" in dense.columns:
        dense["altitude_m"] = pd.to_numeric(dense["altitude_m"], errors="coerce")
        dense["altitude_delta_m_local"] = dense["altitude_m"].diff().fillna(0.0)
        dense["ascent_step_m"] = dense["altitude_delta_m_local"].clip(lower=0.0)
        dense["descent_step_m"] = (-dense["altitude_delta_m_local"].clip(upper=0.0))
        dense["ascent_cumul_from_start_m"] = dense["ascent_step_m"].cumsum()
        dense["descent_cumul_from_start_m"] = dense["descent_step_m"].cumsum()
    else:
        dense["altitude_delta_m_local"] = np.nan
        dense["ascent_step_m"] = np.nan
        dense["descent_step_m"] = np.nan
        dense["ascent_cumul_from_start_m"] = np.nan
        dense["descent_cumul_from_start_m"] = np.nan

    # -------------------------------------------------------------------------
    # Derive speed if missing and time is available
    # -------------------------------------------------------------------------
    if "speed_m_s" not in dense.columns or dense["speed_m_s"].notna().sum() == 0:
        time_delta = dense["time_from_start_s"].diff()
        distance_delta = dense["distance_from_start_m"].diff()
        dense["speed_m_s"] = distance_delta / time_delta.replace(0.0, np.nan)

    return dense


def _prepare_transition_frame(
    activity_df: pd.DataFrame,
    activity_id: int,
    activity_name: str,
    grid_step_m: float,
    transition_horizon_m: float,
) -> pd.DataFrame:
    """
    Convert one activity into a rolling transition dataset.
    """
    source = _prepare_source_activity(activity_df)
    if source.empty:
        return pd.DataFrame()

    horizon_steps = _validate_window_params(grid_step_m, transition_horizon_m)

    dense = _build_dense_grid(source, grid_step_m)
    if dense.empty or len(dense) <= horizon_steps:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Create rolling current-state and future-state views
    # -------------------------------------------------------------------------
    current = dense.iloc[:-horizon_steps].reset_index(drop=True)
    future = dense.iloc[horizon_steps:].reset_index(drop=True)

    out = current.copy()

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    out["activity_id"] = activity_id
    out["activity_name"] = activity_name
    out["sample_id"] = range(1, len(out) + 1)
    out["transition_horizon_m"] = float(transition_horizon_m)

    # -------------------------------------------------------------------------
    # Segment forcing over the horizon
    # -------------------------------------------------------------------------
    out["segment_distance_m"] = float(transition_horizon_m)
    out["distance_delta_m"] = float(transition_horizon_m)

    if "altitude_m" in out.columns and "altitude_m" in future.columns:
        out["altitude_delta_m"] = future["altitude_m"].to_numpy(dtype=float) - out["altitude_m"].to_numpy(dtype=float)
        out["ascent_delta_m"] = out["altitude_delta_m"].clip(lower=0.0)
        out["descent_delta_m"] = (-out["altitude_delta_m"].clip(upper=0.0))
        out["grade_pct"] = (out["altitude_delta_m"] / float(transition_horizon_m)) * 100.0
    else:
        out["altitude_delta_m"] = np.nan
        out["ascent_delta_m"] = np.nan
        out["descent_delta_m"] = np.nan
        out["grade_pct"] = np.nan

    # -------------------------------------------------------------------------
    # Segment duration target
    # -------------------------------------------------------------------------
    out["segment_duration_s"] = future["time_from_start_s"].to_numpy(dtype=float) - out["time_from_start_s"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Next-state targets for the learned dynamical system
    # -------------------------------------------------------------------------
    for col in STATE_TARGET_COLUMNS:
        if col in out.columns and col in future.columns:
            out[f"next_{col}"] = future[col].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Clean invalid rows
    # -------------------------------------------------------------------------
    out["segment_duration_s"] = pd.to_numeric(out["segment_duration_s"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["segment_duration_s"])
    out = out[out["segment_duration_s"] > 0.0]

    # -------------------------------------------------------------------------
    # Final ordering
    # -------------------------------------------------------------------------
    out = out[_order_columns(out.columns)]

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_transition_dataset(
    activities: Sequence[pd.DataFrame],
    activity_names: Sequence[str] | None = None,
    grid_step_m: float = DEFAULT_GRID_STEP_M,
    transition_horizon_m: float = DEFAULT_TRANSITION_HORIZON_M,
) -> pd.DataFrame:
    """
    Build a large rolling transition dataset from multiple historical activities.
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

        prepared = _prepare_transition_frame(
            activity_df=activity_df,
            activity_id=idx,
            activity_name=activity_name,
            grid_step_m=grid_step_m,
            transition_horizon_m=transition_horizon_m,
        )

        if not prepared.empty:
            prepared_frames.append(prepared)

    if not prepared_frames:
        return pd.DataFrame()

    dataset = pd.concat(prepared_frames, ignore_index=True)
    dataset = dataset[_order_columns(dataset.columns)]

    return dataset


def summarize_transition_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """
    Compact summary of the rolling transition dataset.
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

    if "segment_duration_s" in dataset.columns and dataset["segment_duration_s"].notna().any():
        summary["mean_segment_duration_s"] = float(dataset["segment_duration_s"].mean(skipna=True))
        summary["median_segment_duration_s"] = float(dataset["segment_duration_s"].median(skipna=True))

    if "transition_horizon_m" in dataset.columns and dataset["transition_horizon_m"].notna().any():
        summary["transition_horizon_m"] = float(dataset["transition_horizon_m"].iloc[0])

    return summary
  
