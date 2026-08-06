from __future__ import annotations

# -----------------------------------------------------------------------------
# Replay engine
# -----------------------------------------------------------------------------
# This module replays a historical FIT-derived activity from start to finish,
# reconstructs a hidden state at each dense distance step, and can then extract
# rolling transition samples.
#
# Version 1 principles:
#   - replay the activity on a dense distance grid (default 1 m),
#   - reconstruct hidden state sequentially from all prior history,
#   - expose hidden state variables explicitly,
#   - build rolling transition samples over a fixed horizon (default 10 m).
#
# The code is intentionally transparent and configurable so it can later be used
# as the foundation for system identification.
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_GRID_STEP_M = 1.0
DEFAULT_TRANSITION_HORIZON_M = 10.0

OBSERVED_STATE_COLUMNS = [
    "heart_rate_bpm",
    "power",
    "cadence_spm",
    "speed_m_s",
    "step_length_m",
    "vertical_oscillation_mm",
    "stance_time_s",
    "accumulated_power",
]

TERRAIN_COLUMNS = [
    "distance_from_start_m",
    "time_from_start_s",
    "timestamp",
    "altitude_m",
    "altitude_delta_m",
    "ascent_delta_m",
    "descent_delta_m",
    "ascent_cumul_from_start_m",
    "descent_cumul_from_start_m",
    "grade_pct",
]

HIDDEN_STATE_COLUMNS = [
    "current_hr_zone",
    "cardiovascular_debt",
    "mechanical_debt",
    "neuromuscular_debt",
]

ZONE_TIME_COLUMNS = [f"time_in_zone_{i}" for i in range(1, 7)]
ZONE_FRACTION_COLUMNS = [f"fraction_time_in_zone_{i}" for i in range(1, 7)]
ZONE_CONTINUOUS_COLUMNS = [f"continuous_time_spend_in_zone_{i}" for i in range(1, 7)]

REPLAY_OUTPUT_COLUMNS = (
    TERRAIN_COLUMNS
    + OBSERVED_STATE_COLUMNS
    + HIDDEN_STATE_COLUMNS
    + ZONE_TIME_COLUMNS
    + ZONE_FRACTION_COLUMNS
    + ZONE_CONTINUOUS_COLUMNS
)

TRANSITION_TARGET_COLUMNS = (
    TERRAIN_COLUMNS
    + OBSERVED_STATE_COLUMNS
    + HIDDEN_STATE_COLUMNS
    + ZONE_TIME_COLUMNS
    + ZONE_FRACTION_COLUMNS
    + ZONE_CONTINUOUS_COLUMNS
)


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class ReplayConfig:
    """
    Configuration for replaying one historical activity.
    """
    grid_step_m: float = DEFAULT_GRID_STEP_M
    transition_horizon_m: float = DEFAULT_TRANSITION_HORIZON_M

    # If not provided in the activity, these are used as defaults.
    hr_rest: float | None = None
    hr_max: float | None = None
    terrain_technicality: float = 3.0

    # HR zone thresholds expressed as a fraction of HR reserve.
    # Example: 0.60 means HR_rest + 60% of reserve.
    zone_thresholds_pct_of_reserve: tuple[float, float, float, float, float] = (
        0.60,
        0.70,
        0.80,
        0.87,
        0.93,
    )

    # Cardiovascular debt defaults (dimensionless heuristics).
    cardio_zone_build_weights: tuple[float, float, float, float, float, float] = (
        0.001,
        0.004,
        0.010,
        0.020,
        0.035,
        0.050,
    )
    cardio_continuous_time_weight: float = 0.002
    cardio_fraction_time_weight: float = 0.015
    cardio_decay_by_zone: tuple[float, float, float, float, float, float] = (
        0.060,
        0.045,
        0.030,
        0.020,
        0.012,
        0.006,
    )

    # Mechanical debt defaults (dimensionless heuristics).
    mechanical_ascent_weight: float = 0.004
    mechanical_descent_weight: float = 0.006
    mechanical_grade_weight: float = 0.001
    mechanical_technicality_weight: float = 0.003
    mechanical_speed_change_weight: float = 0.002
    mechanical_cadence_change_weight: float = 0.001
    mechanical_step_length_change_weight: float = 0.001
    mechanical_oscillation_change_weight: float = 0.001
    mechanical_stance_change_weight: float = 0.001
    mechanical_decay_rate: float = 0.004

    # Neuromuscular debt defaults (dimensionless heuristics).
    neuromuscular_cadence_deficit_weight: float = 0.002
    neuromuscular_step_length_deficit_weight: float = 0.002
    neuromuscular_oscillation_excess_weight: float = 0.002
    neuromuscular_stance_excess_weight: float = 0.002
    neuromuscular_technicality_weight: float = 0.002
    neuromuscular_decay_rate: float = 0.003


@dataclass
class ReplayState:
    """
    Hidden state reconstructed during replay.
    """
    current_hr_zone: int = 1
    cardiovascular_debt: float = 0.0
    mechanical_debt: float = 0.0
    neuromuscular_debt: float = 0.0

    time_in_zone: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    continuous_time_in_zone: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))

    prev_speed_m_s: float = np.nan
    prev_cadence_spm: float = np.nan
    prev_step_length_m: float = np.nan
    prev_vertical_oscillation_mm: float = np.nan
    prev_stance_time_s: float = np.nan


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _default_activity_name(activity_index: int) -> str:
    return f"activity_{activity_index:03d}"


def _order_columns(columns: Sequence[str]) -> list[str]:
    """
    Order columns so that state information appears first and the rest follows.
    """
    columns = list(columns)
    preferred = (
        ["activity_id", "activity_name", "sample_id"]
        + REPLAY_OUTPUT_COLUMNS
        + [col for col in OBSERVED_STATE_COLUMNS if f"next_{col}" in columns]
        + [col for col in HIDDEN_STATE_COLUMNS if f"next_{col}" in columns]
    )

    # Keep any "next_*" columns together in the end.
    preferred += [
        "segment_distance_m",
        "segment_duration_s",
        "transition_horizon_m",
    ]

    ordered = [col for col in preferred if col in columns]
    remaining = [col for col in columns if col not in ordered]
    return ordered + remaining


def _validate_window_params(grid_step_m: float, transition_horizon_m: float) -> int:
    """
    Validate the rolling window parameters and return horizon length in steps.
    """
    if grid_step_m <= 0:
        raise ValueError("grid_step_m must be greater than 0")
    if transition_horizon_m <= 0:
        raise ValueError("transition_horizon_m must be greater than 0")

    horizon_steps = int(round(transition_horizon_m / grid_step_m))
    if horizon_steps <= 0:
        raise ValueError("transition_horizon_m must be at least one grid step")

    reconstructed = horizon_steps * grid_step_m
    if not np.isclose(reconstructed, transition_horizon_m, atol=1e-9):
        raise ValueError(
            "transition_horizon_m must be an integer multiple of grid_step_m"
        )

    return horizon_steps


def _safe_float(value: Any, default: float = np.nan) -> float:
    """
    Convert a value to float when possible.
    """
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return default
        return float(value)
    except Exception:
        return default


def _prepare_activity_frame(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and sort the source activity.
    """
    if activity_df is None or activity_df.empty:
        return pd.DataFrame()

    df = activity_df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "distance_from_start_m" in df.columns:
        df["distance_from_start_m"] = pd.to_numeric(df["distance_from_start_m"], errors="coerce")
    elif "distance_delta_m" in df.columns:
        delta = pd.to_numeric(df["distance_delta_m"], errors="coerce").fillna(0.0)
        df["distance_from_start_m"] = delta.cumsum()
    else:
        # Last resort: synthetic axis based on row order.
        df["distance_from_start_m"] = np.arange(len(df), dtype=float)

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

    # Shift to origin.
    df["distance_from_start_m"] = df["distance_from_start_m"] - float(df["distance_from_start_m"].iloc[0])

    if "time_from_start_s" in df.columns and df["time_from_start_s"].notna().any():
        df["time_from_start_s"] = pd.to_numeric(df["time_from_start_s"], errors="coerce")
        df["time_from_start_s"] = df["time_from_start_s"] - float(df["time_from_start_s"].iloc[0])

    return df


def _resolve_hr_bounds(source_df: pd.DataFrame, config: ReplayConfig) -> tuple[float, float]:
    """
    Determine HR_rest and HR_max for zone reconstruction.
    """
    hr_series = pd.Series(dtype="float64")
    if "heart_rate_bpm" in source_df.columns:
        hr_series = pd.to_numeric(source_df["heart_rate_bpm"], errors="coerce").dropna()

    hr_rest = config.hr_rest
    hr_max = config.hr_max

    if hr_rest is None:
        hr_rest = float(hr_series.quantile(0.05)) if not hr_series.empty else 60.0
    if hr_max is None:
        hr_max = float(hr_series.quantile(0.95)) if not hr_series.empty else 190.0

    hr_rest = _safe_float(hr_rest, 60.0)
    hr_max = _safe_float(hr_max, 190.0)

    if not np.isfinite(hr_rest):
        hr_rest = 60.0
    if not np.isfinite(hr_max):
        hr_max = 190.0

    if hr_max <= hr_rest:
        hr_max = hr_rest + 100.0

    return hr_rest, hr_max


def _hr_zone_from_bpm(
    hr_bpm: float | None,
    hr_rest: float,
    hr_max: float,
    thresholds: tuple[float, float, float, float, float],
    previous_zone: int = 1,
) -> int:
    """
    Map HR bpm to one of 6 zones.
    """
    if hr_bpm is None or not np.isfinite(hr_bpm):
        return int(previous_zone) if 1 <= int(previous_zone) <= 6 else 1

    reserve = max(hr_max - hr_rest, 1.0)
    thresholds_bpm = [hr_rest + reserve * t for t in thresholds]

    if hr_bpm <= thresholds_bpm[0]:
        return 1
    if hr_bpm <= thresholds_bpm[1]:
        return 2
    if hr_bpm <= thresholds_bpm[2]:
        return 3
    if hr_bpm <= thresholds_bpm[3]:
        return 4
    if hr_bpm <= thresholds_bpm[4]:
        return 5
    return 6


def _current_technicality(row: pd.Series, config: ReplayConfig) -> float:
    """
    Use terrain_technicality if present; otherwise fall back to config default.
    """
    if "terrain_technicality" in row.index:
        value = pd.to_numeric(pd.Series([row["terrain_technicality"]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return float(config.terrain_technicality)


def _cardio_terms(
    zone: int,
    continuous_zone_time: float,
    zone_fraction: float,
    state: ReplayState,
    config: ReplayConfig,
) -> tuple[float, float]:
    """
    Cardiovascular build and decay terms (heuristic, dimensionless).
    """
    idx = max(1, min(6, int(zone))) - 1
    build = (
        config.cardio_zone_build_weights[idx]
        + config.cardio_continuous_time_weight * float(continuous_zone_time)
        + config.cardio_fraction_time_weight * float(zone_fraction)
    )
    decay = config.cardio_decay_by_zone[idx] * state.cardiovascular_debt
    return build, decay


def _mechanical_terms(
    row: pd.Series,
    prev_row: pd.Series | None,
    technicality: float,
    state: ReplayState,
    config: ReplayConfig,
) -> tuple[float, float]:
    """
    Mechanical build and decay terms (heuristic, dimensionless).
    """
    altitude = _safe_float(row.get("altitude_m"), np.nan)
    prev_altitude = _safe_float(prev_row.get("altitude_m"), np.nan) if prev_row is not None else np.nan

    ascent_step = 0.0
    descent_step = 0.0
    if np.isfinite(altitude) and np.isfinite(prev_altitude):
        delta_alt = altitude - prev_altitude
        ascent_step = max(0.0, delta_alt)
        descent_step = max(0.0, -delta_alt)

    grade_pct = abs(_safe_float(row.get("grade_pct"), 0.0))

    speed = _safe_float(row.get("speed_m_s"), np.nan)
    cadence = _safe_float(row.get("cadence_spm"), np.nan)
    step_length = _safe_float(row.get("step_length_m"), np.nan)
    vertical_oscillation = _safe_float(row.get("vertical_oscillation_mm"), np.nan)
    stance_time = _safe_float(row.get("stance_time_s"), np.nan)

    prev_speed = state.prev_speed_m_s
    prev_cadence = state.prev_cadence_spm
    prev_step_length = state.prev_step_length_m
    prev_vertical_oscillation = state.prev_vertical_oscillation_mm
    prev_stance_time = state.prev_stance_time_s

    speed_change = abs(speed - prev_speed) if np.isfinite(speed) and np.isfinite(prev_speed) else 0.0
    cadence_change = abs(cadence - prev_cadence) if np.isfinite(cadence) and np.isfinite(prev_cadence) else 0.0
    step_length_change = abs(step_length - prev_step_length) if np.isfinite(step_length) and np.isfinite(prev_step_length) else 0.0
    oscillation_change = (
        abs(vertical_oscillation - prev_vertical_oscillation)
        if np.isfinite(vertical_oscillation) and np.isfinite(prev_vertical_oscillation)
        else 0.0
    )
    stance_change = (
        abs(stance_time - prev_stance_time)
        if np.isfinite(stance_time) and np.isfinite(prev_stance_time)
        else 0.0
    )

    build = (
        config.mechanical_ascent_weight * ascent_step
        + config.mechanical_descent_weight * descent_step
        + config.mechanical_grade_weight * grade_pct
        + config.mechanical_technicality_weight * technicality
        + config.mechanical_speed_change_weight * speed_change
        + config.mechanical_cadence_change_weight * cadence_change
        + config.mechanical_step_length_change_weight * step_length_change
        + config.mechanical_oscillation_change_weight * oscillation_change
        + config.mechanical_stance_change_weight * stance_change
    )

    decay = config.mechanical_decay_rate * state.mechanical_debt
    return build, decay


def _neuromuscular_terms(
    row: pd.Series,
    prev_row: pd.Series | None,
    technicality: float,
    state: ReplayState,
    config: ReplayConfig,
) -> tuple[float, float]:
    """
    Neuromuscular build and decay terms (heuristic, dimensionless).
    """
    cadence = _safe_float(row.get("cadence_spm"), np.nan)
    step_length = _safe_float(row.get("step_length_m"), np.nan)
    vertical_oscillation = _safe_float(row.get("vertical_oscillation_mm"), np.nan)
    stance_time = _safe_float(row.get("stance_time_s"), np.nan)

    prev_cadence = state.prev_cadence_spm
    prev_step_length = state.prev_step_length_m
    prev_vertical_oscillation = state.prev_vertical_oscillation_mm
    prev_stance_time = state.prev_stance_time_s

    cadence_deficit = 0.0
    step_length_deficit = 0.0
    oscillation_excess = 0.0
    stance_excess = 0.0

    if np.isfinite(cadence) and np.isfinite(prev_cadence) and prev_cadence > 0:
        cadence_deficit = max(0.0, (prev_cadence - cadence) / prev_cadence)

    if np.isfinite(step_length) and np.isfinite(prev_step_length) and prev_step_length > 0:
        step_length_deficit = max(0.0, (prev_step_length - step_length) / prev_step_length)

    if np.isfinite(vertical_oscillation) and np.isfinite(prev_vertical_oscillation) and prev_vertical_oscillation > 0:
        oscillation_excess = max(0.0, (vertical_oscillation - prev_vertical_oscillation) / prev_vertical_oscillation)

    if np.isfinite(stance_time) and np.isfinite(prev_stance_time) and prev_stance_time > 0:
        stance_excess = max(0.0, (stance_time - prev_stance_time) / prev_stance_time)

    build = (
        config.neuromuscular_cadence_deficit_weight * cadence_deficit
        + config.neuromuscular_step_length_deficit_weight * step_length_deficit
        + config.neuromuscular_oscillation_excess_weight * oscillation_excess
        + config.neuromuscular_stance_excess_weight * stance_excess
        + config.neuromuscular_technicality_weight * technicality
    )

    decay = config.neuromuscular_decay_rate * state.neuromuscular_debt
    return build, decay


def _update_zone_counters(
    state: ReplayState,
    current_zone: int,
    delta_time_s: float,
) -> None:
    """
    Update per-zone accumulated time, fractions, and continuous exposure.
    """
    zone_index = max(1, min(6, int(current_zone))) - 1

    if delta_time_s > 0:
        state.time_in_zone[zone_index] += delta_time_s

    # Continuous time in zone: if the runner stays in the same zone, accumulate.
    # If the zone changes, start a new continuous spell.
    if state.current_hr_zone == current_zone:
        state.continuous_time_in_zone[zone_index] += max(0.0, delta_time_s)
    else:
        state.continuous_time_in_zone = np.zeros(6, dtype=float)
        state.continuous_time_in_zone[zone_index] = max(0.0, delta_time_s)

    state.current_hr_zone = int(current_zone)


def _compute_zone_fractions(state: ReplayState) -> np.ndarray:
    """
    Compute fraction of elapsed time in each zone.
    """
    elapsed = float(np.sum(state.time_in_zone))
    if elapsed <= 0:
        return np.zeros(6, dtype=float)
    return state.time_in_zone / elapsed


def _make_state_row(
    row: pd.Series,
    state: ReplayState,
    zone_fraction: np.ndarray,
    technicality: float,
    cardio_build: float,
    cardio_decay: float,
    mechanical_build: float,
    mechanical_decay: float,
    neuromuscular_build: float,
    neuromuscular_decay: float,
) -> dict[str, Any]:
    """
    Assemble one replay output row.
    """
    out: dict[str, Any] = {}

    # Copy observed terrain and runner values.
    for col in TERRAIN_COLUMNS + OBSERVED_STATE_COLUMNS:
        if col in row.index:
            out[col] = row[col]

    out["current_hr_zone"] = int(state.current_hr_zone)
    out["cardiovascular_debt"] = float(max(0.0, state.cardiovascular_debt))
    out["mechanical_debt"] = float(max(0.0, state.mechanical_debt))
    out["neuromuscular_debt"] = float(max(0.0, state.neuromuscular_debt))
    out["terrain_technicality"] = float(technicality)

    # Debug terms help later when we inspect the replay.
    out["cardio_build_term"] = float(cardio_build)
    out["cardio_decay_term"] = float(cardio_decay)
    out["mechanical_build_term"] = float(mechanical_build)
    out["mechanical_decay_term"] = float(mechanical_decay)
    out["neuromuscular_build_term"] = float(neuromuscular_build)
    out["neuromuscular_decay_term"] = float(neuromuscular_decay)

    for i in range(6):
        out[f"time_in_zone_{i + 1}"] = float(state.time_in_zone[i])
        out[f"fraction_time_in_zone_{i + 1}"] = float(zone_fraction[i])
        out[f"continuous_time_spend_in_zone_{i + 1}"] = float(state.continuous_time_in_zone[i])

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def replay_activity(
    activity_df: pd.DataFrame,
    activity_id: int = 1,
    activity_name: str | None = None,
    config: ReplayConfig | None = None,
) -> pd.DataFrame:
    """
    Replay one historical activity on a dense 1 m grid and reconstruct the hidden state.
    """
    config = config or ReplayConfig()

    source = _prepare_activity_frame(activity_df)
    if source.empty:
        return pd.DataFrame()

    hr_rest, hr_max = _resolve_hr_bounds(source, config)

    # Dense interpolation grid.
    max_distance_m = float(source["distance_from_start_m"].max())
    dense_distances = np.arange(0.0, max_distance_m + config.grid_step_m * 0.5, config.grid_step_m)
    dense = pd.DataFrame({"distance_from_start_m": dense_distances})

    # Interpolate time axis.
    if "time_from_start_s" in source.columns and source["time_from_start_s"].notna().any():
        dense["time_from_start_s"] = np.interp(
            dense_distances.astype(float),
            pd.to_numeric(source["distance_from_start_m"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(source["time_from_start_s"], errors="coerce").to_numpy(dtype=float),
        )
    elif "timestamp" in source.columns and source["timestamp"].notna().any():
        ts_source = source[["distance_from_start_m", "timestamp"]].copy()
        ts_source["distance_from_start_m"] = pd.to_numeric(ts_source["distance_from_start_m"], errors="coerce")
        ts_source["timestamp"] = pd.to_datetime(ts_source["timestamp"], errors="coerce")
        ts_source = ts_source.dropna().drop_duplicates(subset=["distance_from_start_m"], keep="last").sort_values("distance_from_start_m")

        if not ts_source.empty:
            x_vals = ts_source["distance_from_start_m"].to_numpy(dtype=float)
            y_vals = ts_source["timestamp"].map(lambda v: v.value).to_numpy(dtype=float)
            interp_vals = np.interp(dense_distances.astype(float), x_vals, y_vals)
            dense["timestamp"] = pd.to_datetime(np.rint(interp_vals).astype("int64"), unit="ns", errors="coerce")
            dense["time_from_start_s"] = (dense["timestamp"] - dense["timestamp"].iloc[0]).dt.total_seconds()
        else:
            dense["time_from_start_s"] = np.arange(len(dense), dtype=float)
    else:
        dense["time_from_start_s"] = np.arange(len(dense), dtype=float)

    # Interpolate useful numeric columns from the source.
    for col in [
        "altitude_m",
        "power",
        "heart_rate_bpm",
        "cadence_spm",
        "speed_m_s",
        "step_length_m",
        "vertical_oscillation_mm",
        "stance_time_s",
        "accumulated_power",
    ]:
        if col in source.columns:
            src = source[["distance_from_start_m", col]].copy()
            src["distance_from_start_m"] = pd.to_numeric(src["distance_from_start_m"], errors="coerce")
            src[col] = pd.to_numeric(src[col], errors="coerce")
            src = src.dropna().drop_duplicates(subset=["distance_from_start_m"], keep="last").sort_values("distance_from_start_m")
            if not src.empty:
                dense[col] = np.interp(
                    dense_distances.astype(float),
                    src["distance_from_start_m"].to_numpy(dtype=float),
                    src[col].to_numpy(dtype=float),
                )

    # Derive terrain fields on dense grid.
    dense["altitude_m"] = pd.to_numeric(dense.get("altitude_m", np.nan), errors="coerce")
    dense["altitude_delta_m"] = dense["altitude_m"].diff().fillna(0.0)
    dense["ascent_delta_m"] = dense["altitude_delta_m"].clip(lower=0.0)
    dense["descent_delta_m"] = (-dense["altitude_delta_m"].clip(upper=0.0))
    dense["ascent_cumul_from_start_m"] = dense["ascent_delta_m"].cumsum()
    dense["descent_cumul_from_start_m"] = dense["descent_delta_m"].cumsum()
    dense["grade_pct"] = np.where(
        dense["distance_from_start_m"].diff().replace(0.0, np.nan).notna(),
        (dense["altitude_delta_m"] / dense["distance_from_start_m"].diff().replace(0.0, np.nan)) * 100.0,
        np.nan,
    )
    dense["grade_pct"] = dense["grade_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Replay sequentially.
    state = ReplayState()
    rows: list[dict[str, Any]] = []

    prev_row: pd.Series | None = None
    prev_time_s: float | None = None

    for idx, row in dense.iterrows():
        row_series = row.copy()

        current_time_s = _safe_float(row_series.get("time_from_start_s"), np.nan)
        if not np.isfinite(current_time_s):
            current_time_s = float(idx) * config.grid_step_m

        delta_time_s = 0.0 if prev_time_s is None else max(0.0, current_time_s - prev_time_s)

        hr_bpm = _safe_float(row_series.get("heart_rate_bpm"), np.nan)
        current_zone = _hr_zone_from_bpm(
            hr_bpm=hr_bpm,
            hr_rest=hr_rest,
            hr_max=hr_max,
            thresholds=config.zone_thresholds_pct_of_reserve,
            previous_zone=state.current_hr_zone,
        )

        # Update zone exposure counters before computing the debt terms.
        _update_zone_counters(state, current_zone, delta_time_s)
        zone_fraction = _compute_zone_fractions(state)

        technicality = _current_technicality(row_series, config)

        cardio_build, cardio_decay = _cardio_terms(
            zone=current_zone,
            continuous_zone_time=float(state.continuous_time_in_zone[current_zone - 1]),
            zone_fraction=float(zone_fraction[current_zone - 1]),
            state=state,
            config=config,
        )

        mechanical_build, mechanical_decay = _mechanical_terms(
            row=row_series,
            prev_row=prev_row,
            technicality=technicality,
            state=state,
            config=config,
        )

        neuromuscular_build, neuromuscular_decay = _neuromuscular_terms(
            row=row_series,
            prev_row=prev_row,
            technicality=technicality,
            state=state,
            config=config,
        )

        # Update hidden debts.
        state.cardiovascular_debt = max(
            0.0,
            state.cardiovascular_debt + delta_time_s * (cardio_build - cardio_decay),
        )
        state.mechanical_debt = max(
            0.0,
            state.mechanical_debt + delta_time_s * (mechanical_build - mechanical_decay),
        )
        state.neuromuscular_debt = max(
            0.0,
            state.neuromuscular_debt + delta_time_s * (neuromuscular_build - neuromuscular_decay),
        )

        # Update last observed runner variables.
        state.prev_speed_m_s = _safe_float(row_series.get("speed_m_s"), np.nan)
        state.prev_cadence_spm = _safe_float(row_series.get("cadence_spm"), np.nan)
        state.prev_step_length_m = _safe_float(row_series.get("step_length_m"), np.nan)
        state.prev_vertical_oscillation_mm = _safe_float(row_series.get("vertical_oscillation_mm"), np.nan)
        state.prev_stance_time_s = _safe_float(row_series.get("stance_time_s"), np.nan)

        out_row = _make_state_row(
            row=row_series,
            state=state,
            zone_fraction=zone_fraction,
            technicality=technicality,
            cardio_build=cardio_build,
            cardio_decay=cardio_decay,
            mechanical_build=mechanical_build,
            mechanical_decay=mechanical_decay,
            neuromuscular_build=neuromuscular_build,
            neuromuscular_decay=neuromuscular_decay,
        )

        out_row["activity_id"] = activity_id
        out_row["activity_name"] = activity_name if activity_name is not None else _default_activity_name(activity_id)
        out_row["sample_id"] = idx + 1

        rows.append(out_row)

        prev_row = row_series
        prev_time_s = current_time_s

    replay_df = pd.DataFrame(rows)
    replay_df = replay_df[_order_columns(replay_df.columns)]

    return replay_df


def build_transition_samples(
    replay_df: pd.DataFrame,
    config: ReplayConfig | None = None,
) -> pd.DataFrame:
    """
    Convert a replay table into rolling transition samples over the configured horizon.
    """
    config = config or ReplayConfig()

    if replay_df is None or replay_df.empty:
        return pd.DataFrame()

    horizon_steps = _validate_window_params(config.grid_step_m, config.transition_horizon_m)

    if len(replay_df) <= horizon_steps:
        return pd.DataFrame()

    current = replay_df.iloc[:-horizon_steps].reset_index(drop=True)
    future = replay_df.iloc[horizon_steps:].reset_index(drop=True)

    out = current.copy()

    # Metadata for transitions.
    out["transition_horizon_m"] = float(config.transition_horizon_m)
    out["segment_distance_m"] = float(config.transition_horizon_m)

    # Segment duration target.
    out["segment_duration_s"] = (
        pd.to_numeric(future["time_from_start_s"], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(current["time_from_start_s"], errors="coerce").to_numpy(dtype=float)
    )

    # Next hidden state targets.
    for col in HIDDEN_STATE_COLUMNS:
        if col in current.columns and col in future.columns:
            out[f"next_{col}"] = future[col].to_numpy()

    # Next observed state targets.
    for col in OBSERVED_STATE_COLUMNS:
        if col in current.columns and col in future.columns:
            out[f"next_{col}"] = future[col].to_numpy()

    # Next terrain / progression columns.
    for col in ["distance_from_start_m", "time_from_start_s", "altitude_m", "altitude_delta_m", "ascent_delta_m", "descent_delta_m", "ascent_cumul_from_start_m", "descent_cumul_from_start_m", "grade_pct"]:
        if col in current.columns and col in future.columns and f"next_{col}" not in out.columns:
            out[f"next_{col}"] = future[col].to_numpy()

    # Clean up.
    out = out.replace([np.inf, -np.inf], np.nan)
    out["segment_duration_s"] = pd.to_numeric(out["segment_duration_s"], errors="coerce")
    out = out.dropna(subset=["segment_duration_s"])
    out = out[out["segment_duration_s"] > 0.0]

    out = out[_order_columns(out.columns)]

    return out


def replay_and_build_transitions(
    activity_df: pd.DataFrame,
    activity_id: int = 1,
    activity_name: str | None = None,
    config: ReplayConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: returns both the replay table and the transition table.
    """
    replay_df = replay_activity(
        activity_df=activity_df,
        activity_id=activity_id,
        activity_name=activity_name,
        config=config,
    )
    transitions_df = build_transition_samples(replay_df, config=config)
    return replay_df, transitions_df


def replay_activities(
    activities: Sequence[pd.DataFrame],
    activity_names: Sequence[str] | None = None,
    config: ReplayConfig | None = None,
) -> pd.DataFrame:
    """
    Replay multiple activities and concatenate the replay tables.
    """
    if activities is None:
        return pd.DataFrame()

    activities = list(activities)
    if len(activities) == 0:
        return pd.DataFrame()

    if activity_names is not None and len(activity_names) != len(activities):
        raise ValueError("activity_names must have the same length as activities.")

    frames: list[pd.DataFrame] = []

    for idx, activity_df in enumerate(activities, start=1):
        name = activity_names[idx - 1] if activity_names is not None else _default_activity_name(idx)
        replay_df = replay_activity(
            activity_df=activity_df,
            activity_id=idx,
            activity_name=name,
            config=config,
        )
        if not replay_df.empty:
            frames.append(replay_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out[_order_columns(out.columns)]
    return out


def build_transition_dataset_from_replay(
    activities: Sequence[pd.DataFrame],
    activity_names: Sequence[str] | None = None,
    config: ReplayConfig | None = None,
) -> pd.DataFrame:
    """
    Replay multiple activities and then extract rolling transition samples.
    """
    if activities is None:
        return pd.DataFrame()

    activities = list(activities)
    if len(activities) == 0:
        return pd.DataFrame()

    if activity_names is not None and len(activity_names) != len(activities):
        raise ValueError("activity_names must have the same length as activities.")

    frames: list[pd.DataFrame] = []

    for idx, activity_df in enumerate(activities, start=1):
        name = activity_names[idx - 1] if activity_names is not None else _default_activity_name(idx)
        replay_df = replay_activity(
            activity_df=activity_df,
            activity_id=idx,
            activity_name=name,
            config=config,
        )
        transition_df = build_transition_samples(replay_df, config=config)
        if not transition_df.empty:
            frames.append(transition_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out[_order_columns(out.columns)]
    return out
  
