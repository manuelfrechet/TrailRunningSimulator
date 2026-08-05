from __future__ import annotations

# -----------------------------------------------------------------------------
# Simulator layer
# -----------------------------------------------------------------------------
# This module applies a learned transition model to a normalized race profile.
#
# Version 1:
#   - iterate through the race profile segment by segment,
#   - predict the next state and segment duration,
#   - accumulate total time,
#   - return the predicted trajectory.
#
# Important stabilization rule:
#   - the simulator does NOT feed its own predicted physiological state back
#     into the next segment yet.
#   - only deterministic race progression and terrain are carried forward.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from system_identification import TransitionModel, predict_next_state


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class SimulationResult:
    segments: pd.DataFrame
    summary: dict[str, Any]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_required_columns(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df is None or profile_df.empty:
        return pd.DataFrame()

    if "distance_from_start_m" not in profile_df.columns:
        raise ValueError("Race profile must contain 'distance_from_start_m'.")

    return profile_df.copy()


def _build_feature_row(
    profile_row: pd.Series,
    current_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge the current deterministic/dynamic state with the current race-profile row.
    Race-profile values override the state for terrain-derived fields.
    """
    row_dict = profile_row.to_dict()
    feature_row = dict(current_state)
    feature_row.update(row_dict)
    return feature_row


def _prepare_current_state(model: TransitionModel) -> dict[str, Any]:
    """
    Start from the learned initial state and enforce a few race-start defaults.
    """
    current_state = dict(model.initial_state)

    # Race starts at zero elapsed time.
    current_state["time_from_start_s"] = 0.0

    # Safe defaults for runner variables if the learned initial state is sparse.
    for col in [
        "heart_rate_bpm",
        "power",
        "cadence_spm",
        "speed_m_s",
        "step_length_m",
        "vertical_oscillation_mm",
        "stance_time_s",
        "accumulated_power",
    ]:
        current_state.setdefault(col, 0.0)

    return current_state


def _format_hhmmss(seconds: float) -> str:
    """
    Format a duration in seconds as HH:MM:SS.
    """
    if seconds is None or not np.isfinite(seconds):
        return ""

    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def simulate_race(
    race_profile_df: pd.DataFrame,
    model: TransitionModel,
) -> SimulationResult:
    """
    Simulate the race profile segment by segment using the learned transition model.
    """
    if model is None:
        raise ValueError("model is required.")

    profile = _ensure_required_columns(race_profile_df)

    if profile.empty:
        return SimulationResult(
            segments=pd.DataFrame(),
            summary={
                "n_segments": 0,
                "total_predicted_time_s": 0.0,
                "mean_segment_duration_s": 0.0,
            },
        )

    segments = profile.copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Initialize the dynamic state at race start.
    # -------------------------------------------------------------------------
    current_state = _prepare_current_state(model)
    cumulative_time_s = 0.0

    output_rows: list[dict[str, Any]] = []

    for _, profile_row in segments.iterrows():
        # ---------------------------------------------------------------------
        # Build model input for this segment
        # ---------------------------------------------------------------------
        current_state["time_from_start_s"] = cumulative_time_s

        feature_row = _build_feature_row(profile_row, current_state)
        feature_df = pd.DataFrame([feature_row])

        # ---------------------------------------------------------------------
        # Predict the next state and the segment duration
        # ---------------------------------------------------------------------
        predicted = predict_next_state(model, feature_df)

        if predicted.empty:
            predicted_duration_s = 1.0
            predicted_state_updates: dict[str, Any] = {}
        else:
            predicted_duration_s = float(predicted["segment_duration_s"].iloc[0])
            if not np.isfinite(predicted_duration_s) or predicted_duration_s <= 0.0:
                predicted_duration_s = 1.0

            predicted_state_updates = predicted.iloc[0].to_dict()

        # ---------------------------------------------------------------------
        # Advance time
        # ---------------------------------------------------------------------
        cumulative_time_s += predicted_duration_s

        # ---------------------------------------------------------------------
        # Build the output row
        # ---------------------------------------------------------------------
        out_row = profile_row.to_dict()
        out_row["current_time_from_start_s"] = current_state.get("time_from_start_s", 0.0)
        out_row["predicted_segment_duration_s"] = predicted_duration_s
        out_row["predicted_cumulative_time_s"] = cumulative_time_s
        out_row["predicted_cumulative_time_hh:mm:ss"] = _format_hhmmss(cumulative_time_s)

        # Store the raw predictions from the learned transition model.
        for target_col, value in predicted_state_updates.items():
            out_row[f"predicted_{target_col}"] = value

        output_rows.append(out_row)

        # ---------------------------------------------------------------------
        # Important stabilization rule
        # ---------------------------------------------------------------------
        # Do NOT feed predicted physiological values back into the next segment
        # yet. This prevents runaway feedback in the first baseline version.
        #
        # We only carry deterministic progression and terrain forward.
        # ---------------------------------------------------------------------
        current_state["time_from_start_s"] = cumulative_time_s

        for col in [
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
        ]:
            if col in profile_row.index:
                current_state[col] = profile_row[col]

    result_df = pd.DataFrame(output_rows)

    total_predicted_time_s = float(result_df["predicted_segment_duration_s"].sum(skipna=True))
    n_segments = int(len(result_df))
    mean_segment_duration_s = (
        float(result_df["predicted_segment_duration_s"].mean(skipna=True))
        if n_segments > 0
        else 0.0
    )

    summary = {
        "n_segments": n_segments,
        "total_predicted_time_s": total_predicted_time_s,
        "mean_segment_duration_s": mean_segment_duration_s,
    }

    return SimulationResult(
        segments=result_df,
        summary=summary,
    )


def summarize_simulation(result: SimulationResult) -> dict[str, Any]:
    if result is None:
        return {}
    return dict(result.summary)
    
