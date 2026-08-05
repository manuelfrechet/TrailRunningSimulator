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

TARGET_TO_STATE = {
    "next_heart_rate_bpm": "heart_rate_bpm",
    "next_power": "power",
    "next_cadence_spm": "cadence_spm",
    "next_speed_m_s": "speed_m_s",
    "next_step_length_m": "step_length_m",
    "next_vertical_oscillation_mm": "vertical_oscillation_mm",
    "next_stance_time_s": "stance_time_s",
    "next_accumulated_power": "accumulated_power",
}


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
    Merge the current dynamic state with the current race-profile row.
    Race-profile values override the dynamic state for terrain-derived fields.
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

    # Use the learned initial state for runner variables, but keep it safe.
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
        out_row["predicted_cumulative_time_hh:mm:ss"] = str(
            pd.to_timedelta(cumulative_time_s, unit="s")
        )

        # Store the raw predictions
        for target_col, value in predicted_state_updates.items():
            out_row[f"predicted_{target_col}"] = value

        output_rows.append(out_row)

        # ---------------------------------------------------------------------
        # Update the dynamic state for the next segment
        # ---------------------------------------------------------------------
        for target_col, state_col in TARGET_TO_STATE.items():
            if target_col in predicted_state_updates:
                current_state[state_col] = predicted_state_updates[target_col]

        # Keep accumulated time updated for the next step.
        current_state["time_from_start_s"] = cumulative_time_s

        # Preserve the current terrain progression fields in the state so they
        # can be available as context in the next prediction.
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
    
