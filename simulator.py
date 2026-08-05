from __future__ import annotations

# -----------------------------------------------------------------------------
# Simulator layer
# -----------------------------------------------------------------------------
# This module applies a learned model to a normalized race profile.
#
# Version 1 goal:
#   - iterate through the 50 m race segments,
#   - predict the duration of each segment,
#   - accumulate total time,
#   - produce a simple simulation output table.
#
# No parsing logic lives here.
# No Streamlit logic lives here.
# No training logic lives here.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from system_identification import SystemIdentificationModel


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Container for the simulation output.

    Attributes
    ----------
    segments:
        DataFrame containing the predicted per-segment trajectory.
    summary:
        Compact dictionary with simulation-level metrics.
    """
    segments: pd.DataFrame
    summary: dict[str, Any]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_required_columns(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the race profile contains the minimum columns required by Version 1.

    For now we require:
      - distance_from_start_m
    """
    if profile_df is None or profile_df.empty:
        return pd.DataFrame()

    if "distance_from_start_m" not in profile_df.columns:
        raise ValueError("Race profile must contain 'distance_from_start_m'.")

    return profile_df.copy()


def _compute_cumulative_time(segment_duration_s: pd.Series) -> pd.Series:
    """
    Compute cumulative race time from segment durations.
    """
    return segment_duration_s.fillna(0.0).cumsum()


def _compute_segment_duration_from_distance(profile_df: pd.DataFrame) -> pd.Series:
    """
    Compute the nominal segment distance.

    In Version 1 the profile is expected to be normalized to 50 m segments,
    but we compute the distance directly so the simulator remains robust.
    """
    distance = pd.to_numeric(profile_df["distance_from_start_m"], errors="coerce")
    segment_distance = distance.diff()
    segment_distance.iloc[0] = np.nan
    return segment_distance


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def simulate_race(
    race_profile_df: pd.DataFrame,
    model: SystemIdentificationModel,
) -> SimulationResult:
    """
    Simulate the race profile segment by segment using the fitted model.

    Parameters
    ----------
    race_profile_df:
        Normalized race profile, typically 50 m segments.
    model:
        Fitted system-identification model used to estimate segment duration.

    Returns
    -------
    SimulationResult
        Per-segment predictions plus a compact summary.
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

    # -------------------------------------------------------------------------
    # Predict a baseline segment duration for every row in the profile.
    # -------------------------------------------------------------------------
    segments = profile.copy().reset_index(drop=True)

    # The simulator currently delegates prediction to the learned baseline model.
    # In later versions, this will become a true state-by-state transition loop.
    segments["predicted_segment_duration_s"] = model.predict(segments)

    # -------------------------------------------------------------------------
    # Cumulative time
    # -------------------------------------------------------------------------
    segments["predicted_cumulative_time_s"] = _compute_cumulative_time(
        segments["predicted_segment_duration_s"]
    )

    # -------------------------------------------------------------------------
    # Convenience columns
    # -------------------------------------------------------------------------
    segments["predicted_segment_distance_m"] = _compute_segment_duration_from_distance(
        segments
    )

    # -------------------------------------------------------------------------
    # Simulation summary
    # -------------------------------------------------------------------------
    total_predicted_time_s = float(
        segments["predicted_segment_duration_s"].sum(skipna=True)
    )
    n_segments = int(len(segments))
    mean_segment_duration_s = (
        float(segments["predicted_segment_duration_s"].mean(skipna=True))
        if n_segments > 0
        else 0.0
    )

    summary = {
        "n_segments": n_segments,
        "total_predicted_time_s": total_predicted_time_s,
        "mean_segment_duration_s": mean_segment_duration_s,
    }

    return SimulationResult(
        segments=segments,
        summary=summary,
    )


def summarize_simulation(result: SimulationResult) -> dict[str, Any]:
    """
    Return the simulation summary in a convenient dictionary form.
    """
    if result is None:
        return {}

    return dict(result.summary)
  
