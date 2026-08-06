from __future__ import annotations

# -----------------------------------------------------------------------------
# Transition dataset builder
# -----------------------------------------------------------------------------
# This module converts replayed FIT activities into rolling transition samples.
#
# The replay engine is responsible for:
#   - reconstructing the hidden state from the full historical trajectory
#   - producing dense replay rows
#
# This module is responsible for:
#   - calling the replay engine
#   - extracting rolling transition samples
#   - providing compact dataset summaries for the app
# -----------------------------------------------------------------------------

from typing import Any, Sequence

import pandas as pd

from replay_engine import (
    ReplayConfig,
    build_transition_dataset_from_replay,
    replay_activity,
    replay_activities,
    replay_and_build_transitions,
)

# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

DEFAULT_GRID_STEP_M = 1.0
DEFAULT_TRANSITION_HORIZON_M = 50.0


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_transition_dataset(
    activities: Sequence[pd.DataFrame],
    activity_names: Sequence[str] | None = None,
    grid_step_m: float = DEFAULT_GRID_STEP_M,
    transition_horizon_m: float = DEFAULT_TRANSITION_HORIZON_M,
    hr_rest: float | None = None,
    hr_max: float | None = None,
    terrain_technicality: float = 3.0,
) -> pd.DataFrame:
    """
    Build a rolling transition dataset from multiple historical activities.

    This uses the replay engine internally so each transition sample is
    reconstructed from the full prior history.
    """
    config = ReplayConfig(
        grid_step_m=grid_step_m,
        transition_horizon_m=transition_horizon_m,
        hr_rest=hr_rest,
        hr_max=hr_max,
        terrain_technicality=terrain_technicality,
    )

    return build_transition_dataset_from_replay(
        activities=activities,
        activity_names=activity_names,
        config=config,
    )


def summarize_transition_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """
    Compact summary of the transition dataset.
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


# -----------------------------------------------------------------------------
# Convenience re-exports
# -----------------------------------------------------------------------------

__all__ = [
    "build_transition_dataset",
    "summarize_transition_dataset",
    "ReplayConfig",
    "replay_activity",
    "replay_activities",
    "replay_and_build_transitions",
]
