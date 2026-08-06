from __future__ import annotations

# -----------------------------------------------------------------------------
# System identification layer
# -----------------------------------------------------------------------------
# This module learns transition laws from the replay-based transition dataset.
#
# Version 1 goal:
#   - select current-state numeric features,
#   - fit one ridge-regularized linear model per transition target,
#   - expose a reusable model that predicts segment duration and next state.
#
# The simulator uses the learned transition law step by step.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Columns and model conventions
# -----------------------------------------------------------------------------

METADATA_COLUMNS = {
    "activity_id",
    "activity_name",
    "sample_id",
    "timestamp",
}

PRIMARY_TARGET = "segment_duration_s"
TARGET_PREFIX = "next_"

DEBUG_COLUMNS = {
    "cardio_build_term",
    "cardio_decay_term",
    "mechanical_build_term",
    "mechanical_decay_term",
    "neuromuscular_build_term",
    "neuromuscular_decay_term",
}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class LinearTargetModel:
    target_column: str
    feature_columns: list[str]
    feature_means: pd.Series
    feature_scales: pd.Series
    coefficients: np.ndarray
    intercept: float
    metrics: dict[str, float]

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if data is None or data.empty:
            return pd.Series(dtype="float64")

        clean_data = _dedupe_columns(data)
        X = _prepare_feature_frame(
            data=clean_data,
            feature_columns=self.feature_columns,
            feature_means=self.feature_means,
            feature_scales=self.feature_scales,
        )

        values = self.intercept + X.to_numpy(dtype=float) @ self.coefficients
        return pd.Series(values, index=clean_data.index, name=self.target_column)


@dataclass
class TransitionModel:
    feature_columns: list[str]
    target_models: dict[str, LinearTargetModel]
    initial_state: dict[str, float]
    training_summary: dict[str, Any]

    def predict_next_state(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        clean_data = _dedupe_columns(data)
        out = pd.DataFrame(index=clean_data.index)

        for target, target_model in self.target_models.items():
            out[target] = target_model.predict(clean_data)

        return out

    def summary(self) -> dict[str, Any]:
        summary = dict(self.training_summary)
        summary["feature_columns"] = list(self.feature_columns)
        summary["target_columns"] = list(self.target_models.keys())
        return summary


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate columns while preserving the first occurrence.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    return df.loc[:, ~df.columns.duplicated()].copy()


def _is_target_column(col: str) -> bool:
    return col == PRIMARY_TARGET or col.startswith(TARGET_PREFIX)


def _select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """
    Select numeric current-state columns used as inputs to the transition model.
    """
    numeric_columns = [
        col
        for col in dataset.columns
        if col not in METADATA_COLUMNS
        and col not in DEBUG_COLUMNS
        and not _is_target_column(col)
        and pd.api.types.is_numeric_dtype(dataset[col])
    ]

    preferred_order = [
        # Terrain / progression
        "distance_from_start_m",
        "time_from_start_s",
        "segment_distance_m",
        "altitude_m",
        "altitude_delta_m",
        "ascent_delta_m",
        "descent_delta_m",
        "ascent_cumul_from_start_m",
        "descent_cumul_from_start_m",
        "grade_pct",
        "terrain_technicality",
        # Runner state
        "heart_rate_bpm",
        "power",
        "cadence_spm",
        "speed_m_s",
        "step_length_m",
        "vertical_oscillation_mm",
        "stance_time_s",
        "accumulated_power",
        # Hidden state / fatigue
        "current_hr_zone",
        "cardiovascular_debt",
        "mechanical_debt",
        "neuromuscular_debt",
    ]

    preferred_order.extend([f"time_in_zone_{i}" for i in range(1, 7)])
    preferred_order.extend([f"fraction_time_in_zone_{i}" for i in range(1, 7)])
    preferred_order.extend([f"continuous_time_spend_in_zone_{i}" for i in range(1, 7)])

    preferred = [col for col in preferred_order if col in numeric_columns]
    remaining = [col for col in numeric_columns if col not in preferred]

    return preferred + remaining


def _select_target_columns(dataset: pd.DataFrame) -> list[str]:
    """
    Select target columns dynamically from the dataset.
    """
    target_columns: list[str] = []

    if PRIMARY_TARGET in dataset.columns and pd.api.types.is_numeric_dtype(dataset[PRIMARY_TARGET]):
        target_columns.append(PRIMARY_TARGET)

    for col in dataset.columns:
        if col.startswith(TARGET_PREFIX) and pd.api.types.is_numeric_dtype(dataset[col]):
            target_columns.append(col)

    # Preserve order and avoid duplicates.
    seen = set()
    ordered_unique: list[str] = []
    for col in target_columns:
        if col not in seen:
            ordered_unique.append(col)
            seen.add(col)

    return ordered_unique


def _prepare_feature_frame(
    data: pd.DataFrame,
    feature_columns: list[str],
    feature_means: pd.Series,
    feature_scales: pd.Series,
) -> pd.DataFrame:
    """
    Build a standardized design matrix from raw data.
    """
    X = pd.DataFrame(index=data.index)

    for col in feature_columns:
        if col in data.columns:
            series = data[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors="coerce")
        else:
            series = pd.Series(np.nan, index=data.index, dtype="float64")

        mean_value = float(feature_means.get(col, 0.0))
        scale_value = float(feature_scales.get(col, 1.0))
        if not np.isfinite(scale_value) or scale_value == 0.0:
            scale_value = 1.0

        X[col] = (series.fillna(mean_value) - mean_value) / scale_value

    return X


def _fit_ridge_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float = 1e-4,
) -> tuple[float, np.ndarray]:
    """
    Fit a ridge-regularized linear model with an explicit intercept.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if len(y) == 0:
        raise ValueError("y is empty.")

    n_samples, n_features = X.shape
    X_aug = np.column_stack([np.ones(n_samples), X])

    penalty = np.eye(n_features + 1)
    penalty[0, 0] = 0.0

    lhs = X_aug.T @ X_aug + ridge_lambda * penalty
    rhs = X_aug.T @ y

    beta = np.linalg.solve(lhs, rhs)
    intercept = float(beta[0])
    coefficients = beta[1:].astype(float)

    return intercept, coefficients


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute simple fit metrics.
    """
    if len(y_true) == 0:
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }

    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }


def _compute_initial_state(dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """
    Estimate a race-start state from the first row of each activity.
    """
    if "activity_id" in dataset.columns:
        sort_cols = [
            c for c in ["activity_id", "distance_from_start_m", "time_from_start_s", "sample_id"]
            if c in dataset.columns
        ]
        if sort_cols:
            first_rows = dataset.sort_values(sort_cols, kind="mergesort").groupby("activity_id", sort=True).head(1)
        else:
            first_rows = dataset.groupby("activity_id", sort=True).head(1)
    else:
        first_rows = dataset.head(1)

    first_rows = _dedupe_columns(first_rows)

    initial_state: dict[str, float] = {}

    for col in feature_columns:
        if col in first_rows.columns:
            series = first_rows[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors="coerce")
            value = series.mean()
            initial_state[col] = float(value) if pd.notna(value) else 0.0
        else:
            initial_state[col] = 0.0

    # Force a few obvious race-start values.
    if "time_from_start_s" in initial_state:
        initial_state["time_from_start_s"] = 0.0
    if "distance_from_start_m" in initial_state:
        initial_state["distance_from_start_m"] = 0.0

    return initial_state


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def fit_system_identification(
    dataset: pd.DataFrame,
    target_column: str = PRIMARY_TARGET,
) -> TransitionModel:
    """
    Fit transition models from the replay-based transition dataset.
    """
    if dataset is None or dataset.empty:
        raise ValueError("dataset is empty.")

    working = _dedupe_columns(dataset)

    # -------------------------------------------------------------------------
    # Select features and targets.
    # -------------------------------------------------------------------------
    feature_columns = _select_feature_columns(working)
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found.")

    target_columns = _select_target_columns(working)
    if not target_columns:
        raise ValueError("No usable target columns found.")

    # -------------------------------------------------------------------------
    # Clean feature matrix.
    # -------------------------------------------------------------------------
    raw_features = working[feature_columns].apply(pd.to_numeric, errors="coerce")
    feature_means = raw_features.mean(axis=0, skipna=True)
    feature_scales = raw_features.std(axis=0, skipna=True).replace(0.0, 1.0)

    # Remove empty columns.
    keep_columns = [col for col in raw_features.columns if raw_features[col].notna().any()]
    feature_columns = keep_columns
    raw_features = raw_features[feature_columns]
    feature_means = feature_means[feature_columns]
    feature_scales = feature_scales[feature_columns].replace(0.0, 1.0)

    X_df = (raw_features.fillna(feature_means) - feature_means) / feature_scales
    X = X_df.to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Fit one model per target.
    # -------------------------------------------------------------------------
    target_models: dict[str, LinearTargetModel] = {}
    target_metrics: dict[str, dict[str, float]] = {}

    for target in target_columns:
        y_series = pd.to_numeric(working[target], errors="coerce")
        valid_mask = y_series.notna() & np.isfinite(X).all(axis=1)

        y = y_series.loc[valid_mask].to_numpy(dtype=float)
        X_target = X[valid_mask.to_numpy(dtype=bool)]

        if len(y) == 0:
            continue

        if len(y) < 2:
            intercept = float(np.mean(y))
            coefficients = np.zeros(X.shape[1], dtype=float)
            y_pred = np.full_like(y, intercept, dtype=float)
        else:
            intercept, coefficients = _fit_ridge_linear_model(X_target, y)
            y_pred = intercept + X_target @ coefficients

        metrics = _compute_metrics(y_true=y, y_pred=y_pred)
        target_metrics[target] = metrics

        target_models[target] = LinearTargetModel(
            target_column=target,
            feature_columns=list(feature_columns),
            feature_means=feature_means,
            feature_scales=feature_scales,
            coefficients=coefficients,
            intercept=intercept,
            metrics={
                "n_samples": int(len(y)),
                **metrics,
            },
        )

    if not target_models:
        raise ValueError("No usable targets could be fitted.")

    # -------------------------------------------------------------------------
    # Estimate initial state from the beginning of historical activities.
    # -------------------------------------------------------------------------
    initial_state = _compute_initial_state(working, feature_columns)

    summary = {
        "n_rows": int(len(working)),
        "n_features": int(len(feature_columns)),
        "n_targets": int(len(target_models)),
        "target_metrics": target_metrics,
    }

    return TransitionModel(
        feature_columns=list(feature_columns),
        target_models=target_models,
        initial_state=initial_state,
        training_summary=summary,
    )


def summarize_system_identification(model: TransitionModel) -> dict[str, Any]:
    if model is None:
        return {}
    return model.summary()


def predict_next_state(
    model: TransitionModel,
    data: pd.DataFrame,
) -> pd.DataFrame:
    if model is None:
        raise ValueError("model is None.")
    return model.predict_next_state(data)
    
