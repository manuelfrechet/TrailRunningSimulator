from __future__ import annotations

# -----------------------------------------------------------------------------
# System identification layer
# -----------------------------------------------------------------------------
# This module learns a first baseline transition law from the training dataset.
#
# Version 1 goal:
#   - take the unified historical training dataset,
#   - select a usable set of numeric features,
#   - fit a simple, transparent model for segment duration,
#   - expose predictions and summary statistics.
#
# No simulation logic lives here.
# No Streamlit logic lives here.
# No parsing logic lives here.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Feature selection
# -----------------------------------------------------------------------------
# We keep the list explicit so the first version remains easy to debug.
# Optional columns are used only if present in the training dataset.
# -----------------------------------------------------------------------------

IGNORED_COLUMNS = {
    "activity_id",
    "activity_name",
    "sample_id",
    "timestamp",
    "segment_duration_s",
    "segment_distance_m",
}

PREFERRED_NUMERIC_FEATURES = [
    # Race progression
    "time_from_start_s",
    "distance_from_start_m",
    "distance_delta_m",
    "fraction_of_race_completed",
    "remaining_distance_m",
    "remaining_ascent_m",
    "remaining_descent_m",
    "distance_since_last_aid_station_m",
    "distance_to_next_aid_station_m",
    # Terrain
    "altitude_m",
    "altitude_delta_m",
    "ascent_delta_m",
    "descent_delta_m",
    "ascent_cumul_from_start_m",
    "descent_cumul_from_start_m",
    "grade_pct",
    # Runner state
    "heart_rate_bpm",
    "power",
    "cadence_spm",
    "speed_m_s",
    "step_length_m",
    "vertical_oscillation_mm",
    "stance_time_s",
    # Fatigue and HR history
    "accumulated_power",
    "time_in_zone_1",
    "time_in_zone_2",
    "time_in_zone_3",
    "time_in_zone_4",
    "time_in_zone_5",
    "time_in_zone_6",
    "fraction_time_in_zone_1",
    "fraction_time_in_zone_2",
    "fraction_time_in_zone_3",
    "fraction_time_in_zone_4",
    "fraction_time_in_zone_5",
    "fraction_time_in_zone_6",
    "continuous_time_spend_in_zone_1",
    "continuous_time_spend_in_zone_2",
    "continuous_time_spend_in_zone_3",
    "continuous_time_spend_in_zone_4",
    "continuous_time_spend_in_zone_5",
    "continuous_time_spend_in_zone_6",
]


# -----------------------------------------------------------------------------
# Model container
# -----------------------------------------------------------------------------

@dataclass
class SystemIdentificationModel:
    """
    Lightweight linear baseline model for segment duration.

    The model is intentionally simple for Version 1:
      - numeric feature selection,
      - mean imputation,
      - standardization,
      - ridge-regularized least squares,
      - transparent coefficients.
    """

    target_column: str
    feature_columns: list[str]
    feature_means: pd.Series
    feature_scales: pd.Series
    coefficients: np.ndarray
    intercept: float
    training_summary: dict[str, Any]

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the target for a dataframe containing the same feature columns
        used at training time.
        """
        if data is None or data.empty:
            return pd.Series(dtype="float64")

        X = _prepare_feature_frame(
            data=data,
            feature_columns=self.feature_columns,
            feature_means=self.feature_means,
            feature_scales=self.feature_scales,
        )

        y_pred = self.intercept + X.to_numpy(dtype=float) @ self.coefficients
        return pd.Series(y_pred, index=data.index, name=f"predicted_{self.target_column}")

    def summary(self) -> dict[str, Any]:
        """
        Return a compact model summary for display or debugging.
        """
        return {
            **self.training_summary,
            "target_column": self.target_column,
            "n_features": len(self.feature_columns),
            "feature_columns": list(self.feature_columns),
        }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _select_numeric_feature_columns(dataset: pd.DataFrame, target_column: str) -> list[str]:
    """
    Select usable numeric columns for the first baseline model.

    Rules:
      - keep numeric columns only,
      - drop obvious metadata and target columns,
      - keep a preferred order,
      - fall back to any remaining numeric columns.
    """
    if dataset is None or dataset.empty:
        return []

    numeric_columns = [
        col
        for col in dataset.columns
        if col not in IGNORED_COLUMNS
        and col != target_column
        and pd.api.types.is_numeric_dtype(dataset[col])
    ]

    preferred = [col for col in PREFERRED_NUMERIC_FEATURES if col in numeric_columns]
    remaining = [col for col in numeric_columns if col not in preferred]

    return preferred + remaining


def _prepare_feature_frame(
    data: pd.DataFrame,
    feature_columns: list[str],
    feature_means: pd.Series,
    feature_scales: pd.Series,
) -> pd.DataFrame:
    """
    Build a numeric design matrix using the training means/scales.

    Missing values are imputed with the training mean.
    """
    if data is None or data.empty:
        return pd.DataFrame(columns=feature_columns)

    X = pd.DataFrame(index=data.index)

    for col in feature_columns:
        if col in data.columns:
            series = pd.to_numeric(data[col], errors="coerce")
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
    ridge_lambda: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """
    Fit a small ridge-regularized linear model.

    The intercept is handled separately and is not regularized.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    n_samples, n_features = X.shape

    # Add intercept column.
    X_aug = np.column_stack([np.ones(n_samples), X])

    # Ridge penalty, excluding intercept.
    penalty = np.eye(n_features + 1)
    penalty[0, 0] = 0.0

    lhs = X_aug.T @ X_aug + ridge_lambda * penalty
    rhs = X_aug.T @ y

    beta = np.linalg.solve(lhs, rhs)

    intercept = float(beta[0])
    coefficients = beta[1:].astype(float)

    return intercept, coefficients


def _compute_training_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute simple metrics for the fitted baseline model.
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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def fit_system_identification(
    dataset: pd.DataFrame,
    target_column: str = "segment_duration_s",
) -> SystemIdentificationModel:
    """
    Fit the first baseline system-identification model from the training dataset.

    This function:
      - selects numeric features,
      - drops rows without target values,
      - imputes missing feature values with the feature mean,
      - standardizes features,
      - fits a ridge-regularized linear baseline,
      - returns a reusable model object.
    """
    if dataset is None or dataset.empty:
        raise ValueError("dataset is empty.")

    if target_column not in dataset.columns:
        raise ValueError(f"target column '{target_column}' is missing from dataset.")

    # -------------------------------------------------------------------------
    # Keep only rows with a known target.
    # -------------------------------------------------------------------------
    working = dataset.copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=[target_column])

    if working.empty:
        raise ValueError(f"No usable rows found for target '{target_column}'.")

    # -------------------------------------------------------------------------
    # Select feature columns.
    # -------------------------------------------------------------------------
    feature_columns = _select_numeric_feature_columns(working, target_column)

    if not feature_columns:
        raise ValueError("No numeric feature columns were found for system identification.")

    # -------------------------------------------------------------------------
    # Build the raw feature matrix.
    # -------------------------------------------------------------------------
    raw_features = working[feature_columns].apply(pd.to_numeric, errors="coerce")

    # -------------------------------------------------------------------------
    # Compute means and scales for imputation + standardization.
    # Columns with zero variance are kept, but their scale is forced to 1.0.
    # -------------------------------------------------------------------------
    feature_means = raw_features.mean(axis=0, skipna=True)
    feature_scales = raw_features.std(axis=0, skipna=True).replace(0.0, 1.0)

    # Fill missing values with means, then standardize.
    X_df = (raw_features.fillna(feature_means) - feature_means) / feature_scales

    # -------------------------------------------------------------------------
    # Remove columns that are all NaN or numerically degenerate after cleaning.
    # -------------------------------------------------------------------------
    keep_columns = [
        col for col in X_df.columns
        if np.isfinite(X_df[col].to_numpy(dtype=float)).any()
    ]

    X_df = X_df[keep_columns]
    feature_means = feature_means[keep_columns]
    feature_scales = feature_scales[keep_columns]

    if X_df.empty or X_df.shape[1] == 0:
        raise ValueError("No valid feature columns remain after cleaning.")

    # -------------------------------------------------------------------------
    # Fit the baseline model.
    # -------------------------------------------------------------------------
    y = working[target_column].to_numpy(dtype=float)
    X = X_df.to_numpy(dtype=float)

    intercept, coefficients = _fit_ridge_linear_model(X=X, y=y)

    y_pred = intercept + X @ coefficients
    metrics = _compute_training_metrics(y_true=y, y_pred=y_pred)

    summary = {
        "n_rows": int(len(working)),
        "n_features": int(X.shape[1]),
        "n_rows_with_target": int(len(y)),
        "n_rows_without_target": int(len(dataset) - len(working)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        **metrics,
    }

    return SystemIdentificationModel(
        target_column=target_column,
        feature_columns=list(X_df.columns),
        feature_means=feature_means,
        feature_scales=feature_scales,
        coefficients=coefficients,
        intercept=intercept,
        training_summary=summary,
    )


def summarize_system_identification(model: SystemIdentificationModel) -> dict[str, Any]:
    """
    Return a compact summary of the fitted model.
    """
    if model is None:
        return {}

    return model.summary()


def predict_segment_duration(
    model: SystemIdentificationModel,
    data: pd.DataFrame,
) -> pd.Series:
    """
    Convenience function for predicting segment duration from a dataframe.
    """
    if model is None:
        raise ValueError("model is None.")

    return model.predict(data)
  
