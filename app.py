from __future__ import annotations

# -----------------------------------------------------------------------------
# Trail Running Simulator - Streamlit App
# -----------------------------------------------------------------------------
# Current responsibilities:
#   1) Build the learning dataset from multiple historical FIT files.
#   2) Build a rolling transition dataset for the first baseline model.
#   3) Fit the first system-identification model.
#   4) Build the future race profile from a GPX file and aid stations.
#   5) Run a first baseline simulation on the normalized race profile.
#   6) Expose diagnostics for the duration model so we can debug it in detail.
# -----------------------------------------------------------------------------

from math import ceil

import numpy as np
import pandas as pd
import streamlit as st

from features import build_features
from gpx_parser import parse_gpx_to_table
from gpx_segments import build_fixed_distance_segments, enhance_race_profile_with_breakpoints
from parser import parse_fit_to_tables
from simulator import simulate_race, summarize_simulation
from system_identification import fit_system_identification
from training_dataset import activity_summary_table, build_training_dataset, summarize_training_dataset
from transition_dataset import build_transition_dataset, summarize_transition_dataset


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

SEGMENT_LENGTH_M = 50.0


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _load_fit_activity(uploaded_file):
    """
    Load one FIT file and convert it to a standardized runner feature dataframe.
    """
    uploaded_file.seek(0)

    # -------------------------------------------------------------------------
    # Raw FIT parsing
    # -------------------------------------------------------------------------
    tables = parse_fit_to_tables(uploaded_file)
    record_df = tables.get("record", pd.DataFrame())

    # -------------------------------------------------------------------------
    # Runner feature construction
    # -------------------------------------------------------------------------
    features_df = build_features(record_df)

    return features_df


def _clean_aid_stations(editor_df: pd.DataFrame, race_length_km: float) -> pd.DataFrame:
    """
    Clean the aid-station input and keep only valid rows.
    """
    aid_stations_df = editor_df.copy()

    # -------------------------------------------------------------------------
    # Standardize text and numeric values
    # -------------------------------------------------------------------------
    aid_stations_df["aid_station_name"] = (
        aid_stations_df["aid_station_name"].astype(str).str.strip()
    )
    aid_stations_df["aid_station_km"] = pd.to_numeric(
        aid_stations_df["aid_station_km"], errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Remove blank rows and invalid entries
    # -------------------------------------------------------------------------
    aid_stations_df = aid_stations_df.dropna(
        subset=["aid_station_name", "aid_station_km"]
    )
    aid_stations_df = aid_stations_df[aid_stations_df["aid_station_name"] != ""]

    # -------------------------------------------------------------------------
    # Enforce race-length bounds
    # -------------------------------------------------------------------------
    aid_stations_df = aid_stations_df[
        (aid_stations_df["aid_station_km"] >= 0.0)
        & (aid_stations_df["aid_station_km"] <= race_length_km)
    ]

    return aid_stations_df.reset_index(drop=True)


def _safe_numeric_corr(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """
    Safe Pearson correlation for numeric columns.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return float("nan")

    pair = df[[x_col, y_col]].copy()
    pair[x_col] = pd.to_numeric(pair[x_col], errors="coerce")
    pair[y_col] = pd.to_numeric(pair[y_col], errors="coerce")
    pair = pair.dropna()

    if len(pair) < 2:
        return float("nan")

    return float(pair[x_col].corr(pair[y_col]))


def _format_hhmmss(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.
    """
    if seconds is None or not np.isfinite(seconds):
        return ""

    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _build_duration_model_diagnostics(
    transition_learning_df: pd.DataFrame,
    system_model,
) -> dict[str, pd.DataFrame]:
    """
    Build diagnostics for the segment-duration model.
    """
    if transition_learning_df is None or transition_learning_df.empty:
        return {}

    duration_model = system_model.target_models.get("segment_duration_s")
    if duration_model is None:
        return {}

    # -------------------------------------------------------------------------
    # Coefficients
    # -------------------------------------------------------------------------
    coeff_df = pd.DataFrame(
        {
            "feature": duration_model.feature_columns,
            "coefficient": duration_model.coefficients,
        }
    )
    coeff_df["abs_coefficient"] = coeff_df["coefficient"].abs()
    coeff_df = coeff_df.sort_values("abs_coefficient", ascending=False).drop(
        columns=["abs_coefficient"]
    )

    # -------------------------------------------------------------------------
    # Correlations with observed segment duration
    # -------------------------------------------------------------------------
    candidate_cols = [
        "time_from_start_s",
        "distance_from_start_m",
        "segment_distance_m",
        "distance_delta_m",
        "altitude_m",
        "altitude_delta_m",
        "ascent_delta_m",
        "descent_delta_m",
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

    corr_rows = []
    for col in candidate_cols:
        if col in transition_learning_df.columns:
            corr_rows.append(
                {
                    "feature": col,
                    "corr_with_segment_duration_s": _safe_numeric_corr(
                        transition_learning_df,
                        col,
                        "segment_duration_s",
                    ),
                    "n_valid_rows": int(
                        transition_learning_df[[col, "segment_duration_s"]]
                        .dropna()
                        .shape[0]
                    ),
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["abs_corr"] = corr_df["corr_with_segment_duration_s"].abs()
        corr_df = corr_df.sort_values("abs_corr", ascending=False).drop(
            columns=["abs_corr"]
        )

    # -------------------------------------------------------------------------
    # Segment duration by grade bins
    # -------------------------------------------------------------------------
    grade_df = pd.DataFrame()
    if "grade_pct" in transition_learning_df.columns:
        grade_working = transition_learning_df.copy()
        grade_working["grade_pct"] = pd.to_numeric(
            grade_working["grade_pct"], errors="coerce"
        )

        bins = [-np.inf, -15, -8, -4, -1, 1, 4, 8, 15, np.inf]
        labels = [
            "<= -15",
            "-15 to -8",
            "-8 to -4",
            "-4 to -1",
            "-1 to 1",
            "1 to 4",
            "4 to 8",
            "8 to 15",
            "> 15",
        ]
        grade_working["grade_bin"] = pd.cut(
            grade_working["grade_pct"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        grade_df = (
            grade_working.groupby("grade_bin", observed=False)["segment_duration_s"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(
                columns={
                    "count": "n",
                    "mean": "mean_segment_duration_s",
                    "median": "median_segment_duration_s",
                }
            )
        )

    # -------------------------------------------------------------------------
    # Synthetic probe: flat vs uphill vs downhill
    # -------------------------------------------------------------------------
    probe_rows = []

    feature_reference = transition_learning_df[duration_model.feature_columns].copy()
    feature_reference = feature_reference.apply(pd.to_numeric, errors="coerce")
    medians = feature_reference.median(axis=0, skipna=True)

    def make_probe(label: str, overrides: dict[str, float]) -> dict[str, float]:
        row = medians.to_dict()
        for key, value in overrides.items():
            if key in duration_model.feature_columns:
                row[key] = value
        return {"scenario": label, **row}

    flat_overrides = {}
    uphill_overrides = {}
    downhill_overrides = {}

    for key in ["grade_pct", "altitude_delta_m", "ascent_delta_m", "descent_delta_m"]:
        if key in duration_model.feature_columns:
            if key == "grade_pct":
                flat_overrides[key] = 0.0
                uphill_overrides[key] = 10.0
                downhill_overrides[key] = -10.0
            elif key == "altitude_delta_m":
                flat_overrides[key] = 0.0
                uphill_overrides[key] = 5.0
                downhill_overrides[key] = -5.0
            elif key == "ascent_delta_m":
                flat_overrides[key] = 0.0
                uphill_overrides[key] = 5.0
                downhill_overrides[key] = 0.0
            elif key == "descent_delta_m":
                flat_overrides[key] = 0.0
                uphill_overrides[key] = 0.0
                downhill_overrides[key] = 5.0

    probe_rows.append(make_probe("flat", flat_overrides))
    probe_rows.append(make_probe("uphill", uphill_overrides))
    probe_rows.append(make_probe("downhill", downhill_overrides))

    probe_df = pd.DataFrame(probe_rows)
    probe_predictions = duration_model.predict(probe_df)
    probe_df["predicted_segment_duration_s"] = probe_predictions.values
    probe_df["predicted_segment_duration_hh:mm:ss"] = probe_df[
        "predicted_segment_duration_s"
    ].apply(_format_hhmmss)

    # -------------------------------------------------------------------------
    # Target summary
    # -------------------------------------------------------------------------
    target_summary_df = pd.DataFrame(
        [
            {
                "n_samples": duration_model.metrics.get("n_samples", np.nan),
                "r2": duration_model.metrics.get("r2", np.nan),
                "mae": duration_model.metrics.get("mae", np.nan),
                "rmse": duration_model.metrics.get("rmse", np.nan),
            }
        ]
    )

    return {
        "coefficients": coeff_df,
        "correlations": corr_df,
        "grade_bins": grade_df,
        "probe": probe_df,
        "target_summary": target_summary_df,
        "feature_columns": pd.DataFrame(
            {"feature_columns_used_by_duration_model": duration_model.feature_columns}
        ),
    }


def _build_diagnostics_export_df(diagnostics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all diagnostics tables into one long CSV-friendly dataframe.
    """
    frames: list[pd.DataFrame] = []

    for table_name, df in diagnostics.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        out = df.copy()
        out.insert(0, "table_name", table_name)
        out.insert(1, "row_in_table", range(1, len(out) + 1))
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------

if "system_model" not in st.session_state:
    st.session_state["system_model"] = None

if "enhanced_race_profile_df" not in st.session_state:
    st.session_state["enhanced_race_profile_df"] = None

if "gpx_file_name" not in st.session_state:
    st.session_state["gpx_file_name"] = None

if "simulation_result" not in st.session_state:
    st.session_state["simulation_result"] = None


# -----------------------------------------------------------------------------
# Page title
# -----------------------------------------------------------------------------

st.title("Trail Running Simulator")


# -----------------------------------------------------------------------------
# 1) Historical learning data
# -----------------------------------------------------------------------------

st.header("1. Historical runner learning data")
st.write(
    "Upload one or more historical .FIT files. The more complete and more granular "
    "the trajectory data, the better the learning dataset will be."
)

uploaded_fit_files = st.file_uploader(
    "Choose one or more .FIT files",
    type=["fit"],
    accept_multiple_files=True,
    key="fit_uploader",
)

training_frames: list[pd.DataFrame] = []
training_names: list[str] = []
skipped_fit_files: list[str] = []

if uploaded_fit_files:
    # -------------------------------------------------------------------------
    # Parse every FIT file and convert it to runner features
    # -------------------------------------------------------------------------
    for uploaded_file in uploaded_fit_files:
        features_df = _load_fit_activity(uploaded_file)

        if features_df.empty:
            skipped_fit_files.append(uploaded_file.name)
        else:
            training_frames.append(features_df)
            training_names.append(uploaded_file.name)

    # -------------------------------------------------------------------------
    # Build the unified training dataset
    # -------------------------------------------------------------------------
    if training_frames:
        training_dataset_df = build_training_dataset(
            training_frames,
            activity_names=training_names,
        )

        training_dataset_summary = summarize_training_dataset(training_dataset_df)
        activity_summary_df = activity_summary_table(training_dataset_df)

        st.success(
            f"Loaded {training_dataset_summary['n_activities']} activities "
            f"and {training_dataset_summary['n_rows']} rows into the training dataset."
        )

        with st.expander("Training dataset summary", expanded=False):
            st.write(f"Activities: {training_dataset_summary['n_activities']}")
            st.write(f"Rows: {training_dataset_summary['n_rows']}")
            st.write(
                f"Rows with segment duration: "
                f"{training_dataset_summary['n_rows_with_segment_duration']}"
            )
            st.write(
                f"Rows missing segment duration: "
                f"{training_dataset_summary['n_rows_missing_segment_duration']}"
            )
            st.write(f"Columns: {training_dataset_summary['n_columns']}")

            if skipped_fit_files:
                st.warning(
                    "These FIT files did not produce usable runner features and were skipped:"
                )
                for name in skipped_fit_files:
                    st.write(f"- {name}")

        with st.expander("Activity summary", expanded=False):
            if activity_summary_df.empty:
                st.warning("No per-activity summary could be built.")
            else:
                st.dataframe(activity_summary_df, width="stretch")

        # ---------------------------------------------------------------------
        # Build the rolling transition learning dataset
        # ---------------------------------------------------------------------
        transition_learning_df = build_transition_dataset(
            training_frames,
            activity_names=training_names,
            grid_step_m=1.0,
            transition_horizon_m=10.0,
        )

        if transition_learning_df.empty:
            st.warning(
                "The rolling transition learning dataset is empty. "
                "The baseline model cannot be fitted yet."
            )
            st.session_state["system_model"] = None
        else:
            transition_learning_summary = summarize_transition_dataset(
                transition_learning_df
            )

            # -----------------------------------------------------------------
            # Fit the first baseline system identification model
            # -----------------------------------------------------------------
            with st.spinner("Fitting baseline system-identification model..."):
                try:
                    system_model = fit_system_identification(transition_learning_df)
                    st.session_state["system_model"] = system_model

                    st.success(
                        "Historical learning completed successfully "
                        f"on {transition_learning_summary['n_rows']} transition samples."
                    )

                    diagnostics = _build_duration_model_diagnostics(
                        transition_learning_df,
                        system_model,
                    )

                    with st.expander("Duration model diagnostics", expanded=False):
                        st.subheader("Target summary")
                        st.dataframe(diagnostics["target_summary"], width="stretch")

                        st.subheader("Features used by the duration model")
                        st.dataframe(diagnostics["feature_columns"], width="stretch")

                        st.subheader("Top coefficients")
                        st.dataframe(diagnostics["coefficients"], width="stretch")

                        st.subheader("Correlations with segment duration")
                        st.dataframe(diagnostics["correlations"], width="stretch")

                        st.subheader("Segment duration by grade bin")
                        st.dataframe(diagnostics["grade_bins"], width="stretch")

                        st.subheader("Synthetic terrain probe")
                        st.dataframe(diagnostics["probe"], width="stretch")

                        # -----------------------------------------------------
                        # Export everything as one CSV so it can be shared
                        # -----------------------------------------------------
                        diagnostics_export_df = _build_diagnostics_export_df(diagnostics)

                        if diagnostics_export_df.empty:
                            st.warning("No diagnostics available to export.")
                        else:
                            st.download_button(
                                label="Download diagnostics CSV",
                                data=diagnostics_export_df.to_csv(index=False),
                                file_name="duration_model_diagnostics.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                except Exception as exc:
                    st.session_state["system_model"] = None
                    st.error(f"System identification failed: {exc}")

    else:
        st.session_state["system_model"] = None
        st.warning(
            "None of the uploaded FIT files produced usable runner features. "
            "Please check that the files contain record messages."
        )
else:
    st.session_state["system_model"] = None
    st.info("Upload one or more FIT files to build the learning dataset.")


# -----------------------------------------------------------------------------
# 2) Future race profile
# -----------------------------------------------------------------------------

st.divider()
st.header("2. Future race profile")
st.write(
    "Upload the GPX of the race you want to simulate. The course will be "
    "normalized to 50 m segments after you enter the aid stations."
)

uploaded_gpx = st.file_uploader(
    "Choose a GPX file",
    type=["gpx"],
    key="gpx_uploader",
)

if uploaded_gpx is None:
    st.info("Upload a GPX file to build the future race profile.")
    st.session_state["enhanced_race_profile_df"] = None
    st.session_state["gpx_file_name"] = None
    st.session_state["simulation_result"] = None

else:
    if st.session_state["gpx_file_name"] != uploaded_gpx.name:
        st.session_state["enhanced_race_profile_df"] = None
        st.session_state["gpx_file_name"] = uploaded_gpx.name
        st.session_state["simulation_result"] = None

    st.success(f"GPX file received: {uploaded_gpx.name}")

    uploaded_gpx.seek(0)
    gpx_raw_df = parse_gpx_to_table(uploaded_gpx)

    # -------------------------------------------------------------------------
    # Raw GPX preview
    # -------------------------------------------------------------------------
    with st.expander("Raw GPX table", expanded=False):
        if gpx_raw_df.empty:
            st.warning("No track points were found in this GPX file.")
        else:
            st.dataframe(gpx_raw_df, width="stretch")

    if gpx_raw_df.empty:
        st.warning("No GPX profile can be built because the raw GPX table is empty.")

    else:
        race_length_km = float(gpx_raw_df["distance_from_start_m"].max()) / 1000.0
        expected_aid_stations = ceil(race_length_km / 10.0)

        # ---------------------------------------------------------------------
        # Aid-station input section
        # ---------------------------------------------------------------------
        with st.expander("Aid stations", expanded=True):
            st.write(
                f"Race length: {race_length_km:.2f} km — "
                f"suggested aid station slots: {expected_aid_stations}"
            )

            aid_station_rows = []

            with st.form(key="aid_station_form"):
                for i in range(expected_aid_stations):
                    col1, col2 = st.columns(2)

                    with col1:
                        station_name = st.text_input(
                            f"Aid-station {i + 1} - name",
                            key=f"aid_name_{i}",
                        )

                    with col2:
                        station_km = st.number_input(
                            f"Aid-station {i + 1} - km",
                            min_value=0.0,
                            max_value=race_length_km,
                            value=0.00,
                            step=0.01,
                            format="%.2f",
                            key=f"aid_km_{i}",
                        )

                    aid_station_rows.append(
                        {
                            "aid_station_name": station_name,
                            "aid_station_km": station_km,
                        }
                    )

                submitted = st.form_submit_button("Build race profile")

            if submitted:
                aid_stations_df = _clean_aid_stations(
                    pd.DataFrame(aid_station_rows),
                    race_length_km,
                )

                st.subheader("Aid stations entered")
                if aid_stations_df.empty:
                    st.warning("No aid stations entered yet.")
                else:
                    st.dataframe(aid_stations_df, width="stretch")

                gpx_segments_df = build_fixed_distance_segments(
                    gpx_raw_df,
                    segment_length_m=SEGMENT_LENGTH_M,
                )

                enhanced_race_profile_df = enhance_race_profile_with_breakpoints(
                    gpx_segments_df,
                    aid_stations_df,
                )

                st.session_state["enhanced_race_profile_df"] = enhanced_race_profile_df

                if st.session_state["system_model"] is None:
                    st.warning(
                        "No learned model is available yet. Upload FIT files first "
                        "to train the simulator."
                    )
                    st.session_state["simulation_result"] = None
                else:
                    try:
                        simulation_result = simulate_race(
                            enhanced_race_profile_df,
                            st.session_state["system_model"],
                        )
                        st.session_state["simulation_result"] = simulation_result
                    except Exception as exc:
                        st.session_state["simulation_result"] = None
                        st.error(f"Simulation failed: {exc}")

        # ---------------------------------------------------------------------
        # Race profile preview
        # ---------------------------------------------------------------------
        if st.session_state["enhanced_race_profile_df"] is not None:
            with st.expander(
                f"Race profile with normalized {SEGMENT_LENGTH_M:.0f}m segments",
                expanded=False,
            ):
                profile_df = st.session_state["enhanced_race_profile_df"]

                if profile_df.empty:
                    st.warning("No enhanced race profile could be built.")
                else:
                    st.dataframe(profile_df, width="stretch")

        # ---------------------------------------------------------------------
        # Simulation output
        # ---------------------------------------------------------------------
        if st.session_state["simulation_result"] is not None:
            with st.expander("Simulation output", expanded=False):
                simulation_result = st.session_state["simulation_result"]
                simulation_summary = summarize_simulation(simulation_result)

                st.write("Simulation summary")
                st.json(simulation_summary)

                if simulation_result.segments.empty:
                    st.warning("No simulated race segments could be produced.")
                else:
                    st.dataframe(simulation_result.segments, width="stretch")
                    
