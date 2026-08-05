from __future__ import annotations

# -----------------------------------------------------------------------------
# Trail Running Simulator - Streamlit App
# -----------------------------------------------------------------------------
# Current responsibilities:
#   1) Build the learning dataset from multiple historical FIT files.
#   2) Fit a first baseline system-identification model.
#   3) Build the future race profile from a GPX file and aid stations.
#
# No simulator loop yet.
# -----------------------------------------------------------------------------

from math import ceil

import pandas as pd
import streamlit as st

from features import build_features
from gpx_parser import parse_gpx_to_table
from gpx_segments import build_fixed_distance_segments, enhance_race_profile_with_breakpoints
from parser import parse_fit_to_tables
from system_identification import (
    fit_system_identification,
    predict_segment_duration,
    summarize_system_identification,
)
from training_dataset import activity_summary_table, build_training_dataset, summarize_training_dataset


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
    Clean the aid-station editor output and keep only valid rows.
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


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------

if "enhanced_race_profile_df" not in st.session_state:
    st.session_state["enhanced_race_profile_df"] = None

if "gpx_file_name" not in st.session_state:
    st.session_state["gpx_file_name"] = None


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

        # ---------------------------------------------------------------------
        # Training dataset summary
        # ---------------------------------------------------------------------
        with st.expander("Training dataset summary", expanded=False):
            st.write("This summary helps verify that the learning data was built correctly.")
            st.json(training_dataset_summary)

            if skipped_fit_files:
                st.warning(
                    "These FIT files did not produce usable runner features and were skipped:"
                )
                for name in skipped_fit_files:
                    st.write(f"- {name}")

        # ---------------------------------------------------------------------
        # Per-activity summary
        # ---------------------------------------------------------------------
        with st.expander("Activity summary", expanded=False):
            if activity_summary_df.empty:
                st.warning("No per-activity summary could be built.")
            else:
                st.dataframe(activity_summary_df, width="stretch")

        # ---------------------------------------------------------------------
        # Dataset preview
        # ---------------------------------------------------------------------
        with st.expander("Training dataset preview", expanded=False):
            st.dataframe(training_dataset_df.head(200), width="stretch")

        # ---------------------------------------------------------------------
        # First baseline system identification
        # ---------------------------------------------------------------------
        with st.spinner("Fitting baseline system-identification model..."):
            try:
                system_model = fit_system_identification(training_dataset_df)
                system_summary = summarize_system_identification(system_model)

                st.success("Baseline system-identification model fitted successfully.")

                with st.expander("System identification summary", expanded=False):
                    st.json(system_summary)

                    coefficients_df = pd.DataFrame(
                        {
                            "feature": system_model.feature_columns,
                            "coefficient": system_model.coefficients,
                        }
                    )
                    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
                    coefficients_df = coefficients_df.sort_values(
                        "abs_coefficient",
                        ascending=False,
                    ).drop(columns=["abs_coefficient"])

                    st.subheader("Model coefficients")
                    st.dataframe(coefficients_df, width="stretch")

                # -----------------------------------------------------------------
                # Quick training preview
                # -----------------------------------------------------------------
                with st.expander("Training prediction preview", expanded=False):
                    preview_df = training_dataset_df.copy()
                    preview_df["predicted_segment_duration_s"] = predict_segment_duration(
                        system_model,
                        preview_df,
                    )

                    preview_columns = [
                        col
                        for col in [
                            "activity_id",
                            "activity_name",
                            "sample_id",
                            "segment_duration_s",
                            "predicted_segment_duration_s",
                        ]
                        if col in preview_df.columns
                    ]

                    st.dataframe(
                        preview_df[preview_columns].head(200),
                        width="stretch",
                    )

            except Exception as exc:
                st.error(f"System identification failed: {exc}")

    else:
        st.warning(
            "None of the uploaded FIT files produced usable runner features. "
            "Please check that the files contain record messages."
        )
else:
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
    # -------------------------------------------------------------------------
    # No GPX means no future race profile to build.
    # -------------------------------------------------------------------------
    st.info("Upload a GPX file to build the future race profile.")
    st.session_state["enhanced_race_profile_df"] = None
    st.session_state["gpx_file_name"] = None

else:
    # -------------------------------------------------------------------------
    # Reset the stored race profile if a different GPX is uploaded.
    # -------------------------------------------------------------------------
    if st.session_state["gpx_file_name"] != uploaded_gpx.name:
        st.session_state["enhanced_race_profile_df"] = None
        st.session_state["gpx_file_name"] = uploaded_gpx.name

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
        # ---------------------------------------------------------------------
        # Course length and aid-station planning
        # ---------------------------------------------------------------------
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

            # -------------------------------------------------------------
            # Use explicit inputs instead of a data editor so tab navigation
            # follows a natural order:
            # aid_name_1 -> aid_km_1 -> aid_name_2 -> aid_km_2 -> ...
            # -------------------------------------------------------------
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

                    # -------------------------------------------------
                    # Keep only rows that have a non-empty name.
                    # The cleaning function will remove anything invalid.
                    # -------------------------------------------------
                    aid_station_rows.append(
                        {
                            "aid_station_name": station_name,
                            "aid_station_km": station_km,
                        }
                    )

                submitted = st.form_submit_button("Build race profile")

            if submitted:
                # -------------------------------------------------------------
                # Clean aid station input
                # -------------------------------------------------------------
                aid_stations_df = pd.DataFrame(aid_station_rows)
                aid_stations_df = _clean_aid_stations(
                    aid_stations_df,
                    race_length_km,
                )

                st.subheader("Aid stations entered")
                if aid_stations_df.empty:
                    st.warning("No aid stations entered yet.")
                else:
                    st.dataframe(aid_stations_df, width="stretch")

                # -------------------------------------------------------------
                # Build normalized race profile
                # -------------------------------------------------------------
                gpx_segments_df = build_fixed_distance_segments(
                    gpx_raw_df,
                    segment_length_m=SEGMENT_LENGTH_M,
                )

                enhanced_race_profile_df = enhance_race_profile_with_breakpoints(
                    gpx_segments_df,
                    aid_stations_df,
                )

                st.session_state["enhanced_race_profile_df"] = enhanced_race_profile_df

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
                    
