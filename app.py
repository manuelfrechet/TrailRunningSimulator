import pandas as pd
import streamlit as st
from math import ceil

from features import build_features
from gpx_parser import parse_gpx_to_table
from gpx_segments import build_fixed_distance_segments
from parser import parse_fit_to_tables

SEGMENT_LENGTH_M = 50.0

st.title("Trail Running Simulator")

# Upload FIT file
st.write("Upload a .FIT file to extract your performance model from previous run")
uploaded_file = st.file_uploader("Choose a .FIT file", type=["fit"], key="fit_uploader")

if uploaded_file is None:
    st.info("Upload a .FIT file to begin.")
    st.stop()

st.success(f"File received: {uploaded_file.name}")

uploaded_file.seek(0)
tables = parse_fit_to_tables(uploaded_file)
record_df = tables.get("record", pd.DataFrame())
features_df = build_features(record_df)

# Display features
st.subheader("Preview your run metrics extracted from uploaded .fit file")
if features_df.empty:
    st.warning("No metrics could be computed.")
else:
    st.dataframe(features_df, width="stretch")

with st.expander("Raw FIT table", expanded=False):
    if record_df.empty:
        st.warning("No record messages were found in this FIT file.")
    else:
        st.dataframe(record_df, width="stretch")

st.divider()

# Upload GPX for next race
st.subheader("Upload your next race .GPX")
uploaded_gpx = st.file_uploader("Choose a GPX file", type=["gpx"], key="gpx_uploader")

if uploaded_gpx is not None:
    st.success(f"GPX file received: {uploaded_gpx.name}")

    uploaded_gpx.seek(0)
    gpx_raw_df = parse_gpx_to_table(uploaded_gpx)

    gpx_segments_df = build_fixed_distance_segments(
        gpx_raw_df,
        segment_length_m=SEGMENT_LENGTH_M,
    )

    with st.expander("Raw GPX table", expanded=False):
        if gpx_raw_df.empty:
            st.warning("No track points were found in this GPX file.")
        else:
            st.dataframe(gpx_raw_df, width="stretch")

    with st.expander(f"Race profile with normalized {SEGMENT_LENGTH_M:.0f}m segments", expanded=False):
        if gpx_segments_df.empty:
            st.warning("No normalized GPX segments could be built.")
        else:
            st.dataframe(gpx_segments_df, width="stretch")

            race_length_km = float(gpx_segments_df["distance_from_start_m"].max()) / 1000.0
            expected_aid_stations = ceil(race_length_km / 10.0)

            st.subheader("Aid stations")
            st.write(
                f"Race length: {race_length_km:.2f} km — suggested aid station slots: {expected_aid_stations}"
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
                            value=0.0,
                            step=0.1,
                            key=f"aid_km_{i}",
                        )

                    if station_name.strip():
                        aid_station_rows.append(
                            {
                                "aid_station_name": station_name.strip(),
                                "aid_station_km": station_km,
                            }
                        )

                submitted = st.form_submit_button("Save aid stations")

            if submitted:
                aid_stations_df = pd.DataFrame(aid_station_rows)

                st.subheader("Aid stations entered")
                if aid_stations_df.empty:
                    st.warning("No aid stations entered yet.")
                else:
                    st.dataframe(aid_stations_df, width="stretch")
                    
