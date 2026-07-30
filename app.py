import pandas as pd
import streamlit as st
from math import ceil

from features import build_features
from gpx_parser import parse_gpx_to_table
from gpx_segments import build_fixed_distance_segments, enhance_race_profile_with_breakpoints
from parser import parse_fit_to_tables

SEGMENT_LENGTH_M = 50.0

st.title("Trail Running Simulator")

# Upload FIT file
st.subheader("Upload a .FIT file to extract your performance model from previous run")
uploaded_file = st.file_uploader("Choose a .FIT file", type=["fit"], key="fit_uploader")

if uploaded_file is None:
    st.info("Upload a .FIT file to begin.")
    st.stop()

st.success(f"File received: {uploaded_file.name}")
uploaded_file.seek(0)
tables = parse_fit_to_tables(uploaded_file)
record_df = tables.get("record", pd.DataFrame())
features_df = build_features(record_df)

with st.expander("Raw FIT table", expanded=False):
    if record_df.empty:
        st.warning("No record messages were found in this FIT file.")
    else:
        st.dataframe(record_df, width="stretch")

st.divider()

# Upload GPX for next race
st.subheader("Upload your next race .GPX")
uploaded_gpx = st.file_uploader("Choose a GPX file", type=["gpx"], key="gpx_uploader")

gpx_raw_df = None
aid_stations_df = None
enhanced_race_profile_df = None

if uploaded_gpx is not None:
    st.success(f"GPX file received: {uploaded_gpx.name}")
    uploaded_gpx.seek(0)
    gpx_raw_df = parse_gpx_to_table(uploaded_gpx)
    
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
    
        with st.expander("Aid stations", expanded=True):
            st.write(
                f"Race length: {race_length_km:.2f} km — suggested aid station slots: {expected_aid_stations}"
            )
    
            default_aid_station_rows = pd.DataFrame(
                {
                    "aid_station_name": [""] * expected_aid_stations,
                    "aid_station_km": [0.00] * expected_aid_stations,
                }
            )
    
            with st.form(key="aid_station_form"):
                aid_station_input_df = st.data_editor(
                    default_aid_station_rows,
                    num_rows="fixed",
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "aid_station_name": st.column_config.TextColumn("aid station name"),
                        "aid_station_km": st.column_config.NumberColumn(
                            "aid station km",
                            min_value=0.0,
                            max_value=race_length_km,
                            step=0.01,
                        ),
                    },
                )
    
                submitted = st.form_submit_button("Build race profile")
    
            if submitted:
                aid_stations_df = aid_station_input_df.copy()
                aid_stations_df["aid_station_name"] = (
                    aid_stations_df["aid_station_name"].astype(str).str.strip()
                )
                aid_stations_df["aid_station_km"] = pd.to_numeric(
                    aid_stations_df["aid_station_km"], errors="coerce"
                )
                aid_stations_df = aid_stations_df.dropna(
                    subset=["aid_station_name", "aid_station_km"]
                )
                aid_stations_df = aid_stations_df[aid_stations_df["aid_station_name"] != ""]
    
                st.subheader("Aid stations entered")
                if aid_stations_df.empty:
                    st.warning("No aid stations entered yet.")
                else:
                    st.dataframe(aid_stations_df, width="stretch")
    
                gpx_segments_df = build_fixed_distance_segments(gpx_raw_df,segment_length_m=SEGMENT_LENGTH_M,)
    
                enhanced_race_profile_df = enhance_race_profile_with_breakpoints(gpx_segments_df,aid_stations_df,)
    
        if enhanced_race_profile_df is not None:
            with st.expander(f"Race profile with normalized {SEGMENT_LENGTH_M:.0f}m segments",expanded=False,):
                if enhanced_race_profile_df.empty:
                    st.warning("No enhanced race profile could be built.")
                else:
                    st.dataframe(enhanced_race_profile_df, width="stretch")
