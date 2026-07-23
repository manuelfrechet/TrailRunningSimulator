import pandas as pd
import streamlit as st

from features import build_features
from parser import parse_fit_to_tables
from gpx_parser import parse_gpx_to_table
from gpx_segments import build_fixed_distance_segments

st.title("Trail Running Simulator")

#Upload for .FIT file
st.write("Upload a FIT file to extract your performance model from this run")
uploaded_file = st.file_uploader("Choose a .FIT file", type=["fit"])

if uploaded_file is None:
    st.info("Upload a FIT file to begin.")
    st.stop()
    
st.success(f"File received: {uploaded_file.name}")

uploaded_file.seek(0)
tables = parse_fit_to_tables(uploaded_file)
record_df = tables.get("record", pd.DataFrame())
features_df = build_features(record_df)

#Display table with raw and features data
st.subheader("Preview: your run metrics at .fit file granularity")
if features_df.empty:
    st.warning("No features could be computed.")
else:
    st.dataframe(features_df, width="stretch")

st.divider()

#Import GPX
st.subheader("Upload your next race .GPX")
uploaded_gpx = st.file_uploader("Choose a GPX file", type=["gpx"], key="gpx_uploader")

if uploaded_gpx is not None:
    st.success(f"GPX file received: {uploaded_gpx.name}")

    uploaded_gpx.seek(0)
    gpx_raw_df = parse_gpx_to_table(uploaded_gpx)

    gpx_segments_df = build_fixed_distance_segments(
        gpx_raw_df,
        segment_length_m=10.0,
    )

    st.subheader("Normalized GPX segments (10 m)")

    if gpx_segments_df.empty:
        st.warning("No track points were found in this GPX file.")
    else:
        st.dataframe(gpx_segments_df, width="stretch")
