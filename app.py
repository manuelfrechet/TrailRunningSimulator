import pandas as pd
import streamlit as st

from features import build_features
from parser import parse_fit_to_tables

st.title("Trail Running Simulator")
st.write("Upload a FIT file to see all useful raw data.")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is None:
    st.info("Upload a FIT file to begin.")
    st.stop()
    
st.success(f"File received: {uploaded_file.name}")

uploaded_file.seek(0)
tables = parse_fit_to_tables(uploaded_file)

record_df = tables.get("record", pd.DataFrame())
#st.subheader("Raw record columns")
#st.write(list(record_df.columns))

features_df = build_features(record_df)

#Display the raw data table
#st.subheader("Preview: Raw records")
#if record_df.empty:
#    st.warning("No record messages were found in this FIT file.")
#else:
#    st.dataframe(record_df.head(50), use_container_width=True)
#    st.write(f"Number of record rows: {len(record_df)}")

#Display table with raw and features data
st.subheader("Preview: featured records")
if features_df.empty:
    st.warning("No features could be computed.")
else:
    st.dataframe(features_df, width="stretch")

#Import GPX
import pandas as pd
import streamlit as st

from features import build_features
from gpx_parser import parse_gpx_to_table
from parser import parse_fit_to_tables

st.title("Trail Running Simulator")

st.subheader("1) Historical race FIT file")
uploaded_fit = st.file_uploader("Choose a FIT file", type=["fit"], key="fit_uploader")

if uploaded_fit is not None:
    st.success(f"FIT file received: {uploaded_fit.name}")

    uploaded_fit.seek(0)
    fit_tables = parse_fit_to_tables(uploaded_fit)
    record_df = fit_tables.get("record", pd.DataFrame())
    features_df = build_features(record_df)

    st.subheader("Features table from FIT")
    if features_df.empty:
        st.warning("No features could be computed from this FIT file.")
    else:
        st.dataframe(features_df, width="stretch")

st.divider()

st.subheader("2) Future race GPX file")
uploaded_gpx = st.file_uploader("Choose a GPX file", type=["gpx"], key="gpx_uploader")

if uploaded_gpx is not None:
    st.success(f"GPX file received: {uploaded_gpx.name}")

    uploaded_gpx.seek(0)
    gpx_df = parse_gpx_to_table(uploaded_gpx)

    st.subheader("Raw GPX table")
    if gpx_df.empty:
        st.warning("No track points were found in this GPX file.")
    else:
        st.dataframe(gpx_df, width="stretch")
