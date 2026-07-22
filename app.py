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
st.subheader("Raw record columns")
st.write(list(record_df.columns))

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
    st.dataframe(features_df.head(20), width="stretch")
