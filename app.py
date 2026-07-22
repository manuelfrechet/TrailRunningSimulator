import pandas as pd
import streamlit as st

from parser import parse_fit_to_tables, tables_to_excel_bytes

st.title("Trail Running Simulator")
st.write("Upload a FIT file to see all useful raw data.")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is not None:
    st.success(f"File received: {uploaded_file.name}")

uploaded_file.seek(0)
tables = parse_fit_to_tables(uploaded_file)

record_df = tables.get("record", pd.DataFrame())

st.subheader("Preview: record messages")

if record_df.empty:
    st.warning("No record messages were found in this FIT file.")
else:
    st.dataframe(record_df.head(50), use_container_width=True)
    st.write(f"Number of record rows: {len(record_df)}")

st.subheader("Download Excel workbook")

excel_bytes = tables_to_excel_bytes(tables)

st.download_button(
    label="Download Excel file",
    data=excel_bytes,
    file_name=uploaded_file.name.replace(".fit", ".xlsx"),
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
