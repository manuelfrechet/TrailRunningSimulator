from **future** import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from parser import parse_fit_to_dataframe

st.set_page_config(
page_title="Trail Running Simulator",
page_icon="🏔️",
layout="centered",
)

st.title("Trail Running Simulator")
st.write("Upload a FIT file and get an Excel export.")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is not None:
st.success(f"File received: {uploaded_file.name}")

```
# Read the uploaded file into memory and give it to the parser
file_bytes = uploaded_file.getvalue()

# Streamlit gives us the file in memory, but our parser expects a file path.
# So we write it temporarily to disk, parse it, then delete the temporary file.
import tempfile
import os

with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp:
    tmp.write(file_bytes)
    temp_fit_path = tmp.name

try:
    df = parse_fit_to_dataframe(temp_fit_path)

    if df.empty:
        st.warning("No track records were found in this FIT file.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Download Excel")
        output = BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="records")

        excel_bytes = output.getvalue()

        st.download_button(
            label="Download Excel file",
            data=excel_bytes,
            file_name=uploaded_file.name.replace(".fit", ".xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

finally:
    if os.path.exists(temp_fit_path):
        os.remove(temp_fit_path)
```
