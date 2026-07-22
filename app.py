import os
import tempfile

import fitdecode
import streamlit as st

st.title("Trail Running Simulator")
st.write("Upload a FIT file to count the track records.")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is not None:
st.success(f"File received: {uploaded_file.name}")

```
# Save the uploaded file temporarily so fitdecode can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp:
    tmp.write(uploaded_file.getvalue())
    temp_fit_path = tmp.name

try:
    record_count = 0

    with fitdecode.FitReader(temp_fit_path) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            if frame.name == "record":
                record_count += 1

    st.subheader("Result")
    st.write(f"Number of record messages found: {record_count}")

finally:
    if os.path.exists(temp_fit_path):
        os.remove(temp_fit_path)
```
