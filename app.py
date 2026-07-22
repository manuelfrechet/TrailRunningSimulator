import fitdecode
import streamlit as st

st.title("Trail Running Simulator")
st.write("Upload a FIT file to count the track records.")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is not None:
    st.success(f"File received: {uploaded_file.name}")

    uploaded_file.seek(0)

    record_count = 0

    with fitdecode.FitReader(uploaded_file) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            if frame.name == "record":
                record_count += 1

    st.subheader("Result")
    st.write(f"Number of record messages found: {record_count}")
