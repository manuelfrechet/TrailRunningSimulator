import streamlit as st

st.title("Trail Running Simulator")

uploaded_file = st.file_uploader("Upload a FIT file", type=["fit"])

if uploaded_file is not None:
    st.success(f"File received: {uploaded_file.name}")
    st.write("Next step: read the file and build the table.")
