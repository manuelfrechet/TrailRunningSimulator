import streamlit as st

st.title("Trail Running Simulator")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is not None:
    st.write(type(uploaded_file))
    st.write(uploaded_file)
