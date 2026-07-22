import pandas as pd
import fitdecode
import streamlit as st

st.title("Trail Running Simulator")
st.write("Upload a FIT file to see the main track data.")

uploaded_file = st.file_uploader("Choose a FIT file", type=["fit"])

if uploaded_file is not None:
    st.success(f"File received: {uploaded_file.name}")

uploaded_file.seek(0)

rows = []

with fitdecode.FitReader(uploaded_file) as fit:
    for frame in fit:
        if frame.frame_type != fitdecode.FIT_FRAME_DATA:
            continue

        if frame.name != "record":
            continue

        row = {
            "timestamp": frame.get_value("timestamp", fallback=None),
            "distance_m": frame.get_value("distance", fallback=None),
            "altitude_m": frame.get_value("enhanced_altitude", fallback=frame.get_value("altitude", fallback=None)),
            "heart_rate_bpm": frame.get_value("heart_rate", fallback=None),
            "cadence_rpm": frame.get_value("cadence", fallback=None),
            "speed_m_s": frame.get_value("enhanced_speed", fallback=frame.get_value("speed", fallback=None)),
        }

        rows.append(row)

df = pd.DataFrame(rows)

if not df.empty and "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

st.subheader("What we found")
st.write(f"Number of record rows: {len(df)}")
