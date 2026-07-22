from __future__ import annotations

import pandas as pd

def build_features(record_df: pd.DataFrame) -> pd.DataFrame:

    df = record_df.copy()
    
    if df.empty:
        return df

    #standardize names
    rename_map = {
        "enhanced_speed": "speed_m_s",
        "enhanced_altitude": "altitude_m",
        "heart_rate": "heart_rate_bpm",
    }
    df = df.rename(columns=rename_map)
    
    # Cadence in steps per minute
    if "cadence" in df.columns:
        cadence = pd.to_numeric(df["cadence"], errors="coerce")
    else:
        cadence = pd.Series(pd.NA, index=df.index, dtype="float64")

    if "fractional_cadence" in df.columns:
        fractional_cadence = pd.to_numeric(df["fractional_cadence"], errors="coerce")
    else:
        fractional_cadence = pd.Series(0.0, index=df.index, dtype="float64")

    df["cadence_spm"] = (cadence + fractional_cadence) * 2.0

    #Time from start
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    #Deltas from previous record
    if "distance" in df.columns:
        df["distance_delta_m"] = pd.to_numeric(df["distance"], errors="coerce").diff()
    else:
        df["distance_delta_m"] = pd.NA
    
    # Ascent / descent deltas from previous record
    df["ascent_delta_m"] = df["altitude_delta_m"].clip(lower=0)
    df["descent_delta_m"] = (-df["altitude_delta_m"].clip(upper=0))

    # Grade percentage
    df["grade_pct"] = (df["altitude_delta_m"] / df["distance_delta_m"]) * 100.0
    df.loc[df["distance_delta_m"] <= 0, "grade_pct"] = pd.NA

    # Speed and pace from deltas
    if "time_from_start_s" in df.columns:
        time_delta_s = df["time_from_start_s"].diff()
        df["speed_m_s_from_distance"] = df["distance_delta_m"] / time_delta_s
        df.loc[time_delta_s <= 0, "speed_m_s_from_distance"] = pd.NA

        df["pace_min_km_from_distance"] = (1000.0 / df["speed_m_s_from_distance"]) / 60.0
        df.loc[df["speed_m_s_from_distance"] <= 0, "pace_min_km_from_distance"] = pd.NA
    else:
        df["speed_m_s_from_distance"] = pd.NA
        df["pace_min_km_from_distance"] = pd.NA

    # Cumulative ascent/descent from start
    df["ascent_cumul_from_start_m"] = df["ascent_delta_m"].fillna(0).cumsum()
    df["descent_cumul_from_start_m"] = df["descent_delta_m"].fillna(0).cumsum()

    return df
