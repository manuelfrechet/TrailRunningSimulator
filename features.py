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
        
    if "distance_m" in df.columns:
        df["distance_delta_m"] = df["distance_m"].diff()
    
    if "altitude_m" in df.columns:
        df["altitude_delta_m"] = df["altitude_m"].diff()
    
    if "distance_m" in df.columns and "timestamp" in df.columns:
        time_delta_s = df["timestamp"].diff().dt.total_seconds()
        distance_delta_m = df["distance_m"].diff()
    
        df["speed_m_s_from_distance"] = distance_delta_m / time_delta_s
        df.loc[time_delta_s <= 0, "speed_m_s_from_distance"] = pd.NA
    
        df["pace_min_km_from_distance"] = (1000.0 / df["speed_m_s_from_distance"]) / 60.0
        df.loc[df["speed_m_s_from_distance"] <= 0, "pace_min_km_from_distance"] = pd.NA
    
    if "altitude_delta_m" in df.columns and "distance_delta_m" in df.columns:
        df["grade_pct"] = (df["altitude_delta_m"] / df["distance_delta_m"]) * 100.0
        df.loc[df["distance_delta_m"] == 0, "grade_pct"] = pd.NA
    
    if "altitude_m" in df.columns:
        df["cumulative_ascent_m"] = df["altitude_delta_m"].clip(lower=0).fillna(0).cumsum()
        df["cumulative_descent_m"] = (-df["altitude_delta_m"].clip(upper=0)).fillna(0).cumsum()
    
    return df
