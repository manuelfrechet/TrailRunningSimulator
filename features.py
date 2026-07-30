from __future__ import annotations

import pandas as pd

def build_features(record_df: pd.DataFrame) -> pd.DataFrame:
    df = record_df.copy()
    
    if df.empty:
        return df
    
    rename_map = {
        "enhanced_speed": "speed_m_s",
        "enhanced_altitude": "altitude_m",
        "heart_rate": "heart_rate_bpm",
        "distance": "distance_from_start_m",
        "step_length": "step_length_m",
        "stance_time": "stance_time_s",
        "vertical_oscillation": "vertical_oscillation_mm",
        "vertical_ratio": "vertical_ratio_pct",
    }
    df = df.rename(columns=rename_map)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["time_from_start_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    else:
        df["time_from_start_s"] = pd.NA
    
    if "distance_from_start_m" in df.columns:
        df["distance_from_start_m"] = pd.to_numeric(df["distance_from_start_m"], errors="coerce")
        df["distance_delta_m"] = df["distance_from_start_m"].diff()
    else:
        df["distance_delta_m"] = pd.NA
    
    if "altitude_m" in df.columns:
        df["altitude_m"] = pd.to_numeric(df["altitude_m"], errors="coerce")
        df["altitude_delta_m"] = df["altitude_m"].diff()
    else:
        df["altitude_delta_m"] = pd.NA
    
    if "altitude_delta_m" in df.columns:
        df["ascent_delta_m"] = df["altitude_delta_m"].clip(lower=0)
        df["descent_delta_m"] = (-df["altitude_delta_m"].clip(upper=0))
        df["ascent_cumul_from_start_m"] = df["ascent_delta_m"].fillna(0).cumsum()
        df["descent_cumul_from_start_m"] = df["descent_delta_m"].fillna(0).cumsum()
    else:
        df["ascent_delta_m"] = pd.NA
        df["descent_delta_m"] = pd.NA
        df["ascent_cumul_from_start_m"] = pd.NA
        df["descent_cumul_from_start_m"] = pd.NA
    
    if "distance_delta_m" in df.columns and "altitude_delta_m" in df.columns:
        df["grade_pct"] = (df["altitude_delta_m"] / df["distance_delta_m"]) * 100.0
        df.loc[df["distance_delta_m"] <= 0, "grade_pct"] = pd.NA
    else:
        df["grade_pct"] = pd.NA
    
    if "cadence" in df.columns:
        cadence = pd.to_numeric(df["cadence"], errors="coerce")
    else:
        cadence = pd.Series(pd.NA, index=df.index, dtype="float64")
    
    if "fractional_cadence" in df.columns:
        fractional_cadence = pd.to_numeric(df["fractional_cadence"], errors="coerce")
    else:
        fractional_cadence = pd.Series(0.0, index=df.index, dtype="float64")
    
    df["cadence_spm"] = (cadence + fractional_cadence) * 2.0
    
    if "time_from_start_s" in df.columns and "distance_from_start_m" in df.columns:
        time_delta_s = df["time_from_start_s"].diff()
        df["speed_m_s_from_distance"] = df["distance_delta_m"] / time_delta_s
        df.loc[time_delta_s <= 0, "speed_m_s_from_distance"] = pd.NA
    
        df["pace_min_km_from_distance"] = (1000.0 / df["speed_m_s_from_distance"]) / 60.0
        df.loc[df["speed_m_s_from_distance"] <= 0, "pace_min_km_from_distance"] = pd.NA
    else:
        df["speed_m_s_from_distance"] = pd.NA
        df["pace_min_km_from_distance"] = pd.NA
    
    display_columns = [
        "timestamp",
        "time_from_start_s",
        "distance_from_start_m",
        "distance_delta_m",
        "altitude_m",
        "altitude_delta_m",
        "ascent_delta_m",
        "descent_delta_m",
        "ascent_cumul_from_start_m",
        "descent_cumul_from_start_m",
        "grade_pct",
        "speed_m_s",
        "speed_m_s_from_distance",
        "pace_min_km_from_distance",
        "heart_rate_bpm",
        "power",
        "accumulated_power",
        "temperature",
        "cadence_spm",
        "step_length_m",
        "stance_time_s",
        "vertical_oscillation_mm",
        "vertical_ratio_pct",
    ]
    
    existing_columns = [c for c in display_columns if c in df.columns]
    return df[existing_columns]
