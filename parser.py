from **future** import annotations

from typing import Any, Optional

import pandas as pd
import fitdecode

SEMICIRCLES_TO_DEGREES = 360.0 / (2**32)

def _safe_value(frame: Any, field_name: str, fallback: Any = None) -> Any:
"""
Read a FIT field safely.

```
fitdecode documents `get_value(..., fallback=...)`, so this lets us ask for
a field even when it may be absent in a given record.
"""
try:
    return frame.get_value(field_name, fallback=fallback)
except Exception:
    return fallback
```

def _semicircles_to_degrees(value: Any) -> Optional[float]:
"""
FIT GPS coordinates are commonly stored as semicircles.

```
If the value is numeric, convert it to decimal degrees.
Otherwise return None.
"""
if value is None:
    return None

if isinstance(value, (int, float)):
    return float(value) * SEMICIRCLES_TO_DEGREES

return None
```

def parse_fit_to_dataframe(file_path: str) -> pd.DataFrame:
"""
Parse one FIT file and return a pandas DataFrame.

```
This first version keeps only record messages (the track points) and extracts
the most useful fields for the historical analysis phase.
"""
rows = []

with fitdecode.FitReader(file_path) as fit:
    for frame in fit:
        # Keep only FIT data messages.
        if frame.frame_type != fitdecode.FIT_FRAME_DATA:
            continue

        # For activity files, the track-point messages are usually named "record".
        # If your file behaves differently, we will adjust after the first test.
        if frame.name != "record":
            continue

        lat_raw = _safe_value(frame, "position_lat")
        lon_raw = _safe_value(frame, "position_long")

        row = {
            "timestamp": _safe_value(frame, "timestamp"),
            "latitude": _semicircles_to_degrees(lat_raw),
            "longitude": _semicircles_to_degrees(lon_raw),
            "distance_m": _safe_value(frame, "distance"),
            "altitude_m": _safe_value(frame, "enhanced_altitude", _safe_value(frame, "altitude")),
            "speed_m_s": _safe_value(frame, "enhanced_speed", _safe_value(frame, "speed")),
            "heart_rate_bpm": _safe_value(frame, "heart_rate"),
            "cadence_rpm": _safe_value(frame, "cadence"),
            "temperature_c": _safe_value(frame, "temperature"),
        }

        rows.append(row)

df = pd.DataFrame(rows)

if not df.empty and "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

return df
```

def fit_to_csv(file_path: str, output_csv_path: str) -> pd.DataFrame:
"""
Convenience helper: parse a FIT file and write a CSV.
"""
df = parse_fit_to_dataframe(file_path)
df.to_csv(output_csv_path, index=False)
return df

def fit_to_excel(file_path: str, output_xlsx_path: str) -> pd.DataFrame:
"""
Convenience helper: parse a FIT file and write an Excel file.
"""
df = parse_fit_to_dataframe(file_path)

```
with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="records")

return df
```
