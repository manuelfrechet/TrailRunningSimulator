from io import BytesIO
from typing import Any, Dict, List

import fitdecode
import pandas as pd

SEMICIRCLES_TO_DEGREES = 180.0 / (2**31)

def _make_unique_key(base_key: str, existing_keys: set[str]) -> str:
    if base_key not in existing_keys:
        return base_key

counter = 2
while f"{base_key}_{counter}" in existing_keys:
    counter += 1
return f"{base_key}_{counter}"

def _semicircles_to_degrees(value: Any) -> Any:
    if value is None:
        return None

if isinstance(value, (int, float)):
    return float(value) * SEMICIRCLES_TO_DEGREES

return value

def _frame_to_row(frame: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    existing_keys: set[str] = set()

for field in frame.fields:
    base_key = str(field.name_or_num)
    key = _make_unique_key(base_key, existing_keys)
    existing_keys.add(key)
    row[key] = field.value

row["message_type"] = frame.name

if "position_lat" in row:
    row["latitude_deg"] = _semicircles_to_degrees(row["position_lat"])

if "position_long" in row:
    row["longitude_deg"] = _semicircles_to_degrees(row["position_long"])

return row

def parse_fit_to_tables(fit_source: Any) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, List[Dict[str, Any]]] = {}

with fitdecode.FitReader(fit_source) as fit:
    for frame in fit:
        if frame.frame_type != fitdecode.FIT_FRAME_DATA:
            continue

        row = _frame_to_row(frame)
        message_type = frame.name

        if message_type not in tables:
            tables[message_type] = []

        tables[message_type].append(row)

dfs: Dict[str, pd.DataFrame] = {}

for message_type, rows in tables.items():
    df = pd.DataFrame(rows)

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    dfs[message_type] = df

return dfs

def tables_to_excel_bytes(tables: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    for sheet_name, df in tables.items():
        safe_sheet_name = sheet_name[:31] if sheet_name else "sheet"
        df.to_excel(writer, index=False, sheet_name=safe_sheet_name)

return output.getvalue()
