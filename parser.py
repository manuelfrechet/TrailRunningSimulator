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

def _order_columns(columns: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []

    for col in PREFERRED_COLUMN_ORDER:
        if col in columns and col not in seen:
            ordered.append(col)
            seen.add(col)

    remaining_named = [col for col in columns if col not in seen and not col.isdigit()]
    ordered.extend(remaining_named)
    seen.update(remaining_named)

    numeric_unknowns = sorted(
        [col for col in columns if col not in seen and col.isdigit()],
        key=lambda x: int(x),
    )
    ordered.extend(numeric_unknowns)

    return ordered

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

        #if not df.empty:
        #    df = df[_order_columns(list(df.columns))]
            
        dfs[message_type] = df

    return dfs
