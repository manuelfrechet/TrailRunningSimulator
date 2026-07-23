from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

PREFERRED_COLUMN_ORDER = [
"track_index",
"segment_index",
"normalized_point_index",
"distance_from_start_m",
"distance_delta_m",
"timestamp",
"latitude_deg",
"longitude_deg",
"altitude_m",
"altitude_delta_m",
"track_name",
]

def _build_target_distances(max_distance_m: float, segment_length_m: float) -> List[float]:
  if segment_length_m <= 0:
    raise ValueError("segment_length_m must be greater than 0")
  
  max_distance_m = max(0.0, float(max_distance_m))
  
  targets: List[float] = []
  current = 0.0
  
  while current < max_distance_m:
      targets.append(round(current, 6))
      current += segment_length_m
  
  targets.append(round(max_distance_m, 6))
  return sorted(set(targets))

def _order_columns(columns: List[str]) -> List[str]:
  seen: set[str] = set()
  ordered: List[str] = []
  
  for col in PREFERRED_COLUMN_ORDER:
      if col in columns and col not in seen:
          ordered.append(col)
          seen.add(col)
  
  remaining = [col for col in columns if col not in seen]
  ordered.extend(remaining)
  
  return ordered

def _interpolate_numeric(group: pd.DataFrame, column: str, target_distances: List[float]) -> pd.Series:
  if column not in group.columns:
    return pd.Series([float("nan")] * len(target_distances), index=target_distances, dtype="float64")
  
  source = group[["distance_from_start_m", column]].copy()
  source["distance_from_start_m"] = pd.to_numeric(source["distance_from_start_m"], errors="coerce")
  source[column] = pd.to_numeric(source[column], errors="coerce")
  source = source.dropna(subset=["distance_from_start_m"])
  source = source.drop_duplicates(subset=["distance_from_start_m"], keep="last").sort_values("distance_from_start_m")
  
  if source.empty:
      return pd.Series([float("nan")] * len(target_distances), index=target_distances, dtype="float64")
  
  indexed = pd.Series(
      source[column].values,
      index=source["distance_from_start_m"].astype(float).values,
      dtype="float64",
  )
  
  combined_index = indexed.index.union(pd.Index(target_distances))
  combined = indexed.reindex(combined_index).sort_index()
  interpolated = combined.interpolate(method="index").ffill().bfill()
  
  return interpolated.reindex(target_distances)

def _interpolate_datetime(group: pd.DataFrame, column: str, target_distances: List[float]) -> pd.Series:
  if column not in group.columns:
    return pd.Series([pd.NaT] * len(target_distances), index=target_distances)
  
  source = group[["distance_from_start_m", column]].copy()
  source["distance_from_start_m"] = pd.to_numeric(source["distance_from_start_m"], errors="coerce")
  source[column] = pd.to_datetime(source[column], errors="coerce")
  source = source.dropna(subset=["distance_from_start_m", column])
  source = source.drop_duplicates(subset=["distance_from_start_m"], keep="last").sort_values("distance_from_start_m")
  
  if source.empty:
      return pd.Series([pd.NaT] * len(target_distances), index=target_distances)
  
  timestamp_ns = pd.Series(
      [pd.Timestamp(value).value if pd.notna(value) else pd.NA for value in source[column]],
      index=source["distance_from_start_m"].astype(float).values,
      dtype="float64",
  )
  
  combined_index = timestamp_ns.index.union(pd.Index(target_distances))
  combined = timestamp_ns.reindex(combined_index).sort_index()
  interpolated = combined.interpolate(method="index").ffill().bfill()
  values = interpolated.reindex(target_distances)
  
  return pd.to_datetime(values, unit="ns", errors="coerce")

def build_fixed_distance_segments(gpx_df: pd.DataFrame, segment_length_m: float = 10.0) -> pd.DataFrame:
  if gpx_df.empty or "distance_from_start_m" not in gpx_df.columns:
    return pd.DataFrame(columns=PREFERRED_COLUMN_ORDER)
  
  working = gpx_df.copy()
  working["distance_from_start_m"] = pd.to_numeric(working["distance_from_start_m"], errors="coerce").round(6)
  working = working.dropna(subset=["distance_from_start_m"])
  
  if working.empty:
      return pd.DataFrame(columns=PREFERRED_COLUMN_ORDER)
  
  group_columns = [col for col in ["track_index", "segment_index"] if col in working.columns]
  
  normalized_groups: List[pd.DataFrame] = []
  
  if group_columns:
      grouped_iter = working.groupby(group_columns, dropna=False, sort=True)
  else:
      grouped_iter = [(None, working)]
  
  for group_key, group in grouped_iter:
      group = group.sort_values("distance_from_start_m").drop_duplicates(subset=["distance_from_start_m"], keep="last")
  
      if group.empty:
          continue
  
      max_distance_m = float(group["distance_from_start_m"].max())
      target_distances = _build_target_distances(max_distance_m, segment_length_m)
  
      out = pd.DataFrame({"distance_from_start_m": target_distances})
      out["normalized_point_index"] = range(1, len(out) + 1)
      out["distance_delta_m"] = out["distance_from_start_m"].diff().fillna(0.0)
      out["timestamp"] = _interpolate_datetime(group, "timestamp", target_distances)
      out["latitude_deg"] = _interpolate_numeric(group, "latitude_deg", target_distances)
      out["longitude_deg"] = _interpolate_numeric(group, "longitude_deg", target_distances)
      out["altitude_m"] = _interpolate_numeric(group, "altitude_m", target_distances)
      out["altitude_delta_m"] = out["altitude_m"].diff().fillna(0.0)
  
      if group_columns:
          if not isinstance(group_key, tuple):
              group_key = (group_key,)
  
          for col, value in zip(group_columns, group_key):
              out[col] = value
  
      if "track_name" in group.columns:
          non_null_track_name = group["track_name"].dropna()
          out["track_name"] = non_null_track_name.iloc[0] if not non_null_track_name.empty else pd.NA
  
      normalized_groups.append(out)
  
  if not normalized_groups:
      return pd.DataFrame(columns=PREFERRED_COLUMN_ORDER)
  
  result = pd.concat(normalized_groups, ignore_index=True)
  
  if not result.empty:
      result = result[_order_columns(list(result.columns))]
  
  return result
