from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

PREFERRED_COLUMN_ORDER = [
"Key_break_points",
"distance_from_start_m",
"altitude_m",
"altitude_delta_m",
"ascent_delta_m",
"descent_delta_m",
"ascent_cumul_from_start_m",
"descent_cumul_from_start_m",
"grade_pct",
"Estimated_running_time",
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
    return pd.Series([pd.NA] * len(target_distances), dtype="float64")
  
  source = group[["distance_from_start_m", column]].copy()
  source["distance_from_start_m"] = pd.to_numeric(source["distance_from_start_m"], errors="coerce")
  source[column] = pd.to_numeric(source[column], errors="coerce")
  source = source.dropna(subset=["distance_from_start_m", column])
  source = source.drop_duplicates(subset=["distance_from_start_m"], keep="last").sort_values("distance_from_start_m")
  
  if source.empty:
      return pd.Series([pd.NA] * len(target_distances), dtype="float64")
  
  x = source["distance_from_start_m"].to_numpy(dtype=float)
  y = source[column].to_numpy(dtype=float)
  targets = np.asarray(target_distances, dtype=float)
  
  interpolated = np.interp(targets, x, y)
  return pd.Series(interpolated, dtype="float64")

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
  
  for _, group in grouped_iter:
      group = group.sort_values("distance_from_start_m").drop_duplicates(subset=["distance_from_start_m"], keep="last")
  
      if group.empty:
          continue
  
      max_distance_m = float(group["distance_from_start_m"].max())
      target_distances = _build_target_distances(max_distance_m, segment_length_m)
  
      out = pd.DataFrame({"distance_from_start_m": target_distances})
      out["Key_break_points"] = ""
      out["Estimated_running_time"] = pd.NA
  
      out["altitude_m"] = _interpolate_numeric(group, "altitude_m", target_distances)
      out["altitude_delta_m"] = out["altitude_m"].diff().fillna(0.0)
  
      step_distance_m = out["distance_from_start_m"].diff().fillna(segment_length_m)
      out["grade_pct"] = np.where(
          step_distance_m > 0,
          (out["altitude_delta_m"] / step_distance_m) * 100.0,
          0.0,
      )
  
      out["ascent_delta_m"] = out["altitude_delta_m"].clip(lower=0.0)
      out["descent_delta_m"] = (-out["altitude_delta_m"].clip(upper=0.0))
      out["ascent_cumul_from_start_m"] = out["ascent_delta_m"].fillna(0.0).cumsum()
      out["descent_cumul_from_start_m"] = out["descent_delta_m"].fillna(0.0).cumsum()
  
      normalized_groups.append(out)
  
  if not normalized_groups:
      return pd.DataFrame(columns=PREFERRED_COLUMN_ORDER)
  
  result = pd.concat(normalized_groups, ignore_index=True)
  
  if not result.empty:
      result = result[_order_columns(list(result.columns))]
  
  return result

def _interpolate_profile_row(reference_df: pd.DataFrame, distance_from_start_m: float) -> Dict[str, Any]:
  row: Dict[str, Any] = {
  "Key_break_points": "",
  "Estimated_running_time": pd.NA,
  "distance_from_start_m": float(distance_from_start_m),
  }
  
  reference = reference_df.copy().sort_values("distance_from_start_m").drop_duplicates(
      subset=["distance_from_start_m"],
      keep="last",
  )
  
  for col in reference.columns:
      if col in {"distance_from_start_m", "Key_break_points", "Estimated_running_time"}:
          continue
  
      if pd.api.types.is_numeric_dtype(reference[col]):
          source = reference[["distance_from_start_m", col]].copy()
          source["distance_from_start_m"] = pd.to_numeric(source["distance_from_start_m"], errors="coerce")
          source[col] = pd.to_numeric(source[col], errors="coerce")
          source = source.dropna(subset=["distance_from_start_m", col]).sort_values("distance_from_start_m")
  
          if source.empty:
              row[col] = pd.NA
          else:
              x = source["distance_from_start_m"].to_numpy(dtype=float)
              y = source[col].to_numpy(dtype=float)
              row[col] = float(np.interp([distance_from_start_m], x, y)[0])
  
  return row

def enhance_race_profile_with_breakpoints(
  race_profile_df: pd.DataFrame,
  aid_stations_df: pd.DataFrame,
  distance_tolerance_m: float = 1e-6,
  ) -> pd.DataFrame:
  if race_profile_df.empty:
  enhanced = race_profile_df.copy()
  if "Key_break_points" not in enhanced.columns:
  enhanced["Key_break_points"] = ""
  if "Estimated_running_time" not in enhanced.columns:
  enhanced["Estimated_running_time"] = pd.NA
  return enhanced
  
  base = race_profile_df.copy().sort_values("distance_from_start_m").reset_index(drop=True)
  
  if "Key_break_points" not in base.columns:
      base["Key_break_points"] = ""
  if "Estimated_running_time" not in base.columns:
      base["Estimated_running_time"] = pd.NA
  
  reference = base.copy().sort_values("distance_from_start_m").reset_index(drop=True)
  
  rows_to_add: List[Dict[str, Any]] = []
  
  if aid_stations_df is not None and not aid_stations_df.empty:
      aid = aid_stations_df.copy()
      aid["aid_station_name"] = aid["aid_station_name"].astype(str).str.strip()
      aid["aid_station_km"] = pd.to_numeric(aid["aid_station_km"], errors="coerce")
      aid = aid.dropna(subset=["aid_station_name", "aid_station_km"])
      aid = aid[aid["aid_station_name"] != ""]
      aid["distance_from_start_m"] = aid["aid_station_km"] * 1000.0
      aid = aid.sort_values("distance_from_start_m")
  
      for _, station in aid.iterrows():
          aid_name = str(station["aid_station_name"]).strip()
          aid_distance_m = float(station["distance_from_start_m"])
  
          mask = np.isclose(
              base["distance_from_start_m"].to_numpy(dtype=float),
              aid_distance_m,
              atol=distance_tolerance_m,
          )
  
          if mask.any():
              idx = base.index[mask][0]
              existing = base.at[idx, "Key_break_points"]
  
              if pd.isna(existing) or str(existing).strip() == "":
                  base.at[idx, "Key_break_points"] = aid_name
              else:
                  base.at[idx, "Key_break_points"] = f"{existing} / {aid_name}"
          else:
              new_row = _interpolate_profile_row(reference, aid_distance_m)
              new_row["Key_break_points"] = aid_name
              rows_to_add.append(new_row)
  
  if rows_to_add:
      base = pd.concat([base, pd.DataFrame(rows_to_add)], ignore_index=True)
      base = base.sort_values("distance_from_start_m").reset_index(drop=True)
  
  if not base.empty:
      base = base[_order_columns(list(base.columns))]
  
  return base
