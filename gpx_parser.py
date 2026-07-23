from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from typing import Any, Dict, List

import gpxpy
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0

PREFERRED_COLUMN_ORDER = [
"point_index",
"track_index",
"segment_index",
"point_index_in_segment",
"timestamp",
"latitude_deg",
"longitude_deg",
"altitude_m",
"altitude_delta_m",
"distance_delta_m",
"distance_from_start_m",
]

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  lat1_rad = radians(lat1)
  lon1_rad = radians(lon1)
  lat2_rad = radians(lat2)
  lon2_rad = radians(lon2)
  
  dlat = lat2_rad - lat1_rad
  dlon = lon2_rad - lon1_rad
  
  a = sin(dlat / 2.0) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2.0) ** 2
  c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
  
  return EARTH_RADIUS_M * c

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

def parse_gpx_to_table(gpx_source: Any) -> pd.DataFrame:
  gpx = gpxpy.parse(gpx_source)
  
  rows: List[Dict[str, Any]] = []
  point_index = 0
  
  for track_index, track in enumerate(gpx.tracks, start=1):
      track_name = getattr(track, "name", None)
  
      for segment_index, segment in enumerate(track.segments, start=1):
          previous_point = None
          cumulative_distance_m = 0.0
  
          for point_index_in_segment, point in enumerate(segment.points, start=1):
              if previous_point is None:
                  distance_delta_m = 0.0
                  altitude_delta_m = 0.0
              else:
                  distance_delta_m = _haversine_m(
                      previous_point.latitude,
                      previous_point.longitude,
                      point.latitude,
                      point.longitude,
                  )
                  cumulative_distance_m += distance_delta_m

              if (
                  previous_point.elevation is None
                  or point.elevation is None
              ):
                  altitude_delat_m = None
              else:
                  altitude_delta_m = point.elevation - previous_point.elevation
  
              row = {
                  "point_index": point_index,
                  "track_index": track_index,
                  "segment_index": segment_index,
                  "point_index_in_segment": point_index_in_segment,
                  "timestamp": point.time,
                  "latitude_deg": point.latitude,
                  "longitude_deg": point.longitude,
                  "altitude_m": point.elevation,
                  "altitude_delta_m": altitude_delta_m,
                  "distance_delta_m": distance_delta_m,
                  "distance_from_start_m": cumulative_distance_m,
              }
  
              rows.append(row)
              point_index += 1
              previous_point = point
  
  df = pd.DataFrame(rows)
  
  if not df.empty and "timestamp" in df.columns:
      df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
  
  if not df.empty:
      df = df[_order_columns(list(df.columns))]
  
  return df
