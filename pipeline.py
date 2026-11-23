from datetime import datetime, timezone
import os
import sqlite3
import numpy as np
import pandas as pd
import requests

DAILY_CSV_DIR = "daily_csv"
DB_PATH = "marine_pipeline.db"
os.makedirs(DAILY_CSV_DIR, exist_ok=True)


def fetch_erddap_json(
    dataset_id: str, variables: list[str], start_iso: str
) -> pd.DataFrame:
  """Fetches real-time or same-day records from Marine Institute ERDDAP JSON API."""
  vars_str = ",".join(variables)
  url = (
      f"https://erddap.marine.ie/erddap/tabledap/{dataset_id}.json?"
      f"{vars_str}&time>={start_iso}"
  )
  resp = requests.get(url, timeout=30)
  resp.raise_for_status()
  payload = resp.json()
  df = pd.DataFrame(
      payload["table"]["rows"], columns=payload["table"]["columnNames"]
  )
  if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], utc=True)
  return df


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Calculates great-circle distance in kilometers between two coordinates."""
  r = 6371.0
  phi1, phi2 = np.radians(lat1), np.radians(lat2)
  dphi = np.radians(lat2 - lat1)
  dlambda = np.radians(lon2 - lon1)
  a = (
      np.sin(dphi / 2.0) ** 2
      + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
  )
  return float(2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def classify_harbour_hazard(
    surge_m: float, wave_height_m: float, wind_knots: float
) -> str:
  """Classifies harbour safety tier based on storm surge anomaly and offshore sea state."""
  if (pd.notna(surge_m) and surge_m >= 0.40) or (
      pd.notna(wave_height_m) and wave_height_m >= 4.5
  ):
    return "Surge Alert"
  if (
      (pd.notna(surge_m) and surge_m >= 0.20)
      or (pd.notna(wave_height_m) and wave_height_m >= 2.5)
      or (pd.notna(wind_knots) and wind_knots >= 25.0)
  ):
    return "Caution"
  return "Normal"


def run_daily_pipeline():
  now_utc = datetime.now(timezone.utc)
  today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
  start_iso = today_start.strftime("%Y-%m-%dT%H:%M:%SZ")
  date_tag = today_start.strftime("%Y-%m-%d")

  # 1. Data Acquisition (5-min coastal tides, astronomical predictions, offshore buoys)
  df_tide = fetch_erddap_json(
      "IrishNationalTideGaugeNetwork",
      ["station_id", "time", "latitude", "longitude", "Water_Level_LAT"],
      start_iso,
  )
  df_pred = fetch_erddap_json(
      "IMI-TidePrediction", ["stationID", "time", "Water_Level"], start_iso
  )
  df_buoy = fetch_erddap_json(
      "IWBNetwork",
      [
          "station_id",
          "time",
          "latitude",
          "longitude",
          "AtmosphericPressure",
          "WindSpeed",
          "WaveHeight",
          "WavePeriod",
          "SeaTemperature",
      ],
      start_iso,
  )

  # Persist daily 5-minute raw CSV backups
  df_tide.to_csv(f"{DAILY_CSV_DIR}/tide_5min_{date_tag}.csv", index=False)
  df_buoy.to_csv(f"{DAILY_CSV_DIR}/buoy_hourly_{date_tag}.csv", index=False)

  # 2. Preprocessing & Feature Engineering
  clean_tide = df_tide.dropna(subset=["Water_Level_LAT"]).copy()
  clean_tide["Water_Level_LAT"] = pd.to_numeric(
      clean_tide["Water_Level_LAT"], errors="coerce"
  )
  clean_tide = clean_tide.dropna(subset=["Water_Level_LAT"]).sort_values(
      ["station_id", "time"]
  )
  clean_tide["rise_rate_m_hr"] = (
      clean_tide.groupby("station_id")["Water_Level_LAT"].diff() * 12.0
  ).round(4)

  pred = df_pred.rename(
      columns={"stationID": "station_id", "Water_Level": "predicted_level_m"}
  ).copy()
  pred["predicted_level_m"] = pd.to_numeric(
      pred["predicted_level_m"], errors="coerce"
  )
  merged = pd.merge(
      clean_tide,
      pred[["station_id", "time", "predicted_level_m"]],
      on=["station_id", "time"],
      how="left",
  )
  merged["storm_surge_m"] = (
      merged["Water_Level_LAT"] - merged["predicted_level_m"]
  ).round(4)

  for col in [
      "AtmosphericPressure",
      "WindSpeed",
      "WaveHeight",
      "WavePeriod",
      "SeaTemperature",
  ]:
    df_buoy[col] = pd.to_numeric(df_buoy[col], errors="coerce")

  buoy_coords = (
      df_buoy[["station_id", "latitude", "longitude"]]
      .dropna()
      .drop_duplicates("station_id")
  )
  nearest_map = {}
  for _, st in (
      merged[["station_id", "latitude", "longitude"]]
      .drop_duplicates("station_id")
      .iterrows()
  ):
    best_buoy, best_dist = None, float("inf")
    for _, b in buoy_coords.iterrows():
      d = haversine_km(
          st["latitude"], st["longitude"], b["latitude"], b["longitude"]
      )
      if d < best_dist:
        best_buoy, best_dist = b["station_id"], d
    nearest_map[st["station_id"]] = (best
