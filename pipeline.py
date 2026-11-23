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
    best_buoy = None
    best_dist = float("inf")
    for _, b in buoy_coords.iterrows():
      d = haversine_km(
          st["latitude"], st["longitude"], b["latitude"], b["longitude"]
      )
      if d < best_dist:
        best_buoy = b["station_id"]
        best_dist = d
    nearest_map[st["station_id"]] = (best_buoy, round(best_dist, 1))

  merged["nearest_buoy_id"] = merged["station_id"].map(
      lambda s: nearest_map.get(s, (None, None))[0]
  )
  merged["buoy_distance_km"] = merged["station_id"].map(
      lambda s: nearest_map.get(s, (None, None))[1]
  )

  enriched = pd.merge_asof(
      merged.sort_values("time"),
      df_buoy[[
          "station_id",
          "time",
          "AtmosphericPressure",
          "WindSpeed",
          "WaveHeight",
          "SeaTemperature",
      ]].rename(columns={"station_id": "nearest_buoy_id"}),
      on="time",
      by="nearest_buoy_id",
      direction="backward",
      tolerance=pd.Timedelta("2h"),
  )
  enriched["hazard_tier"] = enriched.apply(
      lambda r: classify_harbour_hazard(
          r["storm_surge_m"], r["WaveHeight"], r["WindSpeed"]
      ),
      axis=1,
  )

  # 3. Load into Relational SQLite Database
  conn = sqlite3.connect(DB_PATH)
  cur = conn.cursor()
  cur.executescript("""
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            nearest_buoy_id TEXT,
            buoy_distance_km REAL
        );

        CREATE TABLE IF NOT EXISTS buoy_observations (
            station_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            atmospheric_pressure_hpa REAL,
            wind_speed_kn REAL,
            wave_height_m REAL,
            wave_period_s REAL,
            sea_temperature_c REAL,
            PRIMARY KEY (station_id, timestamp_utc)
        );

        CREATE TABLE IF NOT EXISTS harbour_telemetry (
            station_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            observed_level_m REAL NOT NULL,
            predicted_level_m REAL,
            storm_surge_m REAL,
            rise_rate_m_hr REAL,
            wave_height_m REAL,
            wind_speed_kn REAL,
            hazard_tier TEXT NOT NULL,
            PRIMARY KEY (station_id, timestamp_utc),
            FOREIGN KEY (station_id) REFERENCES stations(station_id)
        );
    """)

  stations_df = enriched[[
      "station_id",
      "latitude",
      "longitude",
      "nearest_buoy_id",
      "buoy_distance_km",
  ]].drop_duplicates("station_id")
  cur.executemany(
      "INSERT OR REPLACE INTO stations VALUES (?, ?, ?, ?, ?)",
      stations_df.itertuples(index=False, name=None),
  )

  buoy_load = df_buoy[[
      "station_id",
      "time",
      "AtmosphericPressure",
      "WindSpeed",
      "WaveHeight",
      "WavePeriod",
      "SeaTemperature",
  ]].copy()
  buoy_load["time"] = buoy_load["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  cur.executemany(
      "INSERT OR REPLACE INTO buoy_observations VALUES (?, ?, ?, ?, ?, ?, ?)",
      buoy_load.itertuples(index=False, name=None),
  )

  harbour_load = enriched[[
      "station_id",
      "time",
      "Water_Level_LAT",
      "predicted_level_m",
      "storm_surge_m",
      "rise_rate_m_hr",
      "WaveHeight",
      "WindSpeed",
      "hazard_tier",
  ]].copy()
  harbour_load["time"] = harbour_load["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  cur.executemany(
      "INSERT OR REPLACE INTO harbour_telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
      harbour_load.itertuples(index=False, name=None),
  )

  conn.commit()
  conn.close()
  print(
      f"[{date_tag}] Saved {len(harbour_load)} coastal rows and"
      f" {len(buoy_load)} buoy rows to {DB_PATH}"
  )


if __name__ == "__main__":
  run_daily_pipeline()
