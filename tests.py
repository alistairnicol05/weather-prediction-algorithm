from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests 
import matplotlib.pyplot as plt

import os # to organise files

@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str


MONTREAL = Location(
    name="Montreal, QC",
    lat=45.5017,
    lon=-73.5673,
    timezone="America/Montreal",
)

def fetch_open_meteo_hourly(
    start_date: str,
    end_date: str,
    location: Location = MONTREAL,
    hourly_vars: List[str] | None = None,
) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "cloud_cover",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": location.timezone,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", None)
    if times is None:
        raise RuntimeError(f"Open-Meteo response missing 'hourly.time'. Keys: {payload.keys()}")

    idx = pd.to_datetime(times)
    df = pd.DataFrame(index=idx)
    for k, v in hourly.items():
        if k == "time":
            continue
        df[k] = v
    df.index.name = "time_local"
    return df

# Preprocessing + features
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    df = df.reindex(full_idx)

    df = df.interpolate(limit=6)
    df = df.ffill().bfill()

    rename = {
        "temperature_2m": "T",
        "wind_speed_10m": "W",
        "wind_direction_10m": "Wd",
        "relative_humidity_2m": "RH",
        "surface_pressure": "P",
        "precipitation": "Prec",
        "cloud_cover": "Cloud",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Diurnal + seasonal features
    hour = df.index.hour.to_numpy()
    omega = 2 * math.pi / 24.0
    # TODO: Student can add whatever you like to the dataset here. Example here
    df["sin_day"] = np.sin(omega * hour)
    df["cos_day"] = np.cos(omega * hour)

    doy = df.index.dayofyear.to_numpy()
    omega_y = 2 * math.pi / 365.25
    # TODO: Student can add whatever you like to the dataset here
    df["sin_year"] = np.sin(omega_y * doy)
    df["cos_year"] = np.cos(omega_y * doy)

    return df

# TODO: Students you might find this function useful
def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for L in lags:
        if L <= 0:
            continue
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

# TODO: Students you might find this function useful
def split_train_val(data: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) <= val_hours + 10:
        raise ValueError("Not enough samples for requested validation window.")
    return data.iloc[:-val_hours].copy(), data.iloc[-val_hours:].copy()

def design_matrix(df: pd.DataFrame, horizon: int):

    data = df.copy()

    data = add_lags(data, "T", [1, 2, 3, 6, 12, 24, 48])
    data = add_lags(data, "P", [1, 2, 3, 6, 12, 24, 48])

    feature_cols = [
        "T_lag1", "T_lag2", "T_lag3", "T_lag6", "T_lag12", "T_lag24", "T_lag48",
        "Wd", 
        "RH",
        "P_lag1", "P_lag2", "P_lag3", "P_lag6", "P_lag12", "P_lag24", "P_lag48",
        "Cloud",
        "sin_day", "cos_day",
        "sin_year", "cos_year",
    ]

    data["y_target"] = data["T"].shift(-horizon)
    data = data.dropna(subset=feature_cols + ["y_target"])

    X = data[feature_cols].to_numpy()
    y = data["y_target"].to_numpy()

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    return X, y


def model_training(X_train: np.array, y_train: np.array):

    theta, residuals, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)

    return theta

def prediction(X: np.array, theta: np.array):

    return X @ theta

""" Baseline prediction

def baseline(X_val: np.array):

    return X_val[:,1]

"""
def metric_computation(y_pred: np.ndarray, y_true: np.ndarray):

    error = y_pred - y_true
    RMSE = np.sqrt(np.mean(error**2))
    MAE = np.mean(np.abs(error))

    return RMSE, MAE

def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    # y_baseline: np.ndarray,
    horizon: int,
):
    

    plt.figure(figsize=(12, 4))
    plt.plot(y_true,     label="Observed",  linewidth=1.0)
    plt.plot(y_pred,     label="Predicted", linewidth=1.0, alpha=0.8)
    # plt.plot(y_baseline, label="Baseline",  linewidth=0.8, linestyle="--", alpha=0.6)
    plt.title(f"Temperature forecast — {horizon}h ahead (validation set)")
    plt.ylabel("T [°C]")
    plt.xlabel("Hour index (validation period)")
    plt.legend()
    plt.tight_layout()
    folder = f"figures/{start_date}_to_{end_date}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/forecast_{horizon}h.png")
    plt.close()

    print(f"  Saved figures/forecast_{horizon}h.png")
    

if __name__ == "__main__":
    start_date = "2025-01-01"
    end_date = "2025-03-07"
    
    montreal = Location(
        name="Montreal, QC",
        lat=45.5017,
        lon=-73.5673,
        timezone="America/Montreal",
    )

    print(f"Fetching Open-Meteo hourly data for {montreal.name}...")
    df_raw = fetch_open_meteo_hourly(start_date, end_date, location=montreal)
    print("Preprocessing...")
    df = preprocess(df_raw)

    val_hours = int(len(df)*0.2)
    df_train, df_val = split_train_val(df, val_hours)

    horizons = [1, 3, 6, 12, 24, 48]
    results = []

    for horizon in horizons:

        X_train, y_train = design_matrix(df_train, horizon)
        X_val, y_val = design_matrix(df_val, horizon)

        theta = model_training(X_train, y_train)
        y_pred = prediction(X_val, theta)
        #y_baseline = baseline(X_val)

        rmse_model, mae_model = metric_computation(y_pred, y_val)
        # rmse_baseline, mae_baseline = metric_computation(y_pred, y_baseline)

        print(f"{horizon}h: model RMSE={rmse_model:.2f}, MAE={mae_model:.2f}") # , baseline RMSE={rmse_baseline:.2f}"

        plot_forecast(y_val, y_pred, horizon) # y_baseline

        results.append({
        "Model": "Model 1",
        "Date Range": f"{start_date} to {end_date}",
        "Horizon (h)": horizon,
        "RMSE": rmse_model,
        "MAE": mae_model,
        })

    metric_results = pd.DataFrame(results)
    metric_results.to_excel("Model.xlsx", index = False)
