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

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=pd.DatetimeIndex(df.index).tz)
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
    hour = pd.DatetimeIndex(df.index).hour.to_numpy()
    omega = 2 * math.pi / 24.0
    # TODO: Student can add whatever you like to the dataset here. Example here
    df["sin_day"] = np.sin(omega * hour)
    df["cos_day"] = np.cos(omega * hour)

    doy = pd.DatetimeIndex(df.index).dayofyear.to_numpy()
    omega_y = 2 * math.pi / 365.25
    # TODO: Student can add whatever you like to the dataset here
    df["sin_year"] = np.sin(omega_y * doy)
    df["cos_year"] = np.cos(omega_y * doy)

    # df["dP"] = df["P"].diff() 
    # df["dT"] = df["T"].diff() 
    # df["dRH"] = df["RH"].diff() 
    # df["T_mean_24"] = df["T"].rolling(24).mean()
    # df["T_std_24"] = df["T"].rolling(24).std()
    # df["W_mean_12"] = df["W"].rolling(12).mean()
    # df["P_mean_12"] = df["P"].rolling(12).mean()

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
    data = add_lags(data, "W", [1, 2, 3, 6, 12, 24, 48])
    data = add_lags(data, "Wd", [1, 2, 3, 6, 12, 24, 48])
    data = add_lags(data, "RH", [1, 2, 3, 6, 12, 24, 48])
    data = add_lags(data, "Prec", [1, 2, 3, 6, 12, 24, 48])
    data = add_lags(data, "Cloud", [1, 2, 3])
    # data = add_lags(data, "dP", [1, 2, 3, 6, 12, 24, 48])
    # data = add_lags(data, "dT", [1, 2, 3])
    # data = add_lags(data, "dRH", [1, 2])

    if horizon <= 3:
        feature_cols = [
            "T", "T_lag1", "T_lag3",
            "P", "P_lag1", "P_lag3",
            "W", "W_lag1", "W_lag3",
            "Wd_lag48",
            "Prec", "Prec_lag1", "Prec_lag3",
            "sin_day", "cos_day",
        ]

    elif horizon <= 6:
        feature_cols = [
            "T", "T_lag1", "T_lag3", "T_lag6",
            "P", "P_lag1", "P_lag3", "P_lag6",
            "W", "W_lag1", "W_lag3", "W_lag6",
            "Wd_lag48", "Wd_lag24",
            "Prec", "Prec_lag1", "Prec_lag3", "Prec_lag6",
            "sin_day", "cos_day",
            "sin_year", "cos_year",
        ]

    elif horizon <= 12:
        feature_cols = [
            "T", "T_lag1", "T_lag6", "T_lag12",
            "P", "P_lag1", "P_lag3", "P_lag6", "P_lag12",
            "Wd_lag48", "Wd_lag24", "Wd_lag12",
            "RH_lag48", "RH_lag24", "RH_lag12",
            "sin_day", "cos_day",
            "sin_year", "cos_year",
        ]

    elif horizon <= 24:
        feature_cols = [
            "T", "T_lag1", "T_lag3", "T_lag6", "T_lag12", "T_lag24",
            "P", "P_lag1", "P_lag3", "P_lag6", "P_lag12", "P_lag24",
            "Wd_lag48", "Wd_lag24", "Wd_lag12", "Wd_lag6",
            "RH_lag48", "RH_lag24", "RH_lag12", "RH_lag6",
            "sin_day", "cos_day",
            "sin_year", "cos_year",
        ]

    else:
        feature_cols = [
            "T", "T_lag1", "T_lag3", "T_lag6", "T_lag12", "T_lag24", "T_lag48",
            "P", "P_lag1", "P_lag3", "P_lag6", "P_lag12", "P_lag24", "P_lag48",
            "Wd_lag48", "Wd_lag24", "Wd_lag12", "Wd_lag6", "Wd_lag3",
            "RH_lag48", "RH_lag24", "RH_lag12", "RH_lag6", "RH_lag3",
            "sin_day", "cos_day",
            "sin_year", "cos_year",
        ]
    data["y_target"] = data["T"].shift(-horizon)
    data = data.dropna(subset=feature_cols + ["y_target"])

    X = data[feature_cols].to_numpy()
    y = data["y_target"].to_numpy()

  #  X = np.hstack([np.ones((X.shape[0], 1)), X])

    return X, y, feature_cols

def fit_standardizer(X_std: np.ndarray):
    
    mu = X_std.mean(axis=0)
    sigma = X_std.std(axis=0, ddof=0)

    sigma[sigma==0] = 1

    return mu, sigma

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    X_scaled = (X - mu) / sigma
    X = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    return X

def model_training(X_train: np.ndarray, y_train: np.ndarray):

    theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    return theta

"""
def model_training_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha=1.0):
    n_features = X_train.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  # don't penalize the bias term
    theta = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
    return theta
"""
    
def prediction(X: np.ndarray, theta: np.ndarray):

    return X @ theta

def baseline(X_val_raw: np.ndarray):
    return X_val_raw[:, 0]   # column 0 of the raw matrix = T_lag1 (already in °C)

def metric_computation(y_pred: np.ndarray, y_true: np.ndarray):

    error = y_pred - y_true
    RMSE = np.sqrt(np.mean(error**2))
    MAE = np.mean(np.abs(error))

    return RMSE, MAE

def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    horizon: int,
):
    W_DOUBLE = 7.10
    GOLDEN   = (5**0.5 - 1) / 2

    plt.figure(figsize=(W_DOUBLE, W_DOUBLE*GOLDEN))
    plt.plot(y_true,     label="Observed",  linewidth=1.0)
    plt.plot(y_pred,     label="Predicted", linewidth=1.0, alpha=0.8)
    plt.plot(y_baseline, label="Baseline",  linewidth=0.8, linestyle="--", alpha=0.6)
    plt.title(f"Temperature forecast — {horizon}h ahead (validation set)")
    plt.ylabel("T [°C]")
    plt.xlabel("Hour index (validation period)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    
if __name__ == "__main__":
    start_date = "2025-06-01"
    end_date = "2025-09-01"
    
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

    # alphas = {1: 1, 3: 1, 6: 1, 12: 1, 24: 1000, 48: 1000}
    horizons = [1, 3, 6, 12, 24, 48]
    results = []

    # print("--- Alpha tuning ---")
    """
    for horizon in horizons:
        X_train_raw, y_train_raw, _ = design_matrix(df_train, horizon)
        X_val_raw, y_val_raw, feature_cols     = design_matrix(df_val, horizon)
        mu, sigma   = fit_standardizer(X_train_raw)
        X_train     = apply_standardizer(X_train_raw, mu, sigma)
        X_val       = apply_standardizer(X_val_raw, mu, sigma)
        
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            theta  = model_training_ridge(X_train, y_train_raw, alpha=alpha)
            y_pred = prediction(X_val, theta)
            rmse, _ = metric_computation(y_pred, y_val_raw)
            print(f"  horizon={horizon}h  alpha={alpha}: RMSE={rmse:.2f}")
        """
    from analysis import plot_metrics_vs_horizon, plot_residuals

    feature_cols_dict = {}
    y_true_dict      = {}
    y_pred_dict      = {}

    for horizon in horizons:

        X_train_raw, y_train_raw, _ = design_matrix(df_train, horizon)
        X_val_raw, y_val_raw, feature_cols = design_matrix(df_val, horizon)

        print("Condition number before standardizing", np.linalg.cond(X_train_raw))

        mu, sigma = fit_standardizer(X_train_raw)

        X_train = apply_standardizer(X_train_raw, mu, sigma)
        X_val = apply_standardizer(X_val_raw, mu, sigma)

        print("Condition number after standardizing", np.linalg.cond(X_train))

        theta = model_training(X_train, y_train_raw)
        y_pred = prediction(X_val, theta)
        y_baseline = baseline(X_val_raw)

        rmse_model, mae_model = metric_computation(y_pred, y_val_raw)
        rmse_baseline, mae_baseline = metric_computation(y_pred, y_baseline)

        print(f"{horizon}h: model RMSE={rmse_model:.2f}, MAE={mae_model:.2f}, baseline RMSE={rmse_baseline:.2f}") # , baseline RMSE={rmse_baseline:.2f}"

        plot_forecast(y_val_raw, y_pred, y_baseline, horizon) # y_baseline

        results.append({
        "Model": "Model 1",
        "Date Range": f"{start_date} to {end_date}",
        "Horizon (h)": horizon,
        "RMSE": rmse_model,
        "MAE": mae_model,
        })

        feature_cols_dict[horizon] = feature_cols  # return from design_matrix
        y_true_dict[horizon]       = y_val_raw
        y_pred_dict[horizon]       = y_pred

    metric_results = pd.DataFrame(results)

    plot_metrics_vs_horizon(results)
    plot_residuals(y_true_dict, y_pred_dict)
    # metric_results.to_excel("Model.xlsx", index = False) 

