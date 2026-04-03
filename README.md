# Mech309 — Montreal Temperature Forecasting

A multivariate linear regression model for hourly temperature forecasting in Montreal, QC. The model predicts temperature at multiple forecast horizons (1h, 3h, 6h, 12h, 24h, 48h) using historical meteorological data fetched from the Open-Meteo archive API.

---

## Project Structure

```
Mech309_Project/
├── main.py              # Main model — baseline linear regression
├── sample_code.py       # Reference code provided by course
├── analysis.py          # Plotting utilities (metrics, residuals)
├── requirements.txt     # Python dependencies
├── Model.xlsx           # Condition numbers
└── figures/             # Saved forecast plots
```

---


## Dependencies

```
numpy
scipy
matplotlib
pandas
requests
```

---

## Usage

### Run the model (analysis.py plots will also be obtained)

```bash
python main.py
```

---

## How It Works

### 1. Data Fetching
Hourly meteorological data is fetched from the [Open-Meteo Archive API](https://open-meteo.com/) for Montreal, QC. Variables fetched:
- Temperature (2m)
- Wind speed (10m)
- Wind direction (10m)
- Relative humidity (2m)
- Surface pressure
- Precipitation
- Cloud cover

### 2. Preprocessing
- Duplicate timestamps removed
- Missing values interpolated (limit: 6 hours), then forward/backward filled
- Variables renamed to short codes: `T`, `W`, `Wd`, `RH`, `P`, `Prec`, `Cloud`
- Diurnal cycle encoded as `sin_day`, `cos_day` (24h period)
- Seasonal cycle encoded as `sin_year`, `cos_year` (365.25 day period)

### 3. Feature Engineering
Lag features are added per predictor variable. Lag sets are **horizon-specific** — each horizon uses only the lags that are physically relevant to that forecast distance:

| Horizon | Temperature lags | Pressure lags | Additional variables |
|---|---|---|---|
| 1h, 3h | 1, 3 | 1, 3 | W, Wd, Prec (short lags) |
| 6h | 1, 3, 6 | 1, 3, 6 | W, Wd, Prec |
| 12h | 1, 6, 12 | 1, 3, 6, 12 | RH, Wd (medium lags) |
| 24h | 1, 3, 6, 12, 24 | 1, 3, 6, 12, 24 | RH, Wd (long lags) |
| 48h | 1, 3, 6, 12, 24, 48 | 1, 3, 6, 12, 24, 48 | RH, Wd (longest lags) |

### 4. Design Matrix
For each horizon, `design_matrix()` builds the feature matrix X and target vector y, with lagged columns, and horizon specific variables.

### 5. Standardization
Features are standardized using mean and standard deviation computed, then applied to both train and validation sets:

### 6. Baseline Comparison
A persistence baseline is included — it predicts the most recent observed temperature (`T_lag1`) as the forecast. This serves as a lower bound for model performance.

### 7. Evaluation
Models are evaluated on a held-out validation set (last 20% of data) using:
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error

---

---

## Branch Structure

```
main                   ← stable, reviewed code
feature/iterations     ← model iteration experiments  
feature/pressure_lagtest ← pressure lag and Ridge regression experiments
```

---
