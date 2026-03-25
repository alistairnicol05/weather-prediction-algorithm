# Mech309_Project
Data-Calibrated Local Weather and Air Quality Prediction

# Preprocessing

Consider adding:
- rate of temperature change
- rate of wind change
- (wind direction)
    wind_rad = np.deg2rad(df["Wd"])
    df["Wx"] = df["W"] * np.cos(wind_rad)   # east-west component
    df["Wy"] = df["W"] * np.sin(wind_rad)   # north-south component

# Training/Validation

- We should aim for 80% training, 20% validation (can be modified)