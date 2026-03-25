from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests 
import matplotlib.pyplot as plt

from sample_code import preprocess, fetch_open_meteo_hourly

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

def initial_plot():
    start_date = "2026-01-01"
    end_date = "2026-03-07"
    
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

    plt.figure()
    df["T"].plot(linewidth=1)
    plt.title("Montreal hourly temperature (2m)")
    plt.ylabel("T [°C]")
    plt.tight_layout()
    plt.show()

    return

if __name__ == "__main__":

    initial_plot()
    

