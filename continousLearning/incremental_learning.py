import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


MODEL_FILE = "sarima_model.pkl"
STATE_FILE = "model_state.json"

TARGET = "Global_active_power"

# SARIMA paraméterek
ORDER = (1, 1, 1)
SEASONAL_ORDER = (2, 1, 0, 24)

print("Reading dataset...")

df = pd.read_csv(
    "household_power.txt",
    sep=";",
    na_values="?"
)

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)

df = df.drop(columns=["Date", "Time"])
df = df.set_index("datetime")
df = df.sort_index()

df = df.resample("h").mean().ffill()
df.index.freq = "h"

print("Dataset ready.")

# első futás

if not os.path.exists(MODEL_FILE):

    print("No model found.")
    print("Training initial month...")

    first_month = df.index.to_period("M")[0]

    train = df[df.index.to_period("M") == first_month]

    model = SARIMAX(
        train[TARGET],
        order=ORDER,
        seasonal_order=SEASONAL_ORDER
    )

    results = model.fit(disp=False)

    results.save(MODEL_FILE)

    state = {
        "last_month": str(first_month)
    }

    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

    print("Initial model saved.")

    exit()


# modell betöltés
print("Loading model...")

results = SARIMAXResults.load(MODEL_FILE)

with open(STATE_FILE) as f:
    state = json.load(f)

last_month = pd.Period(state["last_month"])

print(f"Last trained month : {last_month}")

# következő hónap

months = sorted(df.index.to_period("M").unique())

try:
    idx = months.index(last_month)

except ValueError:

    raise RuntimeError("Saved month not found!")

if idx == len(months) - 1:

    print("No new month available.")
    exit()

next_month = months[idx + 1]

print(f"Testing month : {next_month}")

test = df[df.index.to_period("M") == next_month]

# forcast

forecast = results.forecast(
    steps=len(test)
)

print(forecast.head())
print(forecast.describe())

# measure 

mae = mean_absolute_error(
    test[TARGET],
    forecast
)

rmse = root_mean_squared_error(
    test[TARGET],
    forecast
)

print("------------------------------------")
print(test[TARGET].describe())
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print("------------------------------------")

plt.figure(figsize=(15,5))

plt.plot(test.index,
         test[TARGET],
         label="Valós")

plt.plot(test.index,
         forecast,
         label="SARIMA")

plt.legend()
plt.show()

# update

print("Updating model...")

results = results.append(
    test[TARGET],
    refit=True
)

results.save(MODEL_FILE)

state["last_month"] = str(next_month)

with open(STATE_FILE, "w") as f:
    json.dump(state, f)

print("Model updated and saved.")