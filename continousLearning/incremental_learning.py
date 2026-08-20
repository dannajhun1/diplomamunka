import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

MODEL_FILE = "sarima_model.pkl"
STATE_FILE = "model_state.json"

TARGET = "Global_active_power"

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 0, 168)   # heti szezon (168 óra)
TRAIN_WINDOW_DAYS = 45           # sliding window: utolsó 30 nap

print("Reading dataset...")

df = pd.read_csv("household_power.txt", sep=";", na_values="?")

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

# ---------------------------------------------------------
# 1) INITIAL STATE – start AFTER the first full month + 30 days
# ---------------------------------------------------------

if not os.path.exists(STATE_FILE):

    print("No state found, initializing...")

    # első hónap meghatározása
    first_month = df.index.to_period("M")[0]
    first_month_end = first_month.to_timestamp() + pd.offsets.MonthEnd(0)

    # sliding window induljon az első hónap UTÁN
    start_day = first_month_end + pd.Timedelta(days=1)

    # első forecast nap legyen 30 nappal később
    next_day = start_day + pd.Timedelta(days=TRAIN_WINDOW_DAYS)

    # ha nincs elég adat → léptessük addig, amíg van
    while next_day > df.index[-1].normalize():
        start_day += pd.Timedelta(days=1)
        next_day = start_day + pd.Timedelta(days=TRAIN_WINDOW_DAYS)

    state = {
        "next_day": str(next_day.normalize())
    }

    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

    print("Initial state saved.")
    exit()

# ---------------------------------------------------------
# 2) LOAD STATE
# ---------------------------------------------------------

with open(STATE_FILE) as f:
    state = json.load(f)

next_day = pd.to_datetime(state["next_day"]).normalize()
print(f"Forecasting day: {next_day}")

# ---------------------------------------------------------
# 2/A) LOAD MODEL PARAMS IF EXISTS
# ---------------------------------------------------------

params = None
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        params = pickle.load(f)

# ---------------------------------------------------------
# 3) BUILD TRAINING WINDOW
# ---------------------------------------------------------

train_start = next_day - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
train_end = next_day

train = df[(df.index >= train_start) & (df.index < train_end)]

print(f"Training window: {train_start} -> {train.index[-1]}")

# ---------------------------------------------------------
# 4) TRAIN SARIMA ON SLIDING WINDOW
# ---------------------------------------------------------

model = SARIMAX(
    train[TARGET],
    order=ORDER,
    seasonal_order=SEASONAL_ORDER,
    enforce_invertibility=False
)

if params is None:
    print("No model params found → full training.")
    results = model.fit(disp=False)
else:
    print("Model params loaded → filtering.")
    results = model.filter(params)

# ---------------------------------------------------------
# 5) FORECAST 1 DAY (24 hours)
# ---------------------------------------------------------

forecast_steps = 24
forecast = results.forecast(steps=forecast_steps)

print(forecast.head())
print(forecast.describe())

# ---------------------------------------------------------
# 6) MEASURE ERROR ON NEXT DAY
# ---------------------------------------------------------

test = df[df.index.normalize() == next_day]
test = test.iloc[:forecast_steps]

mae = mean_absolute_error(test[TARGET], forecast)
rmse = root_mean_squared_error(test[TARGET], forecast)

print("------------------------------------")
print(test[TARGET].describe())
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print("------------------------------------")

plt.figure(figsize=(15, 5))
plt.plot(test.index, test[TARGET], label="Valós")
plt.plot(test.index, forecast, label="SARIMA")
plt.legend()
plt.show()

# ---------------------------------------------------------
# 7) UPDATE STATE – move to next day
# ---------------------------------------------------------

state["next_day"] = str(next_day + pd.Timedelta(days=1))


params = results.params
with open(MODEL_FILE, "wb") as f:
    pickle.dump(params, f)

print("State updated, ready for next day.")
