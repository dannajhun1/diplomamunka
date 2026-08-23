import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# CONFIG
# ============================================================

RAW_FILE = "household_power.txt"

TARGET = "Global_active_power"

MODEL_ORDER = (1, 1, 1)

INITIAL_TRAIN_DAYS = 7
FORECAST_HOURS = 24

STATE_FILE = "state.json"
PARAMS_FILE = "model_params.pkl"
MODEL_PARAMS_TEXT_FILE = "model_params.txt"
LOG_FILE = "history.jsonl"
RUN_LOG_FILE = "sarima_log.txt"

PLOT_DIR = "forecast_plots"


# ============================================================
# LOG
# ============================================================

def log(message=""):
    """
    Kiírja az üzenetet a konzolra és a log fájlba is.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = f"[{timestamp}] {message}"

    print(line)

    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def separator():
    """
    Elválasztó a logban.
    """

    line = "=" * 70

    print(line)

    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# START LOG
# ============================================================

separator()

log("SARIMA program indítása")
log(f"Target: {TARGET}")
log(f"Model order: {MODEL_ORDER}")
log(f"Kezdeti tanítás: {INITIAL_TRAIN_DAYS} nap")
log(f"Forecast: {FORECAST_HOURS} óra")

separator()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    log("Adatok betöltése...")

    df = pd.read_csv(
        RAW_FILE,
        sep=";",
        na_values="?",
        usecols=["Date", "Time", TARGET]
    )

    log(f"Beolvasott sorok: {len(df)}")

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    df = (
        df[
            ["datetime", TARGET]
        ]
        .set_index("datetime")
        .sort_index()
    )

    log("Perces adatok átalakítása órás adatokra...")

    df = df.resample("h").mean()

    log("Hiányzó értékek kitöltése...")

    df = df.ffill()

    log(f"Órás adatok száma: {len(df)}")
    log(f"Első adat: {df.index.min()}")
    log(f"Utolsó adat: {df.index.max()}")

    return df


df = load_data()


# ============================================================
# EXOGENOUS VARIABLES
# ============================================================

def create_exog(index):

    data = pd.DataFrame(index=index)

    # Óra
    for hour in range(1, 24):
        data[f"hour_{hour}"] = (
            index.hour == hour
        ).astype(int)

    # Hét napja
    for day in range(1, 7):
        data[f"dow_{day}"] = (
            index.dayofweek == day
        ).astype(int)

    return data


log("Exogén változók létrehozása: óra + hét napja")


# ============================================================
# STATE
# ============================================================

if os.path.exists(STATE_FILE):

    log("Meglévő state betöltése...")

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    next_day = pd.Timestamp(
        state["next_day"]
    )

    day_counter = state["day_counter"]

    log(f"State betöltve.")
    log(f"Következő forecast nap: {next_day.date()}")
    log(f"Nap sorszáma: {day_counter}")

else:

    log("Nincs state fájl. Első futás.")

    first_day = df.index.min().normalize()

    next_day = (
        first_day
        + pd.Timedelta(days=INITIAL_TRAIN_DAYS)
    )

    day_counter = 0

    state = {
        "next_day": str(next_day),
        "day_counter": day_counter
    }

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)

    log(f"State létrehozva.")
    log(f"Első forecast nap: {next_day.date()}")


separator()

log(f"FORECAST NAP: {next_day.date()}")
log(f"NAP SORSZÁMA: {day_counter}")

separator()


# ============================================================
# TRAINING DATA
# ============================================================

first_day = df.index.min().normalize()

train_start = first_day
train_end = next_day

train = df[
    (df.index >= train_start)
    &
    (df.index < train_end)
]

log("Tanítóadat létrehozása...")

log(f"Tanítás kezdete: {train.index.min()}")
log(f"Tanítás vége: {train.index.max()}")
log(f"Tanító órák száma: {len(train)}")
log(f"Tanító napok száma: {len(train) / 24:.1f}")


# ============================================================
# EXOGENOUS DATA
# ============================================================

train_exog = create_exog(
    train.index
)

log(
    f"Training exog shape: "
    f"{train_exog.shape}"
)


# ============================================================
# MODEL
# ============================================================

log("SARIMAX modell létrehozása...")

model = SARIMAX(
    train[TARGET],
    exog=train_exog,
    order=MODEL_ORDER,
    enforce_invertibility=False
)


# ============================================================
# CONTINUOUS LEARNING / WARM START
# ============================================================

if os.path.exists(PARAMS_FILE):

    log("Előző modell paramétereinek betöltése...")

    previous_params = pd.read_pickle(
        PARAMS_FILE
    )

    log("Warm start használata.")

    results = model.fit(
        start_params=previous_params,
        disp=False,
        maxiter=50
    )

else:

    log("Nincs korábbi modell.")

    log("Első modell tanítása...")

    results = model.fit(
        disp=False,
        maxiter=50
    )


log("Modell tanítása befejeződött.")

log(f"AIC: {results.aic:.4f}")


# ============================================================
# FORECAST
# ============================================================

log("Forecast készítése...")

forecast_index = pd.date_range(
    start=next_day,
    periods=FORECAST_HOURS,
    freq="h"
)

forecast_exog = create_exog(
    forecast_index
)

forecast = results.forecast(
    steps=FORECAST_HOURS,
    exog=forecast_exog
)

log(
    f"Forecast elkészült: "
    f"{len(forecast)} óra"
)

log(
    f"Forecast kezdete: "
    f"{forecast_index.min()}"
)

log(
    f"Forecast vége: "
    f"{forecast_index.max()}"
)


# ============================================================
# REAL DATA
# ============================================================

log("Valós adatok keresése...")

test = df.loc[
    (df.index >= next_day)
    &
    (
        df.index
        < next_day + pd.Timedelta(days=1)
    ),
    TARGET
]

log(
    f"Valós adatok száma: "
    f"{len(test)}"
)


# ============================================================
# EVALUATION
# ============================================================

if len(test) == FORECAST_HOURS:

    mae = mean_absolute_error(
        test,
        forecast
    )

    rmse = root_mean_squared_error(
        test,
        forecast
    )

    log("Kiértékelés:")

    log(f"MAE :  {mae:.4f}")
    log(f"RMSE:  {rmse:.4f}")
    log(f"AIC :  {results.aic:.4f}")

else:

    mae = None
    rmse = None

    log(
        "Nincs elegendő valós adat "
        "a kiértékeléshez."
    )

    log(
        f"Elérhető: "
        f"{len(test)}/{FORECAST_HOURS} óra"
    )


# ============================================================
# PLOT
# ============================================================

if len(test) == FORECAST_HOURS:

    log("Forecast grafikon készítése...")

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    plt.figure(
        figsize=(15, 5)
    )

    plt.plot(
        test.index,
        test,
        label="Valós"
    )

    plt.plot(
        forecast_index,
        forecast,
        label="Előrejelzés"
    )

    plt.title(
        f"{next_day.date()} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f}"
    )

    plt.xlabel("Idő")
    plt.ylabel(TARGET)

    plt.legend()

    plt.tight_layout()

    plot_file = os.path.join(
        PLOT_DIR,
        f"forecast_{next_day.date()}.png"
    )

    plt.savefig(
        plot_file,
        dpi=150
    )

    plt.close()

    log(
        f"Grafikon mentve: "
        f"{plot_file}"
    )


# ============================================================
# SAVE MODEL PARAMETERS
# ============================================================

log("Modell paramétereinek mentése...")

# Gépi formátum
results.params.to_pickle(PARAMS_FILE)

# Emberileg olvasható formátum
with open(
    MODEL_PARAMS_TEXT_FILE,
    "a",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("SARIMA MODEL PARAMETERS\n")
    f.write("=" * 70 + "\n")

    f.write(f"Dátum: {next_day.date()}\n")
    f.write(f"Nap: {day_counter}\n")
    f.write(f"AIC: {results.aic:.6f}\n")
    f.write(f"Model order: {MODEL_ORDER}\n")

    f.write("\n")
    f.write("Parameters:\n")
    f.write("-" * 70 + "\n")

    for name, value in results.params.items():

        f.write(
            f"{name:<30} : {value:.10f}\n"
        )

    f.write("=" * 70 + "\n")


log(f"Paraméterek mentve: {PARAMS_FILE}")
log(f"Olvasható paraméterek: {MODEL_PARAMS_TEXT_FILE}")


# ============================================================
# HISTORY LOG
# ============================================================

log("Eredmény naplózása...")

history = {
    "date": str(next_day.date()),
    "day_counter": day_counter,
    "train_hours": len(train),
    "train_days": len(train) / 24,
    "aic": float(results.aic),
    "mae": mae,
    "rmse": rmse
}

with open(
    LOG_FILE,
    "a",
    encoding="utf-8"
) as f:

    f.write(
        json.dumps(history) + "\n"
    )

log(
    f"History mentve: "
    f"{LOG_FILE}"
)


# ============================================================
# UPDATE STATE
# ============================================================

log("State frissítése...")

next_day = (
    next_day
    + pd.Timedelta(days=1)
)

state = {
    "next_day": str(next_day),
    "day_counter": day_counter + 1
}

with open(
    STATE_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        state,
        f
    )

log(
    f"Következő forecast: "
    f"{next_day.date()}"
)

log(
    f"Következő nap sorszáma: "
    f"{day_counter + 1}"
)


# ============================================================
# END
# ============================================================

separator()

log("FUTÁS BEFEJEZŐDÖTT")

separator()