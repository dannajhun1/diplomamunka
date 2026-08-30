import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from influxdb_client import InfluxDBClient  # pip install influxdb-client
from dotenv import load_dotenv  # pip install python-dotenv


# ============================================================
# .ENV BETÖLTÉSE
# ============================================================

load_dotenv()  # alapból a .env fájlt keresi a jelenlegi munkakönyvtárban


# ============================================================
# CONFIG
# ============================================================

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

# --- InfluxDB kapcsolat (.env fájlból) ---
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")
INFLUX_MEASUREMENT = os.getenv("INFLUX_MEASUREMENT")

for var_name, var_value in [
    ("INFLUX_TOKEN", INFLUX_TOKEN),
    ("INFLUX_ORG", INFLUX_ORG),
    ("INFLUX_BUCKET", INFLUX_BUCKET),
    ("INFLUX_MEASUREMENT", INFLUX_MEASUREMENT),
]:
    if not var_value:
        raise RuntimeError(
            f"Hiányzó környezeti változó: {var_name} "
            f"(ellenőrizd a .env fájlt)"
        )

# Az adatsor legkorábbi időpontja (csak egyszer kell beállítani).
# Erre azért van szükség, mert nem kérdezzük le az egész adatbázist
# csak azért, hogy megtudjuk, mikor kezdődik az adatsor.
DATA_START = "2006-12-16 17:24:00"


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
# LOAD DATA (InfluxDB-ből, csak a megadott időintervallumra)
# ============================================================

def load_data(start, end):
    """
    Lekérdezi az adatokat InfluxDB-ből a [start, end) intervallumra,
    óránkénti átlaggal (aggregateWindow), NEM az összes adatot.
    """

    log(f"Adatok lekérdezése InfluxDB-ből: {start} -> {end}")

    start_str = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%SZ")

    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start_str}, stop: {end_str})
      |> filter(fn: (r) => r._measurement == "{INFLUX_MEASUREMENT}")
      |> filter(fn: (r) => r._field == "{TARGET}")
      |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
      |> keep(columns: ["_time", "_value"])
      |> sort(columns: ["_time"])
    '''

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    try:
        tables = client.query_api().query(flux)
    finally:
        client.close()

    rows = [
        {"datetime": record.get_time(), TARGET: record.get_value()}
        for table in tables
        for record in table.records
    ]

    df = pd.DataFrame(rows)

    log(f"Beolvasott (óránkénti) sorok: {len(df)}")

    if df.empty:
        log("Nincs adat a megadott intervallumra!")
        return df.set_index(pd.DatetimeIndex([], name="datetime"))

    df = df.set_index("datetime").sort_index()

    # Influx UTC időbélyeget ad vissza, a naiv (tz nélküli) formára hozzuk,
    # hogy a többi rész (create_exog, összehasonlítások) változatlan maradhasson.
    df.index = df.index.tz_localize(None)

    log("Hiányzó órák kitöltése (ffill)...")

    df = df.ffill()

    log(f"Órás adatok száma: {len(df)}")

    if len(df) > 0:
        log(f"Első adat: {df.index.min()}")
        log(f"Utolsó adat: {df.index.max()}")

    return df


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

first_day = pd.Timestamp(DATA_START).normalize()

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
# ADATOK LEKÉRDEZÉSE (csak a tanításhoz + aznapi kiértékeléshez kellő rész)
# ============================================================

train_start = first_day
train_end = next_day

query_start = train_start
query_end = next_day + pd.Timedelta(days=1)  # +1 nap, hogy a teszt (valós) adat is benne legyen

df = load_data(query_start, query_end)


# ============================================================
# TRAINING DATA
# ============================================================

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