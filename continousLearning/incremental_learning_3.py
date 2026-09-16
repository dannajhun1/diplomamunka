import os
import json
import time
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm  # sm.load() - a teljes results objektum betöltéséhez (append-hez kell)

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
RESULTS_FILE = "model_results.pickle"  # a teljes illesztett SARIMAX results objektum (kell az append-hez)
MODEL_PARAMS_TEXT_FILE = "model_params.txt"
LOG_FILE = "history_3.jsonl"
RUN_LOG_FILE = "sarima_log.txt"

PLOT_DIR = "forecast_plots"

# --- Cost-aware retraining ---
# Küszöb (MAE egységben, azaz TARGET mértékegységében, pl. kW), amit ha az
# utolsó lezárt nap valós előrejelzési hibája túllép, akkor a modellt
# újratanítjuk (model.fit). Ha a hiba a küszöb alatt marad, csak
# kiterjesztjük (results.append(refit=False)) a meglévő, fix paraméterű
# modellt - ez lényegesen olcsóbb, mert nincs újraoptimalizálás.
# Ld.: cost-aware-retraining-algorithms repo, src/algorithms/markov.py -> run()
# A küszöböt érdemes historikus adatokon (history.jsonl) hangolni, hasonlóan
# ahhoz, ahogy a repó run_retrain_cost_analysis.py-ja teszi.
RETRAIN_COST = float(os.getenv("RETRAIN_COST", "0.20"))

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
# COST-AWARE RETRAINING DÖNTÉS
# ============================================================
# A cost-aware-retraining-algorithms repó "Markov" stratégiájának online,
# egy körrel késleltetett változata: mivel a mai forecast döntésekor a mai
# nap valós hibáját még nem ismerjük, az UTOLSÓ LEZÁRT nap realizált MAE-jét
# (staleness_cost) használjuk becslésként arra, mennyire "kopott" a jelenlegi
# modell. Ha ez meghaladja a RETRAIN_COST küszöböt, újratanítunk; egyébként
# csak kiterjesztjük a meglévő modellt (nincs paraméter-újraoptimalizálás).

def load_last_mae():
    """Visszaadja az utolsó history.jsonl bejegyzés valós MAE hibáját (None, ha nincs)."""
    if not os.path.exists(LOG_FILE):
        return None
    last_mae = None
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("mae") is not None:
                last_mae = entry["mae"]
    return last_mae


def decide_retrain(staleness_cost, retrain_cost):
    """
    Cost-aware / Markov-döntés: ha a staleness_cost (utolsó ismert valós MAE)
    meghaladja a retrain_cost küszöböt, újratanítás indokolt.
    Ha még nincs korábbi hibaadat, biztonsági okból újratanítunk.
    """
    if staleness_cost is None:
        return True
    return staleness_cost > retrain_cost


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

    # Meddig lett a jelenleg mentett modell betanítva/kiterjesztve.
    # .get(...) a régi state fájlokkal való visszafelé kompatibilitás miatt.
    model_train_end = state.get("model_train_end")
    num_retrains = state.get("num_retrains", 0)

    log(f"State betöltve.")
    log(f"Következő forecast nap: {next_day.date()}")
    log(f"Nap sorszáma: {day_counter}")
    log(f"Eddigi újratanítások száma: {num_retrains}")

else:

    log("Nincs state fájl. Első futás.")

    next_day = (
        first_day
        + pd.Timedelta(days=INITIAL_TRAIN_DAYS)
    )

    day_counter = 0
    model_train_end = None
    num_retrains = 0

    state = {
        "next_day": str(next_day),
        "day_counter": day_counter,
        "model_train_end": model_train_end,
        "num_retrains": num_retrains
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
# COST-AWARE DÖNTÉS: ÚJRATANÍTÁS VAGY KITERJESZTÉS
# ============================================================
# A cost-aware-retraining-algorithms repó "Markov" stratégiájának online,
# egy körrel késleltetett változata: mivel a mai forecast döntésekor a mai
# nap valós hibáját még nem ismerjük, az UTOLSÓ LEZÁRT nap realizált MAE-jét
# (staleness_cost) használjuk becslésként arra, mennyire "kopott" a jelenlegi
# modell. Ha ez meghaladja a RETRAIN_COST küszöböt, újratanítunk; egyébként
# csak kiterjesztjük a meglévő modellt (nincs paraméter-újraoptimalizálás).
#
# FONTOS: ezt a DÖNTÉST szándékosan az InfluxDB-lekérdezés ELŐTT hozzuk meg,
# mert a lekérdezés mérete (és így a lekérdezési overhead) is a döntéstől
# függ: retrain esetén a teljes (növekvő) historikus ablakot kérdezzük le,
# kiterjesztés esetén viszont csak a legutóbbi modellfrissítés óta érkezett
# új adatot - ez az InfluxDB-oldali overhead-et is "cost-aware"-ré teszi,
# nem csak a modell-illesztést.

last_mae = load_last_mae()

# Első futáskor (nincs mentett results objektum) mindenképp tanítani kell.
do_retrain = (not os.path.exists(RESULTS_FILE)) or decide_retrain(last_mae, RETRAIN_COST)

log(f"Utolsó ismert (realizált) MAE: {last_mae}")
log(f"Retrain küszöb: {RETRAIN_COST}")
log(f"Cost-aware döntés -> {'ÚJRATANÍTÁS' if do_retrain else 'KITERJESZTÉS (nincs újratanítás)'}")


# ============================================================
# ADATOK LEKÉRDEZÉSE (csak a tanításhoz + aznapi kiértékeléshez kellő rész)
# ============================================================

train_start = first_day
train_end = next_day

if do_retrain or not model_train_end:
    # Retrain: a teljes (növekvő) historikus ablakot kérdezzük le.
    query_start = train_start
else:
    # Kiterjesztés: csak az utolsó modellfrissítés óta érkezett új adat kell.
    query_start = pd.Timestamp(model_train_end)

query_end = next_day + pd.Timedelta(days=1)  # +1 nap, hogy a teszt (valós) adat is benne legyen

log(f"InfluxDB lekérdezési ablak: {query_start} -> {query_end} "
    f"({'teljes' if query_start == train_start else 'csak delta'})")

_t0 = time.perf_counter()
df = load_data(query_start, query_end)
influx_query_seconds = time.perf_counter() - _t0
influx_rows_fetched = len(df)

log(f"InfluxDB lekérdezés ideje: {influx_query_seconds:.3f} s ({influx_rows_fetched} sor)")


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
# MODEL: ÚJRATANÍTÁS (drágább) VAGY KITERJESZTÉS (olcsóbb)
# ============================================================

_t_model_0 = time.perf_counter()

if do_retrain:

    log("SARIMAX modell létrehozása (újratanítás)...")

    model = SARIMAX(
        train[TARGET],
        exog=train_exog,
        order=MODEL_ORDER,
        enforce_invertibility=False
    )

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

    log("Modell tanítása (újratanítás) befejeződött.")

    num_retrains += 1

else:

    log("Meglévő modell betöltése kiterjesztéshez (nincs újraoptimalizálás)...")

    results_prev = sm.load(RESULTS_FILE)

    # Mivel a lekérdezés (fentebb) már csak a deltát hozta le, a train/train_exog
    # itt pontosan az utolsó modellfrissítés óta érkezett új adat - nem kell
    # külön kiszámolni.
    if len(train) > 0:

        log(f"Kiterjesztés {len(train)} új órával (refit=False)...")

        results = results_prev.append(
            train[TARGET],
            exog=train_exog,
            refit=False
        )

    else:

        log("Nincs új adat a kiterjesztéshez, a korábbi modell változatlan.")

        results = results_prev

    log("Modell kiterjesztése befejeződött.")

model_update_seconds = time.perf_counter() - _t_model_0

log(f"Modell-frissítés ideje ({'retrain' if do_retrain else 'extend'}): "
    f"{model_update_seconds:.3f} s")

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

# Gépi formátum (paraméterek - a warm starthoz kellenek retrain esetén)
results.params.to_pickle(PARAMS_FILE)

# A teljes results objektum (a következő kiterjesztéshez/append-hez kell)
results.save(RESULTS_FILE)

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
    "rmse": rmse,
    "retrained": do_retrain,
    "retrain_cost_threshold": RETRAIN_COST,
    "influx_query_seconds": influx_query_seconds,
    "influx_rows_fetched": influx_rows_fetched,
    "model_update_seconds": model_update_seconds
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
    "day_counter": day_counter + 1,
    "model_train_end": str(train_end),
    "num_retrains": num_retrains
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

log(
    f"Összes újratanítás eddig: "
    f"{num_retrains} / {day_counter + 1} nap"
)


# ============================================================
# END
# ============================================================

separator()

log("FUTÁS BEFEJEZŐDÖTT")

separator()