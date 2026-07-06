import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# =====================================
# CONFIG
# =====================================

ROWS_TO_READ = 10
START_DATE = "2026-06-26"

INPUT_FOLDER = Path("generated")
OUTPUT_FOLDER = Path("timeseries")

PROCESSED_FILE = "processed_generated.txt"

OUTPUT_FOLDER.mkdir(exist_ok=True)

# =====================================
# FILE HANDLING
# =====================================

def get_next_file():

    processed = set()

    if Path(PROCESSED_FILE).exists():
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            processed = {line.strip() for line in f}

    files = sorted(INPUT_FOLDER.glob("generated_*.csv"))

    for file in files:
        if file.name not in processed:
            return file

    return None


def mark_as_processed(file_path):

    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(file_path.name + "\n")

def get_start_datetime():

    files = sorted(OUTPUT_FOLDER.glob("timeseries_*.csv"))

    # Ha még nincs egyetlen timeseries sem
    if not files:
        return datetime.strptime("2026-06-26", "%Y-%m-%d")

    latest_file = files[-1]

    print(f"Last timeseries: {latest_file.name}")

    df_last = pd.read_csv(
        latest_file,
        sep="\t"
    )

    last_time = pd.to_datetime(df_last["_time"].iloc[-1])

    # Következő perc
    return last_time + timedelta(minutes=1)


# =====================================
# MAIN
# =====================================

input_file = get_next_file()

if input_file is None:
    print("Nincs több feldolgozatlan generated fájl.")
    exit()

print(f"Input: {input_file}")

# =====================================
# CSV
# =====================================

df = pd.read_csv(
    input_file,
    sep="\t" #,
    #nrows=ROWS_TO_READ
)

print(df.head())

# =====================================
# TIME SERIES
# =====================================

current_time = get_start_datetime()

print(f"Start time: {current_time}")

timeseries = []

for _, row in df.iterrows():

    demand = row["Demand"]
    duration = int(row["Duration"])

    for _ in range(duration):

        timeseries.append({
            "_time": current_time,
            "Demand": demand
        })

        current_time += timedelta(minutes=1)

timeseries_df = pd.DataFrame(timeseries)

print(timeseries_df.head())

# =====================================
# SAVE
# =====================================

output_file = OUTPUT_FOLDER / (
    f"timeseries_{datetime.now():%Y%m%d_%H%M%S}.csv"
)

timeseries_df.to_csv(
    output_file,
    sep="\t",
    index=False
)

mark_as_processed(input_file)

print(f"Kész: {output_file}")