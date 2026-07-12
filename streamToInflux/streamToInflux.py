import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR.parent / "generate" / "timeseriesCompleted"
PROCESSED_FILE = "processed_influx.txt"

#print(INPUT_FOLDER)
#print(INPUT_FOLDER.exists())
#print(list(INPUT_FOLDER.glob("*")))
load_dotenv()

# InfluxDB Cloud
INFLUX_URL = "https://eu-central-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN = os.getenv("influxdbKey")
print("token:" + INFLUX_TOKEN)
INFLUX_ORG = "IR Research"
INFLUX_BUCKET = "IK_LEARN"

MEASUREMENT = "synthetic_machine"

# =====================================================
# FILE HANDLING
# =====================================================

def get_next_file():

    processed = set()

    if Path(PROCESSED_FILE).exists():
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            processed = {line.strip() for line in f}

    files = sorted(INPUT_FOLDER.glob("timeseries_completed_*.csv"))

    for file in files:
        if file.name not in processed:
            return file

    return None


def mark_as_processed(file_path):

    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(file_path.name + "\n")


# =====================================================
# INFLUX
# =====================================================

client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

#print(client.health())

write_api = client.write_api(write_options=SYNCHRONOUS)

# =====================================================
# MAIN
# =====================================================

input_file = get_next_file()

if input_file is None:
    print("Nincs új feltöltendő fájl.")
    exit()

print(f"Uploading: {input_file.name}")

df = pd.read_csv(input_file)

df["_time"] = pd.to_datetime(df["_time"])

# =====================================================
# WRITE
# =====================================================

points = []

for _, row in df.iterrows():

    point = (
        Point(MEASUREMENT)
        .time(row["_time"], WritePrecision.NS)
    )

    for col in df.columns:

        if col == "_time":
            continue

        value = row[col]

        if col in ["Error", "Ready", "Remote", "Running", "Start"]:

            point.field(col, int(value))

        else:

            point.field(col, float(value))

    points.append(point)

write_api.write(
    bucket=INFLUX_BUCKET,
    org=INFLUX_ORG,
    record=points
)

mark_as_processed(input_file)

print(f"Uploaded {len(points)} records.")