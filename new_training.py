import pandas as pd
import numpy as np
import os
from datetime import datetime

df = pd.read_csv("data.csv")

df["_time"] = pd.to_datetime(df["_time"])

# új szakasz indul, ha a Demand változik
segment_id = (df["Demand"] != df["Demand"].shift()).cumsum()

segments = (
    df.groupby(segment_id)
      .agg(
          demand=("Demand", "first"),
          start=("_time", "first"),
          end=("_time", "last")
      )
)

segments["duration_min"] = (
    (segments["end"] - segments["start"])
    .dt.total_seconds() / 60 + 1
)

print("Segmentation head: ",segments.head(2))

synthetic = []

last_demand = None
repeat_factor = 2

for _ in range(100):

    if last_demand is None or np.random.random() > 0.7:
        row = segments.sample(1).iloc[0]
        last_demand = row["demand"]

    synthetic.append({
        "Demand": 0 if pd.isna(last_demand) else last_demand,
        "Duration": segments.sample(1).iloc[0]["duration_min"] * repeat_factor
    })


synthetic_df = pd.DataFrame(synthetic)

print("Synthethic generation done")
#print(synthetic_df)

os.makedirs("generated", exist_ok=True)

file_name = f"generated/generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

synthetic_df.to_csv(file_name, sep='\t', index=False)

print("Saved to csv:", file_name)