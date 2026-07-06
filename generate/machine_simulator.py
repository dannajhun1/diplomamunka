import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from datetime import datetime


PROCESSED_FILE = "processed_files.txt"
INPUT_FOLDER = Path("timeseries")
OUTPUT_FOLDER = Path("timeseriesCompleted")



class MachineSimulator:
    def __init__(
        self,
        demand_threshold=550,
        error_prob=0.001,
        ramp_up=120.0,
        ramp_down=150.0,
        power_noise=5.0,
        alpha_follow=0.2,
        seed=42
    ):
        np.random.seed(seed)

        self.demand_threshold = demand_threshold
        self.error_prob = error_prob

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.power_noise = power_noise
        self.alpha_follow = alpha_follow

        # state
        self.state = "OFF"

        self.power = 0.0
        self.running = 0.0
        self.ready = 1.0
        self.remote = 1.0
        self.start = 0.0

        self.error = 0.0

        self.newdemand = 0.0
        self.newstart = 0.0
        self.overshoot = False
        self.overshoot_target = 0

        self._newstart_pulse = False

    def step(self, row):
        demand = float(row["Demand"])

        # reset pulse every step
        self.newstart = 0.0

        # -------------------------
        # ERROR STATE
        # -------------------------
        if self.state == "ERROR":
            self.power -= self.ramp_down
            if self.power <= 0:
                self.power = 0
                self.state = "OFF"
                self.error = 0
                self.ready = 1
                self.start = 0
                self.running = 0

            return self._emit(demand)

        # -------------------------
        # RANDOM ERROR TRIGGER
        # -------------------------
        if self.state == "RUNNING":
            if np.random.rand() < self.error_prob:
                self.state = "ERROR"
                self.error = 1
                self.running = 1
                self.ready = 0
                self.start = 0
                self.newstart = 0

                return self._emit(demand)

        # -------------------------
        # OFF STATE
        # -------------------------
        if self.state == "OFF":
            self.power = max(0, self.power - self.ramp_down)
            self.running = 0
            self.ready = 1
            self.start = 0
            self.newstart = 0

            if demand > self.demand_threshold:
                self.state = "STARTING"
                self.newdemand = demand
                self.newstart = 1
                self._newstart_pulse = True

            return self._emit(demand)

        # -------------------------
        # STARTING STATE
        # -------------------------
        if self.state == "STARTING":
            self.power += self.ramp_up + np.random.randn() * self.power_noise
            self.newstart = 1
            self.start = 1

            if self.power >= self.newdemand:
                self.state = "RUNNING"
                self.running = 1
                self.ready = 1
                self.start = 1
                self.newstart = 1
                self.overshoot = True
                self.overshoot_target = self.newdemand * 1.001

            return self._emit(demand)

        # -------------------------
        # RUNNING STATE
        # -------------------------
        if self.state == "RUNNING":

            # Egyszeri overshoot indítás után
            if self.overshoot:

                self.power += self.alpha_follow * (
                    self.overshoot_target - self.power
                )

                # Ha elérte az overshoot célértéket,
                # innentől a Demand-et követi
                if self.power >= self.overshoot_target - 2:
                    self.overshoot = False

            else:
                # Normál szabályozás
                self.power += self.alpha_follow * (
                    demand - self.power
                )

            # Kis zaj
            self.power += np.random.randn() * self.power_noise

            # Állapotok
            self.running = 1
            self.ready = 1
            self.start = 1
            self.newstart = 1

            # Demand követése
            if demand > self.demand_threshold:
                self.newdemand = demand

            # Leállítás
            if demand <= self.demand_threshold:
                self.newdemand = demand
                self.state = "OFF"

            return self._emit(demand)

    def _emit(self, demand):
        return {
            "Demand": demand,
            "Error": self.error,
            "Power": self.power,
            "Ready": self.ready,
            "Remote": self.remote,
            "Running": self.running,
            "Start": self.start,
            "newdemand": self.newdemand,
            "newstart": self.newstart,
        }

    def generate(self, input_csv, output_csv):
        df = pd.read_csv(input_csv, sep="\t", encoding="utf-8-sig")

        df["_time"] = pd.to_datetime(df["_time"])
        df = df.sort_values("_time")

        results = []

        for _, row in df.iterrows():
            results.append(self.step(row))

        out = pd.DataFrame(results)
        out["_time"] = df["_time"].values

        cols = [
            "_time",
            "Demand",
            "Error",
            "Power",
            "Ready",
            "Remote",
            "Running",
            "Start",
            "newdemand",
            "newstart",
        ]

        out = out[cols]

        out.to_csv(output_csv, index=False)

        return out

def get_next_file():
    """
    Visszaadja az első még fel nem dolgozott timeseries fájlt.
    """

    processed = set()

    if Path(PROCESSED_FILE).exists():
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            processed = {line.strip() for line in f}

    files = sorted(INPUT_FOLDER.glob("timeseries_*.csv"))

    for file in files:
        if file.name not in processed:
            return file

    return None


def mark_as_processed(file_path):
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(file_path.name + "\n")


if __name__ == "__main__":

    sim = MachineSimulator()
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    input_file = get_next_file()

    if input_file is None:
        print("Nincs több feldolgozatlan timeseries fájl.")
        exit()

    output_file = OUTPUT_FOLDER / (
        f"timeseries_completed_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    out = sim.generate(
        input_csv=str(input_file),
        output_csv=str(output_file)
    )

    mark_as_processed(input_file)

    print(f"Kész: {output_file}")