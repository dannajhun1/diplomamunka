import subprocess
import time
import sys
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

SCRIPTS = [
    "new_training.py",
    "ts_generation.py",
    "machine_simulator.py",
]

# várakozás két script között (másodperc)
DELAY_BETWEEN_SCRIPTS = 5

# várakozás egy teljes ciklus után
DELAY_BETWEEN_CYCLES = 30


def run_script(script):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR while running {script}")
        return False

    print(f"{script} finished successfully.")
    return True


def main():

    while True:

        print("\nStarting new pipeline...\n")

        for script in SCRIPTS:

            if not Path(script).exists():
                print(f"Missing file: {script}")
                continue

            success = run_script(script)

            if not success:
                print("Pipeline interrupted.")
                break

            print(f"Waiting {DELAY_BETWEEN_SCRIPTS} seconds...\n")
            time.sleep(DELAY_BETWEEN_SCRIPTS)

        print(f"\nPipeline finished.")
        print(f"Next cycle in {DELAY_BETWEEN_CYCLES} seconds...\n")

        time.sleep(DELAY_BETWEEN_CYCLES)


if __name__ == "__main__":
    main()