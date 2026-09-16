"""
Összehasonlítja két history.jsonl futást:
  - baseline: RETRAIN_COST=0 (= mindig újratanít, ez az "eredeti" viselkedés)
  - markov:   RETRAIN_COST=<választott küszöb> (= cost-aware döntés)

Használat:
    python compare_strategies.py baseline_history.jsonl markov_history.jsonl
"""

import sys
import json
import pandas as pd


def load_history(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    total_days = len(df)
    num_retrains = int(df["retrained"].sum()) if "retrained" in df else total_days
    mean_mae = df["mae"].dropna().mean()
    total_mae = df["mae"].dropna().sum()

    result = {
        "stratégia": label,
        "napok száma": total_days,
        "újratanítások száma": num_retrains,
        "újratanítás aránya (%)": round(100 * num_retrains / total_days, 1) if total_days else None,
        "átlagos MAE": round(mean_mae, 4) if pd.notna(mean_mae) else None,
        "összesített MAE": round(total_mae, 4) if pd.notna(total_mae) else None,
    }

    # Overhead mezők - csak akkor, ha a history.jsonl már tartalmazza őket
    # (a régebbi, overhead-mérés előtti futásokban még nincsenek benne).
    if "influx_query_seconds" in df:
        result["InfluxDB idő össz. (s)"] = round(df["influx_query_seconds"].sum(), 2)
    if "influx_rows_fetched" in df:
        result["InfluxDB sorok össz."] = int(df["influx_rows_fetched"].sum())
    if "model_update_seconds" in df:
        result["Modell-frissítés idő össz. (s)"] = round(df["model_update_seconds"].sum(), 2)

    return result


def main():
    if len(sys.argv) != 3:
        print("Használat: python compare_strategies.py baseline_history.jsonl markov_history.jsonl")
        sys.exit(1)

    baseline_path, markov_path = sys.argv[1], sys.argv[2]

    baseline = load_history(baseline_path)
    markov = load_history(markov_path)

    summary = pd.DataFrame([
        summarize(baseline, "Eredeti (mindig retrain)"),
        summarize(markov, "Markov (cost-aware)"),
    ])

    print(summary.to_string(index=False))

    b = summary.iloc[0]
    m = summary.iloc[1]

    if b["újratanítások száma"] > 0:
        retrain_saved_pct = 100 * (1 - m["újratanítások száma"] / b["újratanítások száma"])
        print(f"\nSpórolt újratanítások: {retrain_saved_pct:.1f}%")

    if b["átlagos MAE"]:
        mae_increase_pct = 100 * (m["átlagos MAE"] / b["átlagos MAE"] - 1)
        print(f"MAE változás a baseline-hoz képest: {mae_increase_pct:+.1f}%")


if __name__ == "__main__":
    main()
