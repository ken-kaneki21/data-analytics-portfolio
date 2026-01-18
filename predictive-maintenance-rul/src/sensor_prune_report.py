# src/sensor_prune_report.py
from pathlib import Path
import pandas as pd


def main():
    p = Path("data/processed")
    train = pd.read_parquet(p / "train_fd001_labeled.parquet")

    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]

    # Std across ALL rows (quick global prune indicator)
    stds = train[sensor_cols].std(numeric_only=True).sort_values()

    report = pd.DataFrame({
        "sensor": stds.index,
        "std": stds.values,
        "nunique": [train[c].nunique() for c in stds.index],
    }).sort_values(["std", "nunique"], ascending=[True, True])

    out = p / "sensor_prune_report_fd001.csv"
    report.to_csv(out, index=False)

    print("Saved:", out)
    print("\nLowest-variance sensors (top 10):")
    print(report.head(10).to_string(index=False))

    print("\nHighest-variance sensors (top 10):")
    print(report.tail(10).to_string(index=False))

    # Hard rule suggestion (you can tweak later):
    # std == 0 or nunique <= 2 are basically dead signals
    dead = report[(report["std"] == 0) | (report["nunique"] <= 2)]
    print(f"\nDead sensors candidate count: {len(dead)}")
    if len(dead) > 0:
        print(dead.to_string(index=False))

if __name__ == "__main__":
    main()
