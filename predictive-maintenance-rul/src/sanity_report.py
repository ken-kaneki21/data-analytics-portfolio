# src/sanity_report.py
from pathlib import Path
import pandas as pd


def main():
    p = Path("data/processed")
    train = pd.read_parquet(p / "train_fd001_labeled.parquet")
    test = pd.read_parquet(p / "test_fd001_labeled.parquet")

    print("Shapes:")
    print("train:", train.shape)
    print("test :", test.shape)

    print("\nUnique engines:")
    print("train:", train["unit_number"].nunique())
    print("test :", test["unit_number"].nunique())

    print("\nCycle ranges:")
    print("train cycle min/max:", train["time_in_cycles"].min(), train["time_in_cycles"].max())
    print("test  cycle min/max:", test["time_in_cycles"].min(), test["time_in_cycles"].max())

    print("\nRUL checks:")
    print("train RUL min/max:", train["RUL"].min(), train["RUL"].max())
    print("test  RUL min/max:", test["RUL"].min(), test["RUL"].max())

    # Check monotonicity: within each engine, cycle should increase by 1
    bad = 0
    for df, name in [(train, "train"), (test, "test")]:
        g = df.sort_values(["unit_number", "time_in_cycles"]).groupby("unit_number")["time_in_cycles"]
        diffs = g.diff().dropna()
        bad_steps = (diffs != 1).sum()
        if bad_steps:
            bad += bad_steps
            print(f"\n{name}: WARNING - non-consecutive cycles count = {bad_steps}")
        else:
            print(f"\n{name}: cycles are consecutive for all engines.")

    if bad == 0:
        print("\nSanity report: PASS")
    else:
        print("\nSanity report: FAIL (fix data order/duplicates)")

if __name__ == "__main__":
    main()
