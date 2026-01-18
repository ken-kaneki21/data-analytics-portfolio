# src/features_rolling.py
from pathlib import Path
import pandas as pd

WINDOWS = [5, 10, 20]
ID_COLS = ["unit_number", "time_in_cycles"]


def add_rolling(df: pd.DataFrame, sensor_cols: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(ID_COLS).copy()
    g = df.groupby("unit_number", sort=False)

    for w in windows:
        for col in sensor_cols:
            roll = g[col].rolling(window=w, min_periods=w)
            df[f"{col}_roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_roll_std_{w}"] = roll.std(ddof=0).reset_index(level=0, drop=True)

    return df


def main():
    p = Path("data/processed")

    # Start from base so we don't accidentally roll on lag columns
    train = pd.read_parquet(p / "train_fd001_base.parquet")
    test = pd.read_parquet(p / "test_fd001_base.parquet")

    sensor_cols = sorted([c for c in train.columns if c.startswith("sensor_")])

    train_roll = add_rolling(train, sensor_cols, WINDOWS)
    test_roll = add_rolling(test, sensor_cols, WINDOWS)

    # We will later merge with lag table that already removed first 3 cycles.
    # Rolling needs at least max window cycles.
    min_cycle = max(WINDOWS)
    train_roll = train_roll[train_roll["time_in_cycles"] >= min_cycle].reset_index(drop=True)
    test_roll = test_roll[test_roll["time_in_cycles"] >= min_cycle].reset_index(drop=True)

    added_cols = [c for c in train_roll.columns if "_roll_" in c]
    print("Rolling feature build complete.")
    print("Train shape:", train.shape, "->", train_roll.shape)
    print("Test  shape:", test.shape, "->", test_roll.shape)
    print("Rolling columns added:", len(added_cols))
    print("Example rolling cols:", added_cols[:5])

    train_roll.to_parquet(p / "train_fd001_roll.parquet", index=False)
    test_roll.to_parquet(p / "test_fd001_roll.parquet", index=False)

    print("Saved:")
    print(" -", p / "train_fd001_roll.parquet")
    print(" -", p / "test_fd001_roll.parquet")


if __name__ == "__main__":
    main()
