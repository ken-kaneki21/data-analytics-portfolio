# src/features_lag.py
from pathlib import Path
import pandas as pd

LAGS = [1, 2, 3]

ID_COLS = ["unit_number", "time_in_cycles"]
TARGET = "RUL"


def add_lags(df: pd.DataFrame, feature_cols: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.sort_values(ID_COLS).copy()

    g = df.groupby("unit_number", sort=False)

    for col in feature_cols:
        for k in lags:
            df[f"{col}_lag_{k}"] = g[col].shift(k)

    return df


def main():
    p = Path("data/processed")
    train = pd.read_parquet(p / "train_fd001_base.parquet")
    test = pd.read_parquet(p / "test_fd001_base.parquet")

    # Features to lag: op_settings + sensors (NOT IDs, NOT target)
    feature_cols = [c for c in train.columns if c.startswith("op_setting") or c.startswith("sensor_")]

    train_lag = add_lags(train, feature_cols, LAGS)
    test_lag = add_lags(test, feature_cols, LAGS)

    # Drop rows where any lag is missing (first max(LAGS) cycles of each engine)
    max_lag = max(LAGS)
    train_lag = train_lag[train_lag["time_in_cycles"] > max_lag].reset_index(drop=True)
    test_lag = test_lag[test_lag["time_in_cycles"] > max_lag].reset_index(drop=True)

    # Quick checks
    added_cols = [c for c in train_lag.columns if "_lag_" in c]
    print("Lag feature build complete.")
    print("Train shape:", train.shape, "->", train_lag.shape)
    print("Test  shape:", test.shape, "->", test_lag.shape)
    print("Lag columns added:", len(added_cols))
    print("Example lag cols:", added_cols[:5])

    train_lag.to_parquet(p / "train_fd001_lag.parquet", index=False)
    test_lag.to_parquet(p / "test_fd001_lag.parquet", index=False)
    print("Saved:")
    print(" -", p / "train_fd001_lag.parquet")
    print(" -", p / "test_fd001_lag.parquet")


if __name__ == "__main__":
    main()
