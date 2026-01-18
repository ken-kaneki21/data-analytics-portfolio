# src/merge_features.py
from pathlib import Path
import pandas as pd

KEYS = ["unit_number", "time_in_cycles"]
TARGET = "RUL"

def main():
    p = Path("data/processed")

    train_lag = pd.read_parquet(p / "train_fd001_lag.parquet")
    test_lag = pd.read_parquet(p / "test_fd001_lag.parquet")

    train_roll = pd.read_parquet(p / "train_fd001_roll.parquet")
    test_roll = pd.read_parquet(p / "test_fd001_roll.parquet")

    # only rolling engineered columns
    roll_cols = [c for c in train_roll.columns if "_roll_" in c]

    train = train_lag.merge(train_roll[KEYS + roll_cols], on=KEYS, how="inner")
    test = test_lag.merge(test_roll[KEYS + roll_cols], on=KEYS, how="inner")

    # Sanity
    if TARGET not in train.columns:
        raise ValueError(f"{TARGET} missing. Found: {[c for c in train.columns if 'RUL' in c.upper()]}")

    # ensure no duplicated suffix columns exist
    bad_cols = [c for c in train.columns if c.endswith("_x") or c.endswith("_y")]
    if bad_cols:
        raise ValueError(f"Bad merge: found suffixed cols: {bad_cols[:20]} ...")

    engineered_cols = [c for c in train.columns if ("_lag_" in c) or ("_roll_" in c)]
    nan_train = train[engineered_cols].isna().sum().sum()
    nan_test = test[engineered_cols].isna().sum().sum()
    if nan_train != 0 or nan_test != 0:
        raise ValueError("NaNs present after merge. Fix window/filters.")

    train.to_parquet(p / "train_fd001_features.parquet", index=False)
    test.to_parquet(p / "test_fd001_features.parquet", index=False)

    print("Merge complete (clean).")
    print("Train:", train.shape, "Test:", test.shape)
    print("Engineered cols:", len(engineered_cols))
    print("Saved feature tables.")

if __name__ == "__main__":
    main()
