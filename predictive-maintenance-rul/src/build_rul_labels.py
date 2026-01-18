# src/build_rul_labels.py
from pathlib import Path
import pandas as pd


def main():
    processed_dir = Path("data/processed")

    # Load processed data
    train = pd.read_parquet(processed_dir / "train_fd001.parquet")
    test = pd.read_parquet(processed_dir / "test_fd001.parquet")
    rul_test = pd.read_parquet(processed_dir / "rul_fd001.parquet")

    # ---------- TRAIN RUL ----------
    max_cycle_train = (
        train.groupby("unit_number")["time_in_cycles"]
        .max()
        .reset_index()
        .rename(columns={"time_in_cycles": "max_cycle"})
    )

    train = train.merge(max_cycle_train, on="unit_number", how="left")
    train["RUL"] = train["max_cycle"] - train["time_in_cycles"]
    train.drop(columns=["max_cycle"], inplace=True)

    # ---------- TEST RUL ----------
    max_cycle_test = (
        test.groupby("unit_number")["time_in_cycles"]
        .max()
        .reset_index()
        .rename(columns={"time_in_cycles": "last_cycle"})
    )

    max_cycle_test["RUL_after_last_cycle"] = rul_test["RUL_after_last_cycle"].values
    max_cycle_test["total_cycles"] = (
        max_cycle_test["last_cycle"]
        + max_cycle_test["RUL_after_last_cycle"]
    )

    test = test.merge(
        max_cycle_test[["unit_number", "total_cycles"]],
        on="unit_number",
        how="left",
    )
    test["RUL"] = test["total_cycles"] - test["time_in_cycles"]
    test.drop(columns=["total_cycles"], inplace=True)

    # ---------- SANITY CHECKS ----------
    assert train["RUL"].min() == 0, "Train RUL should reach 0 at failure"
    assert test["RUL"].min() > 0, "Test RUL should never reach 0"

    # Save
    train.to_parquet(processed_dir / "train_fd001_labeled.parquet", index=False)
    test.to_parquet(processed_dir / "test_fd001_labeled.parquet", index=False)

    print("RUL labeling complete.")
    print("Train RUL range:", train["RUL"].min(), "→", train["RUL"].max())
    print("Test  RUL range:", test["RUL"].min(), "→", test["RUL"].max())


if __name__ == "__main__":
    main()
