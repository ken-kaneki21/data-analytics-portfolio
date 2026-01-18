# src/build_base_table.py
from pathlib import Path
import pandas as pd


DROP_SENSORS = {
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
}


def main():
    p = Path("data/processed")

    train = pd.read_parquet(p / "train_fd001_labeled.parquet")
    test = pd.read_parquet(p / "test_fd001_labeled.parquet")

    # Drop dead sensors
    train = train.drop(columns=DROP_SENSORS)
    test = test.drop(columns=DROP_SENSORS)

    # Column order sanity (helps later)
    id_cols = ["unit_number", "time_in_cycles"]
    op_cols = [c for c in train.columns if c.startswith("op_setting")]
    sensor_cols = sorted([c for c in train.columns if c.startswith("sensor_")])
    target_col = ["RUL"]

    ordered_cols = id_cols + op_cols + sensor_cols + target_col
    train = train[ordered_cols]
    test = test[ordered_cols]

    # Save base tables
    train.to_parquet(p / "train_fd001_base.parquet", index=False)
    test.to_parquet(p / "test_fd001_base.parquet", index=False)

    print("Base modeling tables saved.")
    print("Train shape:", train.shape)
    print("Test  shape:", test.shape)
    print("Remaining sensors:", len(sensor_cols))


if __name__ == "__main__":
    main()
