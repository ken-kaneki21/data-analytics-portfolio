# src/split_data.py
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
VAL_FRAC = 0.2

def main():
    p = Path("data/processed")
    df = pd.read_parquet(p / "train_fd001_features.parquet")

    engines = df["unit_number"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(engines)

    n_val = int(len(engines) * VAL_FRAC)
    val_engines = set(engines[:n_val])
    train_engines = set(engines[n_val:])

    train_df = df[df["unit_number"].isin(train_engines)].reset_index(drop=True)
    val_df = df[df["unit_number"].isin(val_engines)].reset_index(drop=True)

    train_df.to_parquet(p / "train_fd001_train.parquet", index=False)
    val_df.to_parquet(p / "train_fd001_val.parquet", index=False)

    print("Split complete.")
    print("Engines total:", len(engines))
    print("Train engines:", len(train_engines), "Val engines:", len(val_engines))
    print("Train rows:", train_df.shape, "Val rows:", val_df.shape)
    print("Val engine IDs (sorted):", sorted(val_engines))

if __name__ == "__main__":
    main()
