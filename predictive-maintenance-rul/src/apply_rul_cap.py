# src/apply_rul_cap.py
from pathlib import Path
import pandas as pd

CAP = 125

def main():
    p = Path("data/processed")
    df = pd.read_parquet(p / "train_fd001_features.parquet")

    df_cap = df.copy()
    df_cap["RUL"] = df_cap["RUL"].clip(upper=CAP)

    out = p / f"train_fd001_features_cap{CAP}.parquet"
    df_cap.to_parquet(out, index=False)

    print(f"Applied RUL cap at {CAP}")
    print("Original RUL max:", df["RUL"].max())
    print("Capped RUL max:", df_cap["RUL"].max())
    print("Saved:", out)

if __name__ == "__main__":
    main()
