# src/check_columns.py
from pathlib import Path
import pandas as pd

def main():
    p = Path("data/processed")
    df = pd.read_parquet(p / "train_fd001_features.parquet")
    print("Columns count:", len(df.columns))
    print("First 30 cols:", df.columns[:30].tolist())
    print("Last 30 cols:", df.columns[-30:].tolist())
    print("\nTarget column candidates:", [c for c in df.columns if "RUL" in c.upper()])

if __name__ == "__main__":
    main()
