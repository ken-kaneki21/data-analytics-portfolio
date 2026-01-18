# src/error_buckets.py
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = Path("data/processed")
    pred = pd.read_csv(p / "rf_backtest_predictions_fd001.csv")

    pred["abs_err"] = (pred["RUL"] - pred["pred_RUL"]).abs()

    bins = [0, 25, 50, 100, 200, 10_000]
    labels = ["0-25", "25-50", "50-100", "100-200", "200+"]

    pred["bucket"] = pd.cut(pred["RUL"], bins=bins, labels=labels, right=True, include_lowest=True)

    summary = pred.groupby("bucket").agg(
        count=("abs_err", "size"),
        MAE=("abs_err", "mean"),
        MedianAE=("abs_err", "median")
    ).reset_index()

    out = p / "rf_error_buckets_fd001.csv"
    summary.to_csv(out, index=False)

    print(summary.to_string(index=False))
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
