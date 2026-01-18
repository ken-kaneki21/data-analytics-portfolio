# src/baselines.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TARGET = "RUL"
DROP_COLS = ["unit_number", "time_in_cycles"]  # keep for splits, drop for model

def within_band(y_true, y_pred, band=20):
    return float(np.mean(np.abs(y_true - y_pred) <= band))

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    w20 = within_band(y_true, y_pred, band=20)
    return mae, rmse, w20

def main():
    p = Path("data/processed")
    train_df = pd.read_parquet(p / "train_fd001_train.parquet")
    val_df = pd.read_parquet(p / "train_fd001_val.parquet")

    X_train = train_df.drop(columns=DROP_COLS + [TARGET])
    y_train = train_df[TARGET].values

    X_val = val_df.drop(columns=DROP_COLS + [TARGET])
    y_val = val_df[TARGET].values

    results = []

    # Baseline 1: Mean predictor
    mean_pred = np.full_like(y_val, y_train.mean(), dtype=float)
    mae, rmse, w20 = eval_metrics(y_val, mean_pred)
    results.append(("MeanPredictor", mae, rmse, w20))

    # Baseline 2: Ridge Regression
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe-ish
        ("model", Ridge(alpha=1.0, random_state=42))
    ])
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    mae, rmse, w20 = eval_metrics(y_val, y_pred)
    results.append(("Ridge", mae, rmse, w20))

    res_df = pd.DataFrame(results, columns=["model", "MAE", "RMSE", "Within±20"])
    out = p / "baseline_results_fd001.csv"
    res_df.to_csv(out, index=False)

    print(res_df.to_string(index=False))
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
