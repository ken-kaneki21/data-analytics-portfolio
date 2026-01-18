# src/model_gbdt.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "RUL"
DROP_COLS = ["unit_number", "time_in_cycles"]

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

    gbdt = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=600,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42
    )

    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_val)

    mae, rmse, w20 = eval_metrics(y_val, y_pred)
    print(f"HGBR | MAE={mae:.3f} RMSE={rmse:.3f} Within±20={w20:.3f}")

    pred_df = val_df[["unit_number", "time_in_cycles", TARGET]].copy()
    pred_df["pred_RUL"] = y_pred
    pred_df.to_csv(p / "val_predictions_hgbr.csv", index=False)
    print("Saved:", p / "val_predictions_hgbr.csv")

if __name__ == "__main__":
    main()
