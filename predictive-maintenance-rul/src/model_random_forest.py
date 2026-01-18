# src/model_random_forest.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

    # Minimal tuning - sensible defaults
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    mae, rmse, w20 = eval_metrics(y_val, y_pred)
    print(f"RandomForest | MAE={mae:.3f} RMSE={rmse:.3f} Within±20={w20:.3f}")

    # Feature importance (top 20)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top20 = imp.head(20)
    out_imp = p / "rf_feature_importance_top20.csv"
    top20.to_csv(out_imp, header=["importance"])
    print("\nTop 10 features:")
    print(top20.head(10).to_string())

    # Save predictions for later dashboard/error analysis
    pred_df = val_df[["unit_number", "time_in_cycles", TARGET]].copy()
    pred_df["pred_RUL"] = y_pred
    pred_df.to_csv(p / "val_predictions_rf.csv", index=False)
    print("\nSaved:")
    print(" -", out_imp)
    print(" -", p / "val_predictions_rf.csv")

if __name__ == "__main__":
    main()
