# src/backtest_rf.py
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
    df = pd.read_parquet(p / "train_fd001_features.parquet")

    engines = np.array(sorted(df["unit_number"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(engines)

    k = 3
    folds = np.array_split(engines, k)

    results = []
    all_preds = []

    for i in range(k):
        val_eng = set(folds[i])
        train_eng = set(np.concatenate([folds[j] for j in range(k) if j != i]))

        train_df = df[df["unit_number"].isin(train_eng)]
        val_df = df[df["unit_number"].isin(val_eng)]

        X_train = train_df.drop(columns=DROP_COLS + [TARGET])
        y_train = train_df[TARGET].values
        X_val = val_df.drop(columns=DROP_COLS + [TARGET])
        y_val = val_df[TARGET].values

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
        results.append((i + 1, len(train_eng), len(val_eng), mae, rmse, w20))

        pred_df = val_df[["unit_number", "time_in_cycles", TARGET]].copy()
        pred_df["pred_RUL"] = y_pred
        pred_df["fold"] = i + 1
        all_preds.append(pred_df)

        print(f"Fold {i+1}: MAE={mae:.3f} RMSE={rmse:.3f} Within±20={w20:.3f}")

    res_df = pd.DataFrame(results, columns=["fold", "train_engines", "val_engines", "MAE", "RMSE", "Within±20"])
    print("\nBacktest summary:")
    print(res_df.to_string(index=False))

    print("\nMean across folds:")
    print(res_df[["MAE", "RMSE", "Within±20"]].mean().to_string())

    out_res = p / "rf_backtest_results_fd001.csv"
    out_pred = p / "rf_backtest_predictions_fd001.csv"
    res_df.to_csv(out_res, index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(out_pred, index=False)

    print("\nSaved:")
    print(" -", out_res)
    print(" -", out_pred)

if __name__ == "__main__":
    main()
