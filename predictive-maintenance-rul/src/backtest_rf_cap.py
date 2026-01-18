# src/backtest_rf_cap.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "RUL"
DROP_COLS = ["unit_number", "time_in_cycles"]
CAP = 125

def within_band(y_true, y_pred, band=20):
    return float(np.mean(np.abs(y_true - y_pred) <= band))

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    w20 = within_band(y_true, y_pred, band=20)
    return mae, rmse, w20

def main():
    p = Path("data/processed")
    df = pd.read_parquet(p / f"train_fd001_features_cap{CAP}.parquet")

    engines = np.array(sorted(df["unit_number"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(engines)

    folds = np.array_split(engines, 3)

    results = []

    for i in range(3):
        val_eng = set(folds[i])
        train_eng = set(np.concatenate([folds[j] for j in range(3) if j != i]))

        train_df = df[df["unit_number"].isin(train_eng)]
        val_df = df[df["unit_number"].isin(val_eng)]

        X_train = train_df.drop(columns=DROP_COLS + [TARGET])
        y_train = train_df[TARGET].values
        X_val = val_df.drop(columns=DROP_COLS + [TARGET])
        y_val = val_df[TARGET].values

        rf = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mae, rmse, w20 = eval_metrics(y_val, y_pred)
        results.append((i + 1, mae, rmse, w20))
        print(f"Fold {i+1}: MAE={mae:.3f} RMSE={rmse:.3f} Within±20={w20:.3f}")

    res = pd.DataFrame(results, columns=["fold", "MAE", "RMSE", "Within±20"])
    print("\nMean across folds:")
    print(res.mean().to_string())

if __name__ == "__main__":
    main()
