# src/train_final_model.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

TARGET = "RUL"
DROP_COLS = ["unit_number", "time_in_cycles"]
CAP = 125

def main():
    p = Path("data/processed")
    df = pd.read_parquet(p / f"train_fd001_features_cap{CAP}.parquet")

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET].values

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    out = p / "rf_final_cap125.joblib"
    joblib.dump(model, out)

    print("Final model trained on full dataset.")
    print("Saved:", out)

if __name__ == "__main__":
    main()
