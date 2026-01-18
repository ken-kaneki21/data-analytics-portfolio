# src/export_dashboard_tables.py
from pathlib import Path
import pandas as pd
import joblib

TARGET = "RUL"
DROP_COLS = ["unit_number", "time_in_cycles"]
CAP = 125
RISK_THRESHOLD = 30

def main():
    p = Path("data/processed")

    model = joblib.load(p / "rf_final_cap125.joblib")

    train = pd.read_parquet(p / f"train_fd001_features_cap{CAP}.parquet")
    test = pd.read_parquet(p / "test_fd001_features.parquet")

    # TRAIN predictions (for analysis/demo)
    X_train = train.drop(columns=DROP_COLS + [TARGET])
    train_out = train[["unit_number", "time_in_cycles", TARGET]].copy()
    train_out["pred_RUL"] = model.predict(X_train)
    train_out["risk_flag"] = (train_out["pred_RUL"] <= RISK_THRESHOLD).astype(int)

    # TEST predictions (deployment-style)
    X_test = test.drop(columns=DROP_COLS + [TARGET])
    test_out = test[["unit_number", "time_in_cycles", TARGET]].copy()
    test_out["pred_RUL"] = model.predict(X_test)
    test_out["risk_flag"] = (test_out["pred_RUL"] <= RISK_THRESHOLD).astype(int)

    train_out.to_csv(p / "dashboard_train_predictions.csv", index=False)
    test_out.to_csv(p / "dashboard_test_predictions.csv", index=False)

    print("Dashboard tables exported:")
    print(" - dashboard_train_predictions.csv")
    print(" - dashboard_test_predictions.csv")

if __name__ == "__main__":
    main()
