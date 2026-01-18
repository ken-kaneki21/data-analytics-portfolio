from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_features.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_scored_drift.csv"

PROXY_ANOMALY_WINDOW = 30

# Drift config
ROLLING_WINDOW = 20       # cycles
MIN_PERIODS = 10
TOP_K_SENSORS = 5         # aggregate top-k absolute z-scores


def sensor_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    return sorted(cols, key=lambda x: int(x[1:]))


def evaluate_on_train(df_train: pd.DataFrame, score_col: str, y_col: str, threshold: float):
    y_true = df_train[y_col].astype(int).values
    y_pred = (df_train[score_col].values >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return p, r, f1


def early_detection_cycles(df_train: pd.DataFrame, score_col: str, threshold: float) -> float:
    records = []
    for eid, g in df_train.groupby("engine_id"):
        g = g.sort_values("cycle")
        near_fail = g[g["cycles_to_failure"] <= PROXY_ANOMALY_WINDOW].copy()
        if near_fail.empty:
            continue
        near_fail["flag"] = (near_fail[score_col] >= threshold).astype(int)
        flagged = near_fail[near_fail["flag"] == 1]
        if flagged.empty:
            records.append(0.0)
        else:
            records.append(float(flagged.iloc[0]["cycles_to_failure"]))
    return float(np.mean(records)) if records else 0.0


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}. Run src/02_features.py first.")

    df = pd.read_csv(IN_PATH)

    required = {"engine_id", "cycle", "split", "cycles_to_failure", "is_proxy_anomaly"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    s_cols = sensor_cols(df)
    if len(s_cols) == 0:
        raise ValueError("No sensor columns found (s1..s21).")

    # Compute rolling z-score per engine per sensor
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # We'll build drift_score by taking top-k abs z-scores across sensors per row
    z_abs_all = []

    for s in s_cols:
        # rolling mean/std within each engine
        roll_mean = df.groupby("engine_id")[s].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()
        )
        roll_std = df.groupby("engine_id")[s].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std()
        ).replace(0, np.nan)

        z = (df[s] - roll_mean) / roll_std
        z_abs_all.append(np.abs(z).to_numpy())

    Z = np.vstack(z_abs_all).T  # shape: (rows, sensors)
    # rows with not enough history will have NaNs; treat them as 0 drift for baseline
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # drift_score = mean of top-k sensor z abs values
    k = min(TOP_K_SENSORS, Z.shape[1])
    topk = np.partition(Z, -k, axis=1)[:, -k:]
    drift_score = topk.mean(axis=1)

    df_out = df.copy()
    df_out["drift_score"] = drift_score

    # Threshold from train-normal distribution (top 1%)
    train_mask = df_out["split"] == "train"
    normal_mask = train_mask & (df_out["cycles_to_failure"] > PROXY_ANOMALY_WINDOW)

    thresh = float(np.quantile(df_out.loc[normal_mask, "drift_score"], 0.99))

    # Evaluate on train
    df_train = df_out.loc[train_mask].copy()
    p, r, f1 = evaluate_on_train(df_train, "drift_score", "is_proxy_anomaly", thresh)
    early = early_detection_cycles(df_train, "drift_score", thresh)

    print("[INFO] Baseline: Rolling Drift (z-score)")
    print(f"[INFO] Rolling window: {ROLLING_WINDOW}, min_periods: {MIN_PERIODS}")
    print(f"[INFO] Drift aggregation: mean(top-{k} abs z-scores)")
    print(f"[INFO] Threshold strategy: 99th percentile of train-normal drift_score")
    print(f"[INFO] Threshold value: {thresh:.6f}\n")

    print("[EVAL on TRAIN (proxy labels)]")
    print(f" Precision: {p:.4f}")
    print(f" Recall   : {r:.4f}")
    print(f" F1       : {f1:.4f}")
    print(f" Early-detect (mean cycles_to_failure at first flag within last {PROXY_ANOMALY_WINDOW} cycles): {early:.2f}")

    df_out.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Output written:\n     {OUT_PATH}")


if __name__ == "__main__":
    main()
