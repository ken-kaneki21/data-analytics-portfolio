from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_features.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_scored_iforest.csv"

PROXY_ANOMALY_WINDOW = 30
RANDOM_STATE = 42

# You can tweak these later; lock them for baseline
N_ESTIMATORS = 300
MAX_SAMPLES = "auto"
CONTAMINATION = "auto"  # we will threshold ourselves anyway
MAX_FEATURES = 1.0


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

    # Only keep sensor columns exactly like s1..s21 (avoid 'split', 'sigma', etc.)
    sensor_cols = []
    for c in df.columns:
        if c.startswith("s") and c[1:].isdigit():
            sensor_cols.append(c)

    sensor_cols = sorted(sensor_cols, key=lambda x: int(x[1:]))
    return op_cols + sensor_cols


def zscore_fit_transform(train: pd.DataFrame, full: pd.DataFrame, cols: list[str]):
    mu = train[cols].mean()
    sigma = train[cols].std().replace(0, 1.0)
    train_z = (train[cols] - mu) / sigma
    full_z = (full[cols] - mu) / sigma
    return train_z, full_z, mu, sigma


def evaluate_on_train(df_train: pd.DataFrame, score_col: str, y_col: str, threshold: float):
    y_true = df_train[y_col].astype(int).values
    y_pred = (df_train[score_col].values >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return p, r, f1


def early_detection_cycles(df_train: pd.DataFrame, score_col: str, threshold: float) -> float:
    """
    For each engine, find the earliest cycle where anomaly is flagged within the proxy anomaly region.
    Return mean cycles-to-failure at first detection (higher = earlier detection).
    """
    records = []
    for eid, g in df_train.groupby("engine_id"):
        g = g.sort_values("cycle")
        # consider only the proxy-anomaly region near failure
        near_fail = g[g["cycles_to_failure"] <= PROXY_ANOMALY_WINDOW].copy()
        if near_fail.empty:
            continue
        near_fail["flag"] = (near_fail[score_col] >= threshold).astype(int)
        flagged = near_fail[near_fail["flag"] == 1]
        if flagged.empty:
            # no detection for this engine
            records.append(0.0)
        else:
            # cycles_to_failure at first flagged point
            records.append(float(flagged.iloc[0]["cycles_to_failure"]))
    return float(np.mean(records)) if records else 0.0


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}. Run src/02_features.py first.")

    df = pd.read_csv(IN_PATH)

    # Basic validation
    required = {"engine_id", "cycle", "split", "cycles_to_failure", "is_proxy_anomaly"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = get_feature_cols(df)

    # Train only on NORMAL region of TRAIN split
    train_mask = df["split"] == "train"
    normal_mask = train_mask & (df["cycles_to_failure"] > PROXY_ANOMALY_WINDOW)

    df_train_normal = df.loc[normal_mask].copy()
    if df_train_normal.empty:
        raise ValueError("No training rows found for normal region. Check PROXY_ANOMALY_WINDOW.")

    # Z-score normalize using train-normal stats
    X_train_z, X_full_z, mu, sigma = zscore_fit_transform(df_train_normal, df, feature_cols)

    # Fit Isolation Forest on normal telemetry
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_z)

    # sklearn: higher score_samples = more normal. We want anomaly_score where higher = more anomalous.
    normality = model.score_samples(X_full_z)
    anomaly_score = -normality

    df_out = df.copy()
    df_out["iforest_anomaly_score"] = anomaly_score

    # Choose threshold from train-normal distribution (top 1% most anomalous among NORMAL training data)
    df_train_normal_out = df_out.loc[normal_mask].copy()
    thresh = float(np.quantile(df_train_normal_out["iforest_anomaly_score"], 0.99))

    # Evaluate on TRAIN using proxy labels
    df_train_out = df_out.loc[train_mask].copy()
    p, r, f1 = evaluate_on_train(df_train_out, "iforest_anomaly_score", "is_proxy_anomaly", thresh)
    early = early_detection_cycles(df_train_out, "iforest_anomaly_score", thresh)

    # Print stats
    print("[INFO] Baseline: Isolation Forest")
    print(f"[INFO] Features used: {len(feature_cols)} ({feature_cols[:5]} ...)")
    print(f"[INFO] Train-normal rows: {len(df_train_normal):,}")
    print(f"[INFO] Threshold strategy: 99th percentile of train-normal anomaly_score")
    print(f"[INFO] Threshold value: {thresh:.6f}\n")

    print("[EVAL on TRAIN (proxy labels)]")
    print(f" Precision: {p:.4f}")
    print(f" Recall   : {r:.4f}")
    print(f" F1       : {f1:.4f}")
    print(f" Early-detect (mean cycles_to_failure at first flag within last {PROXY_ANOMALY_WINDOW} cycles): {early:.2f}")

    # Save
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Output written:\n     {OUT_PATH}")


if __name__ == "__main__":
    main()
