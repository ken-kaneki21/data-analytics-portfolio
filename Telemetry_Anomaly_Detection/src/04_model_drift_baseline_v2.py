from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_features.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_scored_drift_v2.csv"

PROXY_ANOMALY_WINDOW = 30

# v2 drift config
BASELINE_CYCLES = 30     # early life baseline window per engine
TOP_K_SENSORS = 5
THRESH_Q = 0.95          # threshold quantile from train-normal (tunable)


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
    if not s_cols:
        raise ValueError("No sensor columns found (s1..s21).")

    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # Compute per-engine baseline mean/std from first BASELINE_CYCLES cycles
    # Then compute abs z-score vs that baseline for all cycles
    z_abs_all = []

    for s in s_cols:
        base_mean = df.groupby("engine_id").apply(
            lambda g: g.loc[g["cycle"] <= BASELINE_CYCLES, s].mean()
        ).rename(f"{s}_base_mean")
        base_std = df.groupby("engine_id").apply(
            lambda g: g.loc[g["cycle"] <= BASELINE_CYCLES, s].std()
        ).rename(f"{s}_base_std")

        df = df.merge(base_mean, on="engine_id", how="left")
        df = df.merge(base_std, on="engine_id", how="left")

        std = df[f"{s}_base_std"].replace(0, np.nan)
        z = (df[s] - df[f"{s}_base_mean"]) / std
        z_abs_all.append(np.abs(z).to_numpy())

    Z = np.vstack(z_abs_all).T
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(TOP_K_SENSORS, Z.shape[1])
    topk = np.partition(Z, -k, axis=1)[:, -k:]
    drift_score_v2 = topk.mean(axis=1)

    # Build output
    keep_cols = [c for c in df.columns if not c.endswith("_base_mean") and not c.endswith("_base_std")]
    df_out = df[keep_cols].copy()
    df_out["drift_score_v2"] = drift_score_v2

    # Threshold from train-normal distribution
    train_mask = df_out["split"] == "train"
    normal_mask = train_mask & (df_out["cycles_to_failure"] > PROXY_ANOMALY_WINDOW)
    thresh = float(np.quantile(df_out.loc[normal_mask, "drift_score_v2"], THRESH_Q))

    # Evaluate on train
    df_train = df_out.loc[train_mask].copy()
    p, r, f1 = evaluate_on_train(df_train, "drift_score_v2", "is_proxy_anomaly", thresh)
    early = early_detection_cycles(df_train, "drift_score_v2", thresh)

    print("[INFO] Baseline: Early-life Drift v2 (baseline z-score)")
    print(f"[INFO] Baseline cycles per engine: {BASELINE_CYCLES}")
    print(f"[INFO] Drift aggregation: mean(top-{k} abs z-scores)")
    print(f"[INFO] Threshold strategy: {int(THRESH_Q*100)}th percentile of train-normal drift_score_v2")
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
