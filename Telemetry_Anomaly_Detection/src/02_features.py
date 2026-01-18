from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_canonical.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fd001_features.csv"

PROXY_ANOMALY_WINDOW = 30  # last N cycles before failure treated as abnormal (train only)


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}. Run src/01_ingest.py first.")

    df = pd.read_csv(IN_PATH)

    # Validate required columns
    required = {"engine_id", "cycle", "split", "failure_cycle", "cycles_to_failure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # Create proxy anomaly label (train only)
    df["is_proxy_anomaly"] = 0

    train_mask = df["split"] == "train"
    # cycles_to_failure is valid only for train in our ingestion step
    df.loc[train_mask, "is_proxy_anomaly"] = (
        df.loc[train_mask, "cycles_to_failure"] <= PROXY_ANOMALY_WINDOW
    ).astype(int)

    # Sanity checks
    train_anom_rate = df.loc[train_mask, "is_proxy_anomaly"].mean()
    print(f"[INFO] Proxy anomaly window (cycles): {PROXY_ANOMALY_WINDOW}")
    print(f"[STATS] Train anomaly rate: {train_anom_rate:.4f}")

    # Count anomalies by engine (train)
    by_engine = (
        df.loc[train_mask].groupby("engine_id")["is_proxy_anomaly"].sum().describe()
    )
    print("[STATS] Proxy anomalies per train engine (sum over cycles):")
    print(by_engine.to_string())

    df.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Output written:\n     {OUT_PATH}")


if __name__ == "__main__":
    main()
