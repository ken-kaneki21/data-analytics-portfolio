from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FEATURES = PROJECT_ROOT / "data" / "processed" / "fd001_features.csv"
IN_IFOREST = PROJECT_ROOT / "data" / "processed" / "fd001_scored_iforest.csv"
IN_DRIFT = PROJECT_ROOT / "data" / "processed" / "fd001_scored_drift_v2.csv"

OUT_ENSEMBLE = PROJECT_ROOT / "data" / "processed" / "fd001_scored_ensemble.csv"
OUT_ENGINE_SUMMARY = PROJECT_ROOT / "data" / "processed" / "fd001_engine_summary.csv"

PROXY_ANOMALY_WINDOW = 30


def apply_minmax_from_train_normal(df: pd.DataFrame, col: str, normal_mask: pd.Series) -> pd.Series:
    lo = float(df.loc[normal_mask, col].quantile(0.01))
    hi = float(df.loc[normal_mask, col].quantile(0.99))
    s = df[col].clip(lo, hi)
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    return (s - lo) / denom


def prf(y_true: np.ndarray, y_pred: np.ndarray):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def early_detection_cycles(df_train: pd.DataFrame, score_col: str, threshold: float) -> float:
    records = []
    for _, g in df_train.groupby("engine_id"):
        g = g.sort_values("cycle")
        near_fail = g[g["cycles_to_failure"] <= PROXY_ANOMALY_WINDOW].copy()
        if near_fail.empty:
            continue
        flagged = near_fail[near_fail[score_col] >= threshold]
        if flagged.empty:
            records.append(0.0)
        else:
            records.append(float(flagged.iloc[0]["cycles_to_failure"]))
    return float(np.mean(records)) if records else 0.0


def main() -> None:
    for p in [IN_FEATURES, IN_IFOREST, IN_DRIFT]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    df_base = pd.read_csv(IN_FEATURES)
    df_if = pd.read_csv(
        IN_IFOREST,
        usecols=["engine_id", "cycle", "split", "iforest_anomaly_score"],
    )
    df_dr = pd.read_csv(
        IN_DRIFT,
        usecols=["engine_id", "cycle", "split", "drift_score_v2"],
    )

    df = df_base.merge(df_if, on=["engine_id", "cycle", "split"], how="left")
    df = df.merge(df_dr, on=["engine_id", "cycle", "split"], how="left")

    if df["iforest_anomaly_score"].isna().any():
        raise ValueError("NaNs in iforest_anomaly_score after merge.")
    if df["drift_score_v2"].isna().any():
        raise ValueError("NaNs in drift_score_v2 after merge.")

    train_mask = df["split"] == "train"
    normal_mask = train_mask & (df["cycles_to_failure"] > PROXY_ANOMALY_WINDOW)

    # Normalize using train-normal quantiles only
    df["iforest_norm"] = apply_minmax_from_train_normal(df, "iforest_anomaly_score", normal_mask)
    df["drift_norm"] = apply_minmax_from_train_normal(df, "drift_score_v2", normal_mask)

    df_train = df.loc[train_mask].copy()
    y_true = df_train["is_proxy_anomaly"].astype(int).values

    # Grid search weights + threshold quantiles
    weight_grid = np.round(np.linspace(0.0, 1.0, 21), 2)   # 0.00..1.00 step 0.05
    q_grid = np.round(np.linspace(0.90, 0.995, 20), 3)     # 0.900..0.995

    best = {"f1": -1.0, "w": None, "q": None}

    cycles_to_failure_train = df_train["cycles_to_failure"].values
    if_norm_train = df_train["iforest_norm"].values
    dr_norm_train = df_train["drift_norm"].values

    normal_region_mask = cycles_to_failure_train > PROXY_ANOMALY_WINDOW

    for w in weight_grid:
        score = (w * if_norm_train) + ((1.0 - w) * dr_norm_train)
        score_normal = score[normal_region_mask]

        for q in q_grid:
            thresh = float(np.quantile(score_normal, q))
            y_pred = (score >= thresh).astype(int)
            p, r, f1 = prf(y_true, y_pred)

            if f1 > best["f1"]:
                best = {"f1": f1, "w": float(w), "q": float(q)}

    # Apply best settings to full dataset
    w = best["w"]
    df["ensemble_score"] = (w * df["iforest_norm"]) + ((1.0 - w) * df["drift_norm"])

    # Threshold from TRAIN-NORMAL using best quantile
    df_train_full = df.loc[train_mask].copy()
    normal_scores = df_train_full.loc[
        df_train_full["cycles_to_failure"] > PROXY_ANOMALY_WINDOW, "ensemble_score"
    ].values
    thresh = float(np.quantile(normal_scores, best["q"]))

    df["ensemble_flag"] = (df["ensemble_score"] >= thresh).astype(int)

    # Final eval + early detection (train only)
    df_train_out = df.loc[train_mask].copy()
    y_pred_final = (df_train_out["ensemble_score"].values >= thresh).astype(int)
    p, r, f1 = prf(df_train_out["is_proxy_anomaly"].astype(int).values, y_pred_final)
    early = early_detection_cycles(df_train_out, "ensemble_score", thresh)

    print("[INFO] Tuned Ensemble: w*iforest_norm + (1-w)*drift_norm")
    print(f"[INFO] Best weight w (iforest): {w:.2f}  | drift weight: {1.0 - w:.2f}")
    print(f"[INFO] Best threshold quantile (train-normal): {best['q']:.3f}")
    print(f"[INFO] Threshold value: {thresh:.6f}\n")

    print("[EVAL on TRAIN (proxy labels)]")
    print(f" Precision: {p:.4f}")
    print(f" Recall   : {r:.4f}")
    print(f" F1       : {f1:.4f}")
    print(
        f" Early-detect (mean cycles_to_failure at first flag within last {PROXY_ANOMALY_WINDOW} cycles): {early:.2f}"
    )

    # -------- Engine summary (NO groupby.apply) --------
    df_key = df[
        ["split", "engine_id", "cycle", "cycles_to_failure", "ensemble_flag", "ensemble_score"]
    ].copy()

    flagged = df_key[df_key["ensemble_flag"] == 1].copy()

    all_eng = df_key[["split", "engine_id"]].drop_duplicates()

    if flagged.empty:
        eng_first = all_eng.copy()
        eng_first["first_detect_cycle"] = np.nan
        eng_first["first_detect_ctf"] = np.nan
    else:
        idx = flagged.groupby(["split", "engine_id"])["cycle"].idxmin()
        first_rows = flagged.loc[idx, ["split", "engine_id", "cycle", "cycles_to_failure"]].copy()
        first_rows = first_rows.rename(
            columns={"cycle": "first_detect_cycle", "cycles_to_failure": "first_detect_ctf"}
        )
        eng_first = all_eng.merge(first_rows, on=["split", "engine_id"], how="left")

    eng_agg = df_key.groupby(["split", "engine_id"]).agg(
        max_ensemble_score=("ensemble_score", "max"),
        pct_cycles_flagged=("ensemble_flag", "mean"),
        total_cycles=("cycle", "max"),
    ).reset_index()

    engine_summary = eng_first.merge(eng_agg, on=["split", "engine_id"], how="left")
    engine_summary["pct_cycles_flagged"] = (engine_summary["pct_cycles_flagged"] * 100).round(2)
    # -----------------------------------------------

    df.to_csv(OUT_ENSEMBLE, index=False)
    engine_summary.to_csv(OUT_ENGINE_SUMMARY, index=False)

    print("\n[OK] Outputs written:")
    print(f"     {OUT_ENSEMBLE}")
    print(f"     {OUT_ENGINE_SUMMARY}")


if __name__ == "__main__":
    main()
