from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "fd001_canonical.csv"

SUBSET = "FD001"

# NASA CMAPSS has: id, cycle, 3 op settings, 21 sensors (s1..s21)
COLS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)


def _read_cmapss_txt(path: Path) -> pd.DataFrame:
    """
    CMAPSS files are whitespace-delimited and often have trailing empty columns.
    We read with delim_whitespace and then trim to the expected number of columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Trim any trailing empty columns
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS

    # Type coercion
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    for c in df.columns:
        if c not in ("engine_id", "cycle"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic validation
    if df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs found after parsing. Columns with NaNs: {na_cols}")

    return df


def main() -> None:
    train_path = RAW_DIR / f"train_{SUBSET}.txt"
    test_path = RAW_DIR / f"test_{SUBSET}.txt"
    rul_path = RAW_DIR / f"RUL_{SUBSET}.txt"

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Reading train: {train_path}")
    print(f"[INFO] Reading test : {test_path}")
    print(f"[INFO] Reading RUL  : {rul_path}")

    train_df = _read_cmapss_txt(train_path)
    test_df = _read_cmapss_txt(test_path)

    # Add a split column (useful later for pipelines)
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Compute failure_cycle for train (max cycle per engine)
    train_failure = train_df.groupby("engine_id")["cycle"].max().rename("failure_cycle")
    train_df = train_df.merge(train_failure, on="engine_id", how="left")
    train_df["cycles_to_failure"] = train_df["failure_cycle"] - train_df["cycle"]

    # For test, we don't know true failure_cycle from file directly; we can leave blank for now.
    # We will use RUL_FD001 later when needed.
    test_df["failure_cycle"] = pd.NA
    test_df["cycles_to_failure"] = pd.NA

    # Combine into one canonical table
    canonical = pd.concat([train_df, test_df], ignore_index=True)

    # Sort for sanity
    canonical["split"] = pd.Categorical(canonical["split"], categories=["train", "test"], ordered=True)
    canonical = canonical.sort_values(["split", "engine_id", "cycle"]).reset_index(drop=True)

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(OUT_PATH, index=False)

    # Sanity prints
    n_train_eng = train_df["engine_id"].nunique()
    n_test_eng = test_df["engine_id"].nunique()
    print("\n[OK] Output written:")
    print(f"     {OUT_PATH}")
    print("\n[STATS]")
    print(f" Train engines: {n_train_eng}, rows: {len(train_df):,}")
    print(f" Test engines : {n_test_eng}, rows: {len(test_df):,}")
    print(f" Total rows   : {len(canonical):,}")
    print("\n[SAMPLE]")
    print(canonical.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
