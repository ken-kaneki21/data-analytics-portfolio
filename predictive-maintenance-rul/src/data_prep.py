# src/data_prep.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def get_cmapss_columns() -> list[str]:
    # Standard C-MAPSS layout:
    # unit_number, time_in_cycles, op_setting_1..3, sensor_1..21
    cols = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    cols += [f"sensor_{i}" for i in range(1, 22)]
    return cols


def read_cmapss_txt(path: Path) -> pd.DataFrame:
    # Files are space-separated with possible extra trailing spaces
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    expected_cols = len(get_cmapss_columns())

    # Some copies may include an extra empty column due to trailing spaces
    if df.shape[1] > expected_cols:
        df = df.iloc[:, :expected_cols]

    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected column count in {path.name}: got {df.shape[1]}, expected {expected_cols}."
        )

    df.columns = get_cmapss_columns()
    return df


def read_rul_txt(path: Path) -> pd.DataFrame:
    # One RUL value per test engine (in order)
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Again, guard against extra trailing-space columns
    if rul.shape[1] > 1:
        rul = rul.iloc[:, :1]
    rul.columns = ["RUL_after_last_cycle"]
    return rul


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer parquet; fall back to csv if parquet engine not installed
    try:
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
        print(f"Saved: {out_path.with_suffix('.parquet')}")
    except Exception as e:
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Parquet failed ({type(e).__name__}: {e}). Saved CSV instead: {csv_path}")


def main() -> None:
    # Adjust these if your repo uses different casing like Data/Raw
    raw_dir = Path("data/raw/CMAPSSData")
    processed_dir = Path("data/processed")

    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing file: {p}. Fix your folder names/paths before continuing."
            )

    train_df = read_cmapss_txt(train_path)
    test_df = read_cmapss_txt(test_path)
    rul_df = read_rul_txt(rul_path)

    # Minimal type sanity (keeps things clean later)
    # unit_number and time_in_cycles should be int
    train_df["unit_number"] = train_df["unit_number"].astype(int)
    train_df["time_in_cycles"] = train_df["time_in_cycles"].astype(int)
    test_df["unit_number"] = test_df["unit_number"].astype(int)
    test_df["time_in_cycles"] = test_df["time_in_cycles"].astype(int)

    save_df(train_df, processed_dir / "train_fd001")
    save_df(test_df, processed_dir / "test_fd001")
    save_df(rul_df, processed_dir / "rul_fd001")

    print("\nQuick checks:")
    print("train:", train_df.shape, "unique engines:", train_df["unit_number"].nunique())
    print("test :", test_df.shape, "unique engines:", test_df["unit_number"].nunique())
    print("rul  :", rul_df.shape)


if __name__ == "__main__":
    main()
