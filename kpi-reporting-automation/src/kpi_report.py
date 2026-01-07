import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"


REQUIRED_SALES_COLS = {"date", "region", "channel", "orders", "revenue", "cogs", "returns"}
REQUIRED_AD_COLS = {"date", "channel", "ad_spend", "clicks", "leads"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def validate_columns(df: pd.DataFrame, required_cols: set, file_name: str) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        fail(f"{file_name} is missing columns: {sorted(missing)}")


def coerce_numeric(df: pd.DataFrame, cols: list, file_name: str) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[cols].isna().any().any():
        bad_rows = df[df[cols].isna().any(axis=1)]
        fail(
            f"{file_name} has non-numeric values in numeric columns. "
            f"Fix these rows:\n{bad_rows.head(10).to_string(index=False)}"
        )
    return df


def load_and_clean_sales(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_SALES_COLS, path.name)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        fail(f"{path.name} has invalid dates. Check date column format (YYYY-MM-DD).")

    df = coerce_numeric(df, ["orders", "revenue", "cogs", "returns"], path.name)

    # Basic sanity checks
    if (df["orders"] < 0).any() or (df["revenue"] < 0).any() or (df["cogs"] < 0).any() or (df["returns"] < 0).any():
        fail(f"{path.name} has negative values. Fix before reporting.")

    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")  # Monday as start
    return df


def load_and_clean_ads(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_AD_COLS, path.name)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        fail(f"{path.name} has invalid dates. Check date column format (YYYY-MM-DD).")

    df = coerce_numeric(df, ["ad_spend", "clicks", "leads"], path.name)

    if (df["ad_spend"] < 0).any() or (df["clicks"] < 0).any() or (df["leads"] < 0).any():
        fail(f"{path.name} has negative values. Fix before reporting.")

    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    return df


def build_weekly_kpis(sales: pd.DataFrame, ads: pd.DataFrame) -> pd.DataFrame:
    # Aggregate weekly sales KPIs
    weekly_sales = (
        sales.groupby(["week_start", "region", "channel"], as_index=False)
        .agg(
            Orders=("orders", "sum"),
            Revenue=("revenue", "sum"),
            COGS=("cogs", "sum"),
            Returns=("returns", "sum"),
        )
    )
    weekly_sales["Gross Profit"] = weekly_sales["Revenue"] - weekly_sales["COGS"]
    weekly_sales["Gross Margin %"] = (weekly_sales["Gross Profit"] / weekly_sales["Revenue"]).replace([pd.NA, pd.NaT], 0)
    weekly_sales["Return Rate %"] = (weekly_sales["Returns"] / weekly_sales["Orders"]).replace([pd.NA, pd.NaT], 0)

    # Aggregate weekly ads KPIs (channel-level)
    weekly_ads = (
        ads.groupby(["week_start", "channel"], as_index=False)
        .agg(
            Ad_Spend=("ad_spend", "sum"),
            Clicks=("clicks", "sum"),
            Leads=("leads", "sum"),
        )
    )

    # Merge sales + ads on week_start + channel
    merged = weekly_sales.merge(weekly_ads, on=["week_start", "channel"], how="left")
    merged[["Ad_Spend", "Clicks", "Leads"]] = merged[["Ad_Spend", "Clicks", "Leads"]].fillna(0)

    # Efficiency metrics
    merged["CPL"] = (merged["Ad_Spend"] / merged["Leads"]).replace([pd.NA, pd.NaT], 0)
    merged["ROAS"] = (merged["Revenue"] / merged["Ad_Spend"]).replace([pd.NA, pd.NaT], 0)

    # Make week_start readable
    merged["week_start"] = merged["week_start"].dt.date
    return merged


def build_exec_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    # Exec summary aggregated per week (all regions + channels)
    tmp = weekly.copy()
    summary = (
        tmp.groupby(["week_start"], as_index=False)
        .agg(
            Orders=("Orders", "sum"),
            Revenue=("Revenue", "sum"),
            COGS=("COGS", "sum"),
            Returns=("Returns", "sum"),
            Ad_Spend=("Ad_Spend", "sum"),
            Clicks=("Clicks", "sum"),
            Leads=("Leads", "sum"),
        )
    )
    summary["Gross Profit"] = summary["Revenue"] - summary["COGS"]
    summary["Gross Margin %"] = summary["Gross Profit"] / summary["Revenue"]
    summary["Return Rate %"] = summary["Returns"] / summary["Orders"]
    summary["CPL"] = summary["Ad_Spend"] / summary["Leads"]
    summary["ROAS"] = summary["Revenue"] / summary["Ad_Spend"]
    summary = summary.replace([pd.NA, pd.NaT, float("inf")], 0).fillna(0)
    return summary


def export_excel(exec_summary: pd.DataFrame, weekly: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        exec_summary.to_excel(writer, sheet_name="Exec_Summary", index=False)
        weekly.to_excel(writer, sheet_name="Weekly_KPIs", index=False)

    print(f"Saved report: {out_path}")


def main():
    sales_path = INPUT_DIR / "sales_daily.csv"
    ads_path = INPUT_DIR / "ad_spend_daily.csv"

    if not sales_path.exists():
        fail(f"Missing input file: {sales_path}")
    if not ads_path.exists():
        fail(f"Missing input file: {ads_path}")

    sales = load_and_clean_sales(sales_path)
    ads = load_and_clean_ads(ads_path)

    weekly = build_weekly_kpis(sales, ads)
    exec_summary = build_exec_summary(weekly)

    out_file = OUTPUT_DIR / "KPI_Report.xlsx"
    export_excel(exec_summary, weekly, out_file)


if __name__ == "__main__":
    main()
