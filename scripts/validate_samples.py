#!/usr/bin/env python3
"""
Validate that the CSV and XLSX sample packs are identical row-for-row by
(Company, Year, Line Item, Value). Exits 1 with a diff table if not.
"""

import io
import sys
import zipfile
import pandas as pd
from pathlib import Path

# Canonical rev2 packs (required by CI)
CSV_ZIP = Path("samples/sample_financials_rev2_2020_2024.zip")
XLSX_ZIP = Path("samples/sample_financials_rev2_2020_2024_xlsx.zip")
XLSX_SHEET = "Financials_2020_2024"


def load_csv_zip(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Missing CSV pack: {path}"
    rows = []
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            if n.endswith(".csv"):
                rows.append(pd.read_csv(zf.open(n)))
    return pd.concat(rows, ignore_index=True)


def load_xlsx_zip(path: Path, sheet: str) -> pd.DataFrame:
    assert path.exists(), f"Missing XLSX pack: {path}"
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("Financials_2020_2024.xlsx")
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet)


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Company"] = out["Company"].astype(str)
    out["Year"] = out["Year"].astype(int)
    out["Line Item"] = out["Line Item"].astype(str)
    out["Value"] = out["Value"].astype(int)
    return out.sort_values(["Company", "Year", "Line Item"]).reset_index(drop=True)


def main() -> int:
    df_csv = canonicalize(load_csv_zip(CSV_ZIP))
    df_xls = canonicalize(load_xlsx_zip(XLSX_ZIP, XLSX_SHEET))

    if df_csv.equals(df_xls):
        print("OK: CSV and XLSX packs are identical by (Company, Year, Line Item, Value).")
        print(f"Row count: {len(df_csv)}")
        return 0

    merged = df_csv.merge(
        df_xls,
        on=["Company", "Year", "Line Item"],
        how="outer",
        suffixes=("_csv", "_xlsx"),
        indicator=True,
    )
    diff = merged[(merged["_merge"] != "both") | (merged["Value_csv"] != merged["Value_xlsx"])]
    print("DIFF detected between CSV and XLSX sample packs:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None):
        print(diff)
    return 1

if __name__ == "__main__":
    sys.exit(main())
  
