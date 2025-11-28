# tests/test_canonical_financials.py:

import os
import zipfile
import pandas as pd
import pytest

# --- Config (rev2 canonical packs) -----------------------------------------------
CSV_ZIP = os.path.join(os.path.dirname(__file__), "..", "samples", "sample_financials_rev2_2020_2024.zip")

EXPECTED_COLUMNS = ["Company", "Year", "Line Item", "Value"]
EXPECTED_ITEMS = {
    "Total Revenue",
    "Net Profit",
    "Total Assets",
    "Total Liabilities",
    "Shareholders' Equity",
}
EXPECTED_COMPANIES = {"Apple Inc.", "Microsoft", "NVIDIA"}
YEAR_MIN, YEAR_MAX = 2020, 2024


# --- Helpers ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    assert os.path.exists(CSV_ZIP), f"Missing: {CSV_ZIP}"
    parts = []
    with zipfile.ZipFile(CSV_ZIP) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                parts.append(pd.read_csv(zf.open(name)))
    out = pd.concat(parts, ignore_index=True)
    # normalize dtypes to be strict in assertions
    out["Year"] = out["Year"].astype(int)
    out["Value"] = pd.to_numeric(out["Value"], errors="raise")
    out["Company"] = out["Company"].astype(str)
    out["Line Item"] = out["Line Item"].astype(str)
    return out


# --- Tests -----------------------------------------------------------------------
def test_columns(df: pd.DataFrame):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_companies_and_items(df: pd.DataFrame):
    assert set(df["Company"]) == EXPECTED_COMPANIES
    assert set(df["Line Item"]) == EXPECTED_ITEMS


def test_year_bounds(df: pd.DataFrame):
    years = df["Year"].astype(int)
    assert years.between(YEAR_MIN, YEAR_MAX).all()


def test_value_numeric_nonneg(df: pd.DataFrame):
    vals = pd.to_numeric(df["Value"], errors="raise")
    assert (vals >= 0).all(), "Negative Value detected"


def test_unique_keys(df: pd.DataFrame):
    dups = df.duplicated(subset=["Company", "Year", "Line Item"], keep=False)
    assert not dups.any(), f"Duplicate rows:\n{df[dups]}"


def test_balance_identity(df: pd.DataFrame):
    g = df[df["Line Item"].isin({"Total Assets", "Total Liabilities", "Shareholders' Equity"})]
    for (company, year), grp in g.groupby(["Company", "Year"]):
        vals = dict(zip(grp["Line Item"], grp["Value"]))
        assert (
            vals["Total Assets"] == vals["Total Liabilities"] + vals["Shareholders' Equity"]
        ), f"Identity fail: {company} {year}"
