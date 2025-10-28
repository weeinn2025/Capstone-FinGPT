# tests/test_canonical_financials.py:

import os, zipfile, pandas as pd, pytest

CSV_ZIP = os.path.join("samples", "sample_financials_rev2_2020_2024.zip")
EXPECTED_COLUMNS = ["Company", "Year", "Line Item", "Value"]
ITEMS = {
    "Total Revenue",
    "Net Profit",
    "Total Assets",
    "Total Liabilities",
    "Shareholders' Equity",
}
COMPANIES = {"Apple Inc.", "Microsoft", "NVIDIA"}
YMIN, YMAX = 2020, 2024

@pytest.fixture(scope="session")


def df() -> pd.DataFrame:
    assert os.path.exists(CSV_ZIP), f"Missing: {CSV_ZIP}"
    rows = []
    with zipfile.ZipFile(CSV_ZIP) as zf:
        for n in zf.namelist():
            if n.endswith(".csv"):
                rows.append(pd.read_csv(zf.open(n)))
    return pd.concat(rows, ignore_index=True)


def test_columns(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_companies_and_items(df):
    assert set(df["Company"]) <= COMPANIES
    assert set(df["Line Item"]) <= ITEMS


def test_year_bounds(df):
    y = df["Year"].astype(int)
    assert y.between(YMIN, YMAX).all()


def test_value_numeric_nonneg(df):
    v = pd.to_numeric(df["Value"], errors="coerce")
    assert v.notna().all(), "Non-numeric Value detected"
    assert (v >= 0).all(), "Negative Value detected"


def test_unique_keys(df):
    dups = df.duplicated(subset=["Company", "Year", "Line Item"], keep=False)
    assert not dups.any(), f"Duplicate rows:\n{df[dups]}"


def test_balance_identity(df):
    g = df[df["Line Item"].isin({"Total Assets", "Total Liabilities", "Shareholders' Equity"})]
    for (c, y), t in g.groupby(["Company", "Year"]):
        vals = dict(zip(t["Line Item"], t["Value"]))
        assert (
            vals["Total Assets"] == vals["Total Liabilities"] + vals["Shareholders' Equity"]
        ), f"Identity fail: {c} {y}"
