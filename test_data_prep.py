# tests/test_data_prep.py
import zipfile
from pathlib import Path
import pandas as pd

from app import is_allowed_file, read_anytabular, normalize_financial_df


def _sample_df():
    return pd.DataFrame(
        {
            "Company": ["A", "A"],
            "Year": [2021, 2022],
            "LineItem": ["Total Revenue", "Net Income"],
            "Value": [100, 25],
        }
    )


def test_is_allowed_file():
    assert is_allowed_file("x.csv")
    assert is_allowed_file("y.xlsx")
    assert is_allowed_file("z.ZIP")  # case-insensitive
    assert not is_allowed_file("bad.txt")


def test_read_anytabular_csv_xlsx_zip(tmp_path: Path):
    df = _sample_df()

    # CSV
    csvp = tmp_path / "s.csv"
    df.to_csv(csvp, index=False)
    d1 = read_anytabular(csvp)
    assert len(d1) == 2

    # XLSX
    xlp = tmp_path / "s.xlsx"
    df.to_excel(xlp, index=False)
    d2 = read_anytabular(xlp)
    assert list(d2.columns) == list(df.columns)

    # ZIP (with CSV inside)
    zipp = tmp_path / "s.zip"
    with zipfile.ZipFile(zipp, "w") as zf:
        zf.writestr("inner.csv", df.to_csv(index=False))
    d3 = read_anytabular(zipp)
    assert "LineItem" in d3.columns


def test_normalize_financial_df_variants():
    df = pd.DataFrame(
        {
            "firm": ["X"],
            "fy": [2023],
            "account": ["Revenue"],
            "amount": [123.0],
        }
    )
    out, warn = normalize_financial_df(df)
    expected = {"Company", "Year", "LineItem", "Value"}
    assert expected.issubset(out.columns)
    # a warning is fine; we only assert the canonical columns are present
