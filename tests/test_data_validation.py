# tests/test_data_validation.py
import pandas as pd
import pytest

from app_validators import normalize_and_validate, DataValidationError


def test_normalizes_shareholders_equity_label():
    # Curly apostrophe + mixed spacing should normalize to "Shareholders' Equity"
    df = pd.DataFrame(
        {
            "Company": ["TestCo"],
            "Year": [2024],
            "LineItem": ["Shareholders’ Equity"],  # curly apostrophe + extra spaces
            "Value": [12345],
        }
    )
    out = normalize_and_validate(df)
    assert out.loc[0, "LineItem"] == "Shareholders' Equity"


def test_rejects_non_numeric_value():
    # Any unit text/commas/currency that can't be coerced must raise
    df = pd.DataFrame(
        {
            "Company": ["BadCo"],
            "Year": [2024],
            "LineItem": ["Total Assets"],
            "Value": ["3.6E+11 USD"],  # contains unit text -> should fail
        }
    )
    with pytest.raises((ValueError, DataValidationError)):
        normalize_and_validate(df)
