# app_validators.py
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, Tuple

import pandas as pd


__all__ = ["DataValidationError", "normalize_and_validate"]


@dataclass
class DataValidationError(Exception):
    """Raised when an uploaded financials file fails validation."""

    message: str
    sample_rows: pd.DataFrame | None = None

    def __str__(self) -> str:  # pragma: no cover
        return self.message


# Canonical LineItem names we support (keys are lower-cased)
CANON: Dict[str, str] = {
    "total assets": "Total Assets",
    "total liabilities": "Total Liabilities",
    "shareholders' equity": "Shareholders' Equity",
    "total revenue": "Total Revenue",
    "net profit": "Net Profit",
    "net income": "Net Income",
}

# Characters frequently seen in uploads that should be normalized to ASCII apostrophe
_APOSTROPHE_LIKE: Tuple[str, ...] = (
    "\u2019",  # ’
    "\u2018",  # ‘
    "\u2032",  # ′
    "\u02bc",  # ʼ
    "\uff07",  # Fullwidth '
)

# Sometimes users get mojibake like "Shareholdersâ€™ Equity"
# We normalize any run of non-word quotes to a plain apostrophe.
_NON_ASCII_QUOTE_RE = re.compile(r"[^\w\s]")

_MULTI_SPACE_RE = re.compile(r"\s+")


def _to_ascii_apostrophes(s: str) -> str:
    """Replace common Unicode apostrophes (and mojibake) with ASCII '."""
    if not s:
        return s
    s2 = s
    for ch in _APOSTROPHE_LIKE:
        s2 = s2.replace(ch, "'")
    # If it still contains odd punctuation between letters (e.g., â€™), collapse to '
    s2 = re.sub(r"(?<=\w)[^A-Za-z0-9\s](?=\w)", "'", s2)
    return s2


def _normalize_lineitem(raw: str) -> str:
    """
    Normalize a LineItem string to a canonical label in CANON.
    - Trim, collapse spaces, normalize apostrophes.
    - Case-insensitive matching against CANON keys.
    """
    if raw is None:
        return raw
    s = str(raw).strip()
    s = _to_ascii_apostrophes(s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    key = s.lower()
    # Fast path: exact canonical key
    if key in CANON:
        return CANON[key]

    # Soft matching: remove punctuation other than apostrophe and compare
    def _soft(s0: str) -> str:
        s1 = s0.lower()
        s1 = re.sub(r"[^a-z0-9'\s]", " ", s1)
        s1 = _MULTI_SPACE_RE.sub(" ", s1).strip()
        return s1

    soft = _soft(s)
    for k in CANON:
        if _soft(k) == soft:
            return CANON[k]

    # If no match, just return the trimmed/cleaned original (caller may reject)
    return s


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {', '.join(missing)}",
            sample_rows=df.head(5),
        )


def normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an uploaded financials DataFrame and enforce numeric Value.
    Required columns: Company, Year, LineItem (or 'Line Item'), Value

    Returns a **new** cleaned DataFrame with:
      - columns: Company (str), Year (Int64), LineItem (canonical), Value (float)
    Raises DataValidationError on any issue (missing columns, non-numeric Value, unknown LineItem).
    """
    if df is None or df.empty:
        raise DataValidationError("Uploaded file is empty.", None)

    # Accept 'Line Item' or 'LineItem'
    lineitem_col = "LineItem" if "LineItem" in df.columns else ("Line Item" if "Line Item" in df.columns else None)
    _require_columns(df, ["Company", "Year", lineitem_col, "Value"])

    # Make a working copy with just the columns we need
    out = df[["Company", "Year", lineitem_col, "Value"]].copy()
    out.rename(columns={lineitem_col: "LineItem"}, inplace=True)

    # Company: strip
    out["Company"] = out["Company"].astype(str).str.strip()

    # Year → Int64 (nullable), but must be numeric
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")

    # Normalize LineItem text and map to CANON labels
    out["LineItem"] = out["LineItem"].astype(str).map(_normalize_lineitem)

    # Coerce Value to numeric (accept commas, currency symbols if present)
    # Remove common formatting noise, then to_numeric
    cleaned_value = (
        out["Value"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    )
    out["Value"] = pd.to_numeric(cleaned_value, errors="coerce")

    # ---- Hard validations ---------------------------------------------------
    # 1) No missing criticals
    bad_core = out[out["Company"].eq("") | out["Year"].isna() | out["LineItem"].isna()]
    if not bad_core.empty:
        raise DataValidationError(
            "Company/Year/LineItem must be present for all rows.",
            sample_rows=bad_core.head(10),
        )

    # 2) Value must be numeric for all rows
    bad_value = out[out["Value"].isna()]
    if not bad_value.empty:
        raise DataValidationError(
            "Non-numeric values found in 'Value'. Please fix the highlighted rows.",
            sample_rows=bad_value.head(10),
        )

    # 3) LineItem must be among supported canon set
    allowed = set(CANON.values())
    bad_li = out[~out["LineItem"].isin(allowed)]
    if not bad_li.empty:
        raise DataValidationError(
            "Unsupported 'LineItem' labels detected after normalization.",
            sample_rows=bad_li.head(10),
        )

    # All good: sort for stability
    out.sort_values(["Company", "Year", "LineItem"], inplace=True, kind="stable")
    out.reset_index(drop=True, inplace=True)
    return out
