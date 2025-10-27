# scripts/validate_financials.py
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
import zipfile

from app_validators import normalize_and_validate

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".xlsx"))]
            if not members:
                raise SystemExit("ZIP has no CSV/XLSX files.")
            frames = []
            for name in sorted(members):
                with zf.open(name) as fh:
                    if name.lower().endswith(".csv"):
                        frames.append(pd.read_csv(fh))
                    else:
                        frames.append(pd.read_excel(fh, engine="openpyxl"))
        return pd.concat(frames, ignore_index=True, sort=False)
    raise SystemExit(f"Unsupported file type: {path.suffix}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="CSV/XLSX/ZIP to validate")
    args = p.parse_args()

    df = _read_any(Path(args.path))
    _ = normalize_and_validate(df)  # raises on problems
    print("✅ OK: financials validated")

if __name__ == "__main__":
    main()
