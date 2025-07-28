# preprocessing_sample.py

import pandas as pd

# Load full dataset
df = pd.read_csv("financial data sp500 companies.csv")

# Normalize columns
df.columns = df.columns.str.strip().str.lower()

# Show available columns
print("Available columns:", df.columns.tolist())

# Filter for one company (e.g., AAPL) and latest year (e.g., 2021)
# Define company and year to extract
company = "Apple Inc."
ticker = "AAPL"
target_year = "2021"

# Filter rows for AAPL and matching year
# Filter by ticker and year
filtered = df[(df["ticker"] == ticker) & (df["date"].str.startswith(target_year))]

# Ensure at least 1 row found
if filtered.empty:
    print("❌ No data found for", ticker)
    exit()

# Get first record
row = filtered.iloc[0]

# Extract sample values (columns that exist in your dataset)
# Prepare sample summary
sample_rows = [
    [company, target_year, "Total Revenue", row["total revenue"]],
    [company, target_year, "Net Income", row["net income"]],
    # These two don't exist in your dataset, so we’ll remove them
    # [company, target_year, "total assets", row["total assets"]],
    # [company, target_year, "total liabilities", row["total liabilities"]],
]

# Create final sample CSV
df_sample = pd.DataFrame(sample_rows, columns=["Company", "Year", "LineItem", "Value"])
df_sample.to_csv("sample_income_statement.csv", index=False)

print("✅ sample_income_statement.csv created successfully.")
