Capstone‑FinGPT
===============

A lightweight Flask web app to ingest financial statements, preview and clean them, generate a concise AI-powered analysis + charts, and download a PDF report.

🚀 Live demo

Render: https://capstone-fingpt.onrender.com

Render deployment notes:
- ✅   Free tier.
- ✅   First visit after idle may cold-start (a minute or two).
- ✅   If you see “Too many requests” on the home page, refresh once; if it persists, wait ~60s.

📘 Repository

https://github.com/weeinn2025/Capstone-FinGPT

> ⚠️**Disclaimer**
> The sample dataset in this repository is illustrative and may mix calendar and
> fiscal years because companies have different year-ends (e.g., Apple ≈ late Sep,
> Microsoft ≈ Jun 30, NVIDIA ≈ late Jan). Values are simplified or rounded and may not
> match audited filings. Do not use for investment decisions. Always verify
> against the company’s latest 10-K/10-Q or annual report.


Features
========
(Note: “The application is a work in progress with ongoing enhancements, including advanced features, visualization, and AI, with a demo available upon request.”)

1.   **Upload multiple formats - flexible ingest**
     (Accepted file formats for Data Input & Pre-processing)
     - ✅   support **.xlsx**  (via `pandas` + `openpyxl`- same logical columns; multiple sheets are supported (first sheet is read by default))
     - ✅   support **.csv**   (header row required, example: Company,Year,LineItem,Value)
     - ✅   support **.zip**   (uploads a `.zip` that contains one or more CSV/XLSX files - the app reads the **first valid** tabular file inside; demo size ~5 MB)
     - ✅   Try the samples:   `sample_financials_2020_2024_xlsx` (for multi-company all-years line gragh), `sample_companies_2024.csv` (for multi-company grouped bars), `sample_income_statement.csv`, `sample_income_statement.xlsx`, `sample_csv_only.zip`.

3.   **Preview before analysis**
     - ✅   see the **first 10 rows** on a `/preview` screen before analysis, then click **Analyze this file**.

4.   **Best-effort normalization**
     - ✅   Canonical schema (columns): Company | Year | LineItem | Value
     - ✅   `Company | Year | LineItem | Value` (trims headers, coerces numbers, drops blank rows; falls back to raw columns if mapping fails).
     - ✅   Trims headers, coerces numbers, drops blank rows.
     - ✅   Falls back to raw columns if mapping fails.
 
5.   **AI summary**
     - ✅   generates a concise 2–3 sentence narrative via **Gemini** when `GEMINI_*` env vars are set.
     - ✅   app works without AI keys too, shows a friendly “AI disabled” note, and renders summary, chart, and PDF.

6.   **Interactive charts (Plotly) - dashboard**
     - ✅   grouped bars of **Revenue** vs **Net income** for the **latest year**.
     - ✅   Canonicalizes line items so common names map correctly - Synonym mapping (≈):
     -       *  Revenue: “revenue”, “total revenue”, “sales”, “total sales”
     -       *  Net income: “net income”, “net profit”, “profit”
     - ✅   If Plotly JSON is not present, the page falls back to a static PNG.

7.   **PDF export report - Data + Chart + AI analysis**
     - ✅   Includes data table, AI text, and the **same chart** as an image.  
     - ✅   Primary path: Plotly **via Kaleido → PNG → PDF**.  
     - ✅   Fallback: **Matplotlib grouped bars** (no Chrome/Kaleido required).
     - ✅   Renders a bar chart and lets you **download a nicely formatted PDF**.
     - ✅   One-click Download as PDF renders the data + AI text and embeds the same chart as a PNG.
     - ✅   If Kaleido is unavailable, export still works via a Matplotlib grouped snapshot.
     - ✅   To enable Plotly → PNG locally, install Chrome once:
            ```bash
             plotly_get_chrome
     - ✅   “This is what the output looks like” with short narrative - AI analysis:

 
### 📊 Interactive Charts & PDF Export  

- Upload → Normalize → AI Analysis → Chart → PDF Download  
- Live screenshots from the Render demo:

**Revenue & Net Income (Chart 1 – Multi-Year Line Graph):**  
![Revenue vs Net Income (Line)](static/screenshots/chart1.png)

**Revenue & Net Income (Chart 2 – Latest Year Bar Graph):**  
![Revenue vs Net Income (Bar)](static/screenshots/chart2.png)

**Generated PDF Report (includes table, AI analysis & chart):**  
[📄 Download Sample PDF](static/screenshots/report.pdf)

**To Note:**
• Different fiscal year-ends (Apple: late Sep; Microsoft: Jun 30; NVIDIA: late Jan).
• Some of 2024 lines appear to be calendar-year or a later fiscal year instead of each company’s FY2024.
• One Apple row (“Shareholders’ Equity $75B”) seems copied from an earlier year; Apple shows negative equity in FY2024 due to buybacks. 
[annualreports.com]
      
  
8.   **Samples included**
     - ✅   Located in `samples/` to test quickly (CSV/XLSX + ZIP fixtures).

9.   **Safety**
     - ✅   `GET /` (home page) is **not** rate-limited;  
     - ✅   `POST /preview` and `POST /upload` are limited to **10 requests per minute** (demo safety).
     - ✅   For production → configure a shared store or use Redis/Memcached with Flask-Limiter.  

10.  **Clear error messages** for unreadable or invalid files.


How it flows
============

**Mermaid block**
```md
```mermaid
graph LR
A[Upload CSV/XLSX/ZIP] --> B[Read + Normalize]
B --> C[Preview table]
B --> D[AI summary (Gemini)]
B --> E[Plotly interactive charts/grouped bars (latest year)]
E --> F[PNG snapshot via Kaleido]
F --> G[PDF export (WeasyPrint)]


Prerequisites
=============

1.   Python 3.11

2.   Git or GitHub

3.   (Optional) Conda or virtualenv for isolation


Local Setup
===========

1.   Clone your repository

     git clone https://github.com/weeinn2025/Capstone-FinGPT.git
     cd Capstone-FinGPT

2.   Create and activate the environment

     conda create -n ml python=3.11
     conda activate ml
     # or use virtualenv
     
     FLASK_SECRET_KEY=<random_24+_bytes_or_hex>
     # Optional (for AI text)
     GEMINI_API_KEY=<your key>
     GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent


3.   *Install all the dependencies*:
     ```md
     ```bash
     pip install -r requirements.txt

4.   Configure the environment

     Copy .env.example → .env

     Set your FLASK_SECRET_KEY, GEMINI_API_KEY, and GEMINI_URL.

     *  Environment variables:

     GEMINI_API_KEY	(Optional)  --- >  Google Gemini API key
     GEMINI_MODEL	(Optional)  --- >  Model name; e.g., gemini-1.5-flash
     (If these are unset, analysis runs without AI text).

6.   Then Run locally:
     flask run or python app.py

     # Visit http://127.0.0.1:5000


Using the app
=============

1.   First, navigate to the home page and upload a file (.csv, .xlsx, or .zip).

2.   Then, click Preview to see the first 10 rows in your browser.

3.   Click Analyze this file to:
     -  show a normalized summary table.
     -  generate AI analysis/text (if env is set), and the interactive chart.
     -  render a bar chart, line graph

4.   You are able to click Download as PDF to save the report.


Dev & CI
========

- ✅   Lint/format: Black + Flake8 (run locally with black . && flake8).

- ✅   Pre-commit [optional]:
       ```md
       ```bash
       pip install pre-commit
       pre-commit install


Testing & linting
=================

- ✅   Run the linters and tests from the project root:
       ```md
       ```bash
       black . && flake8 . && pytest -q

- ✅   If you don’t have the tools yet:
       ```md
       ```bash
       pip install black flake8 pytest

(Works the same in macOS/Linux terminals and Windows PowerShell.)


Deployment
==========

Render (✅ Done)

1.   Add runtime.txt with:
     *  python-3.11.11
2.   Ensure requirements.txt includes:
     *  gunicorn==20.1.0
3.   Connect your GitHub repo & deploy.  Cold starts expected on free tier.


Troubleshooting
===============

✅   Blank interactive chart:
     *  confirm at least one year has both Revenue and Net income (synonyms above).

✅   PDF uses Matplotlib, if Kaleido not available:
     *  install Chrome via plotly_get_chrome (optional).    

✅   ZIPs not in Git:
     *  ZIPs are ignored globally but allowed in samples/ (see .gitignore).


Project Structure
=================

Capstone-FinGPT/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ Dockerfile
├─ docker-compose.yml
├─ README.md
├─ .env.example
├─ .gitignore
├─ .flake8
├─ .pre-commit-config.yaml
├─ templates/
│  ├─ index.html
│  ├─ preview.html
│  ├─ result.html         # interactive Plotly (or static fallback)
│  └─ pdf.html            # WeasyPrint layout for PDF
├─ samples/
|  ├─ sample_financials_2020_2024.csv_only.zip
|  ├─ sample_financials_2020_2024.xlsx.zip
│  ├─ sample_companies_2024.csv
│  ├─ sample_income_statement.csv
│  ├─ sample_income_statement.xlsx
│  ├─ sample_csv_only.zip
│  ├─ sample_mixed_csv_xlsx.zip
│  └─ sample_xlsx_only.zip
├─ uploads/               # user uploads at runtime (gitignored)
│  └─ .gitkeep
├─ static/screenshots     # optional assets (logos/CSS/JS)
|  ├─ chart.png
|  ├─ pdf_sample.png
│  ├─ pdf_report.pdf
├─ tests/                 # add unit tests here
│  └─ (placeholder)
└─ .github/workflows/
   └─ pr-ci.yml           # PR CI: lint, test, build smoke


License
MIT © 2025

