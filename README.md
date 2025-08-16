Capstone‑FinGPT
===============

A lightweight Flask app to ingest financial statements, preview/clean them, generate a concise AI-powered analysis + chart, and download a PDF report.

🚀 Live demo

Render: https://capstone-fingpt.onrender.com

(Free tier: cold starts can take 2–3 minutes.)

Render deployment notes:
✅   On first visit after idle, Render cold-starts the service (wait 1–2 min).
✅   If you ever see “Too many requests” on the home page, the page is now exempt from rate limiting; refresh the browser once and it should clear. If it doesn’t, wait 60s and try again.


📘 Repository

https://github.com/weeinn2025/Capstone-FinGPT


Features
========
(Note: “interactive” charts aren’t live yet.  Work-in-progress to enhance more features.)

1.   **Upload multiple formats**
     (Accepted file formats for Data Input & Pre-processing)
     ✅  support **.xlxs**  (via `pandas` + `openpyxl`- same logical columns; multiple sheets are supported (first sheet is read by default))
     ✅  support **.csv**   (header row required, example: Company,Year,LineItem,Value)
     ✅  support **.zip**   (uploads a `.zip` that contains one or more CSV/XLSX files - the app reads the **first valid** tabular file inside; demo size ~5 MB)
     ✅  Try the samples:   `sample_income_statement.csv`, `sample_income_statement.xlsx`, `sample_csv_only.zip`.

3.   **Preview before analysis**
     ✅  see the **first 10 rows** on a `/preview` screen before analysis, then click **Analyze this file**.

4.   **Best-effort normalization**
     ✅  to the canonical schema:  
     ✅  `Company | Year | LineItem | Value` (trims headers, coerces numbers, drops blank rows; falls back to raw columns if mapping fails).
 
5.   **AI summary (optional)**
     ✅  generates a concise, 2–3 sentence analysis via **Gemini** when `GEMINI_*` env vars are set.
     ✅  If not set, the app still works and shows a friendly “AI disabled” note, and renders summary, chart, and PDF.

6.   **Chart + report**
     ✅  renders a bar chart (static PNG for now) and lets you **download a nicely formatted PDF**.

7.   **Sensible rate limiting**
     ✅  `GET /` (home) is **not** rate-limited;  
     ✅  `POST /preview` and `POST /upload` are limited to **10 requests per minute** (demo safety).
          (For production, configure a shared store (Redis, Memcached) per Flask-Limiter docs.  

8.   **Clear error messages** for unreadable or invalid files.


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

2.   Create & activate the environment

     conda create -n ml python=3.11
     conda activate ml
     # or use virtualenv

3.   Install all the dependencies

     pip install -r requirements.txt

4.   Configure the environment

     Copy .env.example → .env

     Set your FLASK_SECRET_KEY, GEMINI_API_KEY, and GEMINI_URL.

     Environment variables:
     
     GEMINI_API_KEY	(Optional)  --- >  Google Gemini API key
     GEMINI_MODEL	(Optional)  --- >  Model name; e.g., gemini-1.5-flash
     (If these are unset, analysis runs without AI text).

6.   Then Run locally

     flask run

     Visit http://127.0.0.1:5000


Using the app
=============

1.   First, navigate to the home page and upload a file (.csv, .xlsx, or .zip).

2.   Then, click Preview to see the first 10 rows in your browser.

3.   Click Analyze this file to:
     ✅   show a normalized summary table,
     ✅   generate AI analysis/text (if env is set),
     ✅   render a bar chart,

4.   You are able to click Download as PDF to save the report.


Testing & linting
=================
✅   Run the linters and tests from the project root:

      ```bash
      black . && flake8 . && pytest -q

✅   If you don’t have the tools yet:
      
      pip install black flake8 pytest

(Works the same in macOS/Linux terminals and Windows PowerShell.)
::contentReference[oaicite:0]{index=0}


Deployment
==========
Render (✅ Done)

1.   Add runtime.txt with:

     python-3.11.11

2.   Ensure requirements.txt includes:

     gunicorn==20.1.0

3.   Connect your GitHub repo & deploy.


Project Structure
=================

<img width="756" height="267" alt="image" src="https://github.com/user-attachments/assets/935e3aa1-a01d-4b04-9490-645c9f6ad5ec" />


License

MIT © 2025

