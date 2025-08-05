Capstone‑FinGPT
===============

A lightweight Flask app that allows you to ingest CSV financial data, generate a concise AI‑powered analysis, and let you view, chart, and download the results as a PDF.

🚀 Live Demo

Render: https://capstone-fingpt.onrender.com

📘 Repository

https://github.com/weeinn2025/Capstone-FinGPT

Features
========

1.   Able to upload any financial statement CSV (limited to 5 MB).

2.   Able to generate AI‑driven summary & insights via Gemini API.

3.   Able to provide interactive bar chart of revenue vs. net income.

4.   Able to download a nicely formatted PDF report.

5.   Able to rate‑limited uploads for demo safety.

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

5.   Then Run locally

     flask run

     Visit http://127.0.0.1:5000

Usage
=====

1.   First, navigate to the home page and upload a CSV.

2.   Then, view the AI analysis & chart in your browser.

3.   You are able to click Download as PDF to save the report.

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

Capstone-FinGPT/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ .env (local only)
├─ templates/
│  ├ index.html
│  ├ result.html
│  └ pdf.html
├─ uploads/            # ephemeral file storage
└─ README.md           # this file

License

MIT © 2025

