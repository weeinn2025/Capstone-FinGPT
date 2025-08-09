# ---- 0) Imports ---------------------------------------------------
# (1) Load → (2) read env vars → (3) use them.
import os
import json
import base64
from pathlib import Path
from io import BytesIO

# Ensure matplotlib uses a non-GUI backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    flash,
    send_file,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import requests
from weasyprint import HTML


# ---- 1) Load environment variables ----------------------------------
# Make sure this happens *before* you read from os.environ
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)  # must run before os.environ[...] is used


# ---- 2) Grab secrets / config ---------------------------------------
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = os.environ.get("GEMINI_URL")
GEMINI_AVAILABLE = bool(GEMINI_API_KEY and GEMINI_URL)  # <-- instead of raising

# if not GEMINI_API_KEY or not GEMINI_URL:
#    raise RuntimeError("Missing GEMINI_API_KEY or GEMINI_URL in .env")


# ---- 3) Flask app setup ---------------------------------------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ---- limit uploads to 5 MB ------------------------------------------
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ensure uploads folder exists
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ---- 4) Rate limiter ------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[])
limiter.init_app(app)


# ---- 4a) Health check route --------------------------------------------
@app.get("/health")
@limiter.exempt
def health():
    return {"status": "ok"}, 200


# --- 5) Jinja filter for currency formatting ----------------------------
@app.template_filter("currency")
def currency_filter(val):
    try:
        return "${:,.2f}".format(float(val))
    except (TypeError, ValueError):
        return val


# --- 6) Gemini caller ---------------------------------------------------
def call_gemini(prompt: str) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
    }
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}

    resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # v1 response shape
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        # fallback for older bison payloads
        return data.get("candidates", [{}])[0].get("output", "").strip()


# ---- 7) Upload & AI Analysis Route -----------------------------------------------
@app.route("/", methods=["GET", "POST"])
@limiter.limit("10 per minute")  # limit to 10 uploads per minute
def upload_file():
    if request.method == "POST":
        # --- Validation & CSV loading ---
        if "file" not in request.files:
            flash("No file part in request.")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        # Basic CSV check
        if not file.filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.")
            return redirect(request.url)

        # Save and read
        filepath = UPLOAD_FOLDER / file.filename
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Error reading CSV: {e}")
            return redirect(request.url)

        df = df.head(10)  # limit for demo, show only first 10 rows
        # Convert DataFrame to list of dicts
        summary = df.to_dict(orient="records")

        # --- Build prompt for AI --------
        prompt_lines = [
            f"{r['Company']} {r['Year']} {r['LineItem']}: {r['Value']:,}"
            for r in summary
        ]
        prompt = (
            "Here is a snippet of financial data:\n"
            + "\n".join(prompt_lines)
            + "\n\nPlease provide a concise, 2-3 sentence analysis of these figures."
        )

        # --- Call AI ---------------------
        if GEMINI_AVAILABLE:
            try:
                ai_text = call_gemini(prompt)
            except Exception as e:
                # if it's a requests.HTTPError you can do:
                if hasattr(e, "response"):
                    print("FULL URL:", e.response.request.url)
                    print("STATUS CODE:", e.response.status_code)
                    print("BODY:", e.response.text)
                flash(f"AI call failed: {e}")
                ai_text = None
        else:
            ai_text = "(AI disabled: missing GEMINI_* env vars)"

        # ——— Build bar chart of Revenue vs Net Income ———
        # Expecting summary like:
        # {'Company':'Apple Inc.', 'Year':2021, 'LineItem':'Total Revenue', 'Value':81434000000} # noqa: E501
        # {'Company':'Apple Inc.', 'Year':2021, 'LineItem':'Net Income', 'Value':21744000000} # noqa: E501

        labels = [row["LineItem"] for row in summary]
        values = [row["Value"] for row in summary]

        fig, ax = plt.subplots()
        ax.bar(labels, [v / 1e9 for v in values])  # show in billions
        ax.set_ylabel("Value (USD Billion)")
        ax.set_title(f"{summary[0]['Company']} {summary[0]['Year']}")

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        chart_data = base64.b64encode(buf.read()).decode("ascii")

        # — pass chart_data into template context —
        return render_template(
            "result.html", summary=summary, ai_text=ai_text, chart_data=chart_data
        )

    # GET request
    return render_template("index.html")


# ---- 8) PDF Download Route ------------------------------------------
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    # 1) recover the posted JSON‑encoded context
    summary = json.loads(request.form["summary"])
    ai_text = request.form["ai_text"]
    chart_data = request.form.get("chart_data")  # ← grab our new hidden field

    # 2) render out the PDF template
    html_out = render_template(
        "pdf.html",
        summary=summary,
        ai_text=ai_text,
        chart_data=chart_data,  # ← send it into the PDF template
    )

    # 3) ask WeasyPrint to turn that into a PDF
    # generate a PDF bytes -- > base_url ensures CSS/static links resolve correctly
    # disable outline/bookmarks to avoid the TypeError in WeasyPrint
    pdf_bytes = HTML(
        string=html_out,
        base_url=request.url_root,
    ).write_pdf(outline=False)

    # 4) return the bytes as a downloadable file
    return send_file(
        BytesIO(pdf_bytes),
        as_attachment=True,
        download_name="ai_analysis.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    # enable full tracebacks in the browser - show the full Python error in your browser
    app.run(debug=True)
