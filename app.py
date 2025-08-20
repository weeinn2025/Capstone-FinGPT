# ---- 0) Imports ---------------------------------------------------
# (1) Load → (2) read env vars → (3) use them.
import os
import json
import base64
from pathlib import Path
from io import BytesIO
import numpy as np

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

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
    url_for,
    session,
    flash,
    jsonify,
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


# ---- 4b) Upload helpers: allowed types, readers, normalizer ---------
import zipfile

ALLOWED_EXTENSIONS = {"csv", "xlsx", "zip"}


def is_allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS


def read_zip_first_valid(zip_path: Path) -> pd.DataFrame:
    """Open the first .csv or .xlsx inside a ZIP and return a DataFrame."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(
            [n for n in zf.namelist() if n.lower().endswith((".csv", ".xlsx"))]
        )
        if not members:
            raise ValueError("ZIP has no CSV/XLSX files.")
        name = members[0]
        with zf.open(name) as fh:
            if name.lower().endswith(".csv"):
                return pd.read_csv(fh)
            # xlsx
            return pd.read_excel(fh, engine="openpyxl")


def read_anytabular(path: Path) -> pd.DataFrame:
    """Read CSV/XLSX/ZIP into a DataFrame."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".zip":
        return read_zip_first_valid(path)
    raise ValueError(f"Unsupported file type: {ext}")


_CANON = {
    "company": ["company", "firm", "name"],
    "year": ["year", "fiscal_year", "fy"],
    "lineitem": ["lineitem", "line item", "account", "metric"],
    "value": ["value", "amount", "val", "amount_usd", "usd"],
}


def _find_alias(cols_lower: dict[str, str], aliases: list[str]) -> str | None:
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None


def normalize_financial_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Try to normalize to Company/Year/LineItem/Value (case/space tolerant).
    Returns (normalized_df, warning_message_or_none).
    """
    cols_lower = {c.strip().lower(): c for c in df.columns}

    # first pass: look for friendly aliases
    mapping_src = {
        "Company": _find_alias(cols_lower, _CANON["company"]),
        "Year": _find_alias(cols_lower, _CANON["year"]),
        "LineItem": _find_alias(cols_lower, _CANON["lineitem"]),
        "Value": _find_alias(cols_lower, _CANON["value"]),
    }
    # fallback: exact canonical (case-insensitive) if present
    for k in list(mapping_src.keys()):
        if mapping_src[k] is None and k.lower() in cols_lower:
            mapping_src[k] = cols_lower[k.lower()]

    missing = [k for k, v in mapping_src.items() if v is None]
    warn = None
    if missing:
        warn = f"Could not auto-map columns: {', '.join(missing)}"

    renamed = df.rename(columns={v: k for k, v in mapping_src.items() if v})

    needed = {"Company", "Year", "LineItem", "Value"}
    if not needed.issubset(set(renamed.columns)):
        # return original df with a warning; caller can decide how to proceed
        return df, warn

    out = renamed[["Company", "Year", "LineItem", "Value"]].copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Year", "Value"])
    return out, warn


def build_plotly_chart(clean_df, latest_year=None):
    """
    Returns a Plotly figure: grouped bars of Revenue vs Net income
    for each Company in the selected/latest Year.
    Expects canonical columns: Company | Year | LineItem | Value
    """
    df = clean_df.copy()

    # Use latest year if present
    if "Year" in df.columns and df["Year"].notna().any():
        if latest_year is None:
            latest_year = int(df["Year"].dropna().max())
        df = df[df["Year"] == latest_year]
    else:
        latest_year = None

    # ------ NEW: canonicalize LineItem values ------------------------
    li_norm = (
        df["LineItem"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z]+", " ", regex=True)
        .str.strip()
    )

    df["LI_CANON"] = np.select(
        [
            li_norm.str.contains(r"\b(net income|net profit|profit)\b"),
            li_norm.str.contains(r"\b(revenue|total revenue|sales|total sales)\b"),
        ],
        ["Net income", "Revenue"],
        default=df["LineItem"].astype(str),
    )

    # ---- Build pivot on the canonical names build traces ---------------------------------------
    print("Line items seen:", sorted(df["LI_CANON"].unique()))

    need = df["LI_CANON"].isin(["Revenue", "Net income"])
    pivot = (
        df[need]
        .pivot_table(index="Company", columns="LI_CANON", values="Value", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    fig = go.Figure()
    # x_vals = pivot.index.astype(str).tolist()

    if "Revenue" in pivot.columns:
        fig.add_bar(name="Revenue", x=pivot.index.tolist(), y=pivot["Revenue"].tolist())

    if "Net income" in pivot.columns:
        fig.add_bar(
            name="Net income", x=pivot.index.tolist(), y=pivot["Net income"].tolist()
        )

    title_year = f" — {latest_year}" if latest_year is not None else ""
    fig.update_layout(
        barmode="group",
        title=f"Revenue vs Net income{title_year}",
        xaxis_title="Company",
        yaxis_title="Value",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --- Matplotlib fallback snapshot (multi-company grouped bars) -------
def build_matplotlib_grouped(clean_df, latest_year=None):
    import numpy as np

    df = clean_df.copy()

    # Use latest year if present
    if "Year" in df.columns and df["Year"].notna().any():
        if latest_year is None:
            latest_year = int(df["Year"].dropna().max())
        df = df[df["Year"] == latest_year]
    else:
        latest_year = None

    # Canonicalize line items like the Plotly path
    li_norm = (
        df["LineItem"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z]+", " ", regex=True)
        .str.strip()
    )
    df["LI_CANON"] = np.select(
        [
            li_norm.str.contains(r"\b(net income|net profit|profit)\b"),
            li_norm.str.contains(r"\b(revenue|total revenue|sales|total sales)\b"),
        ],
        ["Net income", "Revenue"],
        default=df["LineItem"].astype(str),
    )

    pivot = (
        df[df["LI_CANON"].isin(["Revenue", "Net income"])]
        .pivot_table(index="Company", columns="LI_CANON", values="Value", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    companies = pivot.index.tolist()
    rev = (
        pivot["Revenue"].tolist()
        if "Revenue" in pivot.columns
        else [0.0] * len(companies)
    )
    ni = (
        pivot["Net income"].tolist()
        if "Net income" in pivot.columns
        else [0.0] * len(companies)
    )

    x = np.arange(len(companies))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, rev, width, label="Revenue")
    ax.bar(x + width / 2, ni, width, label="Net income")
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=25, ha="right")
    title_year = f" — {latest_year}" if latest_year is not None else ""
    ax.set_title(f"Revenue vs Net income{title_year}")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


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
@app.get("/")
@limiter.exempt  # GET homepage is never rate-limited
def index():
    return render_template("index.html")


@app.post("/upload")
@limiter.limit("10 per minute")
def upload_file():
    saved_name = request.form.get("saved_filename")

    if saved_name:
        # Came from preview screen
        filepath = UPLOAD_FOLDER / saved_name
        if not filepath.exists():
            flash("Saved file not found. Please upload again.")
            return redirect(request.url_root)
    else:
        # Fresh upload path (backwards compatible)
        if "file" not in request.files:
            flash("No file part in request.")
            return redirect(request.url_root)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url_root)

        if not is_allowed_file(file.filename):
            flash("Allowed types: .csv, .xlsx, .zip")
            return redirect(request.url_root)

        filepath = UPLOAD_FOLDER / file.filename
        file.save(filepath)

    # Read + normalize (best effort)
    try:
        raw_df = read_anytabular(filepath)
    except Exception as e:
        flash(f"Error reading file: {e}")
        return redirect(request.url_root)

    df_norm, warn = normalize_financial_df(raw_df)
    if warn:
        flash(warn)

    # Use normalized df if it contains the canonical columns; else fallback
    needed = {"Company", "Year", "LineItem", "Value"}
    use_df = df_norm if needed.issubset(set(df_norm.columns)) else raw_df

    df = use_df.head(10)
    summary = df.to_dict(orient="records")

    # --- Build prompt for AI --------
    prompt_lines = [
        f"{r['Company']} {r['Year']} {r['LineItem']}: {r['Value']:,}"
        for r in summary
        if all(k in r for k in ("Company", "Year", "LineItem", "Value"))
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

    # ——— Build interactive Plotly chart + PNG snapshot for PDF (fallback to Matplotlib)———
    # fig_json must always be defined (None means "no interactive chart")
    fig_json = None  # <-- KEY: ensure defined for all paths

    # Use the normalized dataframe (clean_df) for the chart:
    has_canonical = {"Company", "Year", "LineItem", "Value"}.issubset(use_df.columns)

    if has_canonical:
        # 1) Interactive Plotly figure for the page
        fig = build_plotly_chart(use_df)
        fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)  # pass to template

        # 2) Try to produce a PNG for the PDF using Kaleido
        try:
            png_bytes = fig.to_image(format="png", scale=2)  # needs 'kaleido'
            chart_data = base64.b64encode(png_bytes).decode("ascii")
        except Exception as e:
            app.logger.warning(
                f"Plotly->PNG failed (kaleido). Falling back to Matplotlib grouped. Error: {e}"
            )
            chart_data = build_matplotlib_grouped(use_df)

    else:
        # No canonical columns -> still try a grouped fallback using whatever we can
        chart_data = build_matplotlib_grouped(use_df)

    # 3) Render results page (note: this line must be de-dented to function level)
    # Keep `chart_data` (base64 PNG) for PDF and add `fig_json` for the
    # interactive Plotly chart in the template.
    return render_template(
        "result.html",
        summary=summary,
        ai_text=ai_text,
        chart_data=chart_data,  # base64 PNG for PDF
        fig_json=fig_json,  # None => template falls back to PNG <-- NEW
    )


# ---- 7a) Preview route (save file + show first rows) -----------------
@app.post("/preview")
@limiter.limit("10 per minute")
def preview():
    if "file" not in request.files:
        flash("No file part in request.")
        return redirect(request.url_root)

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(request.url_root)

    if not is_allowed_file(file.filename):
        flash("Allowed types: .csv, .xlsx, .zip")
        return redirect(request.url_root)

    # Save to uploads/
    saved_path = UPLOAD_FOLDER / file.filename
    file.save(saved_path)

    try:
        df = read_anytabular(saved_path)
        # show only first 10 rows, raw (pre-normalization)
        preview_rows = df.head(10).to_dict(orient="records")
    except Exception as e:
        flash(f"Could not read file: {e}")
        return redirect(request.url_root)

    return render_template(
        "preview.html",
        saved_filename=saved_path.name,
        preview_rows=preview_rows,
    )


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
