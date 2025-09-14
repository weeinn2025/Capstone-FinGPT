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
    flash,
    send_file,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import requests
from weasyprint import HTML
import zipfile

# ensures engine is available for Excel export
from xlsxwriter.utility import xl_col_to_name


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
storage_uri = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
limiter = Limiter(
    key_func=get_remote_address, default_limits=[], storage_uri=storage_uri
)
limiter.init_app(app)


# ---- 4a) Health check route --------------------------------------------
@app.get("/health")
@limiter.exempt
def health():
    return {"status": "ok"}, 200


# ---- 4b) Upload helpers: allowed types, readers, normalizer ---------
ALLOWED_EXTENSIONS = {"csv", "xlsx", "zip"}


def is_allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS


def read_zip_concat(zip_path: Path) -> pd.DataFrame:
    """Concatenate ALL CSV/XLSX files in a ZIP into one DataFrame."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".xlsx"))]
        if not members:
            raise ValueError("ZIP has no CSV/XLSX files.")
        frames = []
        for name in sorted(members):
            with zf.open(name) as fh:
                if name.lower().endswith(".csv"):
                    frames.append(pd.read_csv(fh))
                else:
                    frames.append(pd.read_excel(fh, engine="openpyxl"))
        return pd.concat(frames, ignore_index=True, sort=False)


def read_anytabular(path: Path) -> pd.DataFrame:
    """Read CSV/XLSX/ZIP into a DataFrame."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".zip":
        return read_zip_concat(path)  # <- was read_zip_first_valid(path)
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


# ---- Charts ----------------------------------------------------------
def build_plotly_multi_year(clean_df):
    """Line chart: Revenue & Net income or profit across all years, per company."""
    df = clean_df.copy()

    li_norm = (
        df["LineItem"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z]+", " ", regex=True)
        .str.strip()
    )
    df["LI_CANON"] = np.select(
        [
            li_norm.str.contains(
                r"\b(?:net income|net profit|profit)\b", regex=True, na=False
            ),
            li_norm.str.contains(
                r"\b(?:revenue|total revenue|sales|total sales)\b",
                regex=True,
                na=False,
            ),
        ],
        ["Net income", "Revenue"],
        default=df["LineItem"].astype(str),
    )

    keep = df["LI_CANON"].isin(["Revenue", "Net income"])
    g = (
        df[keep]
        .groupby(["Year", "Company", "LI_CANON"])["Value"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Company", "LI_CANON"])
    )

    fig = go.Figure()
    for comp in sorted(pd.unique(g["Company"])):
        sub_rev = g[(g["Company"] == comp) & (g["LI_CANON"] == "Revenue")]
        sub_ni = g[(g["Company"] == comp) & (g["LI_CANON"] == "Net income")]

        fig.add_scatter(
            name=f"{comp} — Revenue",
            x=sub_rev["Year"],
            y=sub_rev["Value"],
            mode="lines+markers",
        )
        fig.add_scatter(
            name=f"{comp} — Net income",
            x=sub_ni["Year"],
            y=sub_ni["Value"],
            mode="lines+markers",
            line=dict(dash="dot"),
        )
    fig.update_layout(
        title=dict(
            text="Revenue & Net Income or Profit — All Years", x=0.5, xanchor="center"
        ),
        xaxis_title="Year",
        yaxis_title="Value",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=70, b=120),  # more bottom space for legend
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,  # place legend below chart
            xanchor="center",
            x=0.5,
            traceorder="grouped",
            font=dict(size=11),
            itemwidth=50,
        ),
    )
    fig.update_yaxes(tickformat="~s")
    # default the selector to "All Years"; otherwise figAll would be null
    return fig


# --- Interactive Plotly figure for the page (latest year) ---
def build_plotly_chart(clean_df, latest_year=None):
    """Grouped bars of Revenue vs Net income (or profit) for the selected/latest Year.
    Expects canonical columns: Company | Year | LineItem | Value.
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
            li_norm.str.contains(
                r"\b(?:net income|net profit|profit)\b", regex=True, na=False
            ),
            li_norm.str.contains(
                r"\b(?:revenue|total revenue|sales|total sales)\b", regex=True, na=False
            ),
        ],
        ["Net income", "Revenue"],
        default=df["LineItem"].astype(str),
    )

    # --- Build pivot for grouped bars, on the canonical names build traces -----------
    # print("Line items seen:", sorted(df["LI_CANON"].unique()))

    pivot = (
        df[df["LI_CANON"].isin(["Revenue", "Net income"])]
        .pivot_table(index="Company", columns="LI_CANON", values="Value", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    fig = go.Figure()

    if "Revenue" in pivot.columns:
        fig.add_bar(
            name="Revenue",
            x=pivot.index.tolist(),
            y=pivot["Revenue"].tolist(),
        )
    if "Net income" in pivot.columns:
        fig.add_bar(
            name="Net income",
            x=pivot.index.tolist(),
            y=pivot["Net income"].tolist(),
        )

    title_year = f" — {latest_year}" if latest_year is not None else ""
    fig.update_layout(
        barmode="group",
        title=f"Revenue vs Net income or Profit{title_year}",
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

    ax.set_title(f"Revenue vs Net income or Profit{title_year}")  # ← change this
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# --- 5) Jinja filter for currency formatting ----------------------------------
@app.template_filter("currency")
def currency_filter(val):
    """$ with 2 decimals — used by existing templates (e.g., result.html summary)."""
    try:
        return "${:,.2f}".format(float(val))  # no decimals
    except Exception:
        # if it's NaN/None/str that can't be parsed, return as-is
        return val


@app.template_filter("currency0")
def currency0_filter(val):
    """$ with no decimals — keeps wide PDF table narrow."""
    try:
        return "${:,.0f}".format(float(val))
    except Exception:
        return val


# [ADDED] percentage formatter that prints NaN for missing
@app.template_filter("pct")
def pct_filter(val):
    """percentage with 2 decimals; safe on NaN."""
    try:
        return "{:.2f}%".format(float(val) * 100.0)
    except Exception:
        return "NaN"


# --- 6) Gemini caller -------------------------------------------------------------
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


# [ADDED] ---- Metrics & Alerts ------------------------------------------
# [FIX] Robust metrics that tolerate missing columns and text cells
def compute_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Build Tier-1 metrics and alert flags from a long-form DataFrame with
    columns: Company, Year, LineItem, Value.
    - Normalizes line-item names via an alias map
    - Pivots to wide form per (Company, Year)
    - Computes ratios (net_margin, debt_to_equity, debt_to_assets)
    - Computes YoY growth using groupby-transform (index-safe)
    - Emits alert classes: 'green' | 'red' | None
    Input: df_long with columns Company, Year, LineItem, Value
    Output: wide df with ratios + alert classes (green/red/None)
    """
    d = df_long.copy()

    # Normalize names or keys
    d["Company"] = d["Company"].astype(str).str.strip()
    d["LineItem"] = d["LineItem"].astype(str).str.strip().str.lower()

    # Expand aliases so more files map correctly
    # Canonical alias map (extend as needed)
    aliases = {
        # revenue
        "revenue": "revenue",
        "total revenue": "revenue",
        "sales": "revenue",
        "net sales": "revenue",
        "total sales": "revenue",
        # net income
        "net income": "net_income",
        "net profit": "net_income",
        "profit": "net_income",
        "profit for the year": "net_income",
        "income attributable to shareholders": "net_income",
        "retained profit": "net_income",  # <-- NEW
        # liabilities
        "liabilities": "liabilities",
        "total liabilities": "liabilities",
        # debt (NEW)
        "debt": "debt",
        "total debt": "debt",
        "interest-bearing debt": "debt",
        "total interest bearing debt": "debt",
        "borrowings": "debt",
        "short-term debt": "debt",
        "short term debt": "debt",
        "long-term debt": "debt",
        "long term debt": "debt",
        # assets
        "assets": "assets",
        "total assets": "assets",
        # equity
        "equity": "equity",
        "shareholders' equity": "equity",
        "stockholders' equity": "equity",
        "total equity": "equity",
        "total shareholders' equity": "equity",
        "total stockholders' equity": "equity",
        "total shareholder funds": "equity",  # <-- NEW
    }
    d["key"] = d["LineItem"].map(aliases).fillna(d["LineItem"])

    # Pivot to wide -------------------------------------------------
    wide = (
        d.pivot_table(
            index=["Company", "Year"], columns="key", values="Value", aggfunc="sum"
        )
        .reset_index()
        .sort_values(["Company", "Year"])
    )

    # Helpers (safe numeric ops) -------------------------------------
    def safe_col(frame: pd.DataFrame, name: str) -> pd.Series:
        """Always return a numeric Series (NaN if missing)."""

        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        """Divide with NaN if b is 0/NaN; coerce inputs to numeric."""
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce").replace({0: np.nan})
        return a.divide(b)

    # Pull numeric columns or inputs safely
    rev = safe_col(wide, "revenue")
    ni = safe_col(wide, "net_income")
    liab = safe_col(wide, "liabilities")
    debt = safe_col(wide, "debt")  # NEW - if present, use it
    assets = safe_col(wide, "assets")
    equity = safe_col(wide, "equity")

    # Derive equity if entirely missing but assets & liabilities exist
    if equity.isna().all() and (assets.notna().any() or liab.notna().any()):
        equity = assets - liab
        wide["equity"] = equity  # keep it for Excel or debugging

    # Prefer true debt; otherwise use liabilities as a proxy
    debt_like = debt.combine_first(liab)  # NEW

    # Ratios -----------------------------------------------------------
    wide["revenue"] = rev
    wide["net_income"] = ni
    wide["net_margin"] = safe_div(ni, rev)
    wide["debt_to_equity"] = safe_div(debt_like, equity)  # CHANGED
    wide["debt_to_assets"] = safe_div(debt_like, assets)  # CHANGED

    # ---- Growth (index-safe via transform) ---------------------------
    wide["rev_yoy"] = wide.groupby("Company", sort=False)["revenue"].transform(
        lambda s: s.pct_change()
    )
    wide["ni_yoy"] = wide.groupby("Company", sort=False)["net_income"].transform(
        lambda s: s.pct_change()
    )

    # Alerts -----------------------------------------------------------
    def flag_class(val, threshold_low=None, threshold_high=None, inverse=False):
        if pd.isna(val):
            return None
        if inverse:
            return (
                "red"
                if (threshold_high is not None and val > threshold_high)
                else "green"
            )
        return "red" if (threshold_low is not None and val < threshold_low) else "green"

    wide["alert_liquidity"] = wide["net_margin"].apply(
        lambda x: flag_class(x, threshold_low=0.05)
    )
    wide["alert_leverage"] = wide["debt_to_equity"].apply(
        lambda x: flag_class(x, threshold_high=2, inverse=True)
    )
    wide["alert_revtrend"] = wide["rev_yoy"].apply(
        lambda x: flag_class(x, threshold_low=0)
    )

    return wide


# ---- 7) Upload & AI Analysis Route -----------------------------------------------
@app.get("/")
@limiter.exempt  # GET homepage is never rate-limited
def index():
    return render_template("index.html")


@app.post("/upload")
@limiter.limit("10 per minute")
def upload_file():
    saved_name = request.form.get("saved_filename")

    # ---- locate file (preview or fresh upload) ----
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

    # Debug: see which years your data has
    if "Year" in use_df.columns:
        years_debug = sorted(int(v) for v in pd.unique(use_df["Year"].dropna()))
        app.logger.info("Years found: %s", years_debug)
    else:
        app.logger.info("Years found: <none> (no 'Year' column)")

    # Show ALL rows, Neatly ordered summary for UI /top table
    summary_df = use_df.copy().sort_values(
        ["Company", "Year", "LineItem"], na_position="last"
    )
    summary = summary_df.to_dict(orient="records")

    # Build compact multi-year lines for AI (only Revenue & Net income)
    li_norm_ai = (
        use_df["LineItem"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z]+", " ", regex=True)
        .str.strip()
    )

    ai_df = use_df.assign(
        LI_CANON=np.select(
            [
                li_norm_ai.str.contains(
                    r"\b(?:net income|net profit|profit)\b", regex=True, na=False
                ),
                li_norm_ai.str.contains(
                    r"\b(?:revenue|total revenue|sales|total sales)\b",
                    regex=True,
                    na=False,
                ),
            ],
            ["Net income", "Revenue"],
            default=use_df["LineItem"].astype(str),
        )
    )

    ai_df = (
        ai_df[ai_df["LI_CANON"].isin(["Revenue", "Net income"])]
        .groupby(["Company", "Year", "LI_CANON"], dropna=True)["Value"]
        .sum()
        .reset_index()
        .sort_values(["Company", "Year", "LI_CANON"])
    )

    lines = []
    for (comp, yr), grp in ai_df.groupby(["Company", "Year"]):
        rev = grp.loc[grp["LI_CANON"] == "Revenue", "Value"].sum()
        ni = grp.loc[grp["LI_CANON"] == "Net income", "Value"].sum()
        lines.append(f"{comp} {int(yr)} | Revenue: {rev:,.0f} | Net income: {ni:,.0f}")

    prompt = (
        "Summarize multi-year performance in 3–5 sentences. "
        "Focus on growth/decline and rough margins across years; do not invent data.\n"
        + "\n".join(lines)
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

    # Build interactive Plotly chart + PNG snapshot for PDF (Matplotlib fallback)
    # fig_json must always be defined (None means "no interactive chart")
    # --- Charts (always set fig_json & chart_data) -----------------------
    fig_json = None  # interactive Plotly (latest year) for page
    chart_data = None  # base64 PNG for PDF (latest-year bars)
    years = []  # year dropdown
    figs_by_year_json = "{}"  # per-year figures for client switching
    fig_all_json = "null"  # all-years line chart (Plotly JSON)
    chart_data_all = None  # all-years PNG for PDF (optional)

    needed = {"Company", "Year", "LineItem", "Value"}
    has_canonical = needed.issubset(use_df.columns)

    # All-years line chart if Year exists
    if "Year" in use_df.columns and use_df["Year"].notna().any():
        fig_all = build_plotly_multi_year(use_df)
        fig_all_json = json.dumps(fig_all, cls=PlotlyJSONEncoder)
        try:
            chart_data_all = base64.b64encode(
                fig_all.to_image(format="png", scale=2)  # needs kaleido
            ).decode("ascii")
        except Exception as e:
            app.logger.warning("Plotly->PNG failed (all years): %s", e)
            chart_data_all = None  # fine; PDF will just omit the all-years image

    if has_canonical:
        # Interactive latest-year grouped bars for the page
        fig = build_plotly_chart(use_df)
        fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        # Per-year figures for the year selector
        years = sorted(int(v) for v in pd.unique(use_df["Year"].dropna()))
        if years:
            figs_by_year = {}
            for y in years:
                fy = build_plotly_chart(use_df, latest_year=int(y))
                figs_by_year[str(y)] = json.loads(json.dumps(fy, cls=PlotlyJSONEncoder))
            figs_by_year_json = json.dumps(figs_by_year)

        # PNG for the PDF (fallback to Matplotlib if Kaleido not working)
        try:
            chart_data = base64.b64encode(
                fig.to_image(format="png", scale=2)  # needs kaleido
            ).decode("ascii")
        except Exception as e:
            app.logger.warning(
                "Plotly->PNG failed (latest-year). Using Matplotlib fallback. Error: %s",
                e,
            )
            chart_data = build_matplotlib_grouped(use_df)
    else:
        # Not canonical → still produce a PNG so pdf.html always has an image
        chart_data = build_matplotlib_grouped(use_df)

    # [ADDED] ---- Compute Tier-1 metrics + alerts -----------------------
    metrics = compute_metrics(use_df) if has_canonical else pd.DataFrame()

    # Render results page (this line must be de-dented to function level)
    # Keep `chart_data` (base64 PNG) for PDF and add `fig_json` for the
    # interactive Plotly chart in the template.
    return render_template(
        "result.html",
        summary=summary,
        ai_text=ai_text,
        chart_data=chart_data,  # base64 PNG for PDF (latest-year bars)
        fig_json=fig_json,  # None => template falls back to PNG <-- NEW
        years=years,  # NEW
        figs_by_year_json=figs_by_year_json,  # NEW
        fig_all_json=fig_all_json,  # <-- NEW
        chart_data_all=chart_data_all,  # <-- NEW
        metrics=metrics,  # [ADDED]
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

    # NEW: ← grab both chart images from the form (latest-year bars)
    chart_data = request.form.get("chart_data")
    chart_data_all = request.form.get("chart_data_all")

    # Rebuild df from summary and recompute metrics here
    import pandas as pd

    summary_df = pd.DataFrame(summary)  # has Company, Year, LineItem, Value
    # Ensure types are right (in case)
    summary_df["Year"] = pd.to_numeric(summary_df["Year"], errors="coerce").astype(
        "Int64"
    )
    summary_df["Value"] = pd.to_numeric(summary_df["Value"], errors="coerce")

    # compute_metrics is already defined in app.py
    metrics = compute_metrics(summary_df)

    # 2) render out the PDF template with BOTH images
    html_out = render_template(
        "pdf.html",
        summary=summary,
        ai_text=ai_text,
        chart_data=chart_data,  # ← send it into the PDF template
        chart_data_all=chart_data_all,  # <-- NEW
        metrics=metrics,  # [ADDED]
    )

    # 3) ask WeasyPrint to turn that into a PDF
    # generate a PDF bytes - base_url ensures CSS/static links resolve correctly
    # disable outline/bookmarks to avoid the TypeError in WeasyPrint
    pdf_bytes = HTML(
        string=html_out,
        base_url=request.url_root,
    ).write_pdf()

    # 4) return the bytes as a downloadable file
    return send_file(
        BytesIO(pdf_bytes),
        as_attachment=True,
        download_name="ai_analysis.pdf",
        mimetype="application/pdf",
    )


# ---- 9) Excel Export Route (Tier-1 with native colors) ---------------------------
@app.post("/export_excel")
def export_excel():
    """
    Creates an Excel file with conditional formatting (green/red) for Tier-1 alerts.
    Also formats money as whole numbers and percentages as 2-dp. ( Money=0dp, % = 2dp.)
    Metrics are computed from posted summary to avoid huge hidden fields.
    """
    # Recompute metrics in /download_pdf from summary posted by the page
    summary_json = request.form.get("summary")
    if not summary_json:
        flash("No summary provided for Excel export.")
        return redirect(request.url_root)

    try:
        summary = json.loads(summary_json)
    except Exception as e:
        flash(f"Invalid summary payload for Excel export: {e}")
        return redirect(request.url_root)

    df = pd.DataFrame(summary)
    # ensure types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # reuse your existing compute_metrics()/function that shapes Tier-1 rows:
    metrics = compute_metrics(df)

    # ------------- Build workbook in-memory -------------------------------------------
    output = BytesIO()
    # explicitly tell pandas to use xlsxwriter
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        metrics.to_excel(writer, index=False, sheet_name="Tier1_Ratios_Alerts")
        wb = writer.book
        ws = writer.sheets["Tier1_Ratios_Alerts"]

        # ---------- Formats (define BEFORE set_column/CF) ----------
        header_fmt = wb.add_format(
            {"bold": True, "bg_color": "#EFEFEF", "border": 1, "align": "center"}
        )
        # 0-decimal thousands with commas
        money_fmt = wb.add_format({"num_format": "#,##0"})
        pct_fmt = wb.add_format({"num_format": "0.00%"})

        # Alerts instead of text - fills used by conditional formatting + hides alert text
        hide_text = wb.add_format({"num_format": ";;;"})  # hides alert text
        green_fill = wb.add_format({"bg_color": "#C6EFCE"})
        red_fill = wb.add_format({"bg_color": "#FFC7CE"})
        gray_fill = wb.add_format({"bg_color": "#E5E7EB"})

        # ---------- Column indices (avoid NameError) -----------------------------------
        # (Guard against case differences by using the exact column names produced
        #  by compute_metrics)
        cols = {c: metrics.columns.get_loc(c) for c in metrics.columns}
        ix_company = cols["Company"]
        ix_year = cols["Year"]
        ix_revenue = cols["revenue"]
        ix_net_income = cols["net_income"]
        ix_net_margin = cols["net_margin"]
        ix_debt_equity = cols["debt_to_equity"]
        ix_debt_assets = cols["debt_to_assets"]
        ix_rev_yoy = cols["rev_yoy"]
        ix_ni_yoy = cols["ni_yoy"]
        ix_alert_liq = cols["alert_liquidity"]
        ix_alert_lev = cols["alert_leverage"]
        ix_alert_rev = cols["alert_revtrend"]

        # handy: convert a 0-based index to Excel column letter (A, B, …)
        col_letter = xl_col_to_name

        # total rows incl. header
        nrows = len(metrics) + 1  # +1 header

        # -------- Header row bold/grey ----------------------------------------------
        ws.set_row(0, None, header_fmt)

        # ---------- Column widths / number formats ------------------------------------
        ws.set_column(
            f"{col_letter(ix_company)}:{col_letter(ix_company)}", 18
        )  # Company
        ws.set_column(f"{col_letter(ix_year)}:{col_letter(ix_year)}", 8)  # Year
        ws.set_column(
            f"{col_letter(ix_revenue)}:{col_letter(ix_net_income)}", 16, money_fmt
        )  # Revenue + Net Income (0-dec)
        ws.set_column(
            f"{col_letter(ix_net_margin)}:{col_letter(ix_net_margin)}", 12, pct_fmt
        )  # Net margin
        ws.set_column(
            f"{col_letter(ix_debt_equity)}:{col_letter(ix_debt_assets)}", 14
        )  # Debt ratios
        ws.set_column(
            f"{col_letter(ix_rev_yoy)}:{col_letter(ix_ni_yoy)}", 12, pct_fmt
        )  # YoY %

        # Keep the alert text but hide visually (we will color cells with CF)
        ws.set_column(
            f"{col_letter(ix_alert_liq)}:{col_letter(ix_alert_rev)}", 4, hide_text
        )

        # --- Conditional formatting (CF) for alerts (applied after nrows known) -------
        # GREEN when text == "green"; RED when text == "red"; GRAY when blank
        def cf_eq(col_idx: int, text: str, fmt):
            ws.conditional_format(
                f"{col_letter(col_idx)}2:{col_letter(col_idx)}{nrows}",
                {"type": "cell", "criteria": "==", "value": f'"{text}"', "format": fmt},
            )

        def cf_blank(col_idx: int, fmt):
            ws.conditional_format(
                f"{col_letter(col_idx)}2:{col_letter(col_idx)}{nrows}",
                {"type": "blanks", "format": fmt},
            )

        # Liquidity (green/red/blank->gray)
        cf_eq(ix_alert_liq, "green", green_fill)
        cf_eq(ix_alert_liq, "red", red_fill)
        cf_blank(ix_alert_liq, gray_fill)

        # Leverage (green/red/blank->gray)
        cf_eq(ix_alert_lev, "green", green_fill)
        cf_eq(ix_alert_lev, "red", red_fill)
        cf_blank(ix_alert_lev, gray_fill)

        # Revenue trend (green/red/blank->gray)
        cf_eq(ix_alert_rev, "green", green_fill)
        cf_eq(ix_alert_rev, "red", red_fill)
        cf_blank(ix_alert_rev, gray_fill)

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="Financial_metrics_ratios_Tier1.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    # enable full tracebacks in browser - show full Python error in browser
    app.run(debug=True)
