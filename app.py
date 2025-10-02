# ---- 0) Imports ---------------------------------------------------
# (1) Load → (2) read env vars → (3) use them

import base64
import json
import os
import random
import time
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib

# Ensure matplotlib uses a non-GUI backend

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
from weasyprint import HTML
from xlsxwriter.utility import xl_col_to_name

# ---- 1) Load environment variables ----------------------------------

# Make sure this happens *before* you read from os.environ
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)  # must run before os.environ[...] is used

# ---- 2) Grab secrets / config ---------------------------------------

FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = os.environ.get("GEMINI_URL")

# --- Auto-derive Gemini URL if not supplied & normalize model --------------

GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1").strip()  # v1 or v1beta
GEMINI_MODEL = (os.getenv("GEMINI_MODEL", "") or "").strip()

# normalize short aliases people might set
_alias = GEMINI_MODEL.lower()
if _alias in {
    "",
    "flash",
    "gemini-flash-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
}:
    GEMINI_MODEL = "gemini-2.5-flash"
elif _alias in {
    "pro",
    "gemini-pro-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
}:
    GEMINI_MODEL = "gemini-2.5-pro"

if not GEMINI_URL:
    GEMINI_URL = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
# Consider Gemini available if we at least have a key (URL is auto-built above)
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

# ---- 3) Flask app setup ---------------------------------------------------

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ---- limit uploads to 5 MB ------------------------------------------------

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ensure uploads folder exists
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ---- 4) Rate limiter ------------------------------------------------------

storage_uri = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri=storage_uri,
)
limiter.init_app(app)

# ---- 4a) Health check route ---------------------------------------------------


@app.get("/health")
@limiter.exempt
def health():
    return {"status": "ok"}, 200


@app.get("/ai_status")
@limiter.exempt
def ai_status():
    """Lightweight runtime check (no secrets exposed)."""
    # Don't reveal secrets; only whether they exist
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    model = os.environ.get("GEMINI_MODEL") or "auto: gemini-2.5-flash"
    ver = os.environ.get("GEMINI_API_VERSION", "v1")
    return {"gemini_available": has_key, "model": model, "api_version": ver}, 200


# ---- 4b) Upload helpers: allowed types, readers, normalizer -------------------

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
        return read_zip_concat(path)
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


# ----------- Build Charts -------------------------------------------------------


def build_plotly_multi_year(clean_df: pd.DataFrame) -> go.Figure:
    """Line chart: Revenue & Net income or profit across all years, per company."""
    df = clean_df.copy()

    li_norm = df["LineItem"].astype(str).str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()
    df["LI_CANON"] = np.select(
        [
            li_norm.str.contains(
                r"\b(?:net income|net profit|profit)\b",
                regex=True,
                na=False,
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
            text="Revenue & Net Income or Profit — All Years",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Year",
        yaxis_title="Value",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=70, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            traceorder="grouped",
            font=dict(size=11),
            itemwidth=50,
        ),
    )
    fig.update_yaxes(tickformat="~s")
    return fig


def build_plotly_chart(
    clean_df: pd.DataFrame,
    latest_year: int | None = None,
) -> go.Figure:
    """
    Grouped bars of Revenue vs Net income (or profit) for the selected/latest Year.
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

    li_norm = df["LineItem"].astype(str).str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()
    df["LI_CANON"] = np.select(
        [
            li_norm.str.contains(
                r"\b(?:net income|net profit|profit)\b",
                regex=True,
                na=False,
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

    pivot = (
        df[df["LI_CANON"].isin(["Revenue", "Net income"])]
        .pivot_table(
            index="Company",
            columns="LI_CANON",
            values="Value",
            aggfunc="sum",
        )
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def build_matplotlib_grouped(
    clean_df: pd.DataFrame,
    latest_year: int | None = None,
) -> str:
    """Matplotlib fallback snapshot (multi-company grouped bars)."""
    df = clean_df.copy()

    # Use latest year if present
    if "Year" in df.columns and df["Year"].notna().any():
        if latest_year is None:
            latest_year = int(df["Year"].dropna().max())
        df = df[df["Year"] == latest_year]
    else:
        latest_year = None

    li_norm = df["LineItem"].astype(str).str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()
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
        .pivot_table(
            index="Company",
            columns="LI_CANON",
            values="Value",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )

    companies = pivot.index.tolist()
    rev = pivot["Revenue"].tolist() if "Revenue" in pivot.columns else [0.0] * len(companies)
    ni = pivot["Net income"].tolist() if "Net income" in pivot.columns else [0.0] * len(companies)

    x = np.arange(len(companies))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, rev, width, label="Revenue")
    ax.bar(x + width / 2, ni, width, label="Net income")
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=25, ha="right")
    title_year = f" — {latest_year}" if latest_year is not None else ""
    ax.set_title(f"Revenue vs Net income or Profit{title_year}")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_matplotlib_all_years_line(clean_df: pd.DataFrame) -> str:
    """
    Matplotlib fallback for the 'All years' line chart when Kaleido/Plotly PNG
    export isn't available. Returns a base64-encoded PNG string.
    Plots Revenue and Net income per company across all years.
    """
    df = clean_df.copy()

    li_norm = df["LineItem"].astype(str).str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()
    df["LI_CANON"] = np.select(
        [
            li_norm.str.contains(
                r"\b(?:net income|net profit|profit)\b",
                regex=True,
                na=False,
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
        .groupby(["Year", "Company", "LI_CANON"], dropna=True)["Value"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Company", "LI_CANON"])
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    companies = sorted(pd.unique(g["Company"]))
    for comp in companies:
        sub_rev = g[(g["Company"] == comp) & (g["LI_CANON"] == "Revenue")]
        sub_ni = g[(g["Company"] == comp) & (g["LI_CANON"] == "Net income")]
        ax.plot(
            sub_rev["Year"],
            sub_rev["Value"],
            marker="o",
            linestyle="-",
            label=f"{comp} — Revenue",
        )
        ax.plot(
            sub_ni["Year"],
            sub_ni["Value"],
            marker="o",
            linestyle="--",
            label=f"{comp} — Net income",
        )

    ax.set_title("Revenue & Net Income or Profit — All Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# --- 5) Jinja filters -----------------------------------------------------------


@app.template_filter("currency")
def currency_filter(val):
    """$ with 2 decimals — used by existing templates (e.g., result.html summary)."""
    try:
        return "${:,.2f}".format(float(val))
    except Exception:
        return val


@app.template_filter("currency0")
def currency0_filter(val):
    """$ with no decimals — keeps wide PDF table narrow."""
    try:
        return "${:,.0f}".format(float(val))
    except Exception:
        return val


@app.template_filter("pct")
def pct_filter(val):
    """percentage with 2 decimals; safe on NaN."""
    try:
        return "{:.2f}%".format(float(val) * 100.0)
    except Exception:
        return "NaN"


# --- 6) Call Gemini -------------------------------------------------------------

_GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
_GEMINI_API_VERSION = os.environ.get("GEMINI_API_VERSION", "v1").strip() or "v1"
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def _gemini_endpoint(model: str | None = None, api_version: str | None = None) -> str:
    """Build the REST URL for generateContent."""
    m = (model or _GEMINI_DEFAULT_MODEL).strip()
    ver = (api_version or _GEMINI_API_VERSION).strip()
    return f"https://generativelanguage.googleapis.com/{ver}/models/{m}:generateContent"


def _extract_text(data: dict) -> str:
    """
    Robustly extract text from Gemini responses.
    Supports v1/v1beta shape and very old bison 'output' fallback.
    """
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        txt = " ".join(t for t in texts if t).strip()
        if txt:
            return txt
    except Exception:
        pass

    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
        if txt:
            return txt.strip()
    except Exception:
        pass

    try:
        txt = data.get("candidates", [{}])[0].get("output", "")
        if txt:
            return txt.strip()
    except Exception:
        pass

    return ""


def call_gemini(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 256,
    model: str | None = None,
    api_version: str | None = None,
    timeout: int = 20,
    max_retries: int = 3,
) -> str:
    """
    Resilient Gemini call:
    - Builds URL from env (no GEMINI_URL needed)
    - Retries on 429/5xx with exponential backoff + jitter
    - Parses both modern and legacy response shapes
    """
    if not _GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = _gemini_endpoint(model=model, api_version=api_version)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": _GEMINI_API_KEY}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} {resp.reason}", response=resp)
            resp.raise_for_status()

            data = resp.json()
            txt = _extract_text(data)
            if not txt:
                raise ValueError("Empty AI response")
            return txt

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
            last_err = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            transient = isinstance(
                e,
                (requests.Timeout, requests.ConnectionError),
            ) or status in (429, 500, 502, 503, 504)
            if not transient or attempt == max_retries:
                break
            delay = (2 ** (attempt - 1)) * 0.8 + random.uniform(0, 0.3)
            time.sleep(delay)

    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# [ADDED] ---- Metrics & Alerts -----------------------------------------------------


def compute_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Build Tier-1 metrics and alert flags from a long-form DataFrame with
    columns: Company, Year, LineItem, Value.
    """
    d = df_long.copy()

    d["Company"] = d["Company"].astype(str).str.strip()
    d["LineItem"] = d["LineItem"].astype(str).str.strip().str.lower()

    aliases = {
        "revenue": "revenue",
        "total revenue": "revenue",
        "sales": "revenue",
        "net sales": "revenue",
        "total sales": "revenue",
        "net income": "net_income",
        "net profit": "net_income",
        "profit": "net_income",
        "profit for the year": "net_income",
        "income attributable to shareholders": "net_income",
        "retained profit": "net_income",
        "liabilities": "liabilities",
        "total liabilities": "liabilities",
        "debt": "debt",
        "total debt": "debt",
        "interest-bearing debt": "debt",
        "total interest bearing debt": "debt",
        "borrowings": "debt",
        "short-term debt": "debt",
        "short term debt": "debt",
        "long-term debt": "debt",
        "long term debt": "debt",
        "assets": "assets",
        "total assets": "assets",
        "equity": "equity",
        "shareholders' equity": "equity",
        "stockholders' equity": "equity",
        "total equity": "equity",
        "total shareholders' equity": "equity",
        "total stockholders' equity": "equity",
        "total shareholder funds": "equity",
    }
    d["key"] = d["LineItem"].map(aliases).fillna(d["LineItem"])

    wide = (
        d.pivot_table(
            index=["Company", "Year"],
            columns="key",
            values="Value",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values(["Company", "Year"])
    )

    def safe_col(frame: pd.DataFrame, name: str) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce").replace({0: np.nan})
        return a.divide(b)

    rev = safe_col(wide, "revenue")
    ni = safe_col(wide, "net_income")
    liab = safe_col(wide, "liabilities")
    debt = safe_col(wide, "debt")
    assets = safe_col(wide, "assets")
    equity = safe_col(wide, "equity")

    if equity.isna().all() and (assets.notna().any() or liab.notna().any()):
        equity = assets - liab
        wide["equity"] = equity

    debt_like = debt.where(debt.notna(), liab)

    wide["revenue"] = rev
    wide["net_income"] = ni
    wide["net_margin"] = safe_div(ni, rev)
    wide["debt_to_equity"] = safe_div(debt_like, equity)
    wide["debt_to_assets"] = safe_div(debt_like, assets)

    wide["rev_yoy"] = wide.groupby("Company", sort=False)["revenue"].transform(lambda s: s.pct_change())
    wide["ni_yoy"] = wide.groupby("Company", sort=False)["net_income"].transform(lambda s: s.pct_change())

    def flag_class(val, threshold_low=None, threshold_high=None, inverse=False):
        if pd.isna(val):
            return None
        if inverse:
            return "red" if (threshold_high is not None and val > threshold_high) else "green"
        return "red" if (threshold_low is not None and val < threshold_low) else "green"

    wide["alert_liquidity"] = wide["net_margin"].apply(lambda x: flag_class(x, threshold_low=0.05))
    wide["alert_leverage"] = wide["debt_to_equity"].apply(lambda x: flag_class(x, threshold_high=2, inverse=True))
    wide["alert_revtrend"] = wide["rev_yoy"].apply(lambda x: flag_class(x, threshold_low=0))

    return wide


# ---- 7) Upload & AI Analysis Route -----------------------------------------------


@app.get("/")
@limiter.exempt
def index():
    return render_template("index.html")


@app.post("/upload")
@limiter.limit("10 per minute")
def upload_file():
    saved_name = request.form.get("saved_filename")

    # ---- locate file (preview or fresh upload) ----------------------------------
    if saved_name:
        filepath = UPLOAD_FOLDER / saved_name
        if not filepath.exists():
            flash("Saved file not found. Please upload again.")
            return redirect(request.url_root)
    else:
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

    try:
        raw_df = read_anytabular(filepath)
    except Exception as e:
        flash(f"Error reading file: {e}")
        return redirect(request.url_root)

    df_norm, warn = normalize_financial_df(raw_df)
    if warn:
        flash(warn)

    needed = {"Company", "Year", "LineItem", "Value"}
    use_df = df_norm if needed.issubset(set(df_norm.columns)) else raw_df

    if "Year" in use_df.columns:
        years_debug = sorted(int(v) for v in pd.unique(use_df["Year"].dropna()))
        app.logger.info("Years found: %s", years_debug)
    else:
        app.logger.info("Years found: <none> (no 'Year' column)")

    summary_df = use_df.copy().sort_values(
        ["Company", "Year", "LineItem"],
        na_position="last",
    )
    summary = summary_df.to_dict(orient="records")

    li_norm_ai = use_df["LineItem"].astype(str).str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()

    ai_df = use_df.assign(
        LI_CANON=np.select(
            [
                li_norm_ai.str.contains(
                    r"\b(?:net income|net profit|profit)\b",
                    regex=True,
                    na=False,
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
        "Focus on growth/decline and rough margins across years; do not invent data.\n" + "\n".join(lines)
    )

    if GEMINI_AVAILABLE:
        try:
            ai_text = call_gemini(prompt)
        except Exception as e:
            flash(f"AI call failed: {e}")
            ai_text = None
    else:
        ai_text = "(AI disabled: missing GEMINI_* env vars)"

    # --- Charts (always set fig_json & chart_data) ------------------------------

    fig_json = None  # interactive Plotly (latest year) for page
    chart_data = None  # base64 PNG for PDF (latest-year bars)
    years = []  # year dropdown
    figs_by_year_json = "{}"  # per-year figures for client switching
    fig_all_json = "null"  # all-years line chart (Plotly JSON)
    chart_data_all = None  # all-years PNG for PDF (optional)

    has_canonical = needed.issubset(use_df.columns)

    if "Year" in use_df.columns and use_df["Year"].notna().any():
        fig_all = build_plotly_multi_year(use_df)
        fig_all_json = json.dumps(fig_all, cls=PlotlyJSONEncoder)
        try:
            chart_data_all = base64.b64encode(
                fig_all.to_image(format="png", scale=2),
            ).decode("ascii")
        except Exception as e:
            app.logger.warning(
                "Plotly->PNG failed (all years). Using Matplotlib fallback. Error: %s",
                e,
            )
            chart_data_all = build_matplotlib_all_years_line(use_df)

    if has_canonical:
        fig = build_plotly_chart(use_df)
        fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        years = sorted(int(v) for v in pd.unique(use_df["Year"].dropna()))
        if years:
            figs_by_year = {}
            for y in years:
                fy = build_plotly_chart(use_df, latest_year=int(y))
                figs_by_year[str(y)] = json.loads(json.dumps(fy, cls=PlotlyJSONEncoder))
            figs_by_year_json = json.dumps(figs_by_year)

        try:
            chart_data = base64.b64encode(
                fig.to_image(format="png", scale=2),
            ).decode("ascii")
        except Exception as e:
            app.logger.warning(
                "Plotly->PNG failed (latest-year). Using Matplotlib fallback. Error: %s",
                e,
            )
            chart_data = build_matplotlib_grouped(use_df)
    else:
        chart_data = build_matplotlib_grouped(use_df)

    metrics = compute_metrics(use_df) if has_canonical else pd.DataFrame()

    return render_template(
        "result.html",
        summary=summary,
        ai_text=ai_text,
        chart_data=chart_data,
        fig_json=fig_json,
        years=years,
        figs_by_year_json=figs_by_year_json,
        fig_all_json=fig_all_json,
        chart_data_all=chart_data_all,
        metrics=metrics,
    )


# ---- 7a) Preview route ---------------------------------------------------------


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

    saved_path = UPLOAD_FOLDER / file.filename
    file.save(saved_path)

    try:
        df = read_anytabular(saved_path)
        preview_rows = df.head(10).to_dict(orient="records")
    except Exception as e:
        flash(f"Could not read file: {e}")
        return redirect(request.url_root)

    return render_template(
        "preview.html",
        saved_filename=saved_path.name,
        preview_rows=preview_rows,
    )


# ---- 8) PDF Download Route -----------------------------------------------------


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    summary = json.loads(request.form["summary"])
    ai_text = request.form["ai_text"]

    chart_data = request.form.get("chart_data")
    chart_data_all = request.form.get("chart_data_all")

    summary_df = pd.DataFrame(summary)
    summary_df["Year"] = pd.to_numeric(summary_df["Year"], errors="coerce").astype("Int64")
    summary_df["Value"] = pd.to_numeric(summary_df["Value"], errors="coerce")

    metrics = compute_metrics(summary_df)

    html_out = render_template(
        "pdf.html",
        summary=summary,
        ai_text=ai_text,
        chart_data=chart_data,
        chart_data_all=chart_data_all,
        metrics=metrics,
    )

    pdf_bytes = HTML(
        string=html_out,
        base_url=request.url_root,
    ).write_pdf()

    return send_file(
        BytesIO(pdf_bytes),
        as_attachment=True,
        download_name="ai_analysis.pdf",
        mimetype="application/pdf",
    )


# ---- 9) Excel Export Route -----------------------------------------------------


@app.post("/export_excel")
def export_excel():
    """
    Creates an Excel file with conditional formatting (green/red) for Tier-1 alerts.
    Also formats money as whole numbers and percentages as 2-dp.
    """
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
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    metrics = compute_metrics(df)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        metrics.to_excel(writer, index=False, sheet_name="Tier1_Ratios_Alerts")
        wb = writer.book
        ws = writer.sheets["Tier1_Ratios_Alerts"]

        header_fmt = wb.add_format({"bold": True, "bg_color": "#EFEFEF", "border": 1, "align": "center"})
        money_fmt = wb.add_format({"num_format": "#,##0"})
        pct_fmt = wb.add_format({"num_format": "0.00%"})

        hide_text = wb.add_format({"num_format": ";;;"})
        green_fill = wb.add_format({"bg_color": "#C6EFCE"})
        red_fill = wb.add_format({"bg_color": "#FFC7CE"})
        gray_fill = wb.add_format({"bg_color": "#E5E7EB"})

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

        col_letter = xl_col_to_name
        nrows = len(metrics) + 1

        ws.set_row(0, None, header_fmt)
        ws.set_column(f"{col_letter(ix_company)}:{col_letter(ix_company)}", 18)
        ws.set_column(f"{col_letter(ix_year)}:{col_letter(ix_year)}", 8)
        ws.set_column(
            f"{col_letter(ix_revenue)}:{col_letter(ix_net_income)}",
            16,
            money_fmt,
        )
        ws.set_column(f"{col_letter(ix_net_margin)}:{col_letter(ix_net_margin)}", 12, pct_fmt)
        ws.set_column(
            f"{col_letter(ix_debt_equity)}:{col_letter(ix_debt_assets)}",
            14,
        )
        ws.set_column(f"{col_letter(ix_rev_yoy)}:{col_letter(ix_ni_yoy)}", 12, pct_fmt)
        ws.set_column(f"{col_letter(ix_alert_liq)}:{col_letter(ix_alert_rev)}", 4, hide_text)

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

        cf_eq(ix_alert_liq, "green", green_fill)
        cf_eq(ix_alert_liq, "red", red_fill)
        cf_blank(ix_alert_liq, gray_fill)

        cf_eq(ix_alert_lev, "green", green_fill)
        cf_eq(ix_alert_lev, "red", red_fill)
        cf_blank(ix_alert_lev, gray_fill)

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
