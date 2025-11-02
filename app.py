# ---- 0) Imports ----------------------------------------------------
# (1) Load → (2) read env vars → (3) use them

import re
import time
import random
import hashlib
import base64
import json
import os
import zipfile
from io import BytesIO
from app_validators import normalize_and_validate, DataValidationError, normalize_financials_xlsx
from zipfile import ZipFile
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
    url_for,  # ← add this
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
from weasyprint import HTML
from xlsxwriter.utility import xl_col_to_name

# --- AI mode config / helper -------------------------------------------
AI_MODE_DEFAULT = os.getenv("AI_MODE", "both").strip().lower()
if AI_MODE_DEFAULT not in {"desc", "ratios", "both"}:
    AI_MODE_DEFAULT = "both"


def _resolve_ai_mode(request_form) -> str:
    """Resolve AI analysis mode from form (if any) or env default."""
    mode = (request_form.get("ai_mode") or AI_MODE_DEFAULT).strip().lower()
    return mode if mode in {"desc", "ratios", "both"} else "both"


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

# How many years per company to send to the AI (prompt size control)
AI_YEARS_PER_COMPANY = int(os.getenv("AI_YEARS_PER_COMPANY", "5"))  # default 5

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


@app.get("/ai_smoke")
@limiter.exempt
def ai_smoke():
    try:
        txt = call_gemini_v1("Reply with: OK")
        got = (txt or "").strip()
        ok = got == "OK"
        # 200 only when the check passes; 502 when model responded but not exactly "OK"
        return ({"ok": ok, "got": got}, 200 if ok else 502)
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


# ---- 4b) Upload helpers: allowed types, readers, normalizer -------------------

ALLOWED_EXTENSIONS = {"csv", "xlsx", "zip"}

# 1) Read
raw_df = read_anytabular(upload_path)

# 2) Header aliasing (optional but recommended)
raw_df = _apply_alias_renames(raw_df)  # <-- keep aliasing here

# 3) Canonical validation & normalization
canonical = normalize_and_validate(raw_df)


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
    """
    Load an uploaded file into a single DataFrame.
    - .csv           : read via pandas
    - .xlsx          : read ALL worksheets (multi-sheet aware) and concat
    - .zip           : prefer first .xlsx inside (multi-sheet aware),
                       otherwise concat all .csv files inside
    Raises DataValidationError on unsupported/empty inputs.
    """
    p = Path(path)
    ext = p.suffix.lower()

    # Simple CSV
    if ext == ".csv":
        return pd.read_csv(p)

    # Excel workbook (handles multi-sheet workbooks)
    if ext == ".xlsx":
        return normalize_financials_xlsx(p)

    # ZIP container
    if ext == ".zip":
        with ZipFile(p, "r") as zf:
            # prefer first .xlsx inside
            xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
            if xlsx_names:
                data = zf.read(xlsx_names[0])
                return normalize_financials_xlsx(BytesIO(data))

            # else: concat all CSVs inside
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise DataValidationError(
                    "ZIP does not contain a readable .xlsx workbook or any .csv files."
                )
            frames = []
            for name in csv_names:
                with zf.open(name) as f:
                    frames.append(pd.read_csv(f))
            return pd.concat(frames, ignore_index=True)

    raise DataValidationError(f"Unsupported file type: {ext}")


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


def _apply_alias_renames(df: pd.DataFrame) -> pd.DataFrame:
    # Map lower->original for quick lookup
    cols_lower = {c.lower().strip(): c for c in df.columns}

    company = _find_alias(cols_lower, _CANON["company"])
    year    = _find_alias(cols_lower, _CANON["year"])
    lineit  = _find_alias(cols_lower, _CANON["lineitem"])
    value   = _find_alias(cols_lower, _CANON["value"])

    rename_map = {}
    if company and company != "Company":
        rename_map[company] = "Company"
    if year and year != "Year":
        rename_map[year] = "Year"
    # app_validators accepts "LineItem" or "Line Item"
    if lineit and lineit not in ("LineItem", "Line Item"):
        rename_map[lineit] = "LineItem"
    if value and value != "Value":
        rename_map[value] = "Value"

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


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

# Uses the existing imports at the top: os, json, requests
# GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1")
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Only need to define this (if not already defined once):
GEMINI_BASE = "https://generativelanguage.googleapis.com"


# --- ultra-light LRU-ish cache (per-process, expires ~1 hour) ----------
_AI_CACHE: dict[str, tuple[float, str]] = {}
_AI_CACHE_TTL = 3600.0  # seconds
_AI_CACHE_MAX = 50


def _ai_cache_get(key: str) -> str | None:
    t_v = _AI_CACHE.get(key)
    if not t_v:
        return None
    ts, val = t_v
    if (time.time() - ts) > _AI_CACHE_TTL:
        _AI_CACHE.pop(key, None)
        return None
    return val


def _ai_cache_put(key: str, val: str) -> None:
    # drop oldest if over capacity
    if len(_AI_CACHE) >= _AI_CACHE_MAX:
        _AI_CACHE.pop(next(iter(_AI_CACHE)))
    _AI_CACHE[key] = (time.time(), val)


def _redact_url(s: str) -> str:
    """Scrub any '?key=' or '&key=' query value from a URL string."""
    return re.sub(r'([?&]key=)[^&]+', r'\1REDACTED', s or "")


def _clean_prompt(text: str | None, max_len: int = 120_000) -> str:
    """UTF-8 clean + hard truncate to keep requests within safe token limits."""
    if not text:
        return ""
    return text.encode("utf-8", "ignore").decode("utf-8")[:max_len].strip()


# Heuristic: tidy, clip to last full sentence, ensure sentence-ending punctuation
_SENT_END_RE = re.compile(r'[.!?]["\')\]]*$')


def _postprocess_ai_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    s = re.sub(r'[ \t]+', ' ', s)  # keep newlines, but collapse repeated spaces/tabs
    s = re.sub(r'^[-•\s]+', '', s)  # drop leading bullets/dashes

    # If it doesn’t end with sentence punctuation but contains at least one sentence end,
    # clip to the last complete sentence.
    last_dot = s.rfind(".")
    last_bang = s.rfind("!")
    last_q = s.rfind("?")
    last_end = max(last_dot, last_bang, last_q)

    # If we don't end with punctuation, but we *do* have a prior sentence end,
    # clip to that full sentence (no arbitrary 50-char threshold).
    if not _SENT_END_RE.search(s) and last_end >= 50:
        s = s[: last_end + 1].rstrip()

    # If we *still* don't end cleanly, trim trailing hyphen/dash and add a period.
    s = re.sub(r'[-–—]\s*$', '', s)  # remove trailing hyphen fragments
    if s and not _SENT_END_RE.search(s):
        s += "."

    return s


def call_gemini_v1(
    prompt_text: str,
    temperature: float = 0.2,
    top_p: float = 0.85,
    top_k: int = 32,
    max_tokens: int = 1000,
    _model_override: str | None = None,
    # ↑ (B) raise request timeout to handle cold starts / model latency
    _timeout_s: int = 60,
    # ↑ (C) stronger retry budget
    _max_retries: int = 5,
    random_seed: int | None = None,  # ← add this
) -> str:
    """
    Minimal, v1-compliant request for Gemini. Returns plain text or "".
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    # v1 endpoint shape
    model_to_use = (_model_override or GEMINI_MODEL).strip()
    url_base = f"{GEMINI_BASE}/{GEMINI_API_VERSION}/models"
    headers = {"Content-Type": "application/json"}

    # Hard-truncate very long prompts just in case
    safe_prompt = (prompt_text or "").encode("utf-8", "ignore").decode("utf-8")[:120_000].strip()

    payload = {
        "contents": [{"role": "user", "parts": [{"text": safe_prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "topP": float(top_p),
            "topK": int(top_k),
            "maxOutputTokens": int(max_tokens),
            # "seed" is the field recognized by Gemini for deterministic runs
            **({"seed": int(random_seed)} if random_seed is not None else {}),
        },
    }

    def _once(model_name: str):
        url = f"{url_base}/{model_name}:generateContent?key={GEMINI_API_KEY}"
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=_timeout_s)
        try:
            body = r.json()
        except Exception:
            body = {"_non_json_body": r.text}
        return r.status_code, body, url

    # (C) Exponential backoff with jitter on 429/503 (rate-limit/overload)
    attempts = _max_retries
    backoff = 1.5
    status, body, url = None, None, None
    for i in range(attempts):
        status, body, url = _once(model_to_use)
        if status == 200 or status not in (429, 503):
            break
        # sleep with jitter; cap at ~20s per try
        time.sleep(min(20.0, backoff * (1.0 + random.random())))
        backoff *= 2.0

    # one-shot fallback to flash if still overloaded
    if status in (429, 503) and model_to_use != "gemini-2.5-flash":
        status, body, url = _once("gemini-2.5-flash")

    # One-shot fallback to flash if still overloaded
    if status != 200:
        raise RuntimeError(f"Gemini {status} at {_redact_url(url)}: {json.dumps(body)[:2000]}")

    # Extract first text part
    cands = body.get("candidates") or []
    if not cands:
        return ""
    for p in cands[0].get("content", {}).get("parts", []):
        if "text" in p:
            return p["text"]
    return ""

    # Extract first text part from v1 response
    cands = body.get("candidates") or []
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    for p in parts:
        if "text" in p:
            return p["text"]
    return ""


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

    # Resolve AI analysis mode for this request (form overrides env)

    mode = _resolve_ai_mode(request.form)

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

    # ---- strict normalization & numeric validation (hard-fail on issues)
    try:
        # normalizes “Shareholders’ Equity” style labels and verifies Value is numeric
        use_df = normalize_and_validate(use_df)
    except (ValueError, DataValidationError) as e:
        # friendlier error back to the browser instead of a 500
        flash(str(e))
        return redirect(request.url_root)

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

    ai_text = ""  # make sure it's defined regardless of mode
    # --- AI insight (General summary) ---------------------------------------------------------

    if mode in {"desc", "both"}:
        # 1) Auto-scale years sent to the model (reduces overload with many companies)
        company_count = ai_df["Company"].nunique() if not ai_df.empty else 1
        years_keep = AI_YEARS_PER_COMPANY if company_count <= 2 else min(3, AI_YEARS_PER_COMPANY)

        # Keep {years_keep} years PER COMPANY of Rev/NI
        lines = []
        if not ai_df.empty:
            _tmp = ai_df.sort_values(["Company", "Year"])
            for comp, g in _tmp.groupby("Company", sort=False):
                # build Revenue/Net income pairs per year
                gg = (
                    g.pivot_table(index="Year", columns="LI_CANON", values="Value", aggfunc="sum")
                    .reset_index()
                    .sort_values("Year")
                ).tail(years_keep)
                for _, row in gg.iterrows():
                    yr = int(row.get("Year", 0)) if pd.notna(row.get("Year")) else ""
                    rev = float(row.get("Revenue", 0) or 0)
                    ni = float(row.get("Net income", 0) or 0)
                    lines.append(f"{comp} {yr} | Revenue: {rev:,.0f} | Net income: {ni:,.0f}")

        # Slightly raise overall cap since we now include more years
        if len(lines) > 260:
            lines = lines[-260:]

        # 2) Stronger prompt (keeps output consistent and multi-paragraph)
        prompt = (
            "You are a financial analyst. Write a concise multi-paragraph summary using ONLY the rows below.\n\n"
            f"Cover the last {years_keep} years for each company.\n\n"
            "Requirements:\n"
            "• Write one short paragraph (2–3 sentences) per company.\n"
            "• Focus on growth/decline in Revenue and Net income, and margin direction.\n"
            "• Use clear verbs (rose/fell/improved/softened/stable). No forecasts, no advice, no invented data.\n"
            "• Do not introduce new numbers beyond the table. End each paragraph with a period.\n\n"
            "Data (company-year rows):\n" + "\n".join(lines)
        )

        # Allow a bit more room for 5-year inputs and clean the text
        prompt = _clean_prompt(prompt, max_len=12000)

        # Force pro for richer output; include the model in the cache key
        MODEL_MAIN = "gemini-2.5-pro"
        _k1 = hashlib.sha1((MODEL_MAIN + "|" + prompt).encode("utf-8")).hexdigest()
        ai_text = _ai_cache_get(_k1) or ""

        if not ai_text:
            try:
                # Stronger prompt guarantees separate paragraphs per company.
                prompt = (
                    "You are a financial analyst. Using ONLY the rows below, write a concise yet complete summary.\n\n"
                    f"Cover the last {years_keep} years for each company.\n\n"
                    "Formatting requirements:\n"
                    "• Use a separate paragraph per company, beginning with the company name "
                    "in bold like **Apple Inc.**\n"
                    "• Write 2–4 sentences per company. Do not collapse companies into one paragraph.\n"
                    "• Focus on revenue, net income, and margin direction (rose/fell/improved/softened/stable).\n"
                    "• No forecasts or advice. No new numbers beyond the table. End each paragraph with a period.\n\n"
                    "Data (company-year rows):\n" + "\n".join(lines)
                )
                ai_text = call_gemini_v1(
                    prompt_text=_clean_prompt(prompt, max_len=12000),
                    temperature=0.05,  # steadier
                    top_p=1.0,  # full nucleus to reduce accidental truncation
                    top_k=32,
                    max_tokens=1400,  # more room for multi-paragraph output
                    _model_override=MODEL_MAIN,
                    _timeout_s=90,
                    _max_retries=6,
                )
                _ai_cache_put(_k1, ai_text or "")
                app.logger.info("AI primary call returned length=%s", len(ai_text or ""))

                # Gentle post-processing (no aggressive clipping)
                ai_text = _postprocess_ai_text(ai_text)
                if len(ai_text) < 60 or ai_text.count(" ") < 10:
                    ai_text = ""
            except Exception as e:
                app.logger.warning("AI primary call failed: %s", e)

        # 3a) Fallback with fewer rows if still empty
        if not ai_text or not ai_text.strip():
            # Retry with a small slice; 50 keeps some cross-year signal for 5y/company
            SHORT_LINES = 50
            short_lines = lines[-SHORT_LINES:] if len(lines) > SHORT_LINES else lines
            retry_prompt = "Summarize key trends in 3–5 sentences. Be concise; no advice.\n" + "\n".join(short_lines)
            try:
                ai_text = call_gemini_v1(
                    prompt_text=_clean_prompt(retry_prompt, max_len=4000),
                    temperature=0.2,
                    top_p=0.85,
                    top_k=32,
                    max_tokens=800,
                )
                app.logger.info("AI retry call returned length=%s", len(ai_text or ""))
            except Exception as e:
                app.logger.warning("AI retry call failed: %s", e)

        # 3b) Ultra-short fallback: tiny prompt + smaller output
        if not ai_text or not ai_text.strip():
            short_lines = lines[-20:] if len(lines) > 20 else lines
            short_prompt = (
                "In 2–3 sentences, state the two biggest trends across the companies. "
                "No advice, no invented numbers.\n" + "\n".join(short_lines)
            )
            try:
                ai_text = call_gemini_v1(
                    prompt_text=_clean_prompt(short_prompt, max_len=2000),
                    temperature=0.15,
                    top_p=0.85,
                    top_k=32,
                    max_tokens=420,
                    _model_override="gemini-2.5-flash",  # keep this tiny and fast
                )
                ai_text = _postprocess_ai_text(ai_text)
                if len(ai_text) < 30 or ai_text.count(" ") < 5:
                    ai_text = ""
            except Exception as e:
                app.logger.warning("AI ultra-short fallback failed: %s", e)

        # 4) Final guarantee: multi-sentence deterministic summary if still empty
        if not ai_text or not ai_text.strip():
            try:
                parts = []
                if not ai_df.empty:
                    _t = ai_df.sort_values(["Company", "Year"])
                    for comp, g in _t.groupby("Company", sort=False):
                        years = g["Year"].dropna().astype(int)
                        if years.empty:
                            continue
                        y0, y1 = years.min(), years.max()
                        rev0 = g[(g["Year"] == y0) & (g["LI_CANON"] == "Revenue")]["Value"].sum()
                        rev1 = g[(g["Year"] == y1) & (g["LI_CANON"] == "Revenue")]["Value"].sum()
                        ni0 = g[(g["Year"] == y0) & (g["LI_CANON"] == "Net income")]["Value"].sum()
                        ni1 = g[(g["Year"] == y1) & (g["LI_CANON"] == "Net income")]["Value"].sum()

                        rev_dir = "higher" if rev1 >= rev0 else "lower"
                        ni_dir = "higher" if ni1 >= ni0 else "lower"

                        s1 = f"**{comp}.** Revenue {rev_dir} versus {y0} and Net income {ni_dir} by {y1}."

                        # Optional second sentence, wrapped to satisfy line-length (E501)
                        s2 = (
                            f"Across the period {y0}–{y1}, top-line and bottom-line moved in the same "
                            f"general direction for {comp}, with no forecasts or assumptions."
                        )

                        parts.append(s1 + " " + s2)

                ai_text = " ".join(parts) or "(No AI summary; dataset lacks Revenue/Net income rows.)"
            except Exception as e:
                app.logger.warning("Rule-based summary failed: %s", e)
                ai_text = "(No AI summary due to an unexpected error.)"

    # --- Charts (always set fig_json & chart_data) ------------------------------

    fig_json = None  # interactive Plotly (latest year) for page
    chart_data = None  # base64 PNG for PDF (latest-year bars)
    years = []  # year dropdown
    figs_by_year_json = "{}"  # per-year figures for client switching
    fig_all_json = "null"  # all-years line chart (Plotly JSON)
    chart_data_all = None  # all-years PNG for PDF (optional)

    # NEW: always define, so template render is safe even if we skip ratios block
    ratios_text = ""  # <--- ADD THIS LINE

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

    # [NEW] ---- Build a compact ratios text block (last 5 years per company) and ask Gemini -------

    if mode in {"ratios", "both"} and metrics is not None and not metrics.empty:
        # sort by Company, Year so "last 5" is correct per company
        m = metrics.sort_values(["Company", "Year"]).copy()
        # Auto-scale for ratios as well
        company_count2 = m["Company"].nunique() if not m.empty else 1
        years_keep_ratios = AI_YEARS_PER_COMPANY if company_count2 <= 2 else min(3, AI_YEARS_PER_COMPANY)

        def _fmt_pct(x):
            try:
                x = float(x)
            except Exception:
                return "NaN"
            if pd.isna(x):
                return "NaN"
            return f"{x:.1%}"

        def _fmt_2(x):
            try:
                x = float(x)
            except Exception:
                return "NaN"
            if pd.isna(x):
                return "NaN"
            return f"{x:.2f}"

        lines_ratios = []
        for comp, grp in m.groupby("Company", sort=False):
            tail = grp.tail(years_keep_ratios)
            for _, r in tail.iterrows():
                y = int(r.get("Year", 0)) if pd.notna(r.get("Year")) else ""
                nm = _fmt_pct(r.get("net_margin"))
                de = _fmt_2(r.get("debt_to_equity"))
                da = _fmt_2(r.get("debt_to_assets"))
                ry = _fmt_pct(r.get("rev_yoy"))
                ny = _fmt_pct(r.get("ni_yoy"))
                s1 = f"{comp} {y} | margin={nm} | D/E={de} | "
                s2 = f"D/A~{da} | Rev YoY={ry} | NI YoY={ny}"
                lines_ratios.append(s1 + s2)

        if lines_ratios:
            ratios_prompt = (
                "You are a financial analyst. Using ONLY the Tier-1 ratios below (last 5 years per company), "
                "write a short section titled 'Ratios Focus' to APPEND after the general AI Analysis.\n\n"
                "Formatting:\n"
                "• Begin each company paragraph with the company name in bold like **Apple Inc.**\n"
                "• Write 2–3 sentences PER COMPANY (compact but substantive).\n"
                "• Address clearly: Liquidity (net margin trend), Leverage (level/direction of D/E and D/A), "
                "  Momentum (Rev YoY and NI YoY). Use verbs like improved/softened/stable/eased/rose/fell.\n"
                "• No advice, no forecasts, no numbers not shown; end each paragraph with a period.\n\n"
                "Data (company-year rows):\n" + "\n".join(lines_ratios)
            )
            try:
                ratios_text = call_gemini_v1(
                    prompt_text=_clean_prompt(ratios_prompt, max_len=20000),
                    temperature=0.15,  # lower randomness → tighter, more factual prose
                    top_p=0.9,
                    top_k=32,
                    max_tokens=1100,  # # more space for per-company paragraphs
                    _model_override="gemini-2.5-pro",  # <<< force pro here
                ).strip()
            except Exception as e:
                ratios_text = ""
                app.logger.warning("Ratios AI call failed: %s", e)

            # normalize & gentle guard
            ratios_text = _postprocess_ai_text(ratios_text)
            if len(ratios_text) < 30 or ratios_text.count(" ") < 5:
                # Fallback: deterministic bullets so something always renders
                try:
                    bullets = []
                    for comp, grp in m.groupby("Company", sort=False):
                        tail = grp.tail(years_keep_ratios)
                        nm_trend = "improved" if (tail["net_margin"].diff().mean(skipna=True) or 0) > 0 else "softened"
                        de_mean = tail["debt_to_equity"].mean(skipna=True)
                        da_mean = tail["debt_to_assets"].mean(skipna=True)
                        ry_mean = tail["rev_yoy"].mean(skipna=True)
                        ny_mean = tail["ni_yoy"].mean(skipna=True)
                        bullets.append(
                            f"**{comp}.** Liquidity {nm_trend}; leverage at D/E≈{de_mean:.2f}, D/A≈{da_mean:.2f}; "
                            f"momentum Rev YoY≈{ry_mean:.1%}, NI YoY≈{ny_mean:.1%}."
                        )
                    ratios_text = " ".join(bullets)
                except Exception:
                    ratios_text = ""
            else:
                if not ratios_text.endswith(('.', '!', '?')):
                    ratios_text += '.'
            # (optional) debug
            app.logger.info("ratios_text: chars=%d words=%d", len(ratios_text), len(ratios_text.split()))

    # ---- continue with your existing render_template(...) exactly as before ----------------------

    return render_template(
        "result.html",
        summary=summary,
        ai_text=ai_text,
        ratios_text=ratios_text,  # ← add this <<< must be here
        ai_mode=mode,  # <— add this
        chart_data=chart_data,
        fig_json=fig_json,
        years=years,
        figs_by_year_json=figs_by_year_json,
        fig_all_json=fig_all_json,
        chart_data_all=chart_data_all,
        metrics=metrics,
    )


# ---- 7a) Preview route ------------------------------------------------------------------------


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


# ---- Friendly GET redirects for download endpoints (opened directly in browser) ----


@app.get("/download_pdf")
@limiter.exempt
def download_pdf_get():
    flash("Use the 'Download as PDF' button on the results page.")
    return redirect(url_for("index")), 302


@app.get("/export_excel")
@limiter.exempt
def export_excel_get():
    flash("Use the 'Download Excel' button on the results page.")
    return redirect(url_for("index")), 302


# ---- 8) PDF Download Route --------------------------------------------------------


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    # Data posted from result.html (hidden fields)
    summary = json.loads(request.form.get("summary") or "[]")
    ai_text = (request.form.get("ai_text") or "").strip()
    ratios_text = (request.form.get("ratios_text") or "").strip()

    chart_data = request.form.get("chart_data") or ""
    chart_data_all = request.form.get("chart_data_all") or ""

    # OPTIONAL: add a clean sentence terminator without changing content
    if ai_text and ai_text[-1] not in ".!?":
        ai_text += "."
    if ratios_text and ratios_text[-1] not in ".!?":
        ratios_text += "."

    # Metrics recomputation is fine for the PDF table (doesn't touch ai/ratios text)
    summary_df = pd.DataFrame(summary)
    summary_df["Year"] = pd.to_numeric(summary_df.get("Year"), errors="coerce").astype("Int64")
    summary_df["Value"] = pd.to_numeric(summary_df.get("Value"), errors="coerce")
    metrics = compute_metrics(summary_df)

    html_out = render_template(
        "pdf.html",
        summary=summary,
        ai_text=ai_text,  # ← EXACT string seen on the page
        ratios_text=ratios_text,  # ← EXACT string seen on the page
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
