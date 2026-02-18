# ============================================================
# app.py — Morgan Stanley | Rates Sales Dashboard
# Author  : Mathis de Looze
# Version : 3.0.0
# ============================================================


# ============================================================
# BLOCK 1 — CONFIGURATION
# ============================================================

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="MS | Rates Sales Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _inject_css() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Open Sans', sans-serif; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #00539b;
            border-radius: 6px;
            padding: 12px 16px;
        }
        div[data-testid="metric-container"] label {
            font-size: 11px !important;
            color: #6c757d !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: #002b5c !important;
        }
        </style>
    """, unsafe_allow_html=True)


_inject_css()


# ============================================================
# BLOCK 2 — CONSTANTS & TICKERS
# ============================================================

MS_BLUE  = "#00539b"
MS_DARK  = "#002b5c"
MS_GREY  = "#6c757d"
MS_RED   = "#c0392b"
MS_GREEN = "#27ae60"
MS_BG    = "#F8F9FA"

COUNTRIES = ["US", "UK", "EU", "DE", "FR", "IT", "JP", "CH"]

COUNTRY_LABELS = {
    "US": "🇺🇸 United States",
    "UK": "🇬🇧 United Kingdom",
    "EU": "🇪🇺 Euro Area",
    "DE": "🇩🇪 Germany",
    "FR": "🇫🇷 France",
    "IT": "🇮🇹 Italy",
    "JP": "🇯🇵 Japan",
    "CH": "🇨🇭 Switzerland",
}

# ── Tenors ≥ 1Y only ─────────────────────────────────────────
ALL_TENORS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# ── FRED series IDs per (country, tenor) ────────────────────
# These are the most reliable official sources.
FRED_SERIES: dict[str, dict[str, str]] = {
    "US": {
        "1Y":  "DGS1",
        "2Y":  "DGS2",
        "3Y":  "DGS3",
        "5Y":  "DGS5",
        "7Y":  "DGS7",
        "10Y": "DGS10",
        "20Y": "DGS20",
        "30Y": "DGS30",
    },
    "DE": {
        "2Y":  "IRLTST01DEM156N",
        "10Y": "IRLTLT01DEM156N",
    },
    "FR": {
        "10Y": "IRLTLT01FRM156N",
    },
    "IT": {
        "10Y": "IRLTLT01ITM156N",
    },
    "JP": {
        "10Y": "IRLTLT01JPM156N",
    },
    "EU": {
        "10Y": "IRLTLT01EZM156N",
    },
    "UK": {
        "2Y":  "IRLTST01GBM156N",
        "10Y": "IRLTLT01GBM156N",
    },
    "CH": {},
}

# ── Yahoo Finance tickers per (country, tenor) ───────────────
# Only tenors ≥ 1Y. None = not available.
# US note: ^IRX=13W, ^UST2Y=2Y(plain%), ^FVX=5Y(*10), ^TNX=10Y(*10), ^TYX=30Y(*10)
YAHOO_TICKERS: dict[str, dict[str, str | None]] = {
    "US": {
        "1Y":  "^IRX",      # best proxy (13-week bill, used as short end)
        "2Y":  "^UST2Y",
        "3Y":  None,
        "5Y":  "^FVX",
        "7Y":  None,
        "10Y": "^TNX",
        "20Y": None,
        "30Y": "^TYX",
    },
    "UK": {
        "1Y":  None,
        "2Y":  "GBGB2YR=X",
        "3Y":  None,
        "5Y":  "GBGB5YR=X",
        "7Y":  None,
        "10Y": "GBGB10YR=X",
        "20Y": None,
        "30Y": "GBGB30YR=X",
    },
    "DE": {
        "1Y":  None,
        "2Y":  "DEDE2YR=X",
        "3Y":  None,
        "5Y":  "DEDE5YR=X",
        "7Y":  None,
        "10Y": "DEDE10YR=X",
        "20Y": None,
        "30Y": "DEDE30YR=X",
    },
    "FR": {
        "1Y":  None,
        "2Y":  "FRFR2YR=X",
        "3Y":  None,
        "5Y":  "FRFR5YR=X",
        "7Y":  None,
        "10Y": "FRFR10YR=X",
        "20Y": None,
        "30Y": "FRFR30YR=X",
    },
    "IT": {
        "1Y":  None,
        "2Y":  "ITIT2YR=X",
        "3Y":  None,
        "5Y":  "ITIT5YR=X",
        "7Y":  None,
        "10Y": "ITIT10YR=X",
        "20Y": None,
        "30Y": "ITIT30YR=X",
    },
    "EU": {
        "1Y":  None,
        "2Y":  "DEDE2YR=X",
        "3Y":  None,
        "5Y":  "DEDE5YR=X",
        "7Y":  None,
        "10Y": "DEDE10YR=X",
        "20Y": None,
        "30Y": "DEDE30YR=X",
    },
    "JP": {
        "1Y":  None,
        "2Y":  "JPJP2YR=X",
        "3Y":  None,
        "5Y":  "JPJP5YR=X",
        "7Y":  None,
        "10Y": "JPJP10YR=X",
        "20Y": None,
        "30Y": "JPJP30YR=X",
    },
    "CH": {
        "1Y":  None,
        "2Y":  "CHCH2YR=X",
        "3Y":  None,
        "5Y":  "CHCH5YR=X",
        "7Y":  None,
        "10Y": "CHCH10YR=X",
        "20Y": None,
        "30Y": "CHCH30YR=X",
    },
}

# ── Yahoo scale factors ───────────────────────────────────────
# Multiply raw Yahoo Close value → plain percentage (e.g. 4.05 = 4.05 %)
# ^IRX, ^FVX, ^TNX, ^TYX are quoted as rate * 10  → scale = 0.1
# ^UST2Y and all =X tickers are already in plain % → scale = 1.0
YAHOO_SCALE: dict[str, dict[str, float]] = {
    "US": {
        "1Y":  0.1,   # ^IRX  ÷ 10
        "2Y":  1.0,   # ^UST2Y already %
        "3Y":  1.0,
        "5Y":  0.1,   # ^FVX  ÷ 10
        "7Y":  1.0,
        "10Y": 0.1,   # ^TNX  ÷ 10
        "20Y": 1.0,
        "30Y": 0.1,   # ^TYX  ÷ 10
    },
    **{
        c: {t: 1.0 for t in ALL_TENORS}
        for c in ["UK", "EU", "DE", "FR", "IT", "JP", "CH"]
    },
}

# ── 2Y/10Y spread pair per country ───────────────────────────
SPREAD_KEY: dict[str, tuple[str, str]] = {
    c: ("2Y", "10Y") for c in COUNTRIES
}


# ============================================================
# BLOCK 3 — DATA LAYER  (multi-source: FRED → Yahoo → warning)
# ============================================================

# ── FRED helpers ─────────────────────────────────────────────

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_KEY  = "YOUR_FRED_API_KEY"   # set here or via st.secrets["FRED_API_KEY"]


def _get_fred_key() -> str:
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return _FRED_KEY


def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """
    Fetch one FRED series. Returns a daily-indexed pd.Series in plain %.
    FRED already returns values in plain percent (e.g. 4.05 = 4.05 %).
    Returns empty Series on any failure.
    """
    api_key = _get_fred_key()
    if api_key == "YOUR_FRED_API_KEY":
        # No key configured — skip silently
        return pd.Series(dtype=float)
    try:
        params = {
            "series_id":         series_id,
            "observation_start": start,
            "observation_end":   end,
            "api_key":           api_key,
            "file_type":         "json",
            "frequency":         "d",
        }
        r = requests.get(_FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        records = {
            o["date"]: float(o["value"])
            for o in obs
            if o["value"] != "."
        }
        if not records:
            return pd.Series(dtype=float)
        s = pd.Series(records)
        s.index = pd.to_datetime(s.index)
        s.sort_index(inplace=True)
        return s
    except Exception as exc:
        logger.warning(f"FRED fetch failed for {series_id}: {exc}")
        return pd.Series(dtype=float)


# ── Yahoo helper ─────────────────────────────────────────────

def _fetch_yahoo_series(ticker: str, scale: float, start: str, end: str) -> pd.Series:
    """
    Fetch one Yahoo Finance ticker. Applies scale so result is in plain %.
    Returns empty Series on any failure or empty response.
    """
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.Series(dtype=float)

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if "Close" not in raw.columns:
            return pd.Series(dtype=float)

        s = raw["Close"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        s = s.dropna()
        if s.empty:
            return pd.Series(dtype=float)

        s.index = pd.to_datetime(s.index)
        s = s * scale

        # Sanity-check: a government bond yield should be between -5% and 25%
        s = s[(s > -5) & (s < 25)]
        return s

    except Exception as exc:
        logger.warning(f"Yahoo fetch failed for {ticker}: {exc}")
        return pd.Series(dtype=float)


# ── Master fetch ─────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_yields(country: str, start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch government bond yields ≥ 1Y for a given country.

    Source priority:
      1. FRED  — most reliable, daily, official
      2. Yahoo Finance — fallback

    Parameters
    ----------
    country : str
    start, end : str  — "YYYY-MM-DD"

    Returns
    -------
    df : pd.DataFrame
        index   = DatetimeIndex (daily)
        columns = tenor labels that were successfully fetched
        values  = yield in plain % (e.g. 4.05 = 4.05 %)
    missing : list[str]
        Tenor labels for which NO source returned data.
    """
    fred_map  = FRED_SERIES.get(country, {})
    yahoo_map = YAHOO_TICKERS.get(country, {})
    scale_map = YAHOO_SCALE.get(country, {})

    frames:  dict[str, pd.Series] = {}
    sources: dict[str, str]       = {}   # tenor → source used
    missing: list[str]            = []

    # Cache Yahoo downloads to avoid re-fetching the same ticker
    yahoo_cache: dict[str, pd.Series] = {}

    for tenor in ALL_TENORS:
        series = pd.Series(dtype=float)

        # ── 1. Try FRED ──────────────────────────────────────
        fred_id = fred_map.get(tenor)
        if fred_id:
            series = _fetch_fred_series(fred_id, start, end)
            if not series.empty:
                sources[tenor] = "FRED"
                logger.info(f"[{country}] {tenor} ← FRED ({fred_id})")

        # ── 2. Fallback: Yahoo Finance ────────────────────────
        if series.empty:
            ticker = yahoo_map.get(tenor)
            if ticker:
                if ticker not in yahoo_cache:
                    scale = scale_map.get(tenor, 1.0)
                    yahoo_cache[ticker] = _fetch_yahoo_series(ticker, scale, start, end)
                series = yahoo_cache[ticker].copy()
                if not series.empty:
                    sources[tenor] = "Yahoo"
                    logger.info(f"[{country}] {tenor} ← Yahoo ({ticker})")

        # ── Result ────────────────────────────────────────────
        if not series.empty:
            series.name = tenor
            frames[tenor] = series
        else:
            missing.append(tenor)
            logger.warning(f"[{country}] {tenor}: no data from any source.")

    if not frames:
        return pd.DataFrame(), ALL_TENORS

    df = pd.concat(frames.values(), axis=1)
    ordered = [t for t in ALL_TENORS if t in df.columns]
    df = df[ordered]
    df.dropna(how="all", inplace=True)

    return df, missing


def validate_series(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame [{context}]")
        return pd.DataFrame()
    return df


# ============================================================
# BLOCK 4 — CALCULATION LAYER
# ============================================================

def build_yield_curve(
    yields_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (today, 1m_ago, 1d_ago, 1w_ago) cross-sectional curves."""
    if yields_df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty

    curve_today = yields_df.iloc[-1].dropna()
    last_date   = yields_df.index[-1]

    def _nearest(offset_days: int) -> pd.Series:
        target = last_date - pd.DateOffset(days=offset_days)
        idx    = yields_df.index.get_indexer([target], method="nearest")[0]
        return yields_df.iloc[idx].dropna()

    curve_1d = yields_df.iloc[-2].dropna() if len(yields_df) >= 2 else pd.Series(dtype=float)
    curve_1w = _nearest(7)
    curve_1m = _nearest(30)

    return curve_today, curve_1m, curve_1d, curve_1w


def compute_curve_change_table(yields_df: pd.DataFrame) -> pd.DataFrame:
    """
    Level (plain %) | Δ1D | Δ1W | Δ1M (all in basis points).
    1 bps = 0.01 percentage point.
    """
    if yields_df.empty:
        return pd.DataFrame()

    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)
    rows = []

    for tenor in curve_today.index:
        level = curve_today.get(tenor, np.nan)

        def _bps(ref: pd.Series) -> float:
            if ref.empty:
                return np.nan
            ref_val = ref.get(tenor, np.nan)
            if np.isnan(level) or np.isnan(ref_val):
                return np.nan
            return round((level - ref_val) * 100, 1)   # % → bps

        rows.append({
            "Tenor":     tenor,
            "Level (%)": round(level, 3),
            "Δ1D (bps)": _bps(curve_1d),
            "Δ1W (bps)": _bps(curve_1w),
            "Δ1M (bps)": _bps(curve_1m),
        })

    return pd.DataFrame(rows).set_index("Tenor")


# ============================================================
# BLOCK 5 — VISUALIZATION LAYER
# ============================================================

_MS_THEME = {
    "font_family": "Open Sans, sans-serif",
    "font_color":  "#333333",
    "bg_color":    MS_BG,
    "plot_bg":     "#FFFFFF",
    "grid_color":  "#E8E8E8",
}


def apply_ms_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        font=dict(family=_MS_THEME["font_family"], color=_MS_THEME["font_color"], size=12),
        paper_bgcolor=_MS_THEME["bg_color"],
        plot_bgcolor=_MS_THEME["plot_bg"],
        title=dict(
            text=title,
            font=dict(color=MS_BLUE, size=15),
            x=0, xanchor="left",
        ),
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis=dict(gridcolor=_MS_THEME["grid_color"], linecolor="#cccccc"),
        yaxis=dict(gridcolor=_MS_THEME["grid_color"], linecolor="#cccccc"),
    )
    return fig


def plot_yield_curve(
    curve_today: pd.Series,
    curve_1m: pd.Series,
    country: str,
    label_today: str = "Today",
    label_1m: str = "1M Ago",
) -> go.Figure:
    fig = go.Figure()

    if not curve_1m.empty:
        common = [t for t in curve_1m.index if t in curve_today.index]
        fig.add_trace(go.Scatter(
            x=common,
            y=[curve_1m[t] for t in common],
            mode="lines+markers",
            name=label_1m,
            line=dict(color=MS_GREY, width=1.5, dash="dot"),
            marker=dict(size=5, color=MS_GREY),
            hovertemplate="%{x}: %{y:.3f}%<extra>" + label_1m + "</extra>",
        ))

    if not curve_today.empty:
        fig.add_trace(go.Scatter(
            x=curve_today.index.tolist(),
            y=curve_today.values.tolist(),
            mode="lines+markers",
            name=label_today,
            line=dict(color=MS_BLUE, width=2.5),
            marker=dict(size=7, color=MS_BLUE, symbol="circle"),
            hovertemplate="%{x}: %{y:.3f}%<extra>" + label_today + "</extra>",
        ))

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(title="Tenor", showgrid=True, gridcolor="#E8E8E8"),
        yaxis=dict(
            title="Yield (%)",
            showgrid=True, gridcolor="#E8E8E8",
            tickformat=".2f", ticksuffix="%",
        ),
        height=420,
        hovermode="x unified",
    )
    return apply_ms_theme(
        fig,
        title=f"{COUNTRY_LABELS.get(country, country)} Government Yield Curve",
    )


def plot_yield_history(
    yields_df: pd.DataFrame,
    tenors: list[str],
    country: str,
) -> go.Figure:
    color_ramp = [
        MS_DARK, MS_BLUE, "#2980b9", "#5dade2", MS_GREY,
        "#e67e22", "#8e44ad", "#16a085", MS_RED, "#f39c12",
    ]
    fig = go.Figure()

    for i, tenor in enumerate(tenors):
        if tenor not in yields_df.columns:
            continue
        s = yields_df[tenor].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=tenor,
            line=dict(color=color_ramp[i % len(color_ramp)], width=1.8),
            hovertemplate=f"{tenor}: %{{y:.3f}}%<extra></extra>",
        ))

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#E8E8E8"),
        yaxis=dict(
            title="Yield (%)",
            showgrid=True, gridcolor="#E8E8E8",
            tickformat=".2f", ticksuffix="%",
        ),
        height=380,
        hovermode="x unified",
    )
    return apply_ms_theme(
        fig,
        title=f"{COUNTRY_LABELS.get(country, country)} — Historical Yields",
    )


# ============================================================
# BLOCK 6 — UI COMPONENTS
# ============================================================

def render_header() -> None:
    st.markdown(f"""
        <div style="background:#fff;padding:16px 28px;
                    border-bottom:3px solid {MS_BLUE};border-radius:6px;
                    display:flex;justify-content:space-between;
                    align-items:center;margin-bottom:18px;">
            <div>
                <p style="margin:0;font-size:11px;color:{MS_GREY};font-weight:600;
                          text-transform:uppercase;letter-spacing:1px;">
                    Morgan Stanley — Rates Sales
                </p>
                <h1 style="margin:0;font-size:26px;font-weight:700;color:{MS_BLUE};">
                    Interest Rates Dashboard
                </h1>
            </div>
            <div style="text-align:right;">
                <p style="margin:0;font-size:12px;color:{MS_GREY};">
                    {datetime.now().strftime('%A, %B %d, %Y &nbsp;|&nbsp; %H:%M')} ET
                </p>
                <p style="margin:0;font-size:11px;color:{MS_GREY};">
                    Sources: FRED · Yahoo Finance
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> dict:
    st.sidebar.markdown(
        f"<p style='font-size:13px;font-weight:700;color:{MS_BLUE};"
        f"text-transform:uppercase;letter-spacing:0.8px;'>Dashboard Settings</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 🌍 Country")
    country_display = {COUNTRY_LABELS[c]: c for c in COUNTRIES}
    selected_label  = st.sidebar.selectbox(
        "Select country", list(country_display.keys()), index=0
    )
    selected_country = country_display[selected_label]

    st.sidebar.markdown("### 📅 Date Range")
    end_default   = datetime.today()
    start_default = end_default - timedelta(days=3 * 365)

    start_date = st.sidebar.date_input("Start date", start_default)
    end_date   = st.sidebar.date_input("End date",   end_default)

    if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
        st.sidebar.error("End date must be after start date.")

    st.sidebar.markdown("---")

    # Optional FRED API key input
    st.sidebar.markdown("### 🔑 FRED API Key *(optional)*")
    fred_key_input = st.sidebar.text_input(
        "Enter FRED API key for richer data",
        value="",
        type="password",
        help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html",
    )
    if fred_key_input.strip():
        st.session_state["FRED_API_KEY"] = fred_key_input.strip()

    st.sidebar.markdown("---")
    refresh = st.sidebar.button("🔄 Update All Data", use_container_width=True)
    st.sidebar.markdown(
        f"<p style='font-size:11px;color:{MS_GREY};margin-top:8px;'>"
        f"Cache auto-refreshes every 15 min.</p>",
        unsafe_allow_html=True,
    )

    return {
        "country":    selected_country,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "refresh":    refresh,
    }


def render_error_banner(message: str) -> None:
    st.error(f"⚠️ {message}")


def _style_change_table(df: pd.DataFrame):
    def _colour(val):
        if pd.isna(val):
            return ""
        return (
            f"color:{MS_RED};font-weight:600;" if val > 0
            else f"color:{MS_GREEN};font-weight:600;" if val < 0
            else ""
        )

    return (
        df.style
        .format({
            "Level (%)":  "{:.3f}%",
            "Δ1D (bps)":  lambda v: f"{v:+.1f}" if not pd.isna(v) else "—",
            "Δ1W (bps)":  lambda v: f"{v:+.1f}" if not pd.isna(v) else "—",
            "Δ1M (bps)":  lambda v: f"{v:+.1f}" if not pd.isna(v) else "—",
        })
        .map(_colour, subset=["Δ1D (bps)", "Δ1W (bps)", "Δ1M (bps)"])
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", MS_BLUE), ("color", "white"),
                ("font-weight", "600"), ("font-size", "12px"), ("text-align", "center"),
            ]},
            {"selector": "td", "props": [("font-size", "13px"), ("padding", "6px 12px")]},
        ])
    )


# ============================================================
# BLOCK 7 — TAB MODULES
# ============================================================

def tab_yield_curve(data: dict, params: dict) -> None:
    """
    Tab 1 — Yield Curve (multi-country, ≥ 1Y, multi-source)
    ─────────────────────────────────────────────────────────
    1. Data-source info banner
    2. Missing-tenor warning (if any)
    3. KPI strip — level + Δ1D per tenor
    4. Curve chart (Today vs 1M Ago) + Change table
    5. 2Y/10Y spread card + inversion warning
    6. Historical time-series
    """
    country    = params.get("country", "US")
    yields_df  = data.get("yields", pd.DataFrame())
    missing    = data.get("missing", [])
    flag_label = COUNTRY_LABELS.get(country, country)

    # ── Guard ────────────────────────────────────────────────
    if yields_df.empty:
        render_error_banner(
            f"No yield data available for **{flag_label}**. "
            "Neither FRED nor Yahoo Finance returned valid data for any maturity. "
            "Check your internet connection or try a different date range."
        )
        return

    # ── 1. Source info ───────────────────────────────────────
    available_cols = yields_df.columns.tolist()
    fred_available  = bool(FRED_SERIES.get(country))
    fred_key_set    = st.session_state.get("FRED_API_KEY", _FRED_KEY) != "YOUR_FRED_API_KEY"

    source_note = "FRED + Yahoo Finance" if (fred_available and fred_key_set) else "Yahoo Finance"
    last_date   = yields_df.index[-1].strftime("%B %d, %Y")

    st.markdown(
        f"<p style='color:{MS_GREY};font-size:13px;margin-bottom:6px;'>"
        f"<strong>{flag_label}</strong> &nbsp;·&nbsp; "
        f"Source: <strong>{source_note}</strong> &nbsp;·&nbsp; "
        f"Last session: <strong>{last_date}</strong></p>",
        unsafe_allow_html=True,
    )

    # ── 2. Missing tenor warning ──────────────────────────────
    truly_missing = [t for t in missing if t not in ["1M", "3M", "6M"]]
    if truly_missing:
        st.warning(
            f"⚠️ The following maturities could not be retrieved for {flag_label}: "
            f"**{', '.join(truly_missing)}**. "
            "They are excluded from all charts and tables.",
            icon="📭",
        )

    # ── 3. Derived data ───────────────────────────────────────
    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)
    change_table = compute_curve_change_table(yields_df)

    # ── 4. KPI strip (max 8 per row) ──────────────────────────
    available_tenors = curve_today.index.tolist()
    if available_tenors:
        for row_start in range(0, len(available_tenors), 8):
            row_tenors = available_tenors[row_start: row_start + 8]
            cols = st.columns(len(row_tenors))
            for col, tenor in zip(cols, row_tenors):
                level   = curve_today.get(tenor, np.nan)
                d1d_raw = (
                    change_table.loc[tenor, "Δ1D (bps)"]
                    if tenor in change_table.index else np.nan
                )
                delta_str = (
                    f"{d1d_raw:+.1f} bps"
                    if (d1d_raw is not None and not np.isnan(d1d_raw))
                    else None
                )
                with col:
                    st.metric(
                        label=tenor,
                        value=f"{level:.3f}%" if not np.isnan(level) else "N/A",
                        delta=delta_str,
                    )
            if row_start + 8 < len(available_tenors):
                st.markdown("")

    st.markdown("---")

    # ── 5. Curve chart + Change table ─────────────────────────
    col_chart, col_table = st.columns([2.2, 1.2])

    with col_chart:
        fig_curve = plot_yield_curve(
            curve_today=curve_today,
            curve_1m=curve_1m,
            country=country,
            label_today=f"Today  ({last_date})",
            label_1m="1M Ago",
        )
        st.plotly_chart(fig_curve, use_container_width=True, key="ust_curve")

    with col_table:
        st.markdown(
            f"<p style='font-weight:700;color:{MS_BLUE};font-size:14px;"
            f"margin-bottom:10px;margin-top:6px;'>Yield Changes</p>",
            unsafe_allow_html=True,
        )
        if not change_table.empty:
            st.write(_style_change_table(change_table))
        else:
            st.info("Change data not available.")

    st.markdown("---")

    # ── 6. 2Y/10Y spread + inversion warning ──────────────────
    short_t, long_t = SPREAD_KEY.get(country, ("2Y", "10Y"))
    s_short = curve_today.get(short_t, np.nan)
    s_long  = curve_today.get(long_t,  np.nan)

    if not (np.isnan(s_short) or np.isnan(s_long)):
        spread_bps   = (s_long - s_short) * 100
        spread_color = MS_RED if spread_bps < 0 else MS_GREEN

        card_col, warn_col = st.columns([1, 3])
        with card_col:
            st.markdown(
                f"""
                <div style='padding:14px 18px;
                            border-left:5px solid {spread_color};
                            background:#fff;border-radius:6px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.08);'>
                    <span style='font-size:11px;color:{MS_GREY};font-weight:600;
                                 text-transform:uppercase;letter-spacing:0.5px;'>
                        {short_t}/{long_t} Spread
                    </span><br>
                    <span style='font-size:26px;font-weight:700;color:{spread_color};'>
                        {spread_bps:+.1f} bps
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with warn_col:
            if spread_bps < 0:
                st.warning(
                    f"**Yield Curve Inverted** ({flag_label}) — "
                    f"The {short_t}/{long_t} spread is **{spread_bps:.1f} bps**. "
                    "Historically a leading recession indicator. "
                    "Monitor duration and credit positioning closely.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"Yield curve **upward sloping** ({flag_label}). "
                    f"{short_t}/{long_t}: **{spread_bps:+.1f} bps**.",
                    icon="✅",
                )
    else:
        st.info(
            f"2Y and/or 10Y data not available for {flag_label}. "
            "Spread cannot be computed."
        )

    st.markdown("---")

    # ── 7. Historical time-series ──────────────────────────────
    st.markdown(
        f"<p style='font-weight:700;color:{MS_BLUE};font-size:14px;'>"
        f"Historical Yield Time Series</p>",
        unsafe_allow_html=True,
    )
    tenor_options   = yields_df.columns.tolist()
    default_hist    = [t for t in ["2Y", "10Y"] if t in tenor_options] or tenor_options[:2]
    selected_tenors = st.multiselect(
        "Select tenors to display:",
        options=tenor_options,
        default=default_hist,
        key="yc_tenor_select",
    )
    if selected_tenors:
        st.plotly_chart(
            plot_yield_history(yields_df, selected_tenors, country),
            use_container_width=True,
            key="ust_history",
        )
    else:
        st.info("Select at least one tenor to display the historical chart.")


# ── Placeholder tabs ─────────────────────────────────────────
def tab_spreads(data: dict, params: dict) -> None:
    st.info("🚧 Spread Analysis — coming next.")

def tab_macro(data: dict, params: dict) -> None:
    st.info("🚧 Macro & Central Banks — coming next.")

def tab_carry_roll(data: dict, params: dict) -> None:
    st.info("🚧 Carry & Roll-Down — coming next.")

def tab_scenarios(data: dict, params: dict) -> None:
    st.info("🚧 Scenario Analysis — coming next.")

def tab_news(data: dict, params: dict) -> None:
    st.info("🚧 Rates & Macro News — coming next.")


# ============================================================
# BLOCK 8 — MAIN
# ============================================================

def _load_all_data(params: dict) -> dict:
    """
    Centralised data loading.
    Injects session-state FRED key before fetching.
    """
    country = params["country"]

    # Allow sidebar-entered FRED key to propagate to module-level helper
    global _FRED_KEY
    if "FRED_API_KEY" in st.session_state:
        _FRED_KEY = st.session_state["FRED_API_KEY"]

    data: dict = {}
    with st.spinner(f"Loading {COUNTRY_LABELS.get(country, country)} yield data…"):
        df, missing = fetch_yields(
            country=country,
            start=params["start_date"],
            end=params["end_date"],
        )
        data["yields"]  = validate_series(df, context=f"{country} yields")
        data["missing"] = missing

    return data


def main() -> None:
    render_header()
    params = render_sidebar()

    if params.get("refresh"):
        st.cache_data.clear()

    data = _load_all_data(params)

    tabs = st.tabs([
        "📈 Yield Curve",
        "📊 Spreads",
        "🌍 Macro & Central Banks",
        "💰 Carry & Roll",
        "⚙️ Scenarios",
        "📰 News",
    ])

    with tabs[0]: tab_yield_curve(data, params)
    with tabs[1]: tab_spreads(data, params)
    with tabs[2]: tab_macro(data, params)
    with tabs[3]: tab_carry_roll(data, params)
    with tabs[4]: tab_scenarios(data, params)
    with tabs[5]: tab_news(data, params)


if __name__ == "__main__":
    main()
