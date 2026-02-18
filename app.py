# ============================================================
# app.py — Morgan Stanley | Rates Sales Dashboard
# Author  : Mathis de Looze
# Version : 2.0.0
# ============================================================


# ============================================================
# BLOCK 1 — CONFIGURATION
# ============================================================

import logging
from datetime import datetime, timedelta

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# ── Supported countries ──────────────────────────────────────
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

# ── All tenor labels in display order ───────────────────────
ALL_TENORS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# ── Yahoo Finance tickers per (country, tenor) ───────────────
# None = not available on Yahoo Finance for that maturity
YIELD_TICKERS: dict[str, dict[str, str | None]] = {
    "US": {
        "1M":  "^IRX",
        "3M":  "^IRX",
        "6M":  "^IRX",
        "1Y":  "^IRX",
        "2Y":  "^UST2Y",
        "3Y":  None,
        "5Y":  "^FVX",
        "7Y":  None,
        "10Y": "^TNX",
        "20Y": None,
        "30Y": "^TYX",
    },
    "UK": {
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "GBGB2YR=X",
        "3Y":  None,
        "5Y":  "GBGB5YR=X",
        "7Y":  None,
        "10Y": "GBGB10YR=X",
        "20Y": None,
        "30Y": "GBGB30YR=X",
    },
    "DE": {
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "DEDE2YR=X",
        "3Y":  None,
        "5Y":  "DEDE5YR=X",
        "7Y":  None,
        "10Y": "DEDE10YR=X",
        "20Y": None,
        "30Y": "DEDE30YR=X",
    },
    "FR": {
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "FRFR2YR=X",
        "3Y":  None,
        "5Y":  "FRFR5YR=X",
        "7Y":  None,
        "10Y": "FRFR10YR=X",
        "20Y": None,
        "30Y": "FRFR30YR=X",
    },
    "IT": {
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "ITIT2YR=X",
        "3Y":  None,
        "5Y":  "ITIT5YR=X",
        "7Y":  None,
        "10Y": "ITIT10YR=X",
        "20Y": None,
        "30Y": "ITIT30YR=X",
    },
    "EU": {
        # Euro area — use DE Bunds as core Euro benchmark
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "DEDE2YR=X",
        "3Y":  None,
        "5Y":  "DEDE5YR=X",
        "7Y":  None,
        "10Y": "DEDE10YR=X",
        "20Y": None,
        "30Y": "DEDE30YR=X",
    },
    "JP": {
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "JPJP2YR=X",
        "3Y":  None,
        "5Y":  "JPJP5YR=X",
        "7Y":  None,
        "10Y": "JPJP10YR=X",
        "20Y": None,
        "30Y": "JPJP30YR=X",
    },
    "CH": {
        "1M":  None, "3M": None, "6M": None, "1Y": None,
        "2Y":  "CHCH2YR=X",
        "3Y":  None,
        "5Y":  "CHCH5YR=X",
        "7Y":  None,
        "10Y": "CHCH10YR=X",
        "20Y": None,
        "30Y": "CHCH30YR=X",
    },
}

# ── Scale factors ────────────────────────────────────────────
# Multiply raw Yahoo Close value to obtain plain % (e.g. 4.05 = 4.05%)
# US legacy tickers (^IRX, ^FVX, ^TNX, ^TYX) are quoted as rate * 10
# All =X tickers and ^UST2Y are already in plain %
YIELD_SCALE: dict[str, dict[str, float]] = {
    "US": {
        "1M": 0.1, "3M": 0.1, "6M": 0.1, "1Y": 0.1,
        "2Y": 1.0,
        "3Y": 1.0,
        "5Y": 0.1,
        "7Y": 1.0,
        "10Y": 0.1,
        "20Y": 1.0,
        "30Y": 0.1,
    },
    **{
        country: {tenor: 1.0 for tenor in ALL_TENORS}
        for country in ["UK", "EU", "DE", "FR", "IT", "JP", "CH"]
    },
}

# ── Spread pair per country (short_tenor, long_tenor) ────────
SPREAD_KEY: dict[str, tuple[str, str]] = {
    country: ("2Y", "10Y") for country in COUNTRIES
}

RSS_FEEDS: dict[str, dict[str, str]] = {
    "Financial Times": {
        "Markets": "https://www.ft.com/markets?format=rss",
        "Economy": "https://www.ft.com/global-economy?format=rss",
    },
    "CNBC": {
        "Bonds":   "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "Economy": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    },
}


# ============================================================
# BLOCK 3 — DATA LAYER
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_treasury_yields_yf(
    country: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download government bond yields for one country from Yahoo Finance.

    Deduplicates fetches when multiple tenors share the same ticker
    (e.g. ^IRX covers US 1M/3M/6M/1Y).  Applies per-(country, tenor)
    scale so all returned values are in plain % (e.g. 4.05 = 4.05 %).

    Returns
    -------
    pd.DataFrame
        index   = DatetimeIndex (daily business days)
        columns = tenor labels available for that country
        values  = yield in %
        Fully-NaN rows are dropped.
    """
    tickers_map = YIELD_TICKERS.get(country, {})
    scale_map   = YIELD_SCALE.get(country, {})

    # Group tenors by ticker to avoid duplicate downloads
    ticker_to_tenors: dict[str, list[str]] = {}
    for tenor, ticker in tickers_map.items():
        if ticker is not None:
            ticker_to_tenors.setdefault(ticker, []).append(tenor)

    frames: dict[str, pd.Series] = {}

    for ticker, tenors in ticker_to_tenors.items():
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                logger.warning(f"[{country}] No data returned for {ticker}")
                continue

            # Flatten MultiIndex columns produced by newer yfinance versions
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if "Close" not in raw.columns:
                logger.warning(f"[{country}] 'Close' missing for {ticker}: {raw.columns.tolist()}")
                continue

            series = raw["Close"].squeeze()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            series.index = pd.to_datetime(series.index)

            for tenor in tenors:
                scale = scale_map.get(tenor, 1.0)
                s = series * scale
                s.name = tenor
                frames[tenor] = s

        except Exception as exc:
            logger.error(f"[{country}] Failed fetching {ticker}: {exc}")
            continue

    if not frames:
        logger.warning(f"[{country}] No yield data retrieved.")
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    ordered = [t for t in ALL_TENORS if t in df.columns]
    df = df[ordered]
    df.dropna(how="all", inplace=True)

    return df


def validate_series(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Return df unchanged if valid, else log and return empty DataFrame."""
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
    """
    Extract cross-sectional yield curves at four reference dates.

    Returns
    -------
    (curve_today, curve_1m, curve_1d, curve_1w)
    Each Series indexed by tenor label, values in plain %.
    """
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
    Compute Level | Δ1D | Δ1W | Δ1M per tenor.

    Level   → plain %  (e.g. 4.053)
    Deltas  → basis points  (1 bps = 0.01 percentage point)
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
            # Both values are in %; multiply difference by 100 to get bps
            return (level - ref_val) * 100

        rows.append({
            "Tenor":     tenor,
            "Level (%)": round(level, 3),
            "Δ1D (bps)": round(_bps(curve_1d), 1),
            "Δ1W (bps)": round(_bps(curve_1w), 1),
            "Δ1M (bps)": round(_bps(curve_1m), 1),
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
    """Apply standardised Morgan Stanley visual theme to any Plotly figure."""
    fig.update_layout(
        font=dict(family=_MS_THEME["font_family"], color=_MS_THEME["font_color"], size=12),
        paper_bgcolor=_MS_THEME["bg_color"],
        plot_bgcolor=_MS_THEME["plot_bg"],
        title=dict(
            text=title,
            font=dict(color=MS_BLUE, size=15, family="Open Sans, sans-serif"),
            x=0,
            xanchor="left",
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
    title: str = "",
    label_today: str = "Today",
    label_1m: str = "1M Ago",
) -> go.Figure:
    """Cross-sectional yield curve: Today (MS Blue) vs 1M Ago (Grey dashed)."""
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
            showgrid=True,
            gridcolor="#E8E8E8",
            tickformat=".2f",
            ticksuffix="%",
        ),
        height=420,
        hovermode="x unified",
    )

    auto_title = title or f"{COUNTRY_LABELS.get(country, country)} Government Yield Curve"
    return apply_ms_theme(fig, title=auto_title)


def plot_yield_history(
    yields_df: pd.DataFrame,
    tenors: list[str],
    country: str,
    title: str = "",
) -> go.Figure:
    """Time-series of multiple tenor yields over the selected date range."""
    color_ramp = [MS_DARK, MS_BLUE, "#2980b9", "#5dade2", MS_GREY,
                  "#e67e22", "#8e44ad", "#16a085", "#c0392b", "#f39c12", "#1abc9c"]
    fig = go.Figure()

    for i, tenor in enumerate(tenors):
        if tenor not in yields_df.columns:
            continue
        series = yields_df[tenor].dropna()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=tenor,
            line=dict(color=color_ramp[i % len(color_ramp)], width=1.8),
            hovertemplate=f"{tenor}: %{{y:.3f}}%<extra></extra>",
        ))

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#E8E8E8"),
        yaxis=dict(
            title="Yield (%)",
            showgrid=True,
            gridcolor="#E8E8E8",
            tickformat=".2f",
            ticksuffix="%",
        ),
        height=380,
        hovermode="x unified",
    )

    auto_title = title or f"{COUNTRY_LABELS.get(country, country)} — Historical Yields"
    return apply_ms_theme(fig, title=auto_title)


# ============================================================
# BLOCK 6 — UI COMPONENTS
# ============================================================

def render_header() -> None:
    """Top banner: desk name, title, live timestamp."""
    st.markdown(f"""
        <div style="
            background-color:#ffffff;
            padding:16px 28px;
            border-bottom:3px solid {MS_BLUE};
            border-radius:6px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:18px;
        ">
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
                <p style="margin:0;font-size:11px;color:{MS_GREY};">Data: Yahoo Finance</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> dict:
    """
    Render all sidebar widgets.

    Returns
    -------
    dict
        country, start_date, end_date, refresh
    """
    st.sidebar.markdown(
        f"<p style='font-size:13px;font-weight:700;color:{MS_BLUE};"
        f"text-transform:uppercase;letter-spacing:0.8px;'>Dashboard Settings</p>",
        unsafe_allow_html=True,
    )

    # ── Country ─────────────────────────────────────────────
    st.sidebar.markdown("### 🌍 Country")
    country_display = {COUNTRY_LABELS[c]: c for c in COUNTRIES}
    selected_label  = st.sidebar.selectbox(
        "Select country",
        list(country_display.keys()),
        index=0,
    )
    selected_country = country_display[selected_label]

    # ── Date range ──────────────────────────────────────────
    st.sidebar.markdown("### 📅 Date Range")
    end_default   = datetime.today()
    start_default = end_default - timedelta(days=3 * 365)

    start_date = st.sidebar.date_input("Start date", start_default)
    end_date   = st.sidebar.date_input("End date",   end_default)

    if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
        st.sidebar.error("End date must be after start date.")

    # ── Refresh ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    refresh = st.sidebar.button("🔄 Update All Data", use_container_width=True)
    st.sidebar.markdown(
        f"<p style='font-size:11px;color:{MS_GREY};margin-top:8px;'>"
        f"Cache auto-refreshes every 15 min.<br>"
        f"Click above to force an update.</p>",
        unsafe_allow_html=True,
    )

    return {
        "country":    selected_country,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "refresh":    refresh,
    }


def render_error_banner(message: str) -> None:
    """Unified error display across all tabs."""
    st.error(f"⚠️ {message}")


def _style_change_table(df: pd.DataFrame):
    """Colour-coded styling for the yield change table."""
    def _colour_bps(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return f"color:{MS_RED};font-weight:600;"
        if val < 0:
            return f"color:{MS_GREEN};font-weight:600;"
        return ""

    return (
        df.style
        .format({
            "Level (%)":  "{:.3f}%",
            "Δ1D (bps)":  "{:+.1f}",
            "Δ1W (bps)":  "{:+.1f}",
            "Δ1M (bps)":  "{:+.1f}",
        })
        .map(_colour_bps, subset=["Δ1D (bps)", "Δ1W (bps)", "Δ1M (bps)"])
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", MS_BLUE),
                ("color", "white"),
                ("font-weight", "600"),
                ("font-size", "12px"),
                ("text-align", "center"),
            ]},
            {"selector": "td", "props": [
                ("font-size", "13px"),
                ("padding", "6px 12px"),
            ]},
        ])
    )


# ============================================================
# BLOCK 7 — TAB MODULES
# ============================================================

def tab_yield_curve(data: dict, params: dict) -> None:
    """
    Tab 1 — Yield Curve (multi-country)
    ─────────────────────────────────────
    1. KPI strip   — current level per tenor + Δ1D in bps
    2. Curve chart — Today vs 1M Ago
    3. Change table — Level | Δ1D | Δ1W | Δ1M
    4. 2Y/10Y spread card + inversion warning
    5. Historical time-series (user-selected tenors)
    """
    country    = params.get("country", "US")
    yields_df  = data.get("yields", pd.DataFrame())
    flag_label = COUNTRY_LABELS.get(country, country)

    # ── Guard ────────────────────────────────────────────────
    if yields_df.empty:
        render_error_banner(
            f"No yield data available for {flag_label}. "
            "Several maturities may not be available on Yahoo Finance for this country. "
            "Try a different country or date range."
        )
        return

    # ── Derived data ─────────────────────────────────────────
    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)
    change_table  = compute_curve_change_table(yields_df)
    last_date     = yields_df.index[-1].strftime("%B %d, %Y")

    # ── Sub-header ───────────────────────────────────────────
    st.markdown(
        f"<p style='color:{MS_GREY};font-size:13px;margin-bottom:6px;'>"
        f"<strong>{flag_label}</strong> — "
        f"Last available trading session: <strong>{last_date}</strong></p>",
        unsafe_allow_html=True,
    )

    # ── 1. KPI strip (max 8 per row) ─────────────────────────
    available_tenors = curve_today.index.tolist()
    if available_tenors:
        for row_start in range(0, len(available_tenors), 8):
            row_tenors = available_tenors[row_start : row_start + 8]
            kpi_cols   = st.columns(len(row_tenors))
            for col, tenor in zip(kpi_cols, row_tenors):
                level   = curve_today.get(tenor, np.nan)
                d1d_raw = (
                    change_table.loc[tenor, "Δ1D (bps)"]
                    if tenor in change_table.index else np.nan
                )
                delta_str = f"{d1d_raw:+.1f} bps" if not np.isnan(d1d_raw) else None
                with col:
                    st.metric(
                        label=tenor,
                        value=f"{level:.3f}%" if not np.isnan(level) else "N/A",
                        delta=delta_str,
                    )
            if row_start + 8 < len(available_tenors):
                st.markdown("")

    st.markdown("---")

    # ── 2 & 3. Curve chart + Change table ────────────────────
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

    # ── 4. 2Y/10Y spread + inversion warning ─────────────────
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
                <div style='
                    padding:14px 18px;
                    border-left:5px solid {spread_color};
                    background:#ffffff;
                    border-radius:6px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);
                '>
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
                    f"The {short_t}/{long_t} spread stands at **{spread_bps:.1f} bps**. "
                    "An inverted curve has historically preceded recessions. "
                    "Monitor duration and credit positioning closely.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"Yield curve is **upward sloping** ({flag_label}). "
                    f"{short_t}/{long_t} spread: **{spread_bps:+.1f} bps**.",
                    icon="✅",
                )
    else:
        st.info(
            f"2Y and/or 10Y data not available for {flag_label} on Yahoo Finance. "
            "Spread cannot be computed."
        )

    st.markdown("---")

    # ── 5. Historical time-series ─────────────────────────────
    st.markdown(
        f"<p style='font-weight:700;color:{MS_BLUE};font-size:14px;'>"
        f"Historical Yield Time Series</p>",
        unsafe_allow_html=True,
    )

    tenor_options   = yields_df.columns.tolist()
    default_hist    = [t for t in ["2Y", "10Y"] if t in tenor_options] or tenor_options[:2]
    selected_tenors = st.multiselect(
        label="Select tenors to display:",
        options=tenor_options,
        default=default_hist,
        key="yc_tenor_select",
    )

    if selected_tenors:
        fig_history = plot_yield_history(
            yields_df=yields_df,
            tenors=selected_tenors,
            country=country,
        )
        st.plotly_chart(fig_history, use_container_width=True, key="ust_history")
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
    """Centralised data loading — one call per session / refresh."""
    data: dict = {}
    country    = params["country"]

    with st.spinner(f"Loading {COUNTRY_LABELS.get(country, country)} yield data…"):
        raw = fetch_treasury_yields_yf(
            country=country,
            start=params["start_date"],
            end=params["end_date"],
        )
        data["yields"] = validate_series(raw, context=f"{country} yields")

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
