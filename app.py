# ============================================================
# app.py — Morgan Stanley | Rates Sales Dashboard
# Author  : Mathis de Looze
# Version : 1.0.0
# ============================================================


# ============================================================
# BLOCK 1 — CONFIGURATION
# ============================================================

# --- Imports ------------------------------------------------
import logging
from datetime import datetime, timedelta

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# --- Page Config --------------------------------------------
st.set_page_config(
    page_title="MS | Rates Sales Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logger -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global CSS Injection -----------------------------------
def _inject_css() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #00539b;
            border-radius: 6px;
            padding: 12px 16px;
        }
        div[data-testid="metric-container"] label {
            font-size: 12px !important;
            color: #6c757d !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 20px !important;
            font-weight: 700 !important;
            color: #002b5c !important;
        }
        thead tr th {
            background-color: #00539b !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

_inject_css()


# ============================================================
# BLOCK 2 — CONSTANTS & TICKERS
# ============================================================

# --- Color Palette ------------------------------------------
MS_BLUE  = "#00539b"
MS_DARK  = "#002b5c"
MS_GREY  = "#6c757d"
MS_RED   = "#c0392b"
MS_GREEN = "#27ae60"
MS_BG    = "#F8F9FA"

# --- UST Tickers & Scaling ----------------------------------
# ^IRX, ^FVX, ^TNX, ^TYX : Yahoo quotes as rate * 10 → scale = 0.1
# ^UST2Y                  : Yahoo quotes directly in %  → scale = 1.0
UST_TICKERS: dict[str, str] = {
    "3M":  "^IRX",
    "2Y":  "^UST2Y",
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

UST_SCALE: dict[str, float] = {
    "3M":  0.1,
    "2Y":  1.0,
    "5Y":  0.1,
    "10Y": 0.1,
    "30Y": 0.1,
}

# --- Default Date Range -------------------------------------
_TODAY      = datetime.today()
_DEFAULT_START = (_TODAY - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
_DEFAULT_END   = _TODAY.strftime("%Y-%m-%d")

# --- News Feeds ---------------------------------------------
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
def fetch_treasury_yields_yf(start: str, end: str) -> pd.DataFrame:
    """
    Download UST yields from Yahoo Finance for all defined tenors.

    Normalises each ticker to plain percentage:
      - ^IRX, ^FVX, ^TNX, ^TYX : divided by 10
      - ^UST2Y                  : kept as-is

    Returns
    -------
    pd.DataFrame
        Columns = tenor labels ("3M", "2Y", …), index = DatetimeIndex.
        Rows where ALL columns are NaN are dropped.
    """
    frames: list[pd.Series] = []

    for tenor, ticker in UST_TICKERS.items():
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )

            if raw.empty:
                logger.warning(f"No data for {ticker} ({tenor})")
                continue

            # yfinance may return MultiIndex columns — flatten
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if "Close" not in raw.columns:
                logger.warning(f"'Close' column missing for {ticker}: {raw.columns.tolist()}")
                continue

            series = raw["Close"].squeeze()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            series = series * UST_SCALE.get(tenor, 1.0)
            series.name = tenor
            frames.append(series)

        except Exception as exc:
            logger.error(f"Failed to fetch {ticker} ({tenor}): {exc}")
            continue

    if not frames:
        logger.error("All UST yield fetches failed.")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)

    # Enforce tenor order
    ordered = [t for t in UST_TICKERS.keys() if t in df.columns]
    df = df[ordered]
    df.dropna(how="all", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df


def validate_series(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Return df if valid, else log warning and return empty DataFrame."""
    if df is None or df.empty:
        logger.warning(f"Empty data returned [{context}]")
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
    tuple : (curve_today, curve_1m_ago, curve_1d_ago, curve_1w_ago)
        Each pd.Series is indexed by tenor label, values in %.
        Returns empty Series for missing reference points.
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
    Compute Level | Δ1D | Δ1W | Δ1M table (changes in basis points).

    Returns
    -------
    pd.DataFrame : index = tenor, columns = ["Level (%)", "Δ1D (bps)", "Δ1W (bps)", "Δ1M (bps)"]
    """
    if yields_df.empty:
        return pd.DataFrame()

    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)

    rows = []
    for tenor in curve_today.index:
        level = curve_today.get(tenor, np.nan)

        def _delta(ref: pd.Series) -> float:
            if ref.empty:
                return np.nan
            ref_val = ref.get(tenor, np.nan)
            return (level - ref_val) * 100  # → bps

        rows.append({
            "Tenor":     tenor,
            "Level (%)": round(level, 3),
            "Δ1D (bps)": round(_delta(curve_1d), 1),
            "Δ1W (bps)": round(_delta(curve_1w), 1),
            "Δ1M (bps)": round(_delta(curve_1m), 1),
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
        font=dict(
            family=_MS_THEME["font_family"],
            color=_MS_THEME["font_color"],
            size=12,
        ),
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
    title: str = "US Treasury Yield Curve",
    label_today: str = "Today",
    label_1m: str = "1M Ago",
) -> go.Figure:
    """
    Cross-sectional yield curve: Today (MS Blue) vs 1M Ago (Grey dashed).
    """
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
        xaxis=dict(title="Tenor", showgrid=True),
        yaxis=dict(title="Yield (%)", showgrid=True, tickformat=".2f", ticksuffix="%"),
        height=420,
        hovermode="x unified",
    )
    return apply_ms_theme(fig, title=title)


def plot_yield_history(
    yields_df: pd.DataFrame,
    tenors: list[str],
    title: str = "Yield History — Selected Tenors",
) -> go.Figure:
    """
    Time-series of multiple tenor yields over the selected date range.
    """
    color_ramp = [MS_DARK, MS_BLUE, "#2980b9", "#5dade2", MS_GREY]
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
        xaxis=dict(showgrid=True),
        yaxis=dict(title="Yield (%)", showgrid=True, tickformat=".2f", ticksuffix="%"),
        height=380,
        hovermode="x unified",
    )
    return apply_ms_theme(fig, title=title)


# ============================================================
# BLOCK 6 — UI COMPONENTS
# ============================================================

def render_header() -> None:
    """Top banner: desk name, dashboard title, live timestamp."""
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
                <p style="margin:0; font-size:11px; color:{MS_GREY};
                          font-weight:600; text-transform:uppercase;
                          letter-spacing:1px;">
                    Morgan Stanley — Rates Sales
                </p>
                <h1 style="margin:0; font-size:26px; font-weight:700; color:{MS_BLUE};">
                    Interest Rates Dashboard
                </h1>
            </div>
            <div style="text-align:right;">
                <p style="margin:0; font-size:12px; color:{MS_GREY};">
                    {datetime.now().strftime('%A, %B %d, %Y &nbsp;|&nbsp; %H:%M')} ET
                </p>
                <p style="margin:0; font-size:11px; color:{MS_GREY};">
                    Data: Yahoo Finance
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> dict:
    """
    Render all sidebar widgets.

    Returns
    -------
    dict with keys:
        start_date, end_date, refresh
    """
    st.sidebar.markdown(
        f"<p style='font-size:13px; font-weight:700; color:{MS_BLUE}; "
        f"text-transform:uppercase; letter-spacing:0.8px;'>Dashboard Settings</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 📅 Date Range")
    end_default   = datetime.today()
    start_default = end_default - timedelta(days=3 * 365)

    start_date = st.sidebar.date_input("Start date", start_default)
    end_date   = st.sidebar.date_input("End date",   end_default)

    if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
        st.sidebar.error("End date must be after start date.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Refresh")
    refresh = st.sidebar.button("Update All Data", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<p style='font-size:11px; color:{MS_GREY};'>"
        f"Data refreshes automatically every 15 minutes.<br>"
        f"Click <b>Update</b> to force a refresh.</p>",
        unsafe_allow_html=True,
    )

    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "refresh":    refresh,
    }


def render_error_banner(message: str) -> None:
    """Unified error display."""
    st.error(f"⚠️ {message}")


# ============================================================
# BLOCK 7 — TAB MODULES
# ============================================================

def tab_yield_curve(data: dict, params: dict) -> None:
    """
    Tab 1 — Yield Curve
    ─────────────────────────────────────────────
    Layout:
      1. KPI strip  — current yield per tenor + Δ1D
      2. Curve chart (Today vs 1M Ago) + Change table side-by-side
      3. 2s10s spread card + inversion warning
      4. Historical time-series (user-selected tenors)
    """
    yields_df = data.get("yields", pd.DataFrame())

    # ── Guard ────────────────────────────────────────────────
    if yields_df.empty:
        render_error_banner(
            "Treasury yield data is unavailable. "
            "Check your internet connection or click 'Update All Data'."
        )
        return

    # ── Derived data ─────────────────────────────────────────
    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)
    change_table = compute_curve_change_table(yields_df)
    last_date    = yields_df.index[-1].strftime("%B %d, %Y")

    # ── Sub-header ───────────────────────────────────────────
    st.markdown(
        f"<p style='color:{MS_GREY}; font-size:13px; margin-bottom:6px;'>"
        f"Last available trading session: <strong>{last_date}</strong></p>",
        unsafe_allow_html=True,
    )

    # ── 1. KPI strip ─────────────────────────────────────────
    available_tenors = curve_today.index.tolist()
    if available_tenors:
        kpi_cols = st.columns(len(available_tenors))
        for col, tenor in zip(kpi_cols, available_tenors):
            level   = curve_today.get(tenor, np.nan)
            d1d_raw = change_table.loc[tenor, "Δ1D (bps)"] if tenor in change_table.index else np.nan
            delta_str = f"{d1d_raw:+.1f} bps" if not np.isnan(d1d_raw) else None
            with col:
                st.metric(
                    label=f"{tenor} UST",
                    value=f"{level:.3f}%" if not np.isnan(level) else "N/A",
                    delta=delta_str,
                )

    st.markdown("---")

    # ── 2. Curve chart + Change table ────────────────────────
    col_chart, col_table = st.columns([2.2, 1.2])

    with col_chart:
        fig_curve = plot_yield_curve(
            curve_today=curve_today,
            curve_1m=curve_1m,
            title="US Treasury Yield Curve",
            label_today=f"Today  ({last_date})",
            label_1m="1M Ago",
        )
        st.plotly_chart(fig_curve, use_container_width=True, key="ust_curve")

    with col_table:
        st.markdown(
            f"<p style='font-weight:700; color:{MS_BLUE}; font-size:14px; "
            f"margin-bottom:10px; margin-top:6px;'>Yield Changes</p>",
            unsafe_allow_html=True,
        )

        if not change_table.empty:
            def _colour_bps(val):
                if pd.isna(val):
                    return ""
                if val > 0:
                    return f"color:{MS_RED}; font-weight:600;"
                if val < 0:
                    return f"color:{MS_GREEN}; font-weight:600;"
                return ""

            styled = (
                change_table.style
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
            st.write(styled)
        else:
            st.info("Change data not available.")

    st.markdown("---")

    # ── 3. 2s10s spread + inversion warning ──────────────────
    s2  = curve_today.get("2Y", np.nan)
    s10 = curve_today.get("10Y", np.nan)

    if not np.isnan(s2) and not np.isnan(s10):
        spread_2s10s  = (s10 - s2) * 100   # bps
        spread_color  = MS_RED if spread_2s10s < 0 else MS_GREEN

        card_col, warn_col = st.columns([1, 3])

        with card_col:
            st.markdown(
                f"""
                <div style='
                    padding:14px 18px;
                    border-left:5px solid {spread_color};
                    background:#ffffff;
                    border-radius:6px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                '>
                    <span style='font-size:11px; color:{MS_GREY};
                                 font-weight:600; text-transform:uppercase;
                                 letter-spacing:0.5px;'>2s10s Spread</span><br>
                    <span style='font-size:26px; font-weight:700; color:{spread_color};'>
                        {spread_2s10s:+.1f} bps
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with warn_col:
            if spread_2s10s < 0:
                st.warning(
                    f"**Yield Curve Inverted** — The 2s10s spread stands at "
                    f"**{spread_2s10s:.1f} bps**. An inverted curve has historically "
                    f"been a leading indicator of recession. Monitor duration and "
                    f"credit positioning closely.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"Yield curve is **upward sloping**. "
                    f"2s10s spread: **{spread_2s10s:+.1f} bps**.",
                    icon="✅",
                )

    st.markdown("---")

    # ── 4. Historical time-series ─────────────────────────────
    st.markdown(
        f"<p style='font-weight:700; color:{MS_BLUE}; font-size:14px;'>"
        f"Historical Yield Time Series</p>",
        unsafe_allow_html=True,
    )

    tenor_options    = yields_df.columns.tolist()
    default_tenors   = [t for t in ["2Y", "10Y"] if t in tenor_options] or tenor_options[:2]
    selected_tenors  = st.multiselect(
        label="Select tenors to display:",
        options=tenor_options,
        default=default_tenors,
        key="yc_tenor_select",
    )

    if selected_tenors:
        fig_history = plot_yield_history(
            yields_df=yields_df,
            tenors=selected_tenors,
            title="US Treasury Yields — Historical",
        )
        st.plotly_chart(fig_history, use_container_width=True, key="ust_history")
    else:
        st.info("Select at least one tenor to display the historical chart.")


# Placeholder tabs (to be implemented next)
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
    Centralised data loading — called once per session / refresh.
    Returns a single `data` dict passed to all tab functions.
    """
    data: dict = {}

    with st.spinner("Loading rates data…"):
        raw_yields = fetch_treasury_yields_yf(
            start=params["start_date"],
            end=params["end_date"],
        )
        data["yields"] = validate_series(raw_yields, context="UST Yields")

    return data


def main() -> None:
    render_header()
    params = render_sidebar()

    # Force cache clear when user clicks refresh
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
