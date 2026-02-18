# ============================================================
# app.py — Morgan Stanley | Rates Sales Dashboard
# Author  : [Name]
# Version : 1.0.0
# ============================================================


# ============================================================
# BLOCK 1 — CONFIGURATION
# ============================================================

# --- Imports ------------------------------------------------
import logging
import os
from datetime import datetime, timedelta

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

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
        /* Typography, colors, table styles, metric cards */
        </style>
    """, unsafe_allow_html=True)

_inject_css()

# ============================================================
# YIELD CURVE TAB — Complete Implementation
# Functions to paste into their respective blocks in app.py
# ============================================================

# ============================================================
# BLOCK 2 — ADD TO CONSTANTS (if not already present)
# ============================================================

UST_TICKERS: dict[str, str] = {
    "3M":  "^IRX",
    "2Y":  "^UST2Y",
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

# ^IRX, ^FVX, ^TNX, ^TYX are quoted as rate * 10 → divide by 10
# ^UST2Y is quoted directly in % → no adjustment needed
UST_SCALE: dict[str, float] = {
    "3M":  0.1,
    "2Y":  1.0,
    "5Y":  0.1,
    "10Y": 0.1,
    "30Y": 0.1,
}


# ============================================================
# BLOCK 3 — DATA LAYER
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_treasury_yields_yf(start: str, end: str) -> pd.DataFrame:
    """
    Download UST yields from Yahoo Finance for the defined tenors.

    Handles per-ticker scaling:
      - ^IRX, ^FVX, ^TNX, ^TYX : raw value is rate * 10 → divide by 10
      - ^UST2Y                  : raw value already in % → no change

    Returns
    -------
    pd.DataFrame
        Daily yield data in %, columns = tenor labels (e.g. "3M", "2Y"…),
        index = DatetimeIndex, NaN rows dropped.
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
                logger.warning(f"No data returned for {ticker} ({tenor})")
                continue

            # Prefer "Close"; "Adj Close" not always available for indices
            if "Close" in raw.columns:
                series = raw["Close"].squeeze()
            else:
                logger.warning(f"Unexpected columns for {ticker}: {raw.columns.tolist()}")
                continue

            # Apply per-ticker scaling to normalise to plain %
            scale = UST_SCALE.get(tenor, 1.0)
            series = series * scale
            series.name = tenor
            frames.append(series)

        except Exception as exc:
            logger.error(f"Failed to fetch {ticker} ({tenor}): {exc}")
            continue

    if not frames:
        logger.error("All treasury yield fetches failed.")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)

    # Enforce correct tenor column order
    ordered_cols = [t for t in UST_TICKERS.keys() if t in df.columns]
    df = df[ordered_cols]

    # Drop rows where every column is NaN
    df.dropna(how="all", inplace=True)

    return df


# ============================================================
# BLOCK 4 — CALCULATION LAYER
# ============================================================

def build_yield_curve(
    yields_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extract today's curve, the curve 1 month ago, and 1 day ago
    from a yields DataFrame.

    Parameters
    ----------
    yields_df : pd.DataFrame
        Output of fetch_treasury_yields_yf().

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (curve_today, curve_1m_ago, curve_1d_ago)
        Each Series is indexed by tenor label.
        curve_1d_ago may be None if fewer than 2 rows exist.
    """
    if yields_df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    # Today = last available trading day
    curve_today = yields_df.iloc[-1].dropna()

    # 1-day ago = second-to-last row (if exists)
    if len(yields_df) >= 2:
        curve_1d = yields_df.iloc[-2].dropna()
    else:
        curve_1d = pd.Series(dtype=float)

    # 1-month ago = closest date to (last_date - 30 calendar days)
    last_date = yields_df.index[-1]
    target_1m = last_date - pd.DateOffset(days=30)
    idx_1m = yields_df.index.get_indexer([target_1m], method="nearest")[0]
    curve_1m = yields_df.iloc[idx_1m].dropna()

    # 1-week ago = closest date to (last_date - 7 calendar days)
    target_1w = last_date - pd.DateOffset(days=7)
    idx_1w = yields_df.index.get_indexer([target_1w], method="nearest")[0]
    curve_1w = yields_df.iloc[idx_1w].dropna()

    return curve_today, curve_1m, curve_1d, curve_1w


def compute_curve_change_table(yields_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the summary table shown in tab_yield_curve:
      Level | Δ1D (bps) | Δ1W (bps) | Δ1M (bps)

    Parameters
    ----------
    yields_df : pd.DataFrame
        Full daily yield DataFrame.

    Returns
    -------
    pd.DataFrame
        Rows = tenors, columns = ["Level (%)", "Δ1D (bps)", "Δ1W (bps)", "Δ1M (bps)"]
    """
    if yields_df.empty:
        return pd.DataFrame()

    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)

    common = curve_today.index
    rows = []

    for tenor in common:
        level = curve_today.get(tenor, float("nan"))

        # Changes in basis points (1 bps = 0.01 %)
        d1d = (curve_today.get(tenor, float("nan")) - curve_1d.get(tenor, float("nan"))) * 100 \
              if not curve_1d.empty else float("nan")
        d1w = (curve_today.get(tenor, float("nan")) - curve_1w.get(tenor, float("nan"))) * 100 \
              if not curve_1w.empty else float("nan")
        d1m = (curve_today.get(tenor, float("nan")) - curve_1m.get(tenor, float("nan"))) * 100 \
              if not curve_1m.empty else float("nan")

        rows.append({
            "Tenor":      tenor,
            "Level (%)":  round(level, 3),
            "Δ1D (bps)":  round(d1d, 1),
            "Δ1W (bps)":  round(d1w, 1),
            "Δ1M (bps)":  round(d1m, 1),
        })

    df_table = pd.DataFrame(rows).set_index("Tenor")
    return df_table


# ============================================================
# BLOCK 5 — VISUALIZATION LAYER
# ============================================================

def plot_yield_curve(
    curve_today: pd.Series,
    curve_1m: pd.Series,
    title: str = "US Treasury Yield Curve",
    label_today: str = "Today",
    label_1m: str = "1M Ago",
) -> go.Figure:
    """
    Institutional-grade yield curve chart.

    Two lines:
      - Today   : MS Blue, bold, markers enabled
      - 1M Ago  : MS Grey, dashed, no markers

    Parameters
    ----------
    curve_today : pd.Series
        Index = tenor labels, values = yield in %.
    curve_1m : pd.Series
        Same structure for the comparison date.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # --- 1M ago line (background reference) ---
    if not curve_1m.empty:
        common_1m = [t for t in curve_1m.index if t in curve_today.index]
        fig.add_trace(go.Scatter(
            x=common_1m,
            y=[curve_1m[t] for t in common_1m],
            mode="lines+markers",
            name=label_1m,
            line=dict(color=MS_GREY, width=1.5, dash="dot"),
            marker=dict(size=5, color=MS_GREY),
            hovertemplate="%{x}: %{y:.3f}%<extra>" + label_1m + "</extra>",
        ))

    # --- Today line (primary) ---
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Tenor",
            showgrid=True,
            gridcolor="#E8E8E8",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Yield (%)",
            showgrid=True,
            gridcolor="#E8E8E8",
            tickformat=".2f",
            ticksuffix="%",
            tickfont=dict(size=11),
        ),
        height=420,
        hovermode="x unified",
    )

    fig = apply_ms_theme(fig, title=title)
    return fig


def plot_yield_history(
    yields_df: pd.DataFrame,
    tenors: list[str],
    title: str = "Yield History — Selected Tenors",
) -> go.Figure:
    """
    Time-series chart of selected tenor yields over the full date range.

    Parameters
    ----------
    yields_df : pd.DataFrame
        Full daily yields, columns = tenor labels.
    tenors : list[str]
        Subset of tenors to display.

    Returns
    -------
    go.Figure
    """
    # Colour ramp from MS_DARK → MS_BLUE → MS_GREY for visual hierarchy
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
            line=dict(
                color=color_ramp[i % len(color_ramp)],
                width=1.8,
            ),
            hovertemplate=f"{tenor}: %{{y:.3f}}%<extra></extra>",
        ))

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
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

    fig = apply_ms_theme(fig, title=title)
    return fig


# ============================================================
# BLOCK 7 — TAB MODULE
# ============================================================

def tab_yield_curve(data: dict, params: dict) -> None:
    """
    Tab 1 — Yield Curve
    -------------------
    Sections:
      1. KPI row  — current level for each tenor
      2. Curve chart — Today vs 1M Ago
      3. Change table — Level | Δ1D | Δ1W | Δ1M in bps
      4. Historical time-series of selected tenors
      5. 2s10s inversion warning (if applicable)
    """
    yields_df = data.get("yields", pd.DataFrame())

    # ── Guard: no data ──────────────────────────────────────
    if yields_df.empty:
        render_error_banner(
            "Treasury yield data is unavailable. "
            "Check your internet connection or try refreshing."
        )
        return

    # ── 1. Build curves & change table ──────────────────────
    curve_today, curve_1m, curve_1d, curve_1w = build_yield_curve(yields_df)
    change_table = compute_curve_change_table(yields_df)

    last_date = yields_df.index[-1].strftime("%B %d, %Y")

    # ── 2. Section header ───────────────────────────────────
    st.markdown(
        f"<p style='color:{MS_GREY}; font-size:13px; margin-bottom:4px;'>"
        f"Last available trading day: <strong>{last_date}</strong></p>",
        unsafe_allow_html=True,
    )

    # ── 3. KPI metric row ───────────────────────────────────
    available_tenors = curve_today.index.tolist()
    if available_tenors:
        kpi_cols = st.columns(len(available_tenors))
        for col, tenor in zip(kpi_cols, available_tenors):
            level = curve_today.get(tenor)
            delta_1d = change_table.loc[tenor, "Δ1D (bps)"] if tenor in change_table.index else None

            with col:
                delta_str = (
                    f"{delta_1d:+.1f} bps" if delta_1d is not None and not np.isnan(delta_1d)
                    else None
                )
                st.metric(
                    label=f"{tenor} Treasury",
                    value=f"{level:.3f}%" if level is not None else "N/A",
                    delta=delta_str,
                )

    st.markdown("---")

    # ── 4. Curve chart + Change table (side by side) ────────
    col_chart, col_table = st.columns([2.2, 1.2])

    with col_chart:
        fig_curve = plot_yield_curve(
            curve_today=curve_today,
            curve_1m=curve_1m,
            title="US Treasury Yield Curve",
            label_today=f"Today ({last_date})",
            label_1m="1M Ago",
        )
        st.plotly_chart(fig_curve, use_container_width=True, key="ust_curve")

    with col_table:
        st.markdown(
            f"<p style='font-weight:600; color:{MS_BLUE}; "
            f"font-size:14px; margin-bottom:8px;'>Yield Changes</p>",
            unsafe_allow_html=True,
        )

        if not change_table.empty:
            # Style the table: colour-code Δ columns
            def _colour_bps(val: float) -> str:
                if pd.isna(val):
                    return ""
                if val > 0:
                    return f"color: {MS_RED}; font-weight:600;"
                if val < 0:
                    return f"color: {MS_GREEN}; font-weight:600;"
                return ""

            styled = (
                change_table.style
                .format({
                    "Level (%)":  "{:.3f}%",
                    "Δ1D (bps)":  "{:+.1f}",
                    "Δ1W (bps)":  "{:+.1f}",
                    "Δ1M (bps)":  "{:+.1f}",
                })
                .applymap(_colour_bps, subset=["Δ1D (bps)", "Δ1W (bps)", "Δ1M (bps)"])
                .set_properties(**{"text-align": "center"})
                .set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", MS_BLUE),
                        ("color", "white"),
                        ("font-weight", "600"),
                        ("text-align", "center"),
                        ("font-size", "12px"),
                    ]},
                    {"selector": "td", "props": [
                        ("font-size", "13px"),
                        ("padding", "6px 10px"),
                    ]},
                    {"selector": ".index_name", "props": [("font-weight", "bold")]},
                ])
            )
            st.write(styled)
        else:
            st.info("Change data not available.")

    st.markdown("---")

    # ── 5. 2s10s inversion check & warning ──────────────────
    s2  = curve_today.get("2Y")
    s10 = curve_today.get("10Y")

    if s2 is not None and s10 is not None:
        spread_2s10s = (s10 - s2) * 100  # in bps

        spread_col, _ = st.columns([1, 3])
        with spread_col:
            spread_color = MS_RED if spread_2s10s < 0 else MS_GREEN
            st.markdown(
                f"<div style='padding:12px 16px; border-left: 4px solid {spread_color}; "
                f"background:#fff; border-radius:4px;'>"
                f"<span style='font-size:12px; color:{MS_GREY};'>2s10s Spread</span><br>"
                f"<span style='font-size:22px; font-weight:700; color:{spread_color};'>"
                f"{spread_2s10s:+.1f} bps</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if spread_2s10s < 0:
            st.warning(
                f"⚠️ **Yield Curve Inverted** — The 2s10s spread is currently "
                f"**{spread_2s10s:.1f} bps**. An inverted curve has historically "
                f"preceded economic recessions. Monitor credit and duration positioning closely.",
                icon="📉",
            )
        else:
            st.success(
                f"✅ Yield curve is **upward sloping** — 2s10s spread: {spread_2s10s:+.1f} bps.",
                icon="📈",
            )

    st.markdown("---")

    # ── 6. Historical time-series ───────────────────────────
    st.markdown(
        f"<p style='font-weight:600; color:{MS_BLUE}; font-size:14px;'>"
        f"Historical Yield Time Series</p>",
        unsafe_allow_html=True,
    )

    tenor_options = yields_df.columns.tolist()
    selected_tenors = st.multiselect(
        label="Select tenors to display:",
        options=tenor_options,
        default=["2Y", "10Y"] if "2Y" in tenor_options and "10Y" in tenor_options
                else tenor_options[:2],
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
        st.info("Select at least one tenor to display historical yields.")

