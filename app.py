import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import feedparser
from datetime import datetime, timedelta

# =====================================================================
#  STREAMLIT CONFIG + CSS
# =====================================================================

st.set_page_config(
    page_title="Morgan Stanley – Global Rates Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
body { font-family: 'Open Sans', sans-serif; }
.header {
    background-color:#00539b; padding:22px; border-radius:8px; color:white;
}
h1, h2, h3 { font-weight:600; }
.stButton>button { background-color:#00539b; color:white; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
#  HEADER
# =====================================================================

st.markdown(f"""
<div class='header'>
    <h1>Global Rates & Macro Dashboard – Morgan Stanley</h1>
    <p>{datetime.now().strftime('%A %d %B %Y')}</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 1. YFINANCE ROBUST DOWNLOADER
# =====================================================================

def download_price(ticker: str, period="1y"):
    """
    Télécharge des prix avec gestion d'erreur robuste :
    - 3 tentatives
    - fallback OHLC synthétique pro si échec complet
    """

    for attempt in range(3):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                progress=False,
                timeout=10
            )

            if isinstance(df, pd.DataFrame) and not df.empty:
                return df

        except Exception as e:
            pass  # on ignore l’erreur volontairement

    # ------------------ FALLBACK OHLC ------------------
    # On génère un prix synthétique propre (style MS demo)
    dates = pd.date_range(end=datetime.today(), periods=250)
    base = np.cumsum(np.random.randn(250)) + 100
    fallback = pd.DataFrame({
        "Open": base + np.random.randn(250),
        "High": base + np.random.rand(250),
        "Low": base - np.random.rand(250),
        "Close": base,
        "Volume": np.random.randint(10_000, 100_000, 250)
    }, index=dates)

    st.warning(f"⚠️ {ticker}: données indisponibles → fallback utilisé.")

    return fallback


# =====================================================================
# 2. MOCK YIELD CURVES (propre & scalable)
# =====================================================================

def get_yields(country: str):
    maturities = ["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"]
    base = {
        "USA": 4.9,
        "Germany": 2.3,
        "France": 2.7,
        "Italy": 4.0,
        "UK": 4.2,
        "Spain": 3.3,
        "Japan": 0.9,
        "Canada": 3.9
    }.get(country, 3.0)

    values = base + np.linspace(-0.3, 0.5, len(maturities))
    values += np.random.randn(len(maturities))*0.05

    return pd.DataFrame({
        "Country": country,
        "Maturity": maturities,
        "Yield": values
    })


# =====================================================================
# 3. MACRO (mock pro)
# =====================================================================

def get_macro(indicators):
    out = {}
    dates = pd.date_range(end=datetime.today(), periods=60, freq="M")
    for ind in indicators:
        trend = np.sin(np.linspace(0, 4, 60))*0.5 + 2
        out[ind] = pd.DataFrame({"Date": dates, "Value": trend})
    return out


# =====================================================================
# 4. RSS FEEDS PRO
# =====================================================================

def get_news(keywords):
    feeds = [
        "https://www.ft.com/rss",
        "https://www.reuters.com/rssFeed/financialNews"
    ]

    results = []
    for url in feeds:
        try:
            rss = feedparser.parse(url)
            for e in rss.entries[:20]:
                if any(k.lower() in e.title.lower() for k in keywords.split(",")):
                    results.append({
                        "title": e.title,
                        "link": e.link,
                        "date": e.get("published", ""),
                        "summary": e.get("summary", "")[:250]
                    })
        except:
            pass

    return results


# =====================================================================
# 5. PLOTS
# =====================================================================

def plot_yield_curve(df):
    fig = go.Figure()
    for c in df["Country"].unique():
        sub = df[df["Country"] == c]
        fig.add_trace(go.Scatter(
            x=sub["Maturity"], y=sub["Yield"],
            mode="lines+markers", name=c
        ))
    fig.update_layout(title="Yield Curves", template="plotly_white")
    return fig


def plot_candles(df):
    fig = go.Figure([
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"]
        )
    ])
    fig.update_layout(title="Market Prices", template="plotly_white")
    return fig


# =====================================================================
#                     DASHBOARD SECTIONS
# =====================================================================

section = st.sidebar.radio(
    "Navigation",
    ["🌍 Rates", "📊 Spreads", "🔥 Macro", "📈 Instruments", "📰 News"]
)


# ---------------------------------------------------------------------
# 🌍 1) RATES
# ---------------------------------------------------------------------

if section == "🌍 Rates":
    st.subheader("🌍 Global Rates Monitor")

    countries = ["USA","Germany","France","Italy","UK","Spain","Japan","Canada"]

    selected = st.multiselect("Choisir les pays", countries, default=["USA","Germany"])
    df = pd.concat([get_yields(c) for c in selected])

    st.plotly_chart(plot_yield_curve(df), use_container_width=True)
    st.dataframe(df)


# ---------------------------------------------------------------------
# 📊 2) SPREADS
# ---------------------------------------------------------------------

if section == "📊 Spreads":
    st.subheader("📊 Spread Analyzer – Intra/Inter Countries")

    c1 = st.selectbox("Pays A", ["USA","Germany","France","Italy"])
    c2 = st.selectbox("Pays B", ["Germany","France","Italy","Spain"])

    y1 = get_yields(c1)
    y2 = get_yields(c2)

    merged = y1.merge(y2, on="Maturity", suffixes=(f"_{c1}", f"_{c2}"))
    merged["Spread"] = merged[f"Yield_{c1}"] - merged[f"Yield_{c2}"]

    fig = go.Figure([
        go.Bar(x=merged["Maturity"], y=merged["Spread"])
    ])
    fig.update_layout(title="Yield Spread", template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(merged)


# ---------------------------------------------------------------------
# 🔥 3) MACRO
# ---------------------------------------------------------------------

if section == "🔥 Macro":
    st.subheader("🔥 Macro & Inflation Indicators")

    indicators = ["US CPI","Eurozone HICP","UK CPI","Japan CPI"]
    selected = st.multiselect("Choisir", indicators, default=["US CPI"])

    data = get_macro(selected)

    for k, df in data.items():
        fig = go.Figure([go.Scatter(x=df["Date"], y=df["Value"])])
        fig.update_layout(title=k, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# 📈 4) INSTRUMENTS
# ---------------------------------------------------------------------

if section == "📈 Instruments":
    st.subheader("📈 Fixed Income & Commodities")

    tickers = {
        "UST 10Y Future (ZN)": "ZN=F",
        "Brent": "BZ=F",
        "WTI": "CL=F",
        "Bund Future": "FGBL.DE"
    }

    selected = st.multiselect("Choisir instruments", list(tickers.keys()), default=["UST 10Y Future (ZN)"])

    for name in selected:
        df = download_price(tickers[name])
        st.markdown(f"### {name}")
        st.plotly_chart(plot_candles(df), use_container_width=True)


# ---------------------------------------------------------------------
# 📰 5) NEWS
# ---------------------------------------------------------------------

if section == "📰 News":
    st.subheader("📰 Market News – Rates / Central Banks")

    keywords = st.text_input("Mots-clés", "rates, inflation, ECB, Fed")
    news = get_news(keywords)

    for n in news:
        st.markdown(f"""
        ### [{n['title']}]({n['link']})
        **{n['date']}**  
        {n['summary']}  
        ---
        """)
