import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import feedparser
from datetime import datetime, timedelta

# ============================================================
#                      STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Morgan Stanley – Global Rates Dashboard",
    page_icon="📈",
    layout="wide",
)

# ============================================================
#                      CSS INLINE
# ============================================================
css = """
<style>
body { font-family: 'Open Sans', sans-serif; }
.header {
    background-color: #00539b;
    padding: 20px;
    border-radius: 8px;
    color: white;
    margin-bottom: 20px;
}
h1, h2, h3 { font-weight: 600; }
.stButton>button {
    background-color: #00539b;
    color: white;
    border-radius: 6px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ============================================================
#                      HEADER
# ============================================================
st.markdown(f"""
<div class='header'>
    <h1>Morgan Stanley – Global Rates & Macro Dashboard</h1>
    <h3>Interest Rates Sales Platform</h3>
    <p>{datetime.today().strftime("%A %d %B %Y")}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
#                      SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Settings")
section = st.sidebar.radio(
    "Navigation",
    ["🌍 Global Rates Monitor", "📊 Spread Analyzer",
     "🔥 Inflation & Macro", "📈 Rates Instruments", "📰 Market News"]
)

# ============================================================
#               MOCK DATA (fallback)
# ============================================================
def mock_yields(country):
    maturities = ["1M","3M","6M","1Y","2Y","5Y","10Y","30Y"]
    values = np.linspace(0.5, 3.0, len(maturities)) + np.random.randn(len(maturities)) * 0.1
    return pd.DataFrame({"Maturity": maturities, "Yield": values, "Country": country})

# ============================================================
#               FUNCTIONS: YIELD DATA
# ============================================================
def get_yield_data(countries):
    dfs = []
    for c in countries:
        try:
            df = mock_yields(c)  # replace with real APIs if available
            dfs.append(df)
        except:
            dfs.append(mock_yields(c))
    return pd.concat(dfs, ignore_index=True)

def compute_yield_changes(df):
    return df.groupby("Country")["Yield"].agg(["mean", "min", "max"])

# ============================================================
#               FUNCTIONS: SPREAD DATA
# ============================================================
def compute_spread(c1, c2):
    dates = pd.date_range(end=datetime.today(), periods=200)
    values = np.random.randn(200).cumsum()
    return pd.DataFrame({"Spread": values}, index=dates)

# ============================================================
#               FUNCTIONS: MACRO DATA
# ============================================================
def get_macro(indicators):
    data = {}
    for ind in indicators:
        dates = pd.date_range(end=datetime.today(), periods=60, freq="M")
        values = (np.sin(np.linspace(0, 6, 60)) + 2) * 2
        data[ind] = pd.DataFrame({"Date": dates, "Value": values})
    return data

# ============================================================
#               FUNCTIONS: INSTRUMENT PRICES
# ============================================================
def get_price(ticker):
    try:
        df = yf.download(ticker, period="1y")
        if df.empty:
            return None
        return df
    except:
        return None

# ============================================================
#               FUNCTIONS: RSS NEWS
# ============================================================
def get_news(keywords):
    feeds = ["https://www.reuters.com/rssFeed/financialNews",
             "https://www.ft.com/rss"]
    news = []
    for f in feeds:
        try:
            rss = feedparser.parse(f)
            for e in rss.entries[:15]:
                if any(k.lower() in e.title.lower() for k in keywords.split(",")):
                    news.append({
                        "title": e.title,
                        "link": e.link,
                        "date": e.get("published", ""),
                        "summary": e.get("summary", "")[:250]
                    })
        except:
            pass
    return news

# ============================================================
#                       PLOTTING
# ============================================================
def plot_yield_curve(df):
    fig = go.Figure()
    for c in df["Country"].unique():
        sub = df[df["Country"] == c]
        fig.add_trace(go.Scatter(
            x=sub["Maturity"], y=sub["Yield"],
            mode="lines+markers", name=c
        ))
    fig.update_layout(title="Yield Curve", template="plotly_white")
    return fig

def plot_heatmap(df):
    pivot = df.pivot_table(index="Country", columns="Maturity", values="Yield")
    return px.imshow(pivot, color_continuous_scale="RdBu_r")

def plot_spread(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Spread"]))
    fig.update_layout(title="Spread", template="plotly_white")
    return fig

def plot_macro(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Value"]))
    fig.update_layout(title="Macro Indicator", template="plotly_white")
    return fig

def plot_candles(df):
    return go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    )])

# ============================================================
#                       SECTIONS
# ============================================================

# ---------------------- 1) GLOBAL RATES ----------------------
if section == "🌍 Global Rates Monitor":
    st.subheader("🌍 Global Rates Monitor")

    countries = ["USA", "Germany", "France", "Italy", "UK", "Spain", "Japan", "Canada"]
    selected = st.multiselect("Pays :", countries, default=["USA", "Germany"])

    data = get_yield_data(selected)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_yield_curve(data), use_container_width=True)
    with col2:
        st.plotly_chart(plot_heatmap(data), use_container_width=True)

    st.write("Variations :")
    st.dataframe(compute_yield_changes(data))


# ---------------------- 2) SPREAD ANALYZER ----------------------
if section == "📊 Spread Analyzer":
    st.subheader("📊 Spread Analyzer")

    c1 = st.selectbox("Pays A", ["USA", "Germany"])
    c2 = st.selectbox("Pays B", ["France", "Italy"])

    spread = compute_spread(c1, c2)
    st.plotly_chart(plot_spread(spread), use_container_width=True)


# ---------------------- 3) INFLATION & MACRO ----------------------
if section == "🔥 Inflation & Macro":
    st.subheader("🔥 Inflation & Macro Indicators")

    selected = st.multiselect("Indicateurs", 
        ["US CPI", "Eurozone HICP", "UK CPI", "Japan CPI"], 
        default=["US CPI", "Eurozone HICP"])

    macro = get_macro(selected)

    for name, df in macro.items():
        st.markdown(f"### {name}")
        st.plotly_chart(plot_macro(df), use_container_width=True)


# ---------------------- 4) RATES INSTRUMENTS ----------------------
if section == "📈 Rates Instruments":
    st.subheader("📈 Rates Instruments – Futures, Bonds, Commodities")

    tickers = {
        "UST 10Y": "ZN=F",
        "Bund Future": "FGBL.DE",
        "Brent": "BZ=F",
        "WTI": "CL=F"
    }

    selected = st.multiselect("Instruments :", list(tickers.keys()), default=["UST 10Y"])

    for inst in selected:
        df = get_price(tickers[inst])
        if df is None:
            st.warning(f"Données indisponibles pour {inst}")
            continue
        st.markdown(f"### {inst}")
        st.plotly_chart(plot_candles(df), use_container_width=True)


# ---------------------- 5) NEWS ----------------------
if section == "📰 Market News":
    st.subheader("📰 Market News – Rates & Macro")

    keywords = st.text_input("Mots-clés :", "rates, ECB, Fed, inflation")
    news = get_news(keywords)

    for n in news:
        st.markdown(f"""
        ### {n['link']}
        *{n['date']}*  
        {n['summary']}  
        ---
        """)