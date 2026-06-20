import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="GEX Dashboard", layout="wide")
st.title("📊 SPY Structural Flow Dashboard")
st.caption("Data Engine Core Matrix")

# --- PARAMETERS ---
with st.sidebar:
    st.header("🎛️ Settings")
    symbol = st.text_input("Symbol", "SPY").upper()
    zoom = st.slider("Zoom Window %", 3, 15, 8) / 100.0
    r_rate = st.number_input("Risk Free Rate", value=0.05)
    max_d = st.number_input("Max DTE Filter", value=45)

# --- BASE PRICE FETCH ---
try:
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.get("last_price")
    if price is None or np.isnan(price):
        price = ticker.info.get("regularMarketPrice")
    if price is None or np.isnan(price):
        hist = ticker.history(period="1d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
except Exception:
    price = None

if not price or np.isnan(price):
    st.error(f"❌ Could not retrieve market data for {symbol}.")
    st.stop()

price_key = float(round(price, 2))

# --- OPTION DATA CALCULATION ---
try:
    expirations = ticker.options
    if not expirations:
        st.error("No options data available.")
        st.stop()

    compiled = []
    today = datetime.now().date()

    for exp in expirations:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_dt - today).days
        if dte < 0 or dte > max_d:
            continue

        T = max(dte, 1) / 365.0
        chain = ticker.option_chain(exp)

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            c = df[["strike", "openInterest", "impliedVolatility"]].dropna()
            mask = (c["openInterest"] > 0) & (c["impliedVolatility"] > 0)
            c = c[mask].copy()
            if c.empty:
                continue

            K = c["strike"].values
            iv = c["impliedVolatility"].values
            oi = c["openInterest"].values

            d1 = (np.log(price_key / K) + (r_rate + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            gamma = norm.pdf(d1) / (price_key * iv * np.sqrt(T))
            gamma = np.nan_to_num(gamma, nan=0.0)

            sign = 1.0 if opt_type == "call" else -1.0
            c["GEX"] = oi * gamma * 100 * (price_key**2) * sign
            compiled.append(c[["strike", "GEX"]])

    if not compiled:
        st.warning("No data matched your criteria.")
        st.stop()

    master = pd.concat(compiled, ignore_index=True)
    agg = master.groupby("strike")["GEX"].sum().reset_index()

    # --- UPGRADED TIPPING POINT DECOUPLING LOGIC ---
    agg_df_sorted = agg.sort_values("strike").copy()
    agg_df_sorted["cumulative_GEX"] = agg_df_sorted["GEX"].cumsum()

    cum_g = agg_df_sorted["cumulative_GEX"]
    sign_changes = agg_df_sorted[(cum_g * cum_g.shift(1) < 0)]

    if not sign_changes.empty:
        zero_gamma_strike = float(sign_changes["strike"].iloc[0])
    else:
        # Scan the entire matrix to locate the strike closest to structural neutral (0)
        closest_idx = agg_df_sorted["cumulative_GEX"].abs().idxmin()
        zero_gamma_strike = float(agg_df_sorted.loc[closest_idx, "strike"])

    # --- METRICS & DISPLAY ---
    total_gex = float(agg["GEX"].sum())
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Spot {symbol}", f"${price_key:.2f}")
    col2.metric("Tipping Point (Zero GEX)", f"${zero_gamma_strike:.2f}")
    col3.metric("Total Net GEX", f"${total_gex / 1e6:.2f}M")

    # --- PLOTLY PROFILE ---
    low_b = price_key * (1.0 - zoom)
    high_b = price_key * (1.0 + zoom)
    f_df = agg[(agg["strike"] >= low_b) & (agg["strike"] <= high_b)]

    fig = go.Figure()
    colors = np.where(f_df["GEX"] >= 0, "#2ecc71", "#e74c3c")
    fig.add_trace(go.Bar(x=f_df["strike"], y=f_df["GEX"], marker_color=colors))
    fig.add_vline(x=price_key, line_dash="dash", line_color="#3498db")
    fig.add_vline(x=zero_gamma_strike, line_dash="dot", line_color="#9b59b6")
    fig.update_layout(template="plotly_dark", xaxis_title="Strike", yaxis_title="Net GEX ($)")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Execution Error: {str(e)}")
